//! stage-vol-ui — 舞台体积光材质预览窗口。
//!
//! 由舞台体积光内核驱动控件（control schema 来自内核，非法值被内核拒绝并
//! 反馈到状态栏），实时调整直接写入 GPU uniform。预览区可拖动挡板；
//! 支持参数保存、场景重开、设备资源恢复与参考帧输出。
//!
//! 启动: cargo run --release --bin stage-vol-ui
//! （需要显示服务器与 GPU；无头环境请用 stage-vol-render）

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use eframe::egui;
use stage_vol::cpu::CpuRenderer;
use stage_vol::gpu::GpuRenderer;
use stage_vol::kernel::bench::{run_benchmark, validate_benchmark, BenchmarkReport};
use stage_vol::kernel::gobo::GOBO_NAMES;
use stage_vol::kernel::params::{control_schema, StageVolParams, ViewMode, MAX_STEPS};

const PREVIEW_W: u32 = 960;
const PREVIEW_H: u32 = 540;
/// 预览图横向拖动对应的舞台米数/像素（舞台宽约 12 m）。
const DRAG_METERS_PER_PX: f32 = 12.0 / PREVIEW_W as f32;
const BENCH_MAX_ERROR: f64 = 0.03;
const BENCH_REGRESSION_MARGIN: f64 = 0.005;
/// The benchmark runs the CPU kernel with a converged MAX_STEPS reference per
/// level; at the preview size that would take minutes. A smaller frame still
/// runs the full skip path on every pixel, which is what the evidence checks.
const BENCH_W: u32 = 320;
const BENCH_H: u32 = 180;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 640.0])
            .with_title("舞台体积光 · 材质预览"),
        ..Default::default()
    };
    eframe::run_native(
        "舞台体积光 · 材质预览",
        options,
        Box::new(|cc| Ok(Box::new(StageVolApp::new(cc)?))),
    )
}

struct StageVolApp {
    params: StageVolParams,
    gpu: GpuRenderer,
    render_state: egui_wgpu::RenderState,
    tex_id: egui::TextureId,
    scene_path: String,
    status: String,
    gpu_error_log: Vec<String>,
    last_frame_ms: f64,
    time: f32,
    ref_rx: Option<Receiver<String>>,
    bench_rx: Option<Receiver<Result<BenchmarkReport, String>>>,
    bench_summary: Option<String>,
}

impl StageVolApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Result<StageVolApp, String> {
        let state = cc
            .wgpu_render_state
            .clone()
            .ok_or_else(|| "eframe 未启用 wgpu 渲染状态".to_string())?;
        let device_label = state.adapter.get_info().name;
        let gpu = GpuRenderer::init_shared(
            state.device.clone(),
            state.queue.clone(),
            PREVIEW_W,
            PREVIEW_H,
            device_label,
        )
        .map_err(|e| format!("GPU 初始化失败: {e}"))?;
        let tex_id = state.renderer.write().register_native_texture(
            &state.device,
            gpu.composite_view(),
            wgpu::FilterMode::Linear,
        );
        // 尝试重开上次的场景参数。
        let scene_path = "scenes/ui_session.json".to_string();
        let params = StageVolParams::load_from(PathBuf::from(&scene_path).as_path())
            .unwrap_or_default();
        Ok(StageVolApp {
            params,
            gpu,
            render_state: state,
            tex_id,
            scene_path,
            status: "就绪".into(),
            gpu_error_log: Vec::new(),
            last_frame_ms: 0.0,
            time: 0.0,
            ref_rx: None,
            bench_rx: None,
            bench_summary: None,
        })
    }

    fn re_register_texture(&mut self) {
        self.render_state.renderer.write().free_texture(&self.tex_id);
        self.tex_id = self.render_state.renderer.write().register_native_texture(
            &self.render_state.device,
            self.gpu.composite_view(),
            wgpu::FilterMode::Linear,
        );
    }

    fn save_params(&mut self) {
        let path = PathBuf::from(&self.scene_path);
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        self.status = match self.params.save_to(&path) {
            Ok(()) => format!("参数已保存 → {}", path.display()),
            Err(e) => format!("保存失败: {e}"),
        };
    }

    fn reload_scene(&mut self) {
        let path = PathBuf::from(&self.scene_path);
        self.status = match StageVolParams::load_from(&path) {
            Ok(p) => {
                self.params = p;
                format!("场景已重开 ← {}", path.display())
            }
            Err(e) => format!("场景重开失败: {e}"),
        };
    }

    fn recover_device(&mut self) {
        self.status = match self.gpu.recover() {
            Ok(()) => {
                self.re_register_texture();
                "设备资源已恢复（管线/纹理已按内核参数重建）".into()
            }
            Err(e) => format!("设备恢复失败: {e}"),
        };
    }

    fn render_reference(&mut self) {
        if self.ref_rx.is_some() {
            self.status = "参考帧渲染中…".into();
            return;
        }
        let mut params = self.params.clone();
        params.steps = MAX_STEPS;
        params.view_mode = ViewMode::Composite;
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let frame = CpuRenderer::new(PREVIEW_W, PREVIEW_H).render(&params, 0.0);
            let path = PathBuf::from("out/ui_reference.png");
            let msg = match frame.save_png(&path) {
                Ok(()) => format!("参考帧 (N={MAX_STEPS}) 已保存 → {}", path.display()),
                Err(e) => format!("参考帧保存失败: {e}"),
            };
            let _ = tx.send(msg);
        });
        self.ref_rx = Some(rx);
        self.status = format!("参考帧渲染中 (CPU, N={MAX_STEPS})…");
    }

    fn run_bench(&mut self) {
        if self.bench_rx.is_some() {
            self.status = "光束成本基准运行中…".into();
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let report = run_benchmark(BENCH_W, BENCH_H, "gpu-cpu-kernel");
            let verdict = validate_benchmark(&report, BENCH_MAX_ERROR, BENCH_REGRESSION_MARGIN)
                .and_then(|()| {
                    if report.speed_reported() {
                        Ok(())
                    } else {
                        Err("基准缺少执行证据，拒绝报告速度".to_string())
                    }
                });
            let _ = tx.send(verdict.map(|()| report).map_err(|e| e.to_string()));
        });
        self.bench_rx = Some(rx);
        self.status = "光束成本基准运行中（CPU 内核，含收敛参照）…".into();
    }
}

impl eframe::App for StageVolApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // 推进动画时间并渲染一帧（实时参数经 uniform 上传 GPU）。
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        self.time += dt;
        let t0 = Instant::now();
        if let Err(e) = self.gpu.render_to_textures(&self.params, self.time) {
            self.status = format!("渲染失败: {e}（可尝试“恢复设备资源”）");
        }
        self.last_frame_ms = t0.elapsed().as_secs_f64() * 1000.0;
        for e in self.gpu.take_errors() {
            self.gpu_error_log.push(e);
        }
        self.gpu_error_log.truncate(4);
        if let Some(rx) = &self.ref_rx {
            if let Ok(msg) = rx.try_recv() {
                self.status = msg;
                self.ref_rx = None;
            }
        }
        if let Some(rx) = &self.bench_rx {
            if let Ok(result) = rx.try_recv() {
                self.bench_rx = None;
                match result {
                    Ok(report) => {
                        let worst = report
                            .levels
                            .iter()
                            .map(|l| l.relative_skip_error)
                            .fold(0.0f64, f64::max);
                        let min_speed = report
                            .levels
                            .iter()
                            .map(|l| l.speedup)
                            .fold(f64::INFINITY, f64::min);
                        let saved = report
                            .levels
                            .first()
                            .map(|l| l.skip_cost.sample_saving() * 100.0)
                            .unwrap_or(0.0);
                        self.bench_summary = Some(format!(
                            "基准通过：最差相对亮度误差 {:.4}，样本节省 {:.1}%，最小速度 {:.2}×（{} 档）",
                            worst,
                            saved,
                            min_speed,
                            report.levels.len()
                        ));
                        self.status = "光束成本基准通过".into();
                    }
                    Err(e) => {
                        self.bench_summary = Some(format!("基准未通过: {e}"));
                        self.status = format!("基准未通过: {e}");
                    }
                }
            }
        }

        egui::Panel::right("controls")
            .default_size(300.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| self.controls_ui(ui));
            });

        // Declare the status panel before the central panel so it always gets
        // its space; a central panel declared first would consume the rest.
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("帧耗时 {:.1} ms", self.last_frame_ms));
                ui.separator();
                ui.label(format!("设备: {}", self.gpu.device_label));
                ui.separator();
                ui.label(&self.status);
            });
            if let Some(summary) = &self.bench_summary {
                if summary.starts_with("基准未通过") {
                    ui.colored_label(egui::Color32::LIGHT_RED, summary);
                } else {
                    ui.label(summary);
                }
            }
            // Any GPU validation/uncaptured error is surfaced here instead of
            // being swallowed; the preview keeps running so the stage stays
            // inspectable.
            for e in &self.gpu_error_log {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("GPU: {e}"));
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "灯束遮挡或积分画面可能异常：请降低参数或点击“恢复设备资源”。",
                );
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            let avail = ui.available_size();
            let scale = (avail.x / PREVIEW_W as f32).min(avail.y / PREVIEW_H as f32).max(0.1);
            let size = egui::vec2(PREVIEW_W as f32 * scale, PREVIEW_H as f32 * scale);
            let img = egui::Image::new((self.tex_id, size)).sense(egui::Sense::drag());
            let resp = ui.add(img);
            if resp.dragged() {
                let dx = resp.drag_delta().x * DRAG_METERS_PER_PX / scale;
                let new_x = self.params.occluder_x + dx;
                if let Err(e) = self.params.set_control("occluder_x", new_x) {
                    self.status = format!("内核拒绝: {e}");
                }
            }
        });

        ctx.request_repaint();
    }
}

impl StageVolApp {
    fn controls_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("舞台体积光 · 材质预览");
        ui.separator();

        egui::ComboBox::from_label("视图模式")
            .selected_text(self.params.view_mode.label())
            .show_ui(ui, |ui| {
                for mode in ViewMode::ALL {
                    ui.selectable_value(&mut self.params.view_mode, mode, mode.label());
                }
            });

        ui.add_space(4.0);
        ui.label("内核控件（范围由内核约束）:");
        // Controls come from the kernel schema; rejected values are surfaced.
        for c in control_schema() {
            let Some(mut v) = self.params.get_control(c.id) else { continue };
            let resp = ui.add(
                egui::Slider::new(&mut v, c.min..=c.max)
                    .text(c.label)
                    .step_by(c.step as f64),
            );
            if resp.changed() {
                if let Err(e) = self.params.set_control(c.id, v) {
                    self.status = format!("内核拒绝: {e}");
                }
            }
        }

        ui.separator();
        ui.label("聚光灯 0:");
        let mut rgb = self.params.lights[0].color;
        if egui::color_picker::color_edit_button_rgb(ui, &mut rgb).changed() {
            self.params.lights[0].color = rgb;
        }
        let mut gobo = self.params.lights[0].gobo_id as usize;
        egui::ComboBox::from_label("图案遮片")
            .selected_text(GOBO_NAMES[gobo.min(GOBO_NAMES.len() - 1)])
            .show_ui(ui, |ui| {
                for (i, name) in GOBO_NAMES.iter().enumerate() {
                    ui.selectable_value(&mut gobo, i, *name);
                }
            });
        self.params.lights[0].gobo_id = gobo as u32;

        ui.separator();
        ui.checkbox(&mut self.params.lights[1].enabled, "启用聚光灯 1");
        ui.checkbox(&mut self.params.occluder_visible, "显示挡板");
        ui.checkbox(&mut self.params.occluder_animate, "挡板动画");
        ui.label("提示：在预览图上左右拖动可移动挡板。");

        ui.separator();
        ui.checkbox(&mut self.params.skip_empty_regions, "空区跳过（保守界）");
        ui.label("关闭后使用均匀步进参照，便于对照灯束遮挡与积分。");

        ui.separator();
        ui.label("参数持久化:");
        ui.horizontal(|ui| {
            if ui.button("保存参数").clicked() {
                self.save_params();
            }
            if ui.button("场景重开").clicked() {
                self.reload_scene();
            }
        });
        ui.text_edit_singleline(&mut self.scene_path);
        ui.horizontal(|ui| {
            if ui.button("渲染参考帧").clicked() {
                self.render_reference();
            }
            if ui.button("恢复设备资源").clicked() {
                self.recover_device();
            }
        });
        ui.horizontal(|ui| {
            if ui.button("运行光束成本基准").clicked() {
                self.run_bench();
            }
        });
        if let Some(summary) = &self.bench_summary {
            ui.label(summary);
        }
    }
}
