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
use stage_vol::kernel::gobo::GOBO_NAMES;
use stage_vol::kernel::params::{control_schema, StageVolParams, ViewMode, MAX_STEPS};

const PREVIEW_W: u32 = 960;
const PREVIEW_H: u32 = 540;
/// 预览图横向拖动对应的舞台米数/像素（舞台宽约 12 m）。
const DRAG_METERS_PER_PX: f32 = 12.0 / PREVIEW_W as f32;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
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

        egui::Panel::right("controls")
            .default_size(300.0)
            .show(ui, |ui| {
                ui.heading("舞台体积光 · 材质预览");
                ui.separator();

                // 视图模式
                egui::ComboBox::from_label("视图模式")
                    .selected_text(self.params.view_mode.label())
                    .show_ui(ui, |ui| {
                        for mode in ViewMode::ALL {
                            ui.selectable_value(&mut self.params.view_mode, mode, mode.label());
                        }
                    });

                ui.add_space(4.0);
                ui.label("内核控件（范围由内核约束）:");
                // 控件由内核 control schema 驱动；非法值被内核拒绝并反馈。
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

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("帧耗时 {:.1} ms", self.last_frame_ms));
                ui.separator();
                ui.label(format!("设备: {}", self.gpu.device_label));
                ui.separator();
                ui.label(&self.status);
            });
            for e in &self.gpu_error_log {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("GPU: {e}"));
            }
        });

        ctx.request_repaint();
    }
}
