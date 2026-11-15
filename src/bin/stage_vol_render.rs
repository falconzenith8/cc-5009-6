//! stage-vol-render — headless renderer for the stage volumetric previewer.
//!
//! Runs the test scenes (clear fog, full occlusion, step-count convergence,
//! moving occluder), writes reference high-sample frames and a timing
//! report, and exercises parameter save/reload and device recovery.
//!
//! Examples:
//!   stage-vol-render test                      # all test scenes → out/
//!   stage-vol-render reference -o out/ref.png  # 512-step reference frame
//!   stage-vol-render frame --scene scenes/default.json --view volume
//!   stage-vol-render recover-test              # device resource recovery
//!   stage-vol-render save-scenes               # write scenes/*.json presets
//!   stage-vol-render model-notes               # model boundary statement

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use stage_vol::cpu::{CpuRenderer, Frame};
use stage_vol::gpu::GpuRenderer;
use stage_vol::kernel::bench::{run_benchmark, validate_benchmark};
use stage_vol::kernel::params::{StageVolParams, ViewMode, MAX_STEPS};
use stage_vol::kernel::report::{CheckResult, FrameTimer, FrameTiming, Report, SceneReport};

#[derive(Parser)]
#[command(name = "stage-vol-render", about = "舞台体积光 · 无头渲染与测试")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// Render backend: auto tries GPU and falls back to the CPU reference.
    #[arg(long, default_value = "auto", global = true)]
    backend: BackendArg,
    #[arg(long, default_value = "640", global = true)]
    width: u32,
    #[arg(long, default_value = "360", global = true)]
    height: u32,
    /// Output directory for frames and report.json.
    #[arg(long, default_value = "out", global = true)]
    out: PathBuf,
}

#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run all test scenes and write out/report.json.
    Test,
    /// Render one high-sample reference frame (default: max steps).
    Reference {
        #[arg(long)]
        steps: Option<u32>,
        #[arg(long)]
        scene: Option<PathBuf>,
        #[arg(long, default_value = "composite")]
        view: ViewArg,
        #[arg(long, default_value = "out/reference.png")]
        output: PathBuf,
    },
    /// Render a single frame from a saved scene file.
    Frame {
        #[arg(long)]
        scene: PathBuf,
        #[arg(long, default_value = "composite")]
        view: ViewArg,
        #[arg(long, default_value = "0")]
        time: f32,
        #[arg(long, default_value = "out/frame.png")]
        output: PathBuf,
    },
    /// Drop and recreate GPU device/resources from saved params; outputs
    /// before/after frames must be identical.
    RecoverTest,
    /// Write the bundled scene presets to scenes/*.json.
    SaveScenes,
    /// Print the model boundary statement.
    ModelNotes,
    /// Beam cost benchmark: uniform reference vs empty-region skipping,
    /// staged from full frame through sparse beam to a density discontinuity.
    Bench {
        /// Error tolerance for the benchmark gate (relative luminance).
        #[arg(long, default_value = "0.02")]
        max_error: f64,
        /// Write the benchmark report here.
        #[arg(long, default_value = "out/bench.json")]
        output: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum ViewArg {
    Composite,
    Volume,
    Shadow,
    Opaque,
}

impl ViewArg {
    fn mode(self) -> ViewMode {
        match self {
            ViewArg::Composite => ViewMode::Composite,
            ViewArg::Volume => ViewMode::VolumeOnly,
            ViewArg::Shadow => ViewMode::Shadow,
            ViewArg::Opaque => ViewMode::OpaqueOnly,
        }
    }
}

enum Renderer {
    Cpu(CpuRenderer),
    Gpu(Box<GpuRenderer>),
}

impl Renderer {
    fn render(&mut self, params: &StageVolParams, time: f32) -> Result<Frame> {
        match self {
            Renderer::Cpu(r) => Ok(r.render(params, time)),
            Renderer::Gpu(r) => r.render(params, time),
        }
    }
    fn name(&self) -> &str {
        match self {
            Renderer::Cpu(r) => r.backend_name(),
            Renderer::Gpu(r) => r.backend_name(),
        }
    }
    fn device_label(&self) -> String {
        match self {
            Renderer::Cpu(_) => "cpu-reference".into(),
            Renderer::Gpu(r) => r.device_label.clone(),
        }
    }
    /// Drain GPU validation/uncaptured errors (failure feedback channel).
    fn take_gpu_errors(&self) -> Vec<String> {
        match self {
            Renderer::Cpu(_) => Vec::new(),
            Renderer::Gpu(r) => r.take_errors(),
        }
    }
}

fn make_renderer(backend: BackendArg, w: u32, h: u32) -> Renderer {
    match backend {
        BackendArg::Cpu => Renderer::Cpu(CpuRenderer::new(w, h)),
        BackendArg::Gpu => Renderer::Gpu(Box::new(
            pollster::block_on(GpuRenderer::init_headless(w, h))
                .expect("GPU 后端初始化失败"),
        )),
        BackendArg::Auto => match pollster::block_on(GpuRenderer::init_headless(w, h)) {
            Ok(r) => {
                eprintln!("[backend] GPU 可用: {}", r.device_label);
                Renderer::Gpu(Box::new(r))
            }
            Err(e) => {
                eprintln!("[backend] 无 GPU ({e}); 使用 CPU 参考渲染器");
                Renderer::Cpu(CpuRenderer::new(w, h))
            }
        },
    }
}

fn save_png(frame: &Frame, path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let img: image::RgbaImage =
        image::ImageBuffer::from_raw(frame.width, frame.height, frame.rgba.clone())
            .context("构造图像失败")?;
    img.save(path).map_err(|e| anyhow::anyhow!("写 {} 失败: {e}", path.display()))
}

fn load_or_default(scene: &Option<PathBuf>) -> Result<StageVolParams> {
    match scene {
        Some(p) => StageVolParams::load_from(p).map_err(anyhow::Error::msg),
        None => Ok(StageVolParams::default()),
    }
}

// --- test scenes ---------------------------------------------------------

fn scene_clear() -> StageVolParams {
    let mut p = StageVolParams::default();
    p.fog_sigma = 0.0;
    p
}

fn scene_occluded() -> StageVolParams {
    // Light placed behind an oversized panel, aiming at the camera: the
    // entire beam is blocked, so the volumetric term must vanish.
    let mut p = StageVolParams::default();
    p.lights[0].position = [0.0, 1.2, -2.0];
    p.lights[0].target = [0.0, 1.4, 6.0];
    p.lights[0].range = 12.0;
    p.occluder_scale = 4.0;
    p
}

fn scene_moving() -> StageVolParams {
    let mut p = StageVolParams::default();
    p.occluder_animate = true;
    p
}

fn run_test_suite(r: &mut Renderer, out: &Path) -> Result<Report> {
    let mut report = Report {
        backend: r.name().into(),
        device: r.device_label(),
        scenes: Vec::new(),
    };
    std::fs::create_dir_all(out)?;

    // 1. Clear fog: σ=0 → volume term exactly zero, transmittance one.
    {
        let mut params = scene_clear();
        params.view_mode = ViewMode::VolumeOnly;
        let t = FrameTimer::begin();
        let frame = r.render(&params, 0.0)?;
        let ms = t.finish_ms();
        let mean = frame.mean_volume_luminance();
        let max_t = frame.transmittance.iter().fold(0.0f32, |a, &b| a.max(1.0 - b));
        let png = out.join("clear_volume.png");
        save_png(&frame, &png)?;
        params.view_mode = ViewMode::Composite;
        let frame2 = r.render(&params, 0.0)?;
        save_png(&frame2, &out.join("clear_composite.png"))?;
        report.scenes.push(SceneReport {
            name: "clear".into(),
            description: "无雾 σ=0：体积贡献必须为零、透过率为 1".into(),
            frames: vec![FrameTiming {
                scene: "clear".into(),
                frame: 0,
                steps: params.steps,
                backend: r.name().into(),
                cpu_ms: ms,
            }],
            outputs: vec![png.display().to_string()],
            checks: vec![
                CheckResult {
                    name: "体积亮度为零".into(),
                    passed: mean == 0.0,
                    detail: format!("mean volume luminance = {mean}"),
                },
                CheckResult {
                    name: "透过率为 1".into(),
                    passed: max_t == 0.0,
                    detail: format!("max |1−T| = {max_t}"),
                },
            ],
        });
    }

    // 2. Full occlusion: beam entirely blocked → volume term ≈ 0.
    {
        let mut params = scene_occluded();
        params.view_mode = ViewMode::VolumeOnly;
        let t = FrameTimer::begin();
        let frame = r.render(&params, 0.0)?;
        let ms = t.finish_ms();
        let mean = frame.mean_volume_luminance();
        let max_lum = frame
            .volume
            .iter()
            .map(|v| (v[0] + v[1] + v[2]) / 3.0)
            .fold(0.0f32, f32::max);
        let png = out.join("occluded_volume.png");
        save_png(&frame, &png)?;
        params.view_mode = ViewMode::Composite;
        save_png(&r.render(&params, 0.0)?, &out.join("occluded_composite.png"))?;
        report.scenes.push(SceneReport {
            name: "occluded".into(),
            description: "全遮挡：放大挡板完全挡住光束，体积内光照必须消失".into(),
            frames: vec![FrameTiming {
                scene: "occluded".into(),
                frame: 0,
                steps: params.steps,
                backend: r.name().into(),
                cpu_ms: ms,
            }],
            outputs: vec![png.display().to_string()],
            checks: vec![CheckResult {
                name: "遮挡后体积亮度≈0".into(),
                passed: max_lum < 1e-4,
                detail: format!("mean={mean:.6}, max={max_lum:.6}"),
            }],
        });
    }

    // 3. Convergence: brightness must stabilize as steps grow, not scale
    //    with the sample count.
    {
        let mut params = StageVolParams::default();
        params.view_mode = ViewMode::VolumeOnly;
        let step_counts = [8u32, 16, 32, 64, 128, 256];
        let mut means = Vec::new();
        let mut timings = Vec::new();
        let mut outputs = Vec::new();
        for &n in &step_counts {
            params.steps = n;
            let t = FrameTimer::begin();
            let frame = r.render(&params, 0.0)?;
            let ms = t.finish_ms();
            assert!(frame.all_finite(), "convergence: 非有限值 @ steps={n}");
            means.push(frame.mean_volume_luminance());
            timings.push(FrameTiming {
                scene: "convergence".into(),
                frame: n,
                steps: n,
                backend: r.name().into(),
                cpu_ms: ms,
            });
            if n == 8 || n == 64 || n == 256 {
                let png = out.join(format!("convergence_{n}.png"));
                save_png(&frame, &png)?;
                outputs.push(png.display().to_string());
            }
        }
        let l_hi = means[means.len() - 1];
        let l_lo = means[0];
        let l_mid = means[3]; // 64
        let ratio = if l_lo > 0.0 { l_hi / l_lo } else { f64::INFINITY };
        let drift = if l_hi > 0.0 { (l_hi - l_mid).abs() / l_hi } else { 0.0 };
        report.scenes.push(SceneReport {
            name: "convergence".into(),
            description: "步数收敛：结果应收敛而非随样本数线性变亮".into(),
            frames: timings,
            outputs,
            checks: vec![
                CheckResult {
                    name: "亮度不随步数线性增长".into(),
                    passed: ratio < 2.0,
                    detail: format!("L(256)/L(8) = {ratio:.3} (means={means:.4?})"),
                },
                CheckResult {
                    name: "高步数收敛".into(),
                    passed: drift < 0.15,
                    detail: format!("|L(256)−L(64)|/L(256) = {drift:.4}"),
                },
            ],
        });
    }

    // 4. Moving occluder: animation frames stay finite and the occluder
    //    actually modulates the in-volume light.
    {
        let params = scene_moving();
        let mut timings = Vec::new();
        let mut outputs = Vec::new();
        let mut means = Vec::new();
        for f in 0..8u32 {
            let time = f as f32 * 0.33;
            let t = FrameTimer::begin();
            let frame = r.render(&params, time)?;
            let ms = t.finish_ms();
            if !frame.all_finite() {
                anyhow::bail!("moving: 第 {f} 帧出现非有限值");
            }
            means.push(frame.mean_volume_luminance());
            timings.push(FrameTiming {
                scene: "moving".into(),
                frame: f,
                steps: params.steps,
                backend: r.name().into(),
                cpu_ms: ms,
            });
            if f == 0 || f == 3 || f == 6 {
                let png = out.join(format!("moving_{f}.png"));
                save_png(&frame, &png)?;
                outputs.push(png.display().to_string());
            }
        }
        let min = means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = means.iter().cloned().fold(0.0f64, f64::max);
        report.scenes.push(SceneReport {
            name: "moving".into(),
            description: "移动挡板：8 帧动画，体积光随挡板位置变化且保持有限".into(),
            frames: timings,
            outputs,
            checks: vec![CheckResult {
                name: "挡板移动调制体积光".into(),
                passed: max > 0.0 && (max - min) / max > 0.05,
                detail: format!("mean range = [{min:.4}, {max:.4}]"),
            }],
        });
    }

    // 5. Reference frame at max steps (all view modes).
    {
        let mut params = StageVolParams::default();
        params.steps = MAX_STEPS;
        let mut timings = Vec::new();
        let mut outputs = Vec::new();
        for mode in ViewMode::ALL {
            params.view_mode = mode;
            let t = FrameTimer::begin();
            let frame = r.render(&params, 0.0)?;
            let ms = t.finish_ms();
            let png = out.join(format!("reference_{}_{}.png", mode.index(), mode_name(mode)));
            save_png(&frame, &png)?;
            outputs.push(png.display().to_string());
            timings.push(FrameTiming {
                scene: "reference".into(),
                frame: mode.index(),
                steps: MAX_STEPS,
                backend: r.name().into(),
                cpu_ms: ms,
            });
        }
        report.scenes.push(SceneReport {
            name: "reference".into(),
            description: format!("参考高采样帧 (N={MAX_STEPS})，四种视图"),
            frames: timings,
            outputs,
            checks: vec![],
        });
    }

    let gpu_errors = r.take_gpu_errors();
    if !gpu_errors.is_empty() {
        let detail = gpu_errors.join(" | ");
        for scene in &mut report.scenes {
            scene.checks.push(CheckResult {
                name: "GPU 无错误反馈".into(),
                passed: false,
                detail: detail.chars().take(400).collect(),
            });
        }
    }
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(out.join("report.json"), &json)?;
    Ok(report)
}

fn mode_name(mode: ViewMode) -> &'static str {
    match mode {
        ViewMode::Composite => "composite",
        ViewMode::VolumeOnly => "volume",
        ViewMode::Shadow => "shadow",
        ViewMode::OpaqueOnly => "opaque",
    }
}

fn run_recover_test(backend: BackendArg, w: u32, h: u32, out: &Path) -> Result<()> {
    if matches!(backend, BackendArg::Cpu) {
        println!("recover-test: CPU 后端无设备可恢复，跳过");
        return Ok(());
    }
    let mut gpu = match pollster::block_on(GpuRenderer::init_headless(w, h)) {
        Ok(g) => g,
        Err(e) => {
            println!("recover-test: 无 GPU ({e})，跳过");
            return Ok(());
        }
    };
    let params = StageVolParams::default();
    // 参数保存 → 场景重开：从磁盘重新加载的参数必须复现同一帧。
    std::fs::create_dir_all(out)?;
    let saved = out.join("recover_scene.json");
    params.save_to(&saved)?;
    let reloaded = StageVolParams::load_from(&saved).map_err(anyhow::Error::msg)?;

    let before = gpu.render(&reloaded, 0.0)?;
    gpu.recover()?; // 设备资源恢复
    let after = gpu.render(&reloaded, 0.0)?;

    let identical = before.rgba == after.rgba;
    save_png(&before, &out.join("recover_before.png"))?;
    save_png(&after, &out.join("recover_after.png"))?;
    println!(
        "recover-test: 设备重建后帧一致 = {identical} ({} bytes)",
        before.rgba.len()
    );
    if !identical {
        anyhow::bail!("设备恢复后渲染结果不一致");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.cmd {
        Cmd::ModelNotes => {
            println!("{}", stage_vol::model_notes());
        }
        Cmd::SaveScenes => {
            let dir = Path::new("scenes");
            std::fs::create_dir_all(dir)?;
            StageVolParams::default().save_to(&dir.join("default.json"))?;
            scene_clear().save_to(&dir.join("clear.json"))?;
            scene_occluded().save_to(&dir.join("occluded.json"))?;
            scene_moving().save_to(&dir.join("moving.json"))?;
            println!("已写入 scenes/{{default,clear,occluded,moving}}.json");
        }
        Cmd::Test => {
            let mut r = make_renderer(cli.backend, cli.width, cli.height);
            let report = run_test_suite(&mut r, &cli.out)?;
            let mut failed = 0;
            println!("\n=== 测试报告 (backend={}, device={}) ===", report.backend, report.device);
            for scene in &report.scenes {
                let mean_ms = Report::mean_ms(scene);
                println!("场景 {:<12} 平均帧耗时 {mean_ms:8.2} ms", scene.name);
                for c in &scene.checks {
                    println!("  [{}] {} — {}", if c.passed { "PASS" } else { "FAIL" }, c.name, c.detail);
                    if !c.passed {
                        failed += 1;
                    }
                }
            }
            println!("报告与帧输出见 {}", cli.out.display());
            if failed > 0 {
                anyhow::bail!("{failed} 项检查未通过");
            }
        }
        Cmd::Reference { steps, scene, view, output } => {
            let mut params = load_or_default(scene)?;
            params.steps = steps.unwrap_or(MAX_STEPS).min(MAX_STEPS);
            params.view_mode = view.mode();
            let mut r = make_renderer(cli.backend, cli.width, cli.height);
            let t = FrameTimer::begin();
            let frame = r.render(&params, 0.0)?;
            let ms = t.finish_ms();
            save_png(&frame, output)?;
            println!(
                "参考帧 N={} view={:?} → {} ({:.2} ms, backend={})",
                params.steps,
                params.view_mode,
                output.display(),
                ms,
                r.name()
            );
        }
        Cmd::Frame { scene, view, time, output } => {
            let mut params = StageVolParams::load_from(scene).map_err(anyhow::Error::msg)?;
            params.view_mode = view.mode();
            let mut r = make_renderer(cli.backend, cli.width, cli.height);
            let t = FrameTimer::begin();
            let frame = r.render(&params, *time)?;
            let ms = t.finish_ms();
            save_png(&frame, output)?;
            println!("帧 → {} ({:.2} ms, backend={})", output.display(), ms, r.name());
        }
        Cmd::RecoverTest => {
            run_recover_test(cli.backend, cli.width, cli.height, &cli.out)?;
        }
        Cmd::Bench { max_error, output } => {
            let report = run_benchmark(cli.width, cli.height, "cpu-reference");
            // The benchmark measures the CPU kernel directly; the GPU path
            // shares the same bounds but is validated separately by `test`.
            const REGRESSION_MARGIN: f64 = 0.005;
            if let Err(e) = validate_benchmark(&report, *max_error, REGRESSION_MARGIN) {
                let _ = std::fs::write(
                    output,
                    serde_json::to_string_pretty(&report).unwrap_or_default(),
                );
                anyhow::bail!("光束成本基准未通过: {e}");
            }
            if !report.speed_reported() {
                anyhow::bail!("光束成本基准缺少实际执行证据，拒绝报告速度");
            }
            println!(
                "=== 光束成本基准 (backend={}, {}x{}) ===",
                report.backend, report.width, report.height
            );
            for l in &report.levels {
                println!(
                    "档位 {:<13} N={:<4} σ={:<5} 受光像素 {}/{}",
                    l.level.label(),
                    l.steps,
                    l.fog_sigma,
                    l.lit_pixels,
                    l.total_pixels
                );
                println!(
                    "  样本: 实际 {} / 基准 {} (节省 {:.1}%)；跳过距离 {:.1}%；空区样本 {:.1}%；高密度样本 {:.1}%",
                    l.skip_cost.samples,
                    l.skip_cost.baseline_samples,
                    l.skip_cost.sample_saving() * 100.0,
                    l.skip_cost.skipped_fraction() * 100.0,
                    l.skip_cost.empty_fraction() * 100.0,
                    l.skip_cost.high_density_fraction() * 100.0
                );
                println!(
                    "  亮度误差(对收敛参照): 跳过 平均 {:.3e} 最大 {:.3e} 相对 {:.4}；同预算均匀 平均 {:.3e} 相对 {:.4}",
                    l.skip_error_vs_truth_mean,
                    l.skip_error_vs_truth_max,
                    l.relative_skip_error,
                    l.reference_error_vs_truth_mean,
                    l.relative_reference_error
                );
                println!(
                    "  透过率差 {:.3e}；耗时 跳过 {:.1} ms vs 同预算均匀 {:.1} ms（{:.2}×），收敛参照 {:.1} ms",
                    l.mean_transmittance_delta, l.skip_ms, l.reference_ms, l.speedup, l.truth_ms
                );
                println!(
                    "  证据: rays={} baseline_samples={} samples={} spans={} occlusion_queries={} transmittance_updates={} error_bound_violations={}",
                    l.skip_cost.rays,
                    l.skip_cost.baseline_samples,
                    l.skip_cost.samples,
                    l.skip_cost.spans,
                    l.skip_cost.occlusion_queries,
                    l.skip_cost.transmittance_updates,
                    l.skip_cost.error_bound_violations
                );
            }
            let json = serde_json::to_string_pretty(&report)?;
            if let Some(dir) = output.parent() {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(output, json)?;
            println!("基准报告 → {}", output.display());
        }
    }
    Ok(())
}
