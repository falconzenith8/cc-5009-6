//! Integration tests for the stage volumetric kernel + CPU reference
//! renderer. GPU tests skip gracefully when no adapter is available.

use stage_vol::cpu::CpuRenderer;
use stage_vol::kernel::params::{StageVolParams, ViewMode, MAX_STEPS};

fn render(params: &StageVolParams, w: u32, h: u32) -> stage_vol::cpu::Frame {
    CpuRenderer::new(w, h).render(params, 0.0)
}

#[test]
fn clear_fog_has_zero_volume() {
    let mut p = StageVolParams::default();
    p.fog_sigma = 0.0;
    p.view_mode = ViewMode::VolumeOnly;
    let f = render(&p, 160, 90);
    assert_eq!(f.mean_volume_luminance(), 0.0);
    assert!(f.transmittance.iter().all(|&t| t == 1.0));
}

#[test]
fn full_occlusion_kills_volume() {
    let mut p = StageVolParams::default();
    p.lights[0].position = [0.0, 1.2, -2.0];
    p.lights[0].target = [0.0, 1.4, 6.0];
    p.lights[0].range = 12.0;
    p.occluder_scale = 4.0;
    let f = render(&p, 160, 90);
    let max_lum = f
        .volume
        .iter()
        .map(|v| (v[0] + v[1] + v[2]) / 3.0)
        .fold(0.0f32, f32::max);
    assert!(max_lum < 1e-4, "全遮挡下仍有体积光: {max_lum}");
}

#[test]
fn step_count_converges_not_brightens() {
    let mut p = StageVolParams::default();
    p.view_mode = ViewMode::VolumeOnly;
    let mut means = Vec::new();
    for &n in &[8u32, 32, 128, MAX_STEPS] {
        p.steps = n;
        let f = render(&p, 160, 90);
        assert!(f.all_finite());
        means.push(f.mean_volume_luminance());
    }
    let ratio = means[3] / means[0];
    assert!(ratio < 2.0, "亮度随步数增长: {means:?}");
    let drift = (means[3] - means[2]).abs() / means[3];
    assert!(drift < 0.05, "高步数未收敛: {means:?}");
}

#[test]
fn moving_occluder_modulates_volume() {
    let mut p = StageVolParams::default();
    p.occluder_animate = true;
    let r = CpuRenderer::new(160, 90);
    let a = r.render(&p, 0.0).mean_volume_luminance();
    let b = r.render(&p, 1.65).mean_volume_luminance();
    assert!((a - b).abs() / a.max(b) > 0.02, "挡板移动未改变体积光: {a} vs {b}");
}

#[test]
fn finite_near_camera_and_inside_beam() {
    // Camera inside the beam looking straight at the light.
    let mut p = StageVolParams::default();
    p.camera_pos = [0.0, 2.0, 0.5];
    p.camera_target = [0.0, 4.2, 1.5];
    p.steps = MAX_STEPS;
    let f = render(&p, 160, 90);
    assert!(f.all_finite(), "灯内观察出现非有限值");
    // Camera 5 cm above the floor.
    p.camera_pos = [0.0, 0.05, 2.0];
    p.camera_target = [0.0, 0.5, -2.0];
    let f2 = render(&p, 160, 90);
    assert!(f2.all_finite(), "镜头贴近地面出现非有限值");
}

#[test]
fn params_roundtrip_and_scene_reload() {
    let p = StageVolParams::default();
    let dir = std::env::temp_dir().join(format!("stagevol-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("scene.json");
    p.save_to(&path).unwrap();
    let q = StageVolParams::load_from(&path).unwrap();
    assert_eq!(p, q);
    // Scene reload must reproduce the identical frame.
    let fa = render(&p, 128, 72);
    let fb = render(&q, 128, 72);
    assert_eq!(fa.rgba, fb.rgba);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn invalid_params_are_rejected_with_feedback() {
    let mut p = StageVolParams::default();
    assert!(p.set_control("fog_sigma", 0.9).is_err());
    assert!(p.set_control("steps", 0.0).is_ok()); // clamped to >= 1 by kernel
    assert!(p.set_control("nope", 1.0).is_err());
    p.fog_sigma = -1.0;
    assert!(!p.validate().is_empty());
}

#[test]
fn gpu_matches_cpu_when_adapter_available() {
    let gpu = pollster::block_on(stage_vol::gpu::GpuRenderer::init_headless(160, 90));
    let mut gpu = match gpu {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP gpu_matches_cpu: 无 GPU 适配器 ({e})");
            return;
        }
    };
    let mut p = StageVolParams::default();
    p.view_mode = ViewMode::VolumeOnly;
    let g = gpu.render(&p, 0.0).expect("GPU 渲染失败");
    let c = render(&p, 160, 90);
    assert!(g.all_finite());
    let gm = g.mean_volume_luminance();
    let cm = c.mean_volume_luminance();
    // GPU uses a 1024px shadow map + f16 accumulation; CPU is analytic.
    // They must agree to within a loose tolerance.
    let rel = (gm - cm).abs() / cm.max(1e-6);
    assert!(rel < 0.25, "GPU/CPU 体积亮度偏差过大: gpu={gm} cpu={cm}");
}

#[test]
fn gpu_device_recovery_when_adapter_available() {
    let gpu = pollster::block_on(stage_vol::gpu::GpuRenderer::init_headless(128, 72));
    let mut gpu = match gpu {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP gpu_device_recovery: 无 GPU 适配器 ({e})");
            return;
        }
    };
    let p = StageVolParams::default();
    let before = gpu.render(&p, 0.0).unwrap();
    gpu.recover().expect("设备资源恢复失败");
    let after = gpu.render(&p, 0.0).unwrap();
    assert_eq!(before.rgba, after.rgba, "恢复后帧不一致");
}
