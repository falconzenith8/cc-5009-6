//! CPU reference renderer. Mirrors the GPU pipeline (opaque depth →
//! volumetric integration → composite) with the kernel's analytic
//! intersection and occlusion. Used by tests, reference high-sample frames,
//! and as the fallback backend when no GPU adapter is available.

use crate::kernel::cost::CostBreakdown;
use crate::kernel::integrate::{
    in_scatter_counted, integrate_spans, shade_surface, Camera, PreparedLight,
};
use crate::kernel::math::Vec3;
use crate::kernel::params::{StageVolParams, ViewMode, MAX_RAY_DISTANCE};
use crate::kernel::scene::Scene;
use crate::kernel::summary::SummaryCache;
use crate::kernel::cost::DEFAULT_HIGH_DENSITY_THRESHOLD;

pub struct Frame {
    pub width: u32,
    pub height: u32,
    /// Final display pixels, RGBA8.
    pub rgba: Vec<u8>,
    /// Raw volumetric radiance per pixel (pre-tonemap), for analysis.
    pub volume: Vec<[f32; 3]>,
    /// Transmittance per pixel.
    pub transmittance: Vec<f32>,
    /// Shadow visibility at the first surface (1 = lit, 0 = occluded).
    pub surface_shadow: Vec<f32>,
}

impl Frame {
    pub fn mean_volume_luminance(&self) -> f64 {
        if self.volume.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .volume
            .iter()
            .map(|v| (v[0] + v[1] + v[2]) as f64 / 3.0)
            .sum();
        sum / self.volume.len() as f64
    }

    pub fn all_finite(&self) -> bool {
        self.volume.iter().all(|v| v.iter().all(|x| x.is_finite()))
            && self.transmittance.iter().all(|x| x.is_finite())
    }

    /// Write the display RGBA as PNG.
    pub fn save_png(&self, path: &std::path::Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let img: image::RgbaImage =
            image::ImageBuffer::from_raw(self.width, self.height, self.rgba.clone())
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "图像尺寸不匹配"))?;
        img.save(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, format!("写 {} 失败: {e}", path.display())))
    }
}

pub struct CpuRenderer {
    pub width: u32,
    pub height: u32,
}

impl CpuRenderer {
    pub fn new(width: u32, height: u32) -> Self {
        CpuRenderer { width, height }
    }

    pub fn backend_name(&self) -> &'static str {
        "cpu-reference"
    }

    pub fn render(&self, params: &StageVolParams, time: f32) -> Frame {
        self.render_with_stats(params, time).0
    }

    /// Render with cost evidence. `skip_empty_regions` selects between the
    /// uniform reference integrator (`integrate_volume`, unchanged from the
    /// pre-skip build) and the conservative empty-region skip
    /// (`integrate_spans`). Both share meter units and the same shadow query,
    /// so the second return value is a like-for-like comparison source.
    pub fn render_with_stats(
        &self,
        params: &StageVolParams,
        time: f32,
    ) -> (Frame, CostBreakdown) {
        self.render_inner(params, time, true)
    }

    /// Uniform reference integrator without empty-region skipping. This is
    /// the execution baseline the benchmark compares against.
    pub fn render_reference(&self, params: &StageVolParams, time: f32) -> (Frame, CostBreakdown) {
        self.render_inner(params, time, false)
    }

    fn render_inner(
        &self,
        params: &StageVolParams,
        time: f32,
        skip_empty_regions: bool,
    ) -> (Frame, CostBreakdown) {
        let (w, h) = (self.width, self.height);
        let scene = Scene::build(params.occluder_x_at(time), params.occluder_visible, params.occluder_scale);
        let lights: Vec<PreparedLight> = params.lights.iter().map(PreparedLight::prepare).collect();
        let cam = Camera::from_params(params, w as f32 / h.max(1) as f32);

        let mut summary_cache = SummaryCache::default();
        let summary = summary_cache.get(params, time);

        let mut rgba = vec![0u8; (w * h * 4) as usize];
        let mut volume = vec![[0.0f32; 3]; (w * h) as usize];
        let mut transmittance = vec![1.0f32; (w * h) as usize];
        let mut surface_shadow = vec![1.0f32; (w * h) as usize];
        let mut stats = CostBreakdown::default();
        let mut source = |p: Vec3| in_scatter_counted(&scene, &lights, p, DEFAULT_HIGH_DENSITY_THRESHOLD);

        for py in 0..h {
            for px in 0..w {
                let idx = (py * w + px) as usize;
                let ray = cam.ray(px as f32, py as f32, w as f32, h as f32);
                let hit = scene.intersect(&ray, MAX_RAY_DISTANCE);
                let t_surface = hit.map(|(t, _, _)| t).unwrap_or(MAX_RAY_DISTANCE);

                let (l, tr, ray_stats) = if skip_empty_regions {
                    let spans = summary.spans(&ray, t_surface);
                    integrate_spans(
                        &ray,
                        t_surface,
                        params.fog_sigma,
                        params.steps,
                        params.step_error_tolerance,
                        &mut source,
                        spans.spans(),
                    )
                } else {
                    // Reference path: same update, whole ray as one span, so
                    // the step schedule matches `integrate_volume` bit for bit.
                    let whole = [(0.0f32, t_surface)];
                    integrate_spans(
                        &ray,
                        t_surface,
                        params.fog_sigma,
                        params.steps,
                        1.0,
                        &mut source,
                        &whole,
                    )
                };
                stats.merge(&ray_stats);
                volume[idx] = [l.x, l.y, l.z];
                transmittance[idx] = tr;

                // Shadow factor at the first surface (light 0), for the
                // shadow view mode.
                let mut shadow_vis = 1.0f32;
                if let Some((t, _m, _n)) = hit {
                    let p = ray.at(t);
                    let l0 = &lights[0];
                    let to_light = l0.position - p;
                    let d = to_light.length();
                    if l0.enabled && d > 0.04 {
                        let blocked =
                            scene.occluded(p, to_light * (1.0 / d), d - 0.02);
                        shadow_vis = if blocked { 0.0 } else { 1.0 };
                    }
                }
                surface_shadow[idx] = shadow_vis;

                let color = match params.view_mode {
                    ViewMode::Composite => {
                        let surface = match hit {
                            Some((t, m, n)) => {
                                shade_surface(&scene, &lights, ray.at(t), n, m.albedo())
                            }
                            None => Vec3::ZERO,
                        };
                        surface * tr + l
                    }
                    ViewMode::VolumeOnly => l,
                    ViewMode::Shadow => Vec3::splat(shadow_vis),
                    ViewMode::OpaqueOnly => match hit {
                        Some((t, m, n)) => shade_surface(&scene, &lights, ray.at(t), n, m.albedo()),
                        None => Vec3::ZERO,
                    },
                };
                let rgb = tonemap(color);
                rgba[idx * 4] = rgb[0];
                rgba[idx * 4 + 1] = rgb[1];
                rgba[idx * 4 + 2] = rgb[2];
                rgba[idx * 4 + 3] = 255;
            }
        }
        (Frame { width: w, height: h, rgba, volume, transmittance, surface_shadow }, stats)
    }
}

/// Reinhard + gamma 2.2, shared by the GPU composite shader.
pub fn tonemap(c: Vec3) -> [u8; 3] {
    let f = |x: f32| -> u8 {
        let y = if x.is_finite() { x.max(0.0) } else { 0.0 };
        let mapped = y / (1.0 + y);
        (mapped.powf(1.0 / 2.2) * 255.0 + 0.5).clamp(0.0, 255.0) as u8
    };
    [f(c.x), f(c.y), f(c.z)]
}
