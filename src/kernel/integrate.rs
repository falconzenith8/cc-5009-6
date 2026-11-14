//! Bounded-step volumetric integrator — the single source of truth for the
//! stage volumetric lighting model. The CPU reference renderer calls these
//! functions directly; the WGSL shaders implement the same formulas and are
//! checked against this implementation by the convergence tests.
//!
//! Model (see also `model_notes()`):
//! - uniform thin fog: extinction σ constant everywhere, single scattering
//!   only, isotropic phase folded into light intensity;
//! - per segment the source term S is treated as constant, giving the exact
//!   segment update `L += T·S·(1−e^{−σΔt})`, `T *= e^{−σΔt}` — this converges
//!   with step count instead of accumulating more light with more samples;
//! - all distances in meters, σ in 1/m, every ray is clamped to
//!   [`MAX_RAY_DISTANCE`] so the integral is always bounded.

use super::gobo::gobo;
use super::math::{smoothstep, Mat4, Ray, Vec3};
use super::params::{SpotLight, StageVolParams, MAX_RAY_DISTANCE};
use super::scene::Scene;
use bytemuck::{Pod, Zeroable};

/// Minimum distance used in the 1/d² falloff so a sample exactly at the
/// light position stays finite.
const MIN_LIGHT_DIST: f32 = 0.05;
/// Shadow ray trim at both ends, meters.
const SHADOW_EPS: f32 = 0.02;

#[derive(Clone, Copy, Debug)]
pub struct PreparedLight {
    pub position: Vec3,
    pub direction: Vec3,
    /// color × intensity
    pub source: Vec3,
    pub cos_inner: f32,
    pub cos_outer: f32,
    pub range: f32,
    pub gobo_id: u32,
    pub gobo_scale: f32,
    /// World → light clip (NDC z in [0,1]); used for gobo UV and shadow map.
    pub view_proj: Mat4,
    pub enabled: bool,
}

impl PreparedLight {
    pub fn prepare(l: &SpotLight) -> PreparedLight {
        let position = Vec3::new(l.position[0], l.position[1], l.position[2]);
        let target = Vec3::new(l.target[0], l.target[1], l.target[2]);
        let direction = (target - position).normalized();
        let outer = l.outer_cone_deg.to_radians();
        let view = Mat4::look_at(position, target, Vec3::UP);
        let proj = Mat4::perspective(2.0 * outer, 1.0, 0.1, (l.range * 1.25 + 1.0).max(2.0));
        PreparedLight {
            position,
            direction,
            source: Vec3::new(l.color[0], l.color[1], l.color[2]) * l.intensity,
            cos_inner: l.inner_cone_deg.to_radians().cos(),
            cos_outer: outer.cos(),
            range: l.range,
            gobo_id: l.gobo_id,
            gobo_scale: l.gobo_scale,
            view_proj: proj.mul(&view),
            enabled: l.enabled && l.intensity > 0.0,
        }
    }

    /// Project a world point into light UV. None when outside the frustum.
    pub fn light_uv(&self, p: Vec3) -> Option<[f32; 2]> {
        let clip = self.view_proj.transform_point(p);
        if clip.z <= 0.0 || clip.z >= 1.0 {
            return None;
        }
        let u = clip.x * 0.5 + 0.5;
        let v = 1.0 - (clip.y * 0.5 + 0.5);
        if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
            return None;
        }
        Some([u, v])
    }
}

/// Irradiance-like term from one spotlight at point `p`, including cone
/// falloff, range gate, gobo mask, and (when `shadow` is provided) occlusion.
/// `shadow` returns true when the segment p→light is blocked.
pub fn light_term<F>(light: &PreparedLight, p: Vec3, shadow: F) -> Vec3
where
    F: Fn(Vec3, Vec3, f32) -> bool,
{
    if !light.enabled {
        return Vec3::ZERO;
    }
    let to_light = light.position - p;
    let d = to_light.length();
    if d > light.range {
        return Vec3::ZERO;
    }
    let dir_to_light = if d > 1e-9 { to_light * (1.0 / d) } else { Vec3::ZERO };
    // Angle between beam axis and the direction from the light to p.
    let cosang = (-dir_to_light).dot(light.direction);
    let cone = smoothstep(light.cos_outer, light.cos_inner, cosang);
    if cone <= 0.0 {
        return Vec3::ZERO;
    }
    let atten = 1.0 / (d * d).max(MIN_LIGHT_DIST * MIN_LIGHT_DIST);
    let range_gate = 1.0 - smoothstep(0.8 * light.range, light.range, d);
    let uv = match light.light_uv(p) {
        Some(uv) => uv,
        None => return Vec3::ZERO,
    };
    let g = gobo(light.gobo_id, uv, light.gobo_scale);
    if g <= 0.0 {
        return Vec3::ZERO;
    }
    let blocked = d > 2.0 * SHADOW_EPS && shadow(p, dir_to_light, d - SHADOW_EPS);
    if blocked {
        return Vec3::ZERO;
    }
    light.source * (cone * atten * range_gate * g)
}

/// In-scatter source S(p) summed over all lights (CPU shadow query).
pub fn in_scatter(scene: &Scene, lights: &[PreparedLight], p: Vec3) -> Vec3 {
    let mut s = Vec3::ZERO;
    for l in lights {
        s += light_term(l, p, |p0, dir, max_t| scene.occluded(p0, dir, max_t));
    }
    s
}

/// Bounded-step volume integration along `ray` from the camera to the first
/// opaque surface at `t_surface` (pass `f32::INFINITY` for no surface).
/// Returns (in-scattered radiance L, transmittance T).
///
/// The per-segment update is the exact solution for a constant source term
/// over the segment, so results converge as `steps` grows instead of scaling
/// with it. `steps` is clamped to [1, 512]; distance is clamped to
/// [`MAX_RAY_DISTANCE`]. All outputs are finite for finite inputs.
pub fn integrate_volume<F>(
    ray: &Ray,
    t_surface: f32,
    sigma: f32,
    steps: u32,
    mut source: F,
) -> (Vec3, f32)
where
    F: FnMut(Vec3) -> Vec3,
{
    let t0 = 0.0f32;
    let t1 = t_surface.min(MAX_RAY_DISTANCE);
    let len = t1 - t0;
    if !len.is_finite() || len <= 0.0 {
        return (Vec3::ZERO, 1.0);
    }
    let n = steps.clamp(1, super::params::MAX_STEPS);
    let dt = len / n as f32;
    let sigma = sigma.max(0.0);
    let tr_seg = (-sigma * dt).exp();
    let mut l = Vec3::ZERO;
    let mut t_acc = 1.0f32;
    for i in 0..n {
        let t = t0 + (i as f32 + 0.5) * dt;
        let s = source(ray.at(t));
        if s.is_finite() {
            l += s * (t_acc * (1.0 - tr_seg));
        }
        t_acc *= tr_seg;
    }
    debug_assert!(l.is_finite() && t_acc.is_finite());
    (l, t_acc)
}

/// Lambert-ish opaque shading so the composite view is coherent with the
/// volumetric term (same lights, same shadow query).
pub fn shade_surface(scene: &Scene, lights: &[PreparedLight], p: Vec3, n: Vec3, albedo: Vec3) -> Vec3 {
    let ambient = Vec3::splat(0.03);
    let mut c = albedo * ambient;
    for l in lights {
        let e = light_term(l, p, |p0, dir, max_t| scene.occluded(p0, dir, max_t));
        let to_light = (l.position - p).normalized();
        let ndl = n.dot(to_light).max(0.0);
        c += albedo * e * ndl;
    }
    let _ = scene;
    c
}

/// Pinhole camera matching the GPU uniform layout.
#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    pub view_proj: Mat4,
    pub view_proj_inv: Mat4,
    pub fov_y: f32,
    pub aspect: f32,
}

impl Camera {
    pub fn from_params(params: &StageVolParams, aspect: f32) -> Camera {
        let pos = Vec3::new(params.camera_pos[0], params.camera_pos[1], params.camera_pos[2]);
        let target = Vec3::new(params.camera_target[0], params.camera_target[1], params.camera_target[2]);
        let fov_y = params.camera_fov_deg.to_radians();
        let view = Mat4::look_at(pos, target, Vec3::UP);
        let proj = Mat4::perspective(fov_y, aspect, 0.05, MAX_RAY_DISTANCE * 2.0);
        let view_proj = proj.mul(&view);
        Camera {
            position: pos,
            view_proj,
            view_proj_inv: invert(&view_proj),
            fov_y,
            aspect,
        }
    }

    /// World-space ray through pixel (px, py) of a (w, h) image.
    pub fn ray(&self, px: f32, py: f32, w: f32, h: f32) -> Ray {
        let ndc_x = ((px + 0.5) / w) * 2.0 - 1.0;
        let ndc_y = 1.0 - ((py + 0.5) / h) * 2.0;
        let near = self.view_proj_inv.transform_point(Vec3::new(ndc_x, ndc_y, 0.0));
        let far = self.view_proj_inv.transform_point(Vec3::new(ndc_x, ndc_y, 1.0));
        Ray { origin: self.position, dir: (far - near).normalized() }
    }
}

/// General 4x4 inverse (Gauss-Jordan), adequate for camera/light matrices.
pub fn invert(m: &Mat4) -> Mat4 {
    let mut a = m.m;
    let mut inv = Mat4::IDENTITY.m;
    for col in 0..4 {
        let mut pivot = col;
        for r in (col + 1)..4 {
            if a[r][col].abs() > a[pivot][col].abs() {
                pivot = r;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return Mat4::IDENTITY;
        }
        if pivot != col {
            a.swap(pivot, col);
            inv.swap(pivot, col);
        }
        let d = a[col][col];
        for j in 0..4 {
            a[col][j] /= d;
            inv[col][j] /= d;
        }
        for r in 0..4 {
            if r != col {
                let f = a[r][col];
                for j in 0..4 {
                    a[r][j] -= f * a[col][j];
                    inv[r][j] -= f * inv[col][j];
                }
            }
        }
    }
    Mat4 { m: inv }
}

/// GPU uniform mirror of the model. Layout shared with `common.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuLight {
    pub view_proj: [[f32; 4]; 4],
    pub position_range: [f32; 4],
    pub direction_cos_outer: [f32; 4],
    pub source_cos_inner: [f32; 4],
    pub gobo: [f32; 4], // id, scale, enabled, pad
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuParams {
    pub cam_view_proj: [[f32; 4]; 4],
    pub cam_view_proj_inv: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub lights: [GpuLight; 2],
    pub fog_sigma: f32,
    pub steps: u32,
    pub view_mode: u32,
    pub max_distance: f32,
    pub screen: [f32; 2],
    pub time: f32,
    pub _pad: f32,
}

impl GpuParams {
    pub fn build(params: &StageVolParams, width: u32, height: u32, time: f32) -> GpuParams {
        let aspect = width as f32 / height.max(1) as f32;
        let cam = Camera::from_params(params, aspect);
        let mut lights = [GpuLight {
            view_proj: [[0.0; 4]; 4],
            position_range: [0.0; 4],
            direction_cos_outer: [0.0; 4],
            source_cos_inner: [0.0; 4],
            gobo: [0.0; 4],
        }; super::params::MAX_LIGHTS];
        for (i, l) in params.lights.iter().take(super::params::MAX_LIGHTS).enumerate() {
            let p = PreparedLight::prepare(l);
            lights[i] = GpuLight {
                view_proj: p.view_proj.to_cols_array(),
                position_range: [p.position.x, p.position.y, p.position.z, p.range],
                direction_cos_outer: [p.direction.x, p.direction.y, p.direction.z, p.cos_outer],
                source_cos_inner: [p.source.x, p.source.y, p.source.z, p.cos_inner],
                gobo: [p.gobo_id as f32, p.gobo_scale, if p.enabled { 1.0 } else { 0.0 }, 0.0],
            };
        }
        GpuParams {
            cam_view_proj: cam.view_proj.to_cols_array(),
            cam_view_proj_inv: cam.view_proj_inv.to_cols_array(),
            cam_pos: [cam.position.x, cam.position.y, cam.position.z, 1.0],
            lights,
            fog_sigma: params.fog_sigma,
            steps: params.steps,
            view_mode: params.view_mode.index(),
            max_distance: MAX_RAY_DISTANCE,
            screen: [width as f32, height as f32],
            time,
            _pad: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::scene::Scene;

    fn default_setup() -> (Scene, Vec<PreparedLight>, StageVolParams) {
        let params = StageVolParams::default();
        let scene = Scene::build(params.occluder_x, params.occluder_visible, params.occluder_scale);
        let lights: Vec<_> = params.lights.iter().map(PreparedLight::prepare).collect();
        (scene, lights, params)
    }

    #[test]
    fn integral_converges_not_brightens() {
        let (scene, lights, params) = default_setup();
        let cam = Camera::from_params(&params, 1.0);
        // Ray through the middle of the beam.
        let ray = cam.ray(50.0, 42.0, 100.0, 80.0);
        let t_surface = scene.intersect(&ray, MAX_RAY_DISTANCE).map(|h| h.0).unwrap_or(MAX_RAY_DISTANCE);
        let mut prev: Option<f32> = None;
        for &n in &[8u32, 32, 128, 512] {
            let (l, _t) = integrate_volume(&ray, t_surface, params.fog_sigma, n, |p| in_scatter(&scene, &lights, p));
            let lum = l.x + l.y + l.z;
            assert!(lum.is_finite());
            if let Some(p) = prev {
                // Must not scale with sample count; relative change shrinks.
                let rel = (lum - p).abs() / lum.max(1e-6);
                assert!(rel < 0.35, "n={n}: rel change {rel} too large (not converging)");
            }
            prev = Some(lum);
        }
    }

    #[test]
    fn zero_sigma_gives_zero_light_and_unit_transmittance() {
        let (scene, lights, _params) = default_setup();
        let ray = Ray { origin: Vec3::new(0.0, 2.0, 5.0), dir: Vec3::new(0.0, -0.2, -1.0).normalized() };
        let (l, t) = integrate_volume(&ray, 30.0, 0.0, 64, |p| in_scatter(&scene, &lights, p));
        assert_eq!(l, Vec3::ZERO);
        assert_eq!(t, 1.0);
    }

    #[test]
    fn finite_when_camera_inside_beam_and_at_light() {
        let (scene, lights, params) = default_setup();
        // Ray starting essentially at the light position, looking down.
        let lp = lights[0].position;
        let ray = Ray { origin: lp, dir: Vec3::new(0.0, -1.0, 0.0) };
        let (l, t) = integrate_volume(&ray, 4.0, params.fog_sigma, 128, |p| in_scatter(&scene, &lights, p));
        assert!(l.is_finite() && t.is_finite());
        // Camera 1 cm above the floor looking horizontally.
        let ray2 = Ray { origin: Vec3::new(0.0, 0.01, 1.0), dir: Vec3::new(0.0, 0.0, -1.0) };
        let (l2, t2) = integrate_volume(&ray2, 0.5, params.fog_sigma, 128, |p| in_scatter(&scene, &lights, p));
        assert!(l2.is_finite() && t2.is_finite());
    }

    #[test]
    fn occluder_kills_in_volume_light() {
        let params = StageVolParams::default();
        let lights: Vec<_> = params.lights.iter().map(PreparedLight::prepare).collect();
        // Sample point directly behind the occluder panel relative to light 0.
        let p = Vec3::new(0.0, 1.0, -2.2);
        let open = in_scatter(&Scene::build(10.0, true, 1.0), &lights, p); // panel moved aside
        let blocked = in_scatter(&Scene::build(0.0, true, 1.0), &lights, p);
        assert!(open.length() > 0.0, "expected light at {p:?}");
        assert_eq!(blocked, Vec3::ZERO, "occluder must block in-volume light");
    }
}
