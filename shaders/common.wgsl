// Shared structs and functions for all stage-vol passes.
// Layout mirrors `kernel::integrate::{GpuParams, GpuLight}` exactly.

struct Light {
    view_proj: mat4x4f,
    position_range: vec4f,      // xyz: position (m), w: range (m)
    direction_cos_outer: vec4f, // xyz: beam direction, w: cos(outer)
    source_cos_inner: vec4f,    // rgb: color*intensity, w: cos(inner)
    gobo: vec4f,                // x: gobo id, y: gobo scale, z: enabled
};

struct Params {
    cam_view_proj: mat4x4f,
    cam_view_proj_inv: mat4x4f,
    cam_pos: vec4f,
    lights: array<Light, 2>,
    fog_sigma: f32,             // 1/m
    steps: u32,
    view_mode: u32,
    max_distance: f32,          // m, hard bound for the integral
    screen: vec2f,
    time: f32,
    pad: f32,
};

const MIN_LIGHT_DIST: f32 = 0.05;

fn unproject(inv_vp: mat4x4f, ndc: vec3f) -> vec3f {
    let clip = inv_vp * vec4f(ndc, 1.0);
    return clip.xyz / clip.w;
}

// Procedural gobo — must match kernel::gobo::gobo.
fn gobo(id: u32, uv_in: vec2f, scale: f32) -> f32 {
    let u = uv_in.x - 0.5;
    let v = uv_in.y - 0.5;
    if (id == 1u) {
        let s = sin(u * scale * 6.2831853);
        return select(0.05, 1.0, s > 0.0);
    }
    if (id == 2u) {
        let cu = i32(floor(u * scale));
        let cv = i32(floor(v * scale));
        return select(0.0, 1.0, ((cu + cv) % 2 + 2) % 2 == 0);
    }
    if (id == 3u) {
        let fu = fract(u * scale) - 0.5;
        let fv = fract(v * scale) - 0.5;
        let r = sqrt(fu * fu + fv * fv);
        let ring = clamp(sin(r * 9.0) * 0.5 + 0.5, 0.0, 1.0);
        return select(0.0, ring, r < 0.45);
    }
    if (id == 4u) {
        let r = sqrt(u * u + v * v);
        let a = atan2(v, u);
        let star = 0.30 + 0.10 * cos(a * 5.0);
        return select(0.0, 1.0, r < star);
    }
    return 1.0;
}
