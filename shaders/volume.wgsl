// Fullscreen bounded-step volumetric integration pass.
// Mirrors kernel::integrate::integrate_volume exactly:
//   per segment: L += T·S·(1−e^{−σΔt}); T *= e^{−σΔt}
// so results converge with step count instead of brightening.

struct VolOut {
    @location(0) volume: vec4f, // rgb: in-scattered radiance, a: transmittance
    @location(1) shadow: vec4f, // r: shadow visibility at first surface
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var cam_depth: texture_depth_2d;
@group(0) @binding(2) var shadow0: texture_depth_2d;
@group(0) @binding(3) var shadow1: texture_depth_2d;
@group(0) @binding(4) var cmp_samp: sampler_comparison;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    // Fullscreen triangle.
    let x = f32(i32(i / 2u) * 4 - 1);
    let y = f32(i32(i % 2u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

fn shadow_factor(li: u32, p: vec3f) -> f32 {
    let l = params.lights[li];
    let clip = l.view_proj * vec4f(p, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let bias = 0.0018;
    if (li == 0u) {
        return textureSampleCompareLevel(shadow0, cmp_samp, uv, ndc.z - bias);
    }
    return textureSampleCompareLevel(shadow1, cmp_samp, uv, ndc.z - bias);
}

fn light_term(li: u32, p: vec3f) -> vec3f {
    let l = params.lights[li];
    if (l.gobo.z < 0.5) {
        return vec3f(0.0);
    }
    let to_light = l.position_range.xyz - p;
    let d = length(to_light);
    let range = l.position_range.w;
    if (d > range) {
        return vec3f(0.0);
    }
    let dir_to_light = to_light / max(d, 1e-9);
    let cosang = dot(-dir_to_light, l.direction_cos_outer.xyz);
    let cone = smoothstep(l.direction_cos_outer.w, l.source_cos_inner.w, cosang);
    if (cone <= 0.0) {
        return vec3f(0.0);
    }
    let atten = 1.0 / max(d * d, MIN_LIGHT_DIST * MIN_LIGHT_DIST);
    let range_gate = 1.0 - smoothstep(0.8 * range, range, d);
    let clip = l.view_proj * vec4f(p, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if (ndc.z <= 0.0 || ndc.z >= 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec3f(0.0);
    }
    let g = gobo(u32(l.gobo.x), uv, l.gobo.y);
    if (g <= 0.0) {
        return vec3f(0.0);
    }
    let sh = shadow_factor(li, p);
    return l.source_cos_inner.rgb * (cone * atten * range_gate * g * sh);
}

@fragment
fn fs_volume(@builtin(position) fc: vec4f) -> VolOut {
    let coord = vec2i(fc.xy);
    let depth = textureLoad(cam_depth, coord, 0);
    let ndc_xy = vec2f(
        (fc.x / params.screen.x) * 2.0 - 1.0,
        1.0 - (fc.y / params.screen.y) * 2.0,
    );
    let p_near = unproject(params.cam_view_proj_inv, vec3f(ndc_xy, 0.0));
    let p_far = unproject(params.cam_view_proj_inv, vec3f(ndc_xy, 1.0));
    let ro = params.cam_pos.xyz;
    let rd = normalize(p_far - p_near);

    // Fog/surface depth relationship: the integral stops at the first
    // opaque surface; sky pixels integrate to the hard distance bound.
    var t_surface = params.max_distance;
    var surf_pos = vec3f(0.0);
    var has_surface = false;
    if (depth < 0.99999) {
        surf_pos = unproject(params.cam_view_proj_inv, vec3f(ndc_xy, depth));
        t_surface = min(length(surf_pos - ro), params.max_distance);
        has_surface = true;
    }

    var L = vec3f(0.0);
    var T = 1.0;
    let len = t_surface;
    if (len > 0.0) {
        let n = max(params.steps, 1u);
        let dt = len / f32(n);
        let tr_seg = exp(-max(params.fog_sigma, 0.0) * dt);
        for (var i = 0u; i < n; i = i + 1u) {
            let t = (f32(i) + 0.5) * dt;
            let p = ro + rd * t;
            let s = light_term(0u, p) + light_term(1u, p);
            L = L + s * (T * (1.0 - tr_seg));
            T = T * tr_seg;
        }
    }

    var surf_shadow = 1.0;
    if (has_surface) {
        surf_shadow = shadow_factor(0u, surf_pos);
    }

    var out: VolOut;
    out.volume = vec4f(L, T);
    out.shadow = vec4f(surf_shadow, 0.0, 0.0, 1.0);
    return out;
}
