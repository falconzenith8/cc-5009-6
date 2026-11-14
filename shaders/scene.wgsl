// Opaque scene pass + shadow-map pass.

struct VsOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) albedo: vec3f,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var shadow0: texture_depth_2d;
@group(0) @binding(2) var shadow1: texture_depth_2d;
@group(0) @binding(3) var cmp_samp: sampler_comparison;

@vertex
fn vs_scene(
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) albedo: vec3f,
) -> VsOut {
    var out: VsOut;
    out.clip = params.cam_view_proj * vec4f(pos, 1.0);
    out.world = pos;
    out.normal = normal;
    out.albedo = albedo;
    return out;
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

// Same light model as the volume pass (kernel::integrate::light_term).
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
fn fs_scene(in: VsOut) -> @location(0) vec4f {
    var c = in.albedo * 0.03;
    for (var i = 0u; i < 2u; i = i + 1u) {
        let l = params.lights[i];
        if (l.gobo.z < 0.5) {
            continue;
        }
        let e = light_term(i, in.world);
        let ndl = max(dot(in.normal, normalize(l.position_range.xyz - in.world)), 0.0);
        c = c + in.albedo * e * ndl;
    }
    return vec4f(c, 1.0);
}

// --- shadow-map pass: depth only, from one light's view ---

@group(0) @binding(0) var<uniform> light_vp: mat4x4f;

@vertex
fn vs_shadow(@location(0) pos: vec3f) -> @builtin(position) vec4f {
    return light_vp * vec4f(pos, 1.0);
}
