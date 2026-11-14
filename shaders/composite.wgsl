// Composite pass: combines opaque scene, volumetric term, and shadow
// visualization according to the view mode; applies Reinhard + gamma.

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var opaque_tex: texture_2d<f32>;
@group(0) @binding(2) var volume_tex: texture_2d<f32>;
@group(0) @binding(3) var shadow_tex: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let x = f32(i32(i / 2u) * 4 - 1);
    let y = f32(i32(i % 2u) * 4 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs_composite(@builtin(position) fc: vec4f) -> @location(0) vec4f {
    let coord = vec2i(fc.xy);
    let opaque = textureLoad(opaque_tex, coord, 0).rgb;
    let vol = textureLoad(volume_tex, coord, 0);
    let sh = textureLoad(shadow_tex, coord, 0).r;

    var c: vec3f;
    switch (params.view_mode) {
        case 0u: { c = opaque * vol.a + vol.rgb; }   // 最终合成
        case 1u: { c = vol.rgb; }                    // 体积单独贡献
        case 2u: { c = vec3f(sh); }                  // 阴影
        default: { c = opaque; }                     // 不透明场景
    }
    let mapped = c / (vec3f(1.0) + c);
    let gamma = pow(max(mapped, vec3f(0.0)), vec3f(1.0 / 2.2));
    return vec4f(gamma, 1.0);
}
