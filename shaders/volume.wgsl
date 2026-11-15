// Fullscreen bounded-step volumetric integration pass.
// Mirrors kernel::integrate::integrate_spans:
//   per segment: L += T·S·(1−e^{−σΔt}); T *= e^{−σΔt}
// so results converge with step count instead of brightening.
//
// With `params.skip_empty` the conservative cone∩sphere bound of each enabled
// light is intersected with the camera ray analytically; the complement is
// skipped with an exact transmittance update and no source evaluation. The
// bound matches kernel::summary (cone + range only), so no illumination is
// dropped — samples inside gobo bars are still evaluated. Steps are then
// distributed only over the lit intervals, with the same two-sample floor as
// the CPU path.

struct VolOut {
    @location(0) volume: vec4f, // rgb: in-scattered radiance, a: transmittance
    @location(1) shadow: vec4f, // r: shadow visibility at first surface
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var cam_depth: texture_depth_2d;
@group(0) @binding(2) var shadow0: texture_depth_2d;
@group(0) @binding(3) var shadow1: texture_depth_2d;
@group(0) @binding(4) var cmp_samp: sampler_comparison;

const MAX_SPANS: u32 = 4u;
const MIN_STEPS_PER_SPAN: f32 = 2.0;

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

fn source(p: vec3f) -> vec3f {
    return light_term(0u, p) + light_term(1u, p);
}

// --- conservative ray bounds (mirror kernel::summary) ---

fn bound_contains(li: u32, p: vec3f) -> bool {
    let b = params.lights[li].bound_apex_range;
    let a = params.lights[li].bound_axis_cos;
    let v = p - b.xyz;
    let d2 = dot(v, v);
    if (d2 > b.w * b.w) {
        return false;
    }
    if (d2 < 1e-12) {
        return true;
    }
    let cosang = dot(a.xyz, v) / sqrt(d2);
    return cosang >= a.w - 1e-6;
}

struct Interval {
    lo: f32,
    hi: f32,
};

fn ray_sphere(ro: vec3f, rd: vec3f, center: vec3f, radius: f32) -> Interval {
    let oc = ro - center;
    let a = dot(rd, rd);
    let b = 2.0 * dot(oc, rd);
    let c = dot(oc, oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if (disc <= 0.0 || a < 1e-12) {
        return Interval(1.0, -1.0);
    }
    let sq = sqrt(disc);
    let r1 = (-b - sq) / (2.0 * a);
    let r2 = (-b + sq) / (2.0 * a);
    return Interval(min(r1, r2), max(r1, r2));
}

// Intersection of the camera ray with one light's cone∩sphere, clipped to
// [0, t_end]. The cone is convex and the sphere is convex, so the result is
// one interval; we find it by intersecting the sphere interval with the
// finite cone slab, then probe a point to reject invalid analytic branches.
fn light_interval(li: u32, ro: vec3f, rd: vec3f, t_end: f32) -> Interval {
    let b = params.lights[li].bound_apex_range;
    let a = params.lights[li].bound_axis_cos;
    if (a.w > 1.0) {
        return Interval(1.0, -1.0);
    }
    let sph = ray_sphere(ro, rd, b.xyz, b.w);
    let lo0 = max(sph.lo, 0.0);
    let hi0 = min(sph.hi, t_end);
    if (hi0 <= lo0) {
        return Interval(1.0, -1.0);
    }
    let w = ro - b.xyz;
    let c2 = a.w * a.w;
    let n = dot(a.xyz, rd);
    let m = dot(a.xyz, w);
    let wd = dot(w, rd);
    let ww = dot(w, w);
    let qa2 = n * n - c2;
    let qb = 2.0 * (m * n - c2 * wd);
    let qc = m * m - c2 * ww;
    var clo = lo0;
    var chi = hi0;
    if (abs(qa2) > 1e-7) {
        let disc = qb * qb - 4.0 * qa2 * qc;
        if (qa2 > 0.0) {
            // Outside branch: two side intervals; pick the one overlapping
            // the sphere interval, verifying membership.
            if (disc <= 0.0) {
                // whole ray inside the cone side
            } else {
                let sq = sqrt(disc);
                let r1 = min((-qb - sq) / (2.0 * qa2), (-qb + sq) / (2.0 * qa2));
                let r2 = max((-qb - sq) / (2.0 * qa2), (-qb + sq) / (2.0 * qa2));
                let cand_a_lo = lo0;
                let cand_a_hi = min(hi0, r1);
                let cand_b_lo = max(lo0, r2);
                let cand_b_hi = hi0;
                var found = false;
                if (cand_a_hi > cand_a_lo && bound_contains(li, ro + rd * (0.5 * (cand_a_lo + cand_a_hi)))) {
                    clo = cand_a_lo;
                    chi = cand_a_hi;
                    found = true;
                }
                if (cand_b_hi > cand_b_lo && bound_contains(li, ro + rd * (0.5 * (cand_b_lo + cand_b_hi)))) {
                    if (found) {
                        clo = min(clo, cand_b_lo);
                        chi = max(chi, cand_b_hi);
                    } else {
                        clo = cand_b_lo;
                        chi = cand_b_hi;
                    }
                    found = true;
                }
                if (!found) {
                    return Interval(1.0, -1.0);
                }
            }
        } else if (disc > 0.0) {
            let sq = sqrt(disc);
            let r1 = min((-qb - sq) / (2.0 * qa2), (-qb + sq) / (2.0 * qa2));
            let r2 = max((-qb - sq) / (2.0 * qa2), (-qb + sq) / (2.0 * qa2));
            clo = max(clo, r1);
            chi = min(chi, r2);
        }
    } else if (abs(qb) > 1e-9) {
        let t0 = -qc / qb;
        if (qb > 0.0) {
            clo = max(clo, t0);
        } else {
            chi = min(chi, t0);
        }
    } else if (qc < 0.0) {
        return Interval(1.0, -1.0);
    }
    // Forward nappe: m + t n >= 0.
    if (abs(n) > 1e-9) {
        let tl = -m / n;
        if (n > 0.0) {
            clo = max(clo, tl);
        } else {
            chi = min(chi, tl);
        }
    } else if (m < 0.0) {
        return Interval(1.0, -1.0);
    }
    if (chi <= clo) {
        return Interval(1.0, -1.0);
    }
    if (!bound_contains(li, ro + rd * (0.5 * (clo + chi)))) {
        return Interval(1.0, -1.0);
    }
    return Interval(clo, chi);
}

fn integrate_uniform(ro: vec3f, rd: vec3f, len: f32) -> vec4f {
    var L = vec3f(0.0);
    var T = 1.0;
    let n = max(params.steps, 1u);
    let dt = len / f32(n);
    let tr_seg = exp(-max(params.fog_sigma, 0.0) * dt);
    for (var i = 0u; i < n; i = i + 1u) {
        let t = (f32(i) + 0.5) * dt;
        let s = source(ro + rd * t);
        L = L + s * (T * (1.0 - tr_seg));
        T = T * tr_seg;
    }
    return vec4f(L, T);
}

fn integrate_skip(ro: vec3f, rd: vec3f, len: f32) -> vec4f {
    var ivs: array<Interval, MAX_SPANS>;
    var count = 0u;
    for (var li = 0u; li < 2u; li = li + 1u) {
        if (params.lights[li].gobo.z < 0.5) {
            continue;
        }
        let iv = light_interval(li, ro, rd, len);
        if (iv.hi > iv.lo) {
            if (count < MAX_SPANS) {
                ivs[count] = iv;
                count = count + 1u;
            } else {
                // Overflow: fall back to the full ray (conservative).
                return integrate_uniform(ro, rd, len);
            }
        }
    }
    if (count == 0u) {
        return vec4f(vec3f(0.0), exp(-max(params.fog_sigma, 0.0) * len));
    }
    // Insertion sort by lo (tiny array).
    for (var i = 1u; i < count; i = i + 1u) {
        let key = ivs[i];
        var j = i;
        while (j > 0u && ivs[j - 1u].lo > key.lo) {
            ivs[j] = ivs[j - 1u];
            j = j - 1u;
        }
        ivs[j] = key;
    }
    // Merge overlaps.
    var merged: array<Interval, MAX_SPANS>;
    var mcount = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        if (mcount > 0u && ivs[i].lo <= merged[mcount - 1u].hi + 1e-5) {
            merged[mcount - 1u].hi = max(merged[mcount - 1u].hi, ivs[i].hi);
        } else {
            merged[mcount] = ivs[i];
            mcount = mcount + 1u;
        }
    }
    let sigma = max(params.fog_sigma, 0.0);
    var L = vec3f(0.0);
    var T = 1.0;
    var cursor = 0.0;
    for (var k = 0u; k < mcount; k = k + 1u) {
        let a = max(merged[k].lo, cursor);
        let b = merged[k].hi;
        if (b <= a) {
            continue;
        }
        T = T * exp(-sigma * (a - cursor));
        let width = b - a;
        let budget = max(params.steps, 1u);
        let share = max(f32(budget) / f32(mcount), MIN_STEPS_PER_SPAN);
        let n = u32(ceil(share));
        let dt = width / f32(n);
        let tr_seg = exp(-sigma * dt);
        for (var i = 0u; i < n; i = i + 1u) {
            let t = a + (f32(i) + 0.5) * dt;
            let s = source(ro + rd * t);
            L = L + s * (T * (1.0 - tr_seg));
            T = T * tr_seg;
        }
        cursor = b;
    }
    T = T * exp(-sigma * (len - cursor));
    return vec4f(L, T);
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

    var vol = vec4f(vec3f(0.0), 1.0);
    if (t_surface > 0.0) {
        if (params.skip_empty != 0u) {
            vol = integrate_skip(ro, rd, t_surface);
        } else {
            vol = integrate_uniform(ro, rd, t_surface);
        }
    }

    var surf_shadow = 1.0;
    if (has_surface) {
        surf_shadow = shadow_factor(0u, surf_pos);
    }

    var out: VolOut;
    out.volume = vol;
    out.shadow = vec4f(surf_shadow, 0.0, 0.0, 1.0);
    return out;
}
