//! Conservative influence bounds for the volumetric source term.
//!
//! `light_term` can only be non-zero where an enabled light's beam reaches.
//! Each enabled light is bounded by the intersection of
//!   * a cone around the beam axis with half-angle `outer_cone_deg`, and
//!   * a sphere of radius `range` centred on the light.
//!
//! This bound is a strict superset of the lit region: cone and range gates
//! match the bounds used by the source term, gobo masking and occlusion can
//! only remove light. Skipping the complement therefore never drops
//! illumination. Samples that are inside the bound but masked to zero (gobo
//! bars, cone edges, occluded segments) are still evaluated and counted as
//! empty, so the skip is conservative rather than assuming the bound is lit.
//!
//! The cache key includes every input that can change the bound *or* the
//! medium: light geometry/state, fog extinction (σ), and occluder placement.
//! A light or medium edit rebuilds the bounds; a stale range is never reused.

use crate::kernel::integrate::PreparedLight;
use crate::kernel::math::{Ray, Vec3};
use crate::kernel::params::{StageVolParams, MAX_LIGHTS};

/// Maximum disjoint lit spans reported for one ray. Two lights contribute at
/// most two spans each (cone∩sphere is convex, so one interval per light);
/// the spare room is defensive. Overflow falls back to the whole ray, which
/// is conservative.
pub const MAX_LIT_SPANS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightBound {
    pub apex: Vec3,
    pub axis: Vec3,
    pub cos_outer: f32,
    pub range: f32,
}

impl LightBound {
    fn from_prepared(l: &PreparedLight) -> Option<LightBound> {
        if !l.enabled {
            return None;
        }
        Some(LightBound {
            apex: l.position,
            axis: l.direction,
            cos_outer: l.cos_outer,
            range: l.range,
        })
    }

    /// Conservative membership: cone and range gate only (no gobo/occlusion).
    pub fn contains(&self, p: Vec3) -> bool {
        let v = p - self.apex;
        let d2 = v.dot(v);
        if d2 > self.range * self.range {
            return false;
        }
        if d2 < 1e-12 {
            return true;
        }
        let cosang = self.axis.dot(v) / d2.sqrt();
        // Small epsilon keeps boundary points inside the bound.
        cosang >= self.cos_outer - 1e-6
    }

    /// Ray interval inside this bound, clipped to `[0, t_end]`.
    fn ray_interval(&self, ray: &Ray, t_end: f32) -> Option<(f32, f32)> {
        let sphere = ray_sphere_interval(ray, self.apex, self.range)?;
        let (cands, n) = self.cone_candidates(ray);
        let mut best: Option<(f32, f32)> = None;
        for &(clo, chi) in &cands[..n] {
            let lo = clo.max(sphere.0).max(0.0);
            let hi = chi.min(sphere.1).min(t_end);
            if hi <= lo {
                continue;
            }
            // Guard against analytic edge cases: at least one point of the
            // candidate must actually satisfy the bound.
            let probe = 0.5 * (lo + hi);
            if !self.contains(ray.at(probe)) {
                continue;
            }
            best = Some(match best {
                None => (lo, hi),
                Some(b) => (b.0.min(lo), b.1.max(hi)),
            });
        }
        best
    }

    /// Candidate t-intervals for the infinite cone, before half-space
    /// trimming. At most two intervals, matching the quadratic sign.
    /// Written into a fixed array: this runs once per ray, so it must not
    /// allocate.
    fn cone_candidates(&self, ray: &Ray) -> ([(f32, f32); 2], usize) {
        const BIG: f32 = 1e30;
        let o = ray.origin;
        let d = ray.dir;
        let w = o - self.apex;
        let m = self.axis.dot(w);
        let n = self.axis.dot(d);
        let wd = w.dot(d);
        let ww = w.dot(w);
        let c2 = self.cos_outer * self.cos_outer;
        // (m + t n)² − c²(|w|² + 2t w·d + t²|d|²) ≥ 0
        let qa = n * n - c2;
        let qb = 2.0 * (m * n - c2 * wd);
        let qc = m * m - c2 * ww;
        let mut raw = [(-BIG, BIG); 2];
        let mut nraw = 0usize;
        if qa.abs() > 1e-7 {
            let disc = qb * qb - 4.0 * qa * qc;
            if qa > 0.0 {
                if disc <= 0.0 {
                    raw[0] = (-BIG, BIG);
                    nraw = 1;
                } else {
                    let sq = disc.sqrt();
                    let r1 = ((-qb - sq) / (2.0 * qa)).min((-qb + sq) / (2.0 * qa));
                    let r2 = ((-qb - sq) / (2.0 * qa)).max((-qb + sq) / (2.0 * qa));
                    raw[0] = (-BIG, r1);
                    raw[1] = (r2, BIG);
                    nraw = 2;
                }
            } else if disc > 0.0 {
                let sq = disc.sqrt();
                let r1 = ((-qb - sq) / (2.0 * qa)).min((-qb + sq) / (2.0 * qa));
                let r2 = ((-qb - sq) / (2.0 * qa)).max((-qb + sq) / (2.0 * qa));
                raw[0] = (r1, r2);
                nraw = 1;
            }
        } else if qb.abs() > 1e-9 {
            let t0 = -qc / qb;
            raw[0] = if qb > 0.0 { (t0, BIG) } else { (-BIG, t0) };
            nraw = 1;
        } else if qc >= 0.0 {
            raw[0] = (-BIG, BIG);
            nraw = 1;
        }
        // Keep only the forward nappe: m + t n ≥ 0.
        let mut out = [(-BIG, BIG); 2];
        let mut nout = 0usize;
        for &(mut lo, mut hi) in &raw[..nraw] {
            if n.abs() > 1e-9 {
                let tl = -m / n;
                if n > 0.0 {
                    lo = lo.max(tl);
                } else {
                    hi = hi.min(tl);
                }
            } else if m < 0.0 {
                continue;
            }
            if hi > lo {
                out[nout] = (lo, hi);
                nout += 1;
            }
        }
        (out, nout)
    }
}

fn ray_sphere_interval(ray: &Ray, center: Vec3, radius: f32) -> Option<(f32, f32)> {
    let oc = ray.origin - center;
    let a = ray.dir.dot(ray.dir);
    if a < 1e-12 {
        return None;
    }
    let b = 2.0 * oc.dot(ray.dir);
    let c = oc.dot(oc) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let r1 = (-b - sq) / (2.0 * a);
    let r2 = (-b + sq) / (2.0 * a);
    Some((r1.min(r2), r1.max(r2)))
}

/// Sorted disjoint spans of a ray that can contain non-zero source.
#[derive(Clone, Copy, Debug)]
pub struct SpanSet {
    spans: [(f32, f32); MAX_LIT_SPANS],
    len: usize,
    /// True when the union overflowed or was degenerate; the full ray is used.
    pub full_fallback: bool,
}

impl SpanSet {
    pub fn empty() -> SpanSet {
        SpanSet { spans: [(0.0, 0.0); MAX_LIT_SPANS], len: 0, full_fallback: false }
    }

    pub fn full(t_end: f32) -> SpanSet {
        let mut s = SpanSet::empty();
        if t_end > 0.0 {
            s.spans[0] = (0.0, t_end);
            s.len = 1;
        }
        s.full_fallback = true;
        s
    }

    pub fn spans(&self) -> &[(f32, f32)] {
        &self.spans[..self.len]
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Clone, Debug, PartialEq)]
struct LightKey {
    position: [u32; 3],
    target_axis: [u32; 3],
    outer: u32,
    range: u32,
    enabled: bool,
    intensity: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SummaryKey {
    fog_sigma_bits: u32,
    occluder_x_bits: u32,
    occluder_visible: bool,
    occluder_scale_bits: u32,
    lights: Vec<LightKey>,
}

impl SummaryKey {
    /// Every input that can change the conservative bounds or the medium.
    /// `time` is folded in through the animated occluder position.
    pub fn new(params: &StageVolParams, time: f32) -> SummaryKey {
        let lights = params
            .lights
            .iter()
            .map(|l| {
                let axis = Vec3::new(
                    l.target[0] - l.position[0],
                    l.target[1] - l.position[1],
                    l.target[2] - l.position[2],
                )
                .normalized();
                LightKey {
                    position: [l.position[0].to_bits(), l.position[1].to_bits(), l.position[2].to_bits()],
                    target_axis: [axis.x.to_bits(), axis.y.to_bits(), axis.z.to_bits()],
                    outer: l.outer_cone_deg.to_bits(),
                    range: l.range.to_bits(),
                    enabled: l.enabled && l.intensity > 0.0,
                    intensity: l.intensity.to_bits(),
                }
            })
            .collect();
        SummaryKey {
            fog_sigma_bits: params.fog_sigma.to_bits(),
            occluder_x_bits: params.occluder_x_at(time).to_bits(),
            occluder_visible: params.occluder_visible,
            occluder_scale_bits: params.occluder_scale.to_bits(),
            lights,
        }
    }
}

#[derive(Clone, Debug)]
pub struct InfluenceSummary {
    pub key: SummaryKey,
    bounds: [Option<LightBound>; MAX_LIGHTS],
}

impl InfluenceSummary {
    pub fn build(params: &StageVolParams, time: f32) -> InfluenceSummary {
        let mut bounds: [Option<LightBound>; MAX_LIGHTS] = [None; MAX_LIGHTS];
        for (i, l) in params.lights.iter().take(MAX_LIGHTS).enumerate() {
            let prepared = PreparedLight::prepare(l);
            bounds[i] = LightBound::from_prepared(&prepared);
        }
        InfluenceSummary { key: SummaryKey::new(params, time), bounds }
    }

    pub fn bounds(&self) -> &[Option<LightBound>; MAX_LIGHTS] {
        &self.bounds
    }

    /// Disjoint, sorted, non-zero spans of `ray` that may contain light.
    /// If the union cannot be represented within [`MAX_LIT_SPANS`], the whole
    /// ray is returned so no illumination is dropped.
    pub fn spans(&self, ray: &Ray, t_end: f32) -> SpanSet {
        if !(t_end > 0.0) || !t_end.is_finite() {
            return SpanSet::empty();
        }
        // At most one interval per light (cone∩sphere is convex), so this
        // fixed-size buffer cannot overflow; no heap use on the per-ray path.
        let mut intervals: [(f32, f32); MAX_LIGHTS] = [(0.0, 0.0); MAX_LIGHTS];
        let mut n = 0usize;
        for b in self.bounds.iter().flatten() {
            if let Some(iv) = b.ray_interval(ray, t_end) {
                intervals[n] = iv;
                n += 1;
            }
        }
        if n == 0 {
            return SpanSet::empty();
        }
        // Insertion sort (n ≤ 2).
        for i in 1..n {
            let key = intervals[i];
            let mut j = i;
            while j > 0 && intervals[j - 1].0 > key.0 {
                intervals[j] = intervals[j - 1];
                j -= 1;
            }
            intervals[j] = key;
        }
        let mut set = SpanSet::empty();
        for &(lo, hi) in &intervals[..n] {
            if hi <= lo {
                continue;
            }
            if set.len > 0 {
                let last = &mut set.spans[set.len - 1];
                if lo <= last.1 + 1e-5 {
                    last.1 = last.1.max(hi);
                    continue;
                }
            }
            if set.len == MAX_LIT_SPANS {
                return SpanSet::full(t_end);
            }
            set.spans[set.len] = (lo, hi);
            set.len += 1;
        }
        set
    }
}

/// Caches one summary per parameter state, counting rebuilds and reuses so
/// the benchmark and UI can show that invalidation actually happened.
pub struct SummaryCache {
    summary: Option<InfluenceSummary>,
    pub rebuilds: u64,
    pub reuses: u64,
}

impl Default for SummaryCache {
    fn default() -> Self {
        SummaryCache { summary: None, rebuilds: 0, reuses: 0 }
    }
}

impl SummaryCache {
    /// Return a summary valid for `params`/`time`. Rebuilds when the key
    /// changed (any light or medium edit), otherwise reuses. Never returns
    /// bounds from a stale parameter state.
    pub fn get(&mut self, params: &StageVolParams, time: f32) -> &InfluenceSummary {
        let key = SummaryKey::new(params, time);
        let stale = self.summary.as_ref().map_or(true, |s| s.key != key);
        if stale {
            self.summary = Some(InfluenceSummary::build(params, time));
            self.rebuilds += 1;
        } else {
            self.reuses += 1;
        }
        self.summary.as_ref().expect("summary built above")
    }

    pub fn is_valid_for(&self, params: &StageVolParams, time: f32) -> bool {
        self.summary
            .as_ref()
            .is_some_and(|s| s.key == SummaryKey::new(params, time))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::integrate::{in_scatter, PreparedLight};

    fn params() -> StageVolParams {
        StageVolParams::default()
    }

    #[test]
    fn bound_contains_every_lit_sample() {
        let p = params();
        let scene = crate::kernel::scene::Scene::build(10.0, false, 1.0);
        let prepared: Vec<_> = p.lights.iter().map(PreparedLight::prepare).collect();
        let summary = InfluenceSummary::build(&p, 0.0);
        let bounds: Vec<_> = summary.bounds().iter().flatten().copied().collect();
        assert!(!bounds.is_empty());
        let mut checked = 0;
        for i in 0..24 {
            for j in 0..24 {
                for k in 0..24 {
                    let q = Vec3::new(
                        -5.0 + 10.0 * i as f32 / 23.0,
                        0.0 + 4.5 * j as f32 / 23.0,
                        -3.4 + 5.9 * k as f32 / 23.0,
                    );
                    if in_scatter(&scene, &prepared, q).length() > 0.0 {
                        assert!(
                            bounds.iter().any(|b| b.contains(q)),
                            "lit sample outside conservative bound: {q:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "test grid found no lit samples");
    }

    #[test]
    fn spans_cover_lit_segments_and_skip_dark_ones() {
        let p = params();
        let scene = crate::kernel::scene::Scene::build(p.occluder_x, p.occluder_visible, p.occluder_scale);
        let prepared: Vec<_> = p.lights.iter().map(PreparedLight::prepare).collect();
        let camera = crate::kernel::integrate::Camera::from_params(&p, 1.0);
        let ray = camera.ray(50.0, 42.0, 100.0, 80.0);
        let summary = InfluenceSummary::build(&p, 0.0);
        let set = summary.spans(&ray, 40.0);
        assert!(!set.is_empty());
        // Every lit sample along the ray lies inside a span.
        let mut lit_seen = 0;
        for i in 0..400 {
            let t = 40.0 * i as f32 / 399.0;
            if in_scatter(&scene, &prepared, ray.at(t)).length() > 0.0 {
                lit_seen += 1;
                assert!(
                    set.spans().iter().any(|&(a, b)| t >= a && t <= b),
                    "lit t={t} outside spans {:?}",
                    set.spans()
                );
            }
        }
        assert!(lit_seen > 0, "ray through the beam should be lit somewhere");
        // A ray pointing away from the stage beam must skip everything.
        let away = Ray { origin: Vec3::new(0.0, 2.0, 20.0), dir: Vec3::new(0.0, 1.0, 0.0).normalized() };
        let dark = summary.spans(&away, 40.0);
        assert!(dark.is_empty(), "dark ray should have no lit spans: {:?}", dark.spans());
    }

    #[test]
    fn summary_invalidates_on_light_and_medium_changes() {
        let mut cache = SummaryCache::default();
        let mut p = params();
        {
            cache.get(&p, 0.0);
        }
        assert_eq!(cache.rebuilds, 1);
        cache.get(&p, 0.0);
        assert_eq!(cache.reuses, 1);
        assert!(cache.is_valid_for(&p, 0.0));

        p.lights[0].outer_cone_deg = 24.0;
        assert!(!cache.is_valid_for(&p, 0.0));
        cache.get(&p, 0.0);
        assert_eq!(cache.rebuilds, 2);

        p.fog_sigma *= 2.0;
        assert!(!cache.is_valid_for(&p, 0.0));
        cache.get(&p, 0.0);
        assert_eq!(cache.rebuilds, 3);

        p.occluder_x = 0.5;
        cache.get(&p, 0.0);
        assert_eq!(cache.rebuilds, 4);
    }

    #[test]
    fn unused_light_does_not_bound() {
        let mut p = params();
        p.lights[0].enabled = false;
        p.lights[1].enabled = false;
        let summary = InfluenceSummary::build(&p, 0.0);
        assert!(summary.bounds().iter().all(|b| b.is_none()));
        let camera = crate::kernel::integrate::Camera::from_params(&p, 1.0);
        let set = summary.spans(&camera.ray(50.0, 42.0, 100.0, 80.0), 40.0);
        assert!(set.is_empty());
    }

    #[test]
    fn animated_occluder_rebuilds_stale_summary() {
        // The summary must not be reused across a timeline edit even when the
        // shape key (lights/fog) is unchanged, because the occluder animates.
        let mut p = params();
        p.occluder_animate = true;
        let mut cache = SummaryCache::default();
        cache.get(&p, 0.0);
        assert!(!cache.is_valid_for(&p, 0.5));
        cache.get(&p, 0.5);
        assert_eq!(cache.rebuilds, 2);
    }
}
