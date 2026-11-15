//! Cost accounting for the bounded-step volume integrator.
//!
//! The counters here are the execution evidence used by the beam cost
//! benchmark: how many rays ran, how many samples the unskipped reference
//! would have taken, how many the optimized path actually took, how much
//! distance was skipped, and how the samples split between zero-contribution
//! and high-density regions. A default/zeroed value is therefore *not* a
//! successful measurement; callers must check `rays`, `samples` and
//! `baseline_samples` before trusting an error number.

use serde::{Deserialize, Serialize};

/// Samples whose source luminance reaches this value are counted as
/// "high density" in the default benchmark. The value is in the crate's
/// radiance-like units (spot intensity / m² before cone and mask factors).
pub const DEFAULT_HIGH_DENSITY_THRESHOLD: f32 = 1.0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Pixels (rays) integrated.
    pub rays: u64,
    /// Samples the uniform reference integrator would have evaluated:
    /// `rays × steps`.
    pub baseline_samples: u64,
    /// Samples actually evaluated in the lit spans.
    pub samples: u64,
    /// Evaluated lit-span samples whose source term is exactly zero
    /// (gobo bars, cone edge, occluded samples inside the conservative bound).
    pub empty_samples: u64,
    /// Evaluated samples whose source luminance is at or above
    /// `high_density_threshold`.
    pub high_density_samples: u64,
    /// Distance skipped without evaluating the source term, meters.
    pub skipped_meters: f64,
    /// Distance actually integrated inside lit spans, meters.
    pub integrated_meters: f64,
    /// Shadow-ray queries issued (same meter units as the reference path).
    pub occlusion_queries: u64,
    /// Transmittance multiplications, including exact gap updates.
    pub transmittance_updates: u64,
    /// Number of lit spans integrated.
    pub spans: u64,
    /// Rays where the error-bound step request exceeded the step budget.
    pub error_bound_violations: u64,
    /// Conservative summary cache rebuilds.
    pub summary_rebuilds: u64,
    /// Conservative summary cache reuses.
    pub summary_reuses: u64,
}

impl CostBreakdown {
    pub fn empty_fraction(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.empty_samples as f64 / self.samples as f64
        }
    }

    pub fn high_density_fraction(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.high_density_samples as f64 / self.samples as f64
        }
    }

    pub fn sample_saving(&self) -> f64 {
        if self.baseline_samples == 0 {
            0.0
        } else {
            1.0 - self.samples as f64 / self.baseline_samples as f64
        }
    }

    pub fn skipped_fraction(&self) -> f64 {
        let total = self.skipped_meters + self.integrated_meters;
        if total <= 0.0 {
            0.0
        } else {
            self.skipped_meters / total
        }
    }

    pub fn merge(&mut self, o: &CostBreakdown) {
        self.rays += o.rays;
        self.baseline_samples += o.baseline_samples;
        self.samples += o.samples;
        self.empty_samples += o.empty_samples;
        self.high_density_samples += o.high_density_samples;
        self.skipped_meters += o.skipped_meters;
        self.integrated_meters += o.integrated_meters;
        self.occlusion_queries += o.occlusion_queries;
        self.transmittance_updates += o.transmittance_updates;
        self.spans += o.spans;
        self.error_bound_violations += o.error_bound_violations;
        self.summary_rebuilds += o.summary_rebuilds;
        self.summary_reuses += o.summary_reuses;
    }

    /// True when the counters show that integration actually executed.
    /// Used by the benchmark to reject zeroed/default measurements.
    pub fn has_execution_evidence(&self) -> bool {
        self.rays > 0 && self.baseline_samples > 0 && self.samples > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_not_evidence() {
        assert!(!CostBreakdown::default().has_execution_evidence());
    }

    #[test]
    fn merge_adds_and_fractions_are_finite() {
        let mut a = CostBreakdown {
            rays: 10,
            baseline_samples: 100,
            samples: 40,
            empty_samples: 10,
            high_density_samples: 5,
            skipped_meters: 30.0,
            integrated_meters: 10.0,
            ..Default::default()
        };
        a.merge(&a.clone());
        assert_eq!(a.rays, 20);
        assert_eq!(a.samples, 80);
        assert!((a.sample_saving() - 0.6).abs() < 1e-12);
        assert!((a.empty_fraction() - 0.25).abs() < 1e-12);
        assert!((a.skipped_fraction() - 0.75).abs() < 1e-12);
        assert!(a.has_execution_evidence());
    }
}
