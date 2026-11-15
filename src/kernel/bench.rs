//! Beam cost benchmark: like-for-like comparison of the uniform reference
//! integrator and the empty-region-skipping integrator.
//!
//! Load is increased step by step — full frame, sparse beam, then a density
//! discontinuity (fog σ change) — and each level records
//!
//! * the step counts the optimized path actually ran (`CostBreakdown`),
//! * the luminance error against a converged ground truth (`MAX_STEPS`
//!   uniform integration) per level, for the skip path *and* for the uniform
//!   path at the same user step budget, so the skip path is never credited
//!   for error it did not earn,
//! * wall time for both paths. Speed is only reported after the full skip
//!   path has executed.
//!
//! The benchmark refuses to report success on default/zeroed counters: every
//! level carries execution evidence and the caller checks
//! [`CostBreakdown::has_execution_evidence`].

use serde::{Deserialize, Serialize};
use std::time::Instant;

use crate::cpu::{CpuRenderer, Frame};
use crate::kernel::cost::CostBreakdown;
use crate::kernel::params::{StageVolParams, ViewMode, MAX_STEPS};

/// One load level of the benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadLevel {
    /// Full frame at the default step count (the original workload).
    FullFrame,
    /// Fewer steps so the beam's sparse pixels expose the empty-region skip.
    SparseBeam,
    /// Sparse beam plus a density discontinuity: the medium changes and the
    /// step schedule must react instead of reusing a stale range.
    DensityJump,
}

impl LoadLevel {
    pub fn label(self) -> &'static str {
        match self {
            LoadLevel::FullFrame => "full-frame",
            LoadLevel::SparseBeam => "sparse-beam",
            LoadLevel::DensityJump => "density-jump",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelResult {
    pub level: LoadLevel,
    pub steps: u32,
    pub fog_sigma: f32,
    /// Pixels where either path produced non-zero volume radiance.
    pub lit_pixels: u64,
    pub total_pixels: u64,
    /// Cost evidence from the optimized (skip) path.
    pub skip_cost: CostBreakdown,
    /// Cost evidence from the uniform path at the same step budget.
    pub reference_cost: CostBreakdown,
    /// Mean/max absolute luminance error of the skip path vs ground truth.
    pub skip_error_vs_truth_mean: f64,
    pub skip_error_vs_truth_max: f64,
    /// Mean/max absolute luminance error of the uniform path vs ground truth.
    pub reference_error_vs_truth_mean: f64,
    pub reference_error_vs_truth_max: f64,
    /// Relative error of the skip path, `skip_mean / truth_mean`.
    pub relative_skip_error: f64,
    /// Relative error of the uniform path, `reference_mean / truth_mean`.
    pub relative_reference_error: f64,
    /// Mean transmittance difference skip vs uniform (behavior beyond light).
    pub mean_transmittance_delta: f64,
    pub skip_ms: f64,
    pub reference_ms: f64,
    pub truth_ms: f64,
    pub speedup: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub backend: String,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<LevelResult>,
    pub notes: Vec<String>,
}

impl BenchmarkReport {
    pub fn all_levels_have_evidence(&self) -> bool {
        self.levels.iter().all(|l| {
            l.skip_cost.has_execution_evidence()
                && l.reference_cost.has_execution_evidence()
                && l.total_pixels > 0
        })
    }

    pub fn max_relative_skip_error(&self) -> f64 {
        self.levels.iter().map(|l| l.relative_skip_error).fold(0.0, f64::max)
    }

    /// Speedup is only meaningful once every level's skip path executed.
    pub fn speed_reported(&self) -> bool {
        self.all_levels_have_evidence() && self.levels.iter().all(|l| l.skip_ms > 0.0)
    }
}

fn luminance(v: [f32; 3]) -> f64 {
    (v[0] as f64 + v[1] as f64 + v[2] as f64) / 3.0
}

fn mean_luminance(frame: &Frame) -> f64 {
    if frame.volume.is_empty() {
        return 0.0;
    }
    let sum: f64 = frame.volume.iter().map(|v| luminance(*v)).sum();
    sum / frame.volume.len() as f64
}

fn error_vs_truth(frame: &Frame, truth: &Frame) -> (f64, f64) {
    let n = frame.volume.len().min(truth.volume.len()).max(1);
    let mut sum = 0.0;
    let mut max: f64 = 0.0;
    for i in 0..n {
        let a = luminance(frame.volume[i]);
        let b = luminance(truth.volume[i]);
        let d = (a - b).abs();
        sum += d;
        max = max.max(d);
    }
    (sum / n as f64, max)
}

fn mean_transmittance_delta(a: &Frame, b: &Frame) -> f64 {
    let n = a.transmittance.len().min(b.transmittance.len()).max(1);
    let sum: f64 = (0..n)
        .map(|i| (a.transmittance[i] as f64 - b.transmittance[i] as f64).abs())
        .sum();
    sum / n as f64
}

/// Render `params` three ways: converged ground truth, uniform at the user
/// budget, and the optimized skip path. Error is measured against the truth,
/// speed against the matched-budget uniform path.
///
/// Timings interleave the two budgets over several repetitions and keep the
/// best time for each, so a single scheduler hiccup cannot flip the sign of
/// the comparison. Truth (which only feeds the error gate) is timed once.
fn run_level(level: LoadLevel, params: &StageVolParams, w: u32, h: u32) -> LevelResult {
    const TIMING_REPEATS: usize = 7;
    let cpu = CpuRenderer::new(w, h);

    let mut truth_params = params.clone();
    truth_params.steps = MAX_STEPS;
    let t_truth = Instant::now();
    let truth = cpu.render(&truth_params, 0.0);
    let truth_ms = t_truth.elapsed().as_secs_f64() * 1000.0;

    let mut reference: Option<Frame> = None;
    let mut skip: Option<Frame> = None;
    let mut reference_ms = f64::INFINITY;
    let mut skip_ms = f64::INFINITY;
    let mut reference_cost = CostBreakdown::default();
    let mut skip_cost = CostBreakdown::default();
    for _ in 0..TIMING_REPEATS {
        let t_ref = Instant::now();
        let (r, rc) = cpu.render_reference(params, 0.0);
        reference_ms = reference_ms.min(t_ref.elapsed().as_secs_f64() * 1000.0);
        reference = Some(r);
        reference_cost = rc;

        let t_skip = Instant::now();
        let (s, sc) = cpu.render_with_stats(params, 0.0);
        skip_ms = skip_ms.min(t_skip.elapsed().as_secs_f64() * 1000.0);
        skip = Some(s);
        skip_cost = sc;
    }
    let reference = reference.expect("timing loop ran");
    let skip = skip.expect("timing loop ran");

    let npix = (w * h) as usize;
    let mut lit_pixels = 0u64;
    for i in 0..npix {
        if luminance(reference.volume[i]) != 0.0 || luminance(skip.volume[i]) != 0.0 {
            lit_pixels += 1;
        }
    }

    let (skip_mean, skip_max) = error_vs_truth(&skip, &truth);
    let (ref_mean, ref_max) = error_vs_truth(&reference, &truth);
    let truth_mean = mean_luminance(&truth);
    let rel_skip = if truth_mean > 1e-12 { skip_mean / truth_mean } else { 0.0 };
    let rel_ref = if truth_mean > 1e-12 { ref_mean / truth_mean } else { 0.0 };
    let speedup = if skip_ms > 0.0 { reference_ms / skip_ms } else { 0.0 };

    LevelResult {
        level,
        steps: params.steps,
        fog_sigma: params.fog_sigma,
        lit_pixels,
        total_pixels: npix as u64,
        skip_cost,
        reference_cost,
        skip_error_vs_truth_mean: skip_mean,
        skip_error_vs_truth_max: skip_max,
        reference_error_vs_truth_mean: ref_mean,
        reference_error_vs_truth_max: ref_max,
        relative_skip_error: rel_skip,
        relative_reference_error: rel_ref,
        mean_transmittance_delta: mean_transmittance_delta(&skip, &reference),
        skip_ms,
        reference_ms,
        truth_ms,
        speedup,
    }
}

/// Run the staged benchmark. Load increases in this order:
/// 1. full frame at the default step count (original workload),
/// 2. sparse beam at a low step count, where the beam occupies a small
///    fraction of the frame,
/// 3. density jump: low-σ and high-σ settings at the same step budget, so a
///    stale step schedule or stale bound would show as an error jump.
pub fn run_benchmark(w: u32, h: u32, backend: &str) -> BenchmarkReport {
    let mut levels = Vec::new();
    let mut notes = vec![
        "亮度误差以 MAX_STEPS 均匀积分为收敛参照；跳过路径与同预算均匀路径分别测量，避免把参照自身的近似当成跳过路径的精度。".into(),
        "步数为优化路径实际执行的样本数；参考路径的 baseline_samples 为同预算均匀积分。".into(),
        "透过率按无单位累积，遮挡查询的 max_t 仍为米制单位，与改动前一致。".into(),
    ];

    let mut p = StageVolParams::default();
    p.view_mode = ViewMode::VolumeOnly;
    levels.push(run_level(LoadLevel::FullFrame, &p, w, h));
    notes.push(
        "每档速度取 7 次交错渲染的最小值（跳过/同预算均匀交替），避免单次调度抖动翻转结论。".into(),
    );

    p.steps = 24;
    p.fog_sigma = 0.09;
    levels.push(run_level(LoadLevel::SparseBeam, &p, w, h));

    p.steps = 24;
    p.fog_sigma = 0.03;
    levels.push(run_level(LoadLevel::DensityJump, &p, w, h));
    p.fog_sigma = 0.35;
    levels.push(run_level(LoadLevel::DensityJump, &p, w, h));
    notes.push(format!("密度突变档位：σ=0.03 与 σ=0.35（同一 N={}）。", p.steps));

    BenchmarkReport {
        backend: backend.into(),
        width: w,
        height: h,
        levels,
        notes,
    }
}

/// Hard gate for the benchmark. Rejects a report that:
/// * has no levels or lacks execution evidence (zeroed/default counters),
/// * has a skip-path error above `max_relative_error`,
/// * has a skip path *worse* than the matched-budget uniform path by more
///   than `regression_margin` (that would be a behavior regression, not a
///   step/error trade).
pub fn validate_benchmark(
    report: &BenchmarkReport,
    max_relative_error: f64,
    regression_margin: f64,
) -> Result<(), String> {
    if report.levels.is_empty() {
        return Err("基准没有档位（不接受空基准）".into());
    }
    if !report.all_levels_have_evidence() {
        return Err("基准缺少执行证据（rays/samples/baseline_samples 为零）".into());
    }
    if report.max_relative_skip_error() > max_relative_error {
        return Err(format!(
            "跳过路径亮度误差 {:.4} 超过容差 {:.4}",
            report.max_relative_skip_error(),
            max_relative_error
        ));
    }
    for l in &report.levels {
        if l.relative_skip_error > l.relative_reference_error + regression_margin {
            return Err(format!(
                "档位 {} 跳过路径误差 {:.4} 劣于同预算均匀路径 {:.4}",
                l.level.label(),
                l.relative_skip_error,
                l.relative_reference_error
            ));
        }
    }
    Ok(())
}
