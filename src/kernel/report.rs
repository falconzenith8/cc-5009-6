//! Frame timing and test-scene reports, serialized to `out/report.json`.

use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameTiming {
    pub scene: String,
    pub frame: u32,
    pub steps: u32,
    pub backend: String,
    pub cpu_ms: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneReport {
    pub name: String,
    pub description: String,
    pub frames: Vec<FrameTiming>,
    pub outputs: Vec<String>,
    pub checks: Vec<CheckResult>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Report {
    pub backend: String,
    pub device: String,
    pub scenes: Vec<SceneReport>,
}

impl Report {
    pub fn mean_ms(scene: &SceneReport) -> f64 {
        if scene.frames.is_empty() {
            return 0.0;
        }
        scene.frames.iter().map(|f| f.cpu_ms).sum::<f64>() / scene.frames.len() as f64
    }
}

pub struct FrameTimer {
    start: Instant,
}

impl FrameTimer {
    pub fn begin() -> FrameTimer {
        FrameTimer { start: Instant::now() }
    }
    pub fn finish_ms(self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
}
