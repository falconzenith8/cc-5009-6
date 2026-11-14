//! Procedural gobo (pattern mask) evaluated in light-projection UV space.
//! The WGSL shader implements the same functions bit-for-bit; keep ids and
//! formulas in sync with `shaders/common.wgsl`.

pub const GOBO_NAMES: [&str; 5] = ["无遮片", "条形格栅", "棋盘格", "圆环阵列", "星形孔"];

/// uv in [0,1]^2 light-projection space, scale = tiling factor.
/// Returns a multiplier in [0, 1].
pub fn gobo(id: u32, uv: [f32; 2], scale: f32) -> f32 {
    let u = uv[0] - 0.5;
    let v = uv[1] - 0.5;
    match id {
        1 => {
            // Vertical slats: half the beam blocked.
            let s = (u * scale * core::f32::consts::TAU).sin();
            if s > 0.0 { 1.0 } else { 0.05 }
        }
        2 => {
            // Checkerboard.
            let cu = (u * scale).floor() as i32;
            let cv = (v * scale).floor() as i32;
            if (cu + cv).rem_euclid(2) == 0 { 1.0 } else { 0.0 }
        }
        3 => {
            // Ring array: bright rings around cell centers.
            let fu = (u * scale).fract() - 0.5;
            let fv = (v * scale).fract() - 0.5;
            let r = (fu * fu + fv * fv).sqrt();
            let ring = ((r * 9.0).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
            if r < 0.45 { ring } else { 0.0 }
        }
        4 => {
            // Star aperture: angular modulation around the beam axis.
            let r = (u * u + v * v).sqrt();
            let a = v.atan2(u);
            let star = 0.30 + 0.10 * (a * 5.0).cos();
            if r < star { 1.0 } else { 0.0 }
        }
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gobo_range_and_blocking() {
        for id in 0..GOBO_NAMES.len() as u32 {
            let mut blocked = false;
            for i in 0..64 {
                for j in 0..64 {
                    let g = gobo(id, [i as f32 / 63.0, j as f32 / 63.0], 4.0);
                    assert!((0.0..=1.0).contains(&g), "gobo {id} out of range");
                    if g < 0.5 {
                        blocked = true;
                    }
                }
            }
            if id != 0 {
                assert!(blocked, "gobo {id} should block part of the beam");
            }
        }
    }
}
