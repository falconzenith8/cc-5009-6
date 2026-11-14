//! Procedural stage: floor deck, back/side/top curtains, and a movable
//! occluder panel. Geometry is analytic (rects + one box) so the CPU
//! reference renderer intersects rays exactly, and the same description
//! generates triangles for the GPU raster passes.

use super::math::{Ray, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Material {
    Floor,
    Curtain,
    Occluder,
}

impl Material {
    pub fn albedo(self) -> Vec3 {
        match self {
            Material::Floor => Vec3::new(0.32, 0.30, 0.28),
            Material::Curtain => Vec3::new(0.42, 0.07, 0.09),
            Material::Occluder => Vec3::new(0.55, 0.55, 0.58),
        }
    }
}

/// Axis-aligned rectangle. `axis` is the normal axis (0=x, 1=y, 2=z).
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub axis: u8,
    /// Normal direction along the axis (+1 or -1).
    pub sign: f32,
    /// Plane offset along the axis.
    pub offset: f32,
    /// Bounds on the other two axes.
    pub min_u: f32,
    pub max_u: f32,
    pub min_v: f32,
    pub max_v: f32,
    pub material: Material,
}

/// Axis-aligned box (the occluder panel).
#[derive(Clone, Copy, Debug)]
pub struct Panel {
    pub min: Vec3,
    pub max: Vec3,
    pub material: Material,
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub rects: Vec<Rect>,
    pub panel: Panel,
}

pub const PANEL_WIDTH: f32 = 1.1;
pub const PANEL_HEIGHT: f32 = 2.3;
pub const PANEL_THICKNESS: f32 = 0.06;
pub const PANEL_Z: f32 = -0.9;

impl Scene {
    /// Build the stage with the occluder panel centered at `occluder_x`,
    /// scaled by `scale` (width/height; thickness stays constant).
    pub fn build(occluder_x: f32, occluder_visible: bool, scale: f32) -> Scene {
        let rects = vec![
            // Stage deck (floor), normal +Y.
            Rect { axis: 1, sign: 1.0, offset: 0.0, min_u: -6.0, max_u: 6.0, min_v: -4.0, max_v: 2.5, material: Material::Floor },
            // Back curtain, normal +Z (facing the audience/camera).
            Rect { axis: 2, sign: 1.0, offset: -3.5, min_u: -6.0, max_u: 6.0, min_v: 0.0, max_v: 4.6, material: Material::Curtain },
            // Side curtains (legs), normals pointing toward stage center.
            Rect { axis: 0, sign: -1.0, offset: 5.2, min_u: -3.5, max_u: 2.5, min_v: 0.0, max_v: 4.6, material: Material::Curtain },
            Rect { axis: 0, sign: 1.0, offset: -5.2, min_u: -3.5, max_u: 2.5, min_v: 0.0, max_v: 4.6, material: Material::Curtain },
            // Top border curtain (valance), normal -Y hanging above the back.
            Rect { axis: 1, sign: -1.0, offset: 4.6, min_u: -6.0, max_u: 6.0, min_v: -4.0, max_v: 2.5, material: Material::Curtain },
        ];
        let panel = if occluder_visible {
            let hw = PANEL_WIDTH * 0.5 * scale;
            let hgt = PANEL_HEIGHT * scale;
            Panel {
                min: Vec3::new(occluder_x - hw, 0.0, PANEL_Z - PANEL_THICKNESS * 0.5),
                max: Vec3::new(occluder_x + hw, hgt, PANEL_Z + PANEL_THICKNESS * 0.5),
                material: Material::Occluder,
            }
        } else {
            // Degenerate, never-hit panel far outside the stage.
            Panel { min: Vec3::splat(1e6), max: Vec3::splat(1e6 + 0.01), material: Material::Occluder }
        };
        Scene { rects, panel }
    }

    /// Nearest intersection with any scene surface. Returns (t, material, normal).
    pub fn intersect(&self, ray: &Ray, t_max: f32) -> Option<(f32, Material, Vec3)> {
        let mut best: Option<(f32, Material, Vec3)> = None;
        let mut consider = |t: f32, m: Material, n: Vec3| {
            if t > 1e-4 && t < t_max && best.map_or(true, |(bt, _, _)| t < bt) {
                best = Some((t, m, n));
            }
        };
        for r in &self.rects {
            if let Some(t) = intersect_rect(ray, r) {
                let n = match r.axis {
                    0 => Vec3::new(r.sign, 0.0, 0.0),
                    1 => Vec3::new(0.0, r.sign, 0.0),
                    _ => Vec3::new(0.0, 0.0, r.sign),
                };
                consider(t, r.material, n);
            }
        }
        if let Some((t, n)) = intersect_box(ray, &self.panel) {
            consider(t, Material::Occluder, n);
        }
        best
    }

    /// True if the segment p0->p1 is blocked by any surface (shadow query).
    /// `max_t` is the distance from p0 to the light; anything closer blocks.
    pub fn occluded(&self, p0: Vec3, dir: Vec3, max_t: f32) -> bool {
        let ray = Ray { origin: p0, dir };
        self.intersect(&ray, max_t).is_some()
    }

    /// Triangle soup for GPU passes: [pos.xyz, normal.xyz, albedo.rgb] × vertices.
    pub fn triangles(&self) -> Vec<f32> {
        let mut out = Vec::new();
        for r in &self.rects {
            push_rect(&mut out, r);
        }
        push_box(&mut out, &self.panel);
        out
    }
}

fn intersect_rect(ray: &Ray, r: &Rect) -> Option<f32> {
    let (o, d) = (ray.origin, ray.dir);
    let (oa, da) = match r.axis {
        0 => (o.x, d.x),
        1 => (o.y, d.y),
        _ => (o.z, d.z),
    };
    if da.abs() < 1e-9 {
        return None;
    }
    let t = (r.offset - oa) / da;
    if t <= 0.0 {
        return None;
    }
    let p = o + d * t;
    let (u, v) = match r.axis {
        0 => (p.z, p.y),
        1 => (p.x, p.z),
        _ => (p.x, p.y),
    };
    if u >= r.min_u && u <= r.max_u && v >= r.min_v && v <= r.max_v {
        Some(t)
    } else {
        None
    }
}

fn intersect_box(ray: &Ray, b: &Panel) -> Option<(f32, Vec3)> {
    let o = ray.origin;
    let d = ray.dir;
    let mut tmin = f32::NEG_INFINITY;
    let mut tmax = f32::INFINITY;
    let mut axis_hit = 0u8;
    let mut sign_hit = 0.0f32;
    for axis in 0..3 {
        let (oa, da, mn, mx) = match axis {
            0 => (o.x, d.x, b.min.x, b.max.x),
            1 => (o.y, d.y, b.min.y, b.max.y),
            _ => (o.z, d.z, b.min.z, b.max.z),
        };
        if da.abs() < 1e-9 {
            if oa < mn || oa > mx {
                return None;
            }
            continue;
        }
        let inv = 1.0 / da;
        let mut t0 = (mn - oa) * inv;
        let mut t1 = (mx - oa) * inv;
        let mut sign = -1.0f32;
        if t0 > t1 {
            core::mem::swap(&mut t0, &mut t1);
            sign = 1.0;
        }
        if t0 > tmin {
            tmin = t0;
            axis_hit = axis;
            sign_hit = sign;
        }
        tmax = tmax.min(t1);
        if tmin > tmax {
            return None;
        }
    }
    if tmin <= 0.0 {
        return None;
    }
    let n = match axis_hit {
        0 => Vec3::new(sign_hit, 0.0, 0.0),
        1 => Vec3::new(0.0, sign_hit, 0.0),
        _ => Vec3::new(0.0, 0.0, sign_hit),
    };
    Some((tmin, n))
}

fn push_quad(out: &mut Vec<f32>, corners: [Vec3; 4], normal: Vec3, albedo: Vec3) {
    let idx = [0usize, 1, 2, 0, 2, 3];
    for &i in &idx {
        let c = corners[i];
        out.extend_from_slice(&[c.x, c.y, c.z, normal.x, normal.y, normal.z, albedo.x, albedo.y, albedo.z]);
    }
}

fn push_rect(out: &mut Vec<f32>, r: &Rect) {
    let albedo = r.material.albedo();
    let (normal, corners) = match r.axis {
        0 => {
            let n = Vec3::new(r.sign, 0.0, 0.0);
            let c = [
                Vec3::new(r.offset, r.min_v, r.min_u),
                Vec3::new(r.offset, r.min_v, r.max_u),
                Vec3::new(r.offset, r.max_v, r.max_u),
                Vec3::new(r.offset, r.max_v, r.min_u),
            ];
            (n, c)
        }
        1 => {
            let n = Vec3::new(0.0, r.sign, 0.0);
            let c = [
                Vec3::new(r.min_u, r.offset, r.min_v),
                Vec3::new(r.max_u, r.offset, r.min_v),
                Vec3::new(r.max_u, r.offset, r.max_v),
                Vec3::new(r.min_u, r.offset, r.max_v),
            ];
            (n, c)
        }
        _ => {
            let n = Vec3::new(0.0, 0.0, r.sign);
            let c = [
                Vec3::new(r.min_u, r.min_v, r.offset),
                Vec3::new(r.max_u, r.min_v, r.offset),
                Vec3::new(r.max_u, r.max_v, r.offset),
                Vec3::new(r.min_u, r.max_v, r.offset),
            ];
            (n, c)
        }
    };
    // Winding so the front face matches the normal (CCW when viewed from the
    // normal side). wgpu uses CCW front faces by default.
    push_quad(out, corners, normal, albedo);
}

fn push_box(out: &mut Vec<f32>, b: &Panel) {
    let a = b.material.albedo();
    let (mn, mx) = (b.min, b.max);
    // +Y top
    push_quad(out, [
        Vec3::new(mn.x, mx.y, mn.z), Vec3::new(mx.x, mx.y, mn.z),
        Vec3::new(mx.x, mx.y, mx.z), Vec3::new(mn.x, mx.y, mx.z),
    ], Vec3::new(0.0, 1.0, 0.0), a);
    // -Y bottom
    push_quad(out, [
        Vec3::new(mn.x, mn.y, mx.z), Vec3::new(mx.x, mn.y, mx.z),
        Vec3::new(mx.x, mn.y, mn.z), Vec3::new(mn.x, mn.y, mn.z),
    ], Vec3::new(0.0, -1.0, 0.0), a);
    // +Z front
    push_quad(out, [
        Vec3::new(mn.x, mn.y, mx.z), Vec3::new(mx.x, mn.y, mx.z),
        Vec3::new(mx.x, mx.y, mx.z), Vec3::new(mn.x, mx.y, mx.z),
    ], Vec3::new(0.0, 0.0, 1.0), a);
    // -Z back
    push_quad(out, [
        Vec3::new(mx.x, mn.y, mn.z), Vec3::new(mn.x, mn.y, mn.z),
        Vec3::new(mn.x, mx.y, mn.z), Vec3::new(mx.x, mx.y, mn.z),
    ], Vec3::new(0.0, 0.0, -1.0), a);
    // +X
    push_quad(out, [
        Vec3::new(mx.x, mn.y, mx.z), Vec3::new(mx.x, mn.y, mn.z),
        Vec3::new(mx.x, mx.y, mn.z), Vec3::new(mx.x, mx.y, mx.z),
    ], Vec3::new(1.0, 0.0, 0.0), a);
    // -X
    push_quad(out, [
        Vec3::new(mn.x, mn.y, mn.z), Vec3::new(mn.x, mn.y, mx.z),
        Vec3::new(mn.x, mx.y, mx.z), Vec3::new(mn.x, mx.y, mn.z),
    ], Vec3::new(-1.0, 0.0, 0.0), a);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_hit_from_camera_height() {
        let scene = Scene::build(0.0, true, 1.0);
        // Steep enough to land on the deck (front edge z=2.5).
        let ray = Ray { origin: Vec3::new(0.0, 2.0, 5.0), dir: Vec3::new(0.0, -1.0, -1.5).normalized() };
        let hit = scene.intersect(&ray, 100.0);
        assert!(matches!(hit, Some((_, Material::Floor, _))));
    }

    #[test]
    fn panel_blocks_segment() {
        // Segment ends past the panel (z=-0.9) but before the back curtain.
        let scene = Scene::build(0.0, true, 1.0);
        let p0 = Vec3::new(0.0, 1.0, 2.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        assert!(scene.occluded(p0, dir, 3.5));
        let scene_hidden = Scene::build(0.0, false, 1.0);
        assert!(!scene_hidden.occluded(p0, dir, 3.5));
    }
}
