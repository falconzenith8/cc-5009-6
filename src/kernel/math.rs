//! Minimal 3D math shared by the CPU reference renderer and GPU buffer layouts.
//! Units are meters throughout the crate; angles are radians unless a
//! parameter name says `_deg`.

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3::new(0.0, 0.0, 0.0);
    pub const ONE: Vec3 = Vec3::new(1.0, 1.0, 1.0);
    pub const UP: Vec3 = Vec3::new(0.0, 1.0, 0.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3 { x, y, z }
    }
    pub fn splat(v: f32) -> Self {
        Vec3::new(v, v, v)
    }
    pub fn dot(self, o: Vec3) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalized(self) -> Vec3 {
        let l = self.length();
        if l > 1e-12 {
            self * (1.0 / l)
        } else {
            Vec3::ZERO
        }
    }
    pub fn min(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }
    pub fn max(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }
    pub fn clamp(self, lo: f32, hi: f32) -> Vec3 {
        Vec3::new(self.x.clamp(lo, hi), self.y.clamp(lo, hi), self.z.clamp(lo, hi))
    }
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl core::ops::Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl core::ops::Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl core::ops::Mul<f32> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f32) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }
}
impl core::ops::Mul<Vec3> for Vec3 {
    type Output = Vec3;
    fn mul(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x * o.x, self.y * o.y, self.z * o.z)
    }
}
impl core::ops::Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        Vec3::new(-self.x, -self.y, -self.z)
    }
}
impl core::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}

/// Row-major 4x4 matrix; `self * v` treats `v` as a column vector.
#[derive(Clone, Copy, Debug)]
pub struct Mat4 {
    pub m: [[f32; 4]; 4],
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    pub fn mul(&self, o: &Mat4) -> Mat4 {
        let mut r = Mat4 { m: [[0.0; 4]; 4] };
        for i in 0..4 {
            for j in 0..4 {
                r.m[i][j] = self.m[i][0] * o.m[0][j]
                    + self.m[i][1] * o.m[1][j]
                    + self.m[i][2] * o.m[2][j]
                    + self.m[i][3] * o.m[3][j];
            }
        }
        r
    }

    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let m = &self.m;
        let x = m[0][0] * p.x + m[0][1] * p.y + m[0][2] * p.z + m[0][3];
        let y = m[1][0] * p.x + m[1][1] * p.y + m[1][2] * p.z + m[1][3];
        let z = m[2][0] * p.x + m[2][1] * p.y + m[2][2] * p.z + m[2][3];
        let w = m[3][0] * p.x + m[3][1] * p.y + m[3][2] * p.z + m[3][3];
        if w.abs() > 1e-12 {
            Vec3::new(x / w, y / w, z / w)
        } else {
            Vec3::new(x, y, z)
        }
    }

    pub fn transform_dir(&self, d: Vec3) -> Vec3 {
        let m = &self.m;
        Vec3::new(
            m[0][0] * d.x + m[0][1] * d.y + m[0][2] * d.z,
            m[1][0] * d.x + m[1][1] * d.y + m[1][2] * d.z,
            m[2][0] * d.x + m[2][1] * d.y + m[2][2] * d.z,
        )
    }

    /// Right-handed view matrix looking from `eye` toward `target`, with the
    /// camera looking down -Z in view space (matches `perspective`).
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalized();
        let s = f.cross(up).normalized();
        let u = s.cross(f);
        Mat4 {
            m: [
                [s.x, s.y, s.z, -s.dot(eye)],
                [u.x, u.y, u.z, -u.dot(eye)],
                [-f.x, -f.y, -f.z, f.dot(eye)],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Perspective projection, right-handed, NDC z in [0, 1] (wgpu convention).
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let nf = 1.0 / (near - far);
        Mat4 {
            m: [
                [f / aspect, 0.0, 0.0, 0.0],
                [0.0, f, 0.0, 0.0],
                [0.0, 0.0, far * nf, far * near * nf],
                [0.0, 0.0, -1.0, 0.0],
            ],
        }
    }

    /// Column-major float array, the layout wgpu uniform buffers expect.
    pub fn to_cols_array(&self) -> [[f32; 4]; 4] {
        let m = &self.m;
        [
            [m[0][0], m[1][0], m[2][0], m[3][0]],
            [m[0][1], m[1][1], m[2][1], m[3][1]],
            [m[0][2], m[1][2], m[2][2], m[3][2]],
            [m[0][3], m[1][3], m[2][3], m[3][3]],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
}

impl Ray {
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn look_at_places_target_on_minus_z() {
        let eye = Vec3::new(0.0, 2.0, 6.0);
        let target = Vec3::new(0.0, 1.0, 0.0);
        let view = Mat4::look_at(eye, target, Vec3::UP);
        let tv = view.transform_point(target);
        assert!(tv.x.abs() < 1e-4 && tv.y.abs() < 1e-4 && tv.z < 0.0);
    }

    #[test]
    fn perspective_maps_near_to_zero() {
        let p = Mat4::perspective(1.0, 1.0, 0.1, 100.0);
        let v = p.transform_point(Vec3::new(0.0, 0.0, -0.1));
        assert!(v.z.abs() < 1e-5);
        let f = p.transform_point(Vec3::new(0.0, 0.0, -100.0));
        assert!((f.z - 1.0).abs() < 1e-4);
    }
}
