use three_d::*;

use crate::gui::colormap3d::{BlendMode3d, ColorMode3d, Colormap3d};
use crate::gui::point_cloud::{PointCloudGeometry, PointCloudMaterial};

/// Which generation pass produced a layer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerKind {
    Bifurcation,
    Wide,
}

/// One renderable point-cloud layer with its own material.
pub struct PointCloudLayer {
    pub kind: LayerKind,
    pub geometry: PointCloudGeometry,
    pub material: PointCloudMaterial,
    pub visible: bool,
    pub count: usize,
    pub cpu_positions: Vec<Vec3>,
    pub cpu_colors: Vec<three_d::Srgba>,
}

/// Visual indicator linking a c-value between the 2D and 3D views.
pub struct LinkOverlay {
    pub c_re: f32,
    pub c_im: f32,
    pub z_min: f32,
    pub z_max: f32,
    /// The GPU mesh for the vertical line.
    pub line_mesh: Option<Gm<Mesh, ColorMaterial>>,
}

/// Clipping volume (axis-aligned box in Re(c), Im(c), Re(z) space).
pub struct ClipBox {
    pub re_min: f32,
    pub re_max: f32,
    pub im_min: f32,
    pub im_max: f32,
    pub z_min: f32,
    pub z_max: f32,
}

impl Default for ClipBox {
    fn default() -> Self {
        Self {
            re_min: -10.0,
            re_max: 10.0,
            im_min: -10.0,
            im_max: 10.0,
            z_min: -10.0,
            z_max: 10.0,
        }
    }
}

/// Track whether the AABB has been set with actual data.
fn aabb_is_empty(aabb: &AxisAlignedBoundingBox) -> bool {
    // EMPTY has min > max, so check that
    let min = aabb.min();
    let max = aabb.max();
    min.x > max.x || min.y > max.y || min.z > max.z
}

/// Holds all 3D scene state: layers, overlay, AABB, visual settings.
pub struct Scene3D {
    pub bifurcation: Option<PointCloudLayer>,
    pub wide: Option<PointCloudLayer>,
    pub link: Option<LinkOverlay>,
    pub aabb: AxisAlignedBoundingBox,

    // Global shader controls
    pub color_mode: ColorMode3d,
    pub colormap: Colormap3d,
    pub blend_mode: BlendMode3d,
    pub clip: ClipBox,

    // Screenshot request flag
    pub screenshot_requested: bool,
}

impl Scene3D {
    pub fn new() -> Self {
        Self {
            bifurcation: None,
            wide: None,
            link: None,
            aabb: AxisAlignedBoundingBox::EMPTY,
            color_mode: ColorMode3d::ReC,
            colormap: Colormap3d::Magma,
            blend_mode: BlendMode3d::Transparency,
            clip: ClipBox::default(),
            screenshot_requested: false,
        }
    }

    /// Whether the AABB is empty / uninitialized.
    pub fn aabb_is_empty(&self) -> bool {
        aabb_is_empty(&self.aabb)
    }

    /// Iterate over layers that are present and visible.
    pub fn visible_layers(&self) -> Vec<&PointCloudLayer> {
        let mut layers = Vec::with_capacity(2);
        if let Some(l) = &self.bifurcation {
            if l.visible {
                layers.push(l);
            }
        }
        if let Some(l) = &self.wide {
            if l.visible {
                layers.push(l);
            }
        }
        layers
    }

    /// Iterate over mutable layers that are present and visible.
    pub fn visible_layers_mut(&mut self) -> Vec<&mut PointCloudLayer> {
        let mut layers = Vec::with_capacity(2);
        if let Some(l) = &mut self.bifurcation {
            if l.visible {
                layers.push(l);
            }
        }
        if let Some(l) = &mut self.wide {
            if l.visible {
                layers.push(l);
            }
        }
        layers
    }

    /// Compute AABB from all visible layer positions.
    /// Call this after updating layer geometry.
    pub fn compute_aabb_from_positions(&mut self, positions: &[Vec3]) {
        if positions.is_empty() {
            self.aabb = AxisAlignedBoundingBox::EMPTY;
            return;
        }
        let mut min = positions[0];
        let mut max = positions[0];
        for p in positions.iter().skip(1) {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        self.aabb = AxisAlignedBoundingBox::new_with_positions(&[min, max]);
    }

    /// Merge two sets of positions to compute a combined AABB.
    pub fn compute_aabb_merged(&mut self, pos_a: &[Vec3], pos_b: &[Vec3]) {
        let all: Vec<Vec3> = pos_a.iter().chain(pos_b.iter()).copied().collect();
        self.compute_aabb_from_positions(&all);
    }

    /// Suggest camera parameters that frame the current AABB nicely.
    /// Returns (eye, target, up).
    pub fn suggested_camera(&self) -> (Vec3, Vec3, Vec3) {
        if self.aabb_is_empty() {
            return (
                vec3(-1.0, -5.0, 1.5),
                vec3(-1.0, 0.0, 0.0),
                vec3(0.0, 0.0, 1.0),
            );
        }
        let min = self.aabb.min();
        let max = self.aabb.max();
        let center = (min + max) * 0.5;
        let size = max - min;

        // Match Python brot.py defaults:
        // target = center biased toward chaotic region
        let target = if self.bifurcation.is_some() {
            // Bias slightly toward the left (lower Re(c)) where bifurcation is dense
            vec3(center.x - size.x * 0.1, center.y, center.z * 0.3)
        } else {
            center
        };

        let eye = vec3(
            target.x - size.x * 0.3,
            target.y - 2.5 * size.y.max(size.x),
            target.z + 0.4 * size.z.max(0.5),
        );

        let up = vec3(0.0, 0.0, 1.0);
        (eye, target, up)
    }

    /// Reset clip box to match the current AABB (with small margin).
    pub fn reset_clip_to_aabb(&mut self) {
        if self.aabb_is_empty() {
            self.clip = ClipBox::default();
            return;
        }
        let min = self.aabb.min();
        let max = self.aabb.max();
        let margin = 0.1;
        self.clip = ClipBox {
            re_min: min.x - margin,
            re_max: max.x + margin,
            im_min: min.y - margin,
            im_max: max.y + margin,
            z_min: min.z - margin,
            z_max: max.z + margin,
        };
    }

    /// Build the link overlay geometry for a given c-value.
    pub fn update_link_overlay(&mut self, context: &Context, c_re: f32, c_im: f32) {
        let z_min = if self.aabb_is_empty() {
            -2.0
        } else {
            self.aabb.min().z - 0.1
        };
        let z_max = if self.aabb_is_empty() {
            2.0
        } else {
            self.aabb.max().z + 0.1
        };

        // Create a thin cylinder for the vertical line
        let mut cpu_mesh = CpuMesh::cylinder(16);
        let length = z_max - z_min;
        let radius = 0.01;
        
        // CpuMesh::cylinder goes from 0 to 1 along the x-axis.
        // We scale its x-length to match our z-range, and its y/z to the radius.
        // Then rotate around Y by -90 degrees so the local X axis points along the world Z axis.
        let transform = Mat4::from_translation(vec3(c_re, c_im, z_min))
            * Mat4::from_angle_y(degrees(-90.0))
            * Mat4::from_nonuniform_scale(length, radius, radius);
            
        cpu_mesh.transform(transform).unwrap();

        let mesh = Gm::new(
            Mesh::new(context, &cpu_mesh),
            ColorMaterial {
                color: Srgba::new(255, 50, 50, 255),
                ..Default::default()
            },
        );

        self.link = Some(LinkOverlay {
            c_re,
            c_im,
            z_min,
            z_max,
            line_mesh: Some(mesh),
        });
    }

    /// Clear the link overlay.
    pub fn clear_link_overlay(&mut self) {
        self.link = None;
    }
}
