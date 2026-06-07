use three_d::*;

use crate::gui::colormap3d::{BlendMode3d, ColorMode3d, Colormap3d, COLORMAP_GLSL};

pub struct PointCloudGeometry {
    context: Context,
    position_buffer: VertexBuffer<Vec3>,
    color_buffer: VertexBuffer<Vec4>,
    count: usize,
}

impl PointCloudGeometry {
    pub fn new(context: &Context) -> Self {
        Self {
            context: context.clone(),
            position_buffer: VertexBuffer::new(context),
            color_buffer: VertexBuffer::new(context),
            count: 0,
        }
    }

    pub fn update(&mut self, positions: &[Vec3], colors: &[Srgba]) {
        self.count = positions.len();
        if self.count > 0 {
            self.position_buffer.fill(positions);
            let color_data: Vec<Vec4> = colors.iter().map(|c| c.to_linear_srgb()).collect();
            self.color_buffer.fill(&color_data);
        }
    }

    pub fn update_raw(&mut self, positions: &[Vec3], colors: &[Vec4]) {
        self.count = positions.len();
        if self.count > 0 {
            self.position_buffer.fill(positions);
            self.color_buffer.fill(colors);
        }
    }
}

impl Geometry for PointCloudGeometry {
    fn draw(&self, viewer: &dyn Viewer, program: &Program, render_states: RenderStates) {
        if self.count == 0 {
            return;
        }

        program.use_uniform("viewProjection", viewer.projection() * viewer.view());
        program.use_uniform("modelMatrix", Mat4::identity());

        program.use_vertex_attribute("position", &self.position_buffer);
        program.use_vertex_attribute("color", &self.color_buffer);

        program.draw_with(render_states, viewer.viewport(), || {
            unsafe {
                use glow::HasContext;
                // Enable point size control from shader
                self.context.enable(glow::PROGRAM_POINT_SIZE);
                self.context.draw_arrays(glow::POINTS, 0, self.count as i32);
                self.context.disable(glow::PROGRAM_POINT_SIZE);
            }
        });
    }

    fn vertex_shader_source(&self) -> String {
        "
        uniform mat4 viewProjection;
        uniform mat4 modelMatrix;
        uniform float pointSize;
        uniform float tailEmphasis;
        uniform float tailScale;
        uniform float pointSizeReference;

        // Clipping planes
        uniform vec3 clipMin;
        uniform vec3 clipMax;

        in vec3 position;
        in vec4 color;

        out vec4 fragColor;
        out float tailWeight;
        out vec3 worldPos;

        void main() {
            // Clip planes — discard points outside the clip box
            if (any(lessThan(position, clipMin)) || any(greaterThan(position, clipMax))) {
                gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
                gl_PointSize = 0.0;
                return;
            }

            gl_Position = viewProjection * modelMatrix * vec4(position, 1.0);
            worldPos = position;

            float height = abs(position.z);
            tailWeight = clamp(height / max(tailScale, 1.0e-4), 0.0, 1.0);

            // Distance-based point size attenuation
            float distScale = clamp(pointSizeReference / max(-gl_Position.w, 0.01), 0.4, 4.0);
            gl_PointSize = pointSize * distScale * (1.0 + tailEmphasis * tailWeight);

            fragColor = color;
        }
        "
        .to_string()
    }

    fn id(&self) -> GeometryId {
        GeometryId(0x7FFF) // Use the last available custom geometry ID
    }

    fn render_with_material(
        &self,
        material: &dyn Material,
        viewer: &dyn Viewer,
        lights: &[&dyn Light],
    ) {
        render_with_material(&self.context, viewer, self, material, lights).unwrap();
    }

    fn render_with_effect(
        &self,
        material: &dyn Effect,
        viewer: &dyn Viewer,
        lights: &[&dyn Light],
        color_texture: Option<ColorTexture>,
        depth_texture: Option<DepthTexture>,
    ) {
        render_with_effect(
            &self.context,
            viewer,
            self,
            material,
            lights,
            color_texture,
            depth_texture,
        ).unwrap();
    }

    fn aabb(&self) -> AxisAlignedBoundingBox {
        AxisAlignedBoundingBox::INFINITE
    }
}

pub struct PointCloudMaterial {
    pub point_size: f32,
    pub opacity: f32,
    /// Enlarge + brighten points far from z = 0 (the vertical orbit "rays").
    pub tail_emphasis: f32,
    /// |z| value at which tail boost reaches full strength.
    pub tail_scale: f32,
    pub is_transparent: bool,
    /// Reference distance for perspective point size scaling.
    pub point_size_reference: f32,

    // Shader-driven visual controls
    pub color_mode: ColorMode3d,
    pub colormap: Colormap3d,
    pub blend_mode: BlendMode3d,
    /// If true, the fragment shader uses the colormap + color mode instead of vertex colors.
    pub use_shader_coloring: bool,

    // Clip planes
    pub clip_min: Vec3,
    pub clip_max: Vec3,

    // Scene AABB for normalization in color modes
    pub scene_min: Vec3,
    pub scene_max: Vec3,
}

impl Default for PointCloudMaterial {
    fn default() -> Self {
        Self {
            point_size: 1.0,
            opacity: 1.0,
            tail_emphasis: 2.5,
            tail_scale: 0.35,
            is_transparent: true,
            point_size_reference: 5.0,
            color_mode: ColorMode3d::ReC,
            colormap: Colormap3d::Magma,
            blend_mode: BlendMode3d::Transparency,
            use_shader_coloring: true,
            clip_min: vec3(-10.0, -10.0, -10.0),
            clip_max: vec3(10.0, 10.0, 10.0),
            scene_min: vec3(-2.5, -1.15, -2.0),
            scene_max: vec3(0.55, 1.15, 2.0),
        }
    }
}

impl Material for PointCloudMaterial {
    fn fragment_shader_source(&self, _lights: &[&dyn Light]) -> String {
        let mut src = String::from(
            "
        uniform float opacity;
        uniform float tailEmphasis;
        uniform int colorMode;
        uniform int colormapId;
        uniform int useShaderColoring;
        uniform vec3 sceneMin;
        uniform vec3 sceneMax;

        in vec4 fragColor;
        in float tailWeight;
        in vec3 worldPos;
        out vec4 outColor;
        ",
        );

        // Inject colormap GLSL functions
        src.push_str(COLORMAP_GLSL);

        src.push_str(
            "
        void main() {
            // Soft circular points read better than square GL_POINTS splats.
            vec2 uv = gl_PointCoord - vec2(0.5);
            float dist2 = dot(uv, uv);
            if (dist2 > 0.25) {
                discard;
            }
            float soft = exp(-dist2 * 10.0);

            float boost = 1.0 + tailEmphasis * tailWeight;

            vec3 rgb;
            if (useShaderColoring == 1) {
                // Compute scalar t from color mode
                float t;
                vec3 range = sceneMax - sceneMin;
                if (colorMode == 0) {
                    // ReC: position.x normalized by bbox
                    t = (worldPos.x - sceneMin.x) / max(range.x, 1e-6);
                } else if (colorMode == 1) {
                    // PhaseC: atan2(Im(c), Re(c))
                    float phase = atan(worldPos.y, worldPos.x);
                    t = (phase + 3.14159265359) / (2.0 * 3.14159265359);
                } else {
                    // HeightZ: position.z normalized by z-range
                    t = (worldPos.z - sceneMin.z) / max(range.z, 1e-6);
                }
                t = clamp(t, 0.0, 1.0);
                rgb = applyColormap(colormapId, t) * boost;
            } else {
                rgb = fragColor.rgb * boost;
            }

            float alpha = opacity * soft * boost;
            outColor = vec4(rgb, alpha);
        }
        ",
        );

        src
    }

    fn use_uniforms(&self, program: &Program, _viewer: &dyn Viewer, _lights: &[&dyn Light]) {
        program.use_uniform("pointSize", self.point_size);
        program.use_uniform("opacity", self.opacity);
        program.use_uniform("tailEmphasis", self.tail_emphasis);
        program.use_uniform("tailScale", self.tail_scale);
        program.use_uniform("pointSizeReference", self.point_size_reference);
        program.use_uniform("colorMode", self.color_mode.uniform_id());
        program.use_uniform("colormapId", self.colormap.uniform_id());
        program.use_uniform(
            "useShaderColoring",
            if self.use_shader_coloring { 1i32 } else { 0i32 },
        );
        program.use_uniform("clipMin", self.clip_min);
        program.use_uniform("clipMax", self.clip_max);
        program.use_uniform("sceneMin", self.scene_min);
        program.use_uniform("sceneMax", self.scene_max);
    }

    fn render_states(&self) -> RenderStates {
        let blend = match self.blend_mode {
            BlendMode3d::Transparency => Blend::TRANSPARENCY,
            BlendMode3d::Additive => Blend::ADD,
            BlendMode3d::Opaque => Blend::Disabled,
        };
        RenderStates {
            write_mask: WriteMask::COLOR,
            depth_test: DepthTest::Always,
            blend,
            ..Default::default()
        }
    }

    fn material_type(&self) -> MaterialType {
        if self.is_transparent {
            MaterialType::Transparent
        } else {
            MaterialType::Opaque
        }
    }

    fn id(&self) -> EffectMaterialId {
        EffectMaterialId(0x4FFF) // Use a custom material ID
    }
}
