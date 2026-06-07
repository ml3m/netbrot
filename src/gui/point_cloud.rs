use three_d::*;

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

        in vec3 position;
        in vec4 color;

        out vec4 fragColor;
        out float tailWeight;

        void main() {
            gl_Position = viewProjection * modelMatrix * vec4(position, 1.0);
            float height = abs(position.z);
            tailWeight = clamp(height / max(tailScale, 1.0e-4), 0.0, 1.0);
            gl_PointSize = pointSize * (1.0 + tailEmphasis * tailWeight);
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
}

impl Default for PointCloudMaterial {
    fn default() -> Self {
        Self {
            point_size: 1.0,
            opacity: 1.0,
            tail_emphasis: 2.5,
            tail_scale: 0.35,
            is_transparent: true,
        }
    }
}

impl Material for PointCloudMaterial {
    fn fragment_shader_source(&self, _lights: &[&dyn Light]) -> String {
        "
        uniform float opacity;
        uniform float tailEmphasis;

        in vec4 fragColor;
        in float tailWeight;
        out vec4 outColor;

        void main() {
            // Soft circular points read better than square GL_POINTS splats.
            vec2 uv = gl_PointCoord - vec2(0.5);
            float dist2 = dot(uv, uv);
            if (dist2 > 0.25) {
                discard;
            }
            float soft = exp(-dist2 * 10.0);

            float boost = 1.0 + tailEmphasis * tailWeight;
            vec3 rgb = fragColor.rgb * boost;
            float alpha = fragColor.a * opacity * soft * boost;
            outColor = vec4(rgb, alpha);
        }
        "
        .to_string()
    }

    fn use_uniforms(&self, program: &Program, _viewer: &dyn Viewer, _lights: &[&dyn Light]) {
        program.use_uniform("pointSize", self.point_size);
        program.use_uniform("opacity", self.opacity);
        program.use_uniform("tailEmphasis", self.tail_emphasis);
        program.use_uniform("tailScale", self.tail_scale);
    }

    fn render_states(&self) -> RenderStates {
        RenderStates {
            write_mask: WriteMask::COLOR,
            depth_test: DepthTest::Always, // Points drawn over each other
            blend: Blend::TRANSPARENCY,    // Additive blending for density
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
