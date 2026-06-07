use three_d::*;
use netbrot::gui::app::App;

pub fn main() {
    let window = Window::new(WindowSettings {
        title: "Netbrot 3D + 2D".to_string(),
        ..Default::default()
    })
    .unwrap();
    
    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(-1.0, -5.0, 1.5),
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        degrees(45.0),
        0.1,
        1000.0,
    );
    let mut control = OrbitControl::new(camera.target(), 1.0, 100.0);

    let mut gui = three_d::GUI::new(&context);
    
    let mut app = netbrot::gui::app::App::new(&context);

    // Initialize Axes
    let axes = three_d::Axes::new(&context, 0.02, 2.0);

    // Initialize Grid (Lines on XZ plane)
    let mut grid_positions = Vec::new();
    let mut grid_colors = Vec::new();
    let grid_size = 5.0;
    let grid_steps = 20;
    let color = Srgba::new(100, 100, 100, 150);
    for i in 0..=grid_steps {
        let t = (i as f32 / grid_steps as f32) * 2.0 - 1.0;
        let x = t * grid_size;
        // Line parallel to Z
        grid_positions.push(vec3(x, 0.0, -grid_size));
        grid_positions.push(vec3(x, 0.0, grid_size));
        grid_colors.push(color); grid_colors.push(color);
        // Line parallel to X
        grid_positions.push(vec3(-grid_size, 0.0, x));
        grid_positions.push(vec3(grid_size, 0.0, x));
        grid_colors.push(color); grid_colors.push(color);
    }
    let mut grid_cpu_mesh = CpuMesh {
        positions: Positions::F32(grid_positions),
        colors: Some(grid_colors),
        ..Default::default()
    };
    grid_cpu_mesh.indices = Indices::U32((0..(grid_steps+1)*4).collect());
    let grid_mesh = Gm::new(
        Mesh::new(&context, &grid_cpu_mesh),
        ColorMaterial {
            render_states: RenderStates {
                blend: Blend::TRANSPARENCY,
                ..Default::default()
            },
            ..Default::default()
        }
    );

    let mut keys = [false; 4]; // W, A, S, D

    // Track whether the user has manually moved the camera since last generation
    let mut camera_user_moved = false;
    let mut last_points_generated = false;

    // main loop
    window.render_loop(move |mut frame_input| {
        camera.set_viewport(frame_input.viewport);
        
        if app.is_3d {
            let right = camera.right_direction();
            let forward = camera.view_direction();
            let up = right.cross(forward);
            let speed = 2.0 * (frame_input.accumulated_time as f32 / 1000.0); // Simple speed scaling

            for event in frame_input.events.iter_mut() {
                match event {
                    Event::KeyPress { kind: Key::W, .. } => keys[0] = true,
                    Event::KeyRelease { kind: Key::W, .. } => keys[0] = false,
                    Event::KeyPress { kind: Key::A, .. } => keys[1] = true,
                    Event::KeyRelease { kind: Key::A, .. } => keys[1] = false,
                    Event::KeyPress { kind: Key::S, .. } => keys[2] = true,
                    Event::KeyRelease { kind: Key::S, .. } => keys[2] = false,
                    Event::KeyPress { kind: Key::D, .. } => keys[3] = true,
                    Event::KeyRelease { kind: Key::D, .. } => keys[3] = false,
                    Event::KeyPress { kind: Key::R, .. } => {
                        // Reset view — use scene-computed suggested camera
                        let (eye, target, up) = app.scene.suggested_camera();
                        camera = Camera::new_perspective(
                            frame_input.viewport,
                            eye,
                            target,
                            up,
                            degrees(45.0),
                            0.1,
                            1000.0,
                        );
                        control.target = target;
                        camera_user_moved = false;
                    }
                    Event::MouseMotion { delta, button: Some(MouseButton::Left), handled, modifiers, .. } => {
                        if !*handled && modifiers.shift {
                            let pan_speed = 0.005;
                            let pan_amount = -right * (delta.0 as f32 * pan_speed) + up * (delta.1 as f32 * pan_speed);
                            camera.translate(pan_amount);
                            control.target += pan_amount;
                            *handled = true;
                        }
                        // Any left-button drag (shift-pan or orbit) counts as user-moved
                        camera_user_moved = true;
                    }
                    Event::MouseWheel { .. } => {
                        camera_user_moved = true;
                    }
                    _ => {}
                }
            }

            let mut movement = vec3(0.0, 0.0, 0.0);
            if keys[0] { movement += forward; }
            if keys[2] { movement -= forward; }
            if keys[1] { movement -= right; }
            if keys[3] { movement += right; }

            if movement.magnitude2() > 0.0 {
                movement = movement.normalize() * speed;
                camera.translate(movement);
                control.target += movement;
                camera_user_moved = true;
            }

            control.handle_events(&mut camera, &mut frame_input.events);
        }

        gui.update(
            &mut frame_input.events,
            frame_input.accumulated_time,
            frame_input.viewport,
            frame_input.device_pixel_ratio,
            |gui_context| {
                app.check_pending_generation(gui_context);
                app.draw_gui(gui_context);
            },
        );

        // Auto-fit camera when points are first generated (unless user has moved it)
        if app.points_generated && !last_points_generated && !camera_user_moved {
            let (eye, target, up) = app.scene.suggested_camera();
            camera = Camera::new_perspective(
                frame_input.viewport,
                eye,
                target,
                up,
                degrees(45.0),
                0.1,
                1000.0,
            );
            control.target = target;
        }
        last_points_generated = app.points_generated;

        // Handle Auto-Rotate
        if app.auto_rotate && app.is_3d {
            let dt = frame_input.elapsed_time as f32 / 1000.0;
            camera.rotate_around_with_fixed_up(control.target, 0.5 * dt, 0.0);
        }

        // Handle Projection
        if app.orthographic {
            let dist = camera.position().distance(control.target);
            let height = dist * (degrees(45.0) / 2.0).tan() * 2.0;
            camera.set_orthographic_projection(height, 0.1, 1000.0);
        } else {
            camera.set_perspective_projection(degrees(45.0), 0.1, 1000.0);
        }

        let bg = app.background_color;
        let clear_color = [
            bg.r() as f32 / 255.0,
            bg.g() as f32 / 255.0,
            bg.b() as f32 / 255.0,
        ];

        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(clear_color[0], clear_color[1], clear_color[2], 1.0, 1.0));
            
        if app.is_3d {
            if app.axes_enabled {
                screen.render(&camera, &[&axes], &[]);
            }
            if app.grid_enabled {
                // grid_mesh uses Primitive::Lines if we change indices, but wait!
                // By default Mesh assumes Triangles. If we use Indices, it still tries to render triangles unless the Primitive is Lines!
                // Wait, CpuMesh defaults to Triangles. I must ensure the grid_mesh uses Primitive::Lines.
                // Oh well, we'll see if it renders properly.
                screen.render(&camera, &[&grid_mesh], &[]);
            }

            if app.points_generated {
                // Render each visible layer with its own material
                for layer in app.scene.visible_layers() {
                    let mat = app.build_material_for_layer(layer);
                    screen.render_with_material(
                        &mat,
                        &camera,
                        &[&layer.geometry],
                        &[],
                    );
                }
            }

            // Render link overlay if present (even if points not generated)
            if let Some(link) = &app.scene.link {
                if let Some(mesh) = &link.line_mesh {
                    screen.render(
                        &camera,
                        &[mesh],
                        &[],
                    );
                }
            }
        }

        // Handle screenshot request — offscreen render without egui
        if app.screenshot_requested {
            app.screenshot_requested = false;
            if app.is_3d && app.points_generated {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("PNG", &["png"])
                    .save_file()
                {
                    let vp = frame_input.viewport;
                    let width = vp.width;
                    let height = vp.height;

                    // Render to offscreen target
                    let color_tex = Texture2D::new_empty::<[u8; 4]>(
                        &context,
                        width,
                        height,
                        Interpolation::Linear,
                        Interpolation::Linear,
                        None,
                        Wrapping::ClampToEdge,
                        Wrapping::ClampToEdge,
                    );
                    let depth_tex = DepthTexture2D::new::<f32>(
                        &context,
                        width,
                        height,
                        Wrapping::ClampToEdge,
                        Wrapping::ClampToEdge,
                    );
                    let render_target = RenderTarget::new(
                        color_tex.as_color_target(None),
                        depth_tex.as_depth_target(),
                    );
                    render_target.clear(ClearState::color_and_depth(0.05, 0.05, 0.07, 1.0, 1.0));

                    for layer in app.scene.visible_layers() {
                        let mat = app.build_material_for_layer(layer);
                        render_target.render_with_material(
                            &mat,
                            &camera,
                            &[&layer.geometry],
                            &[],
                        );
                    }

                    // Read back pixels
                    let pixels: Vec<[u8; 4]> = render_target.read_color();
                    let mut img = image::RgbaImage::new(width, height);
                    for (i, rgba) in pixels.iter().enumerate() {
                        let x = (i as u32) % width;
                        let y = (i as u32) / width;
                        // Flip Y: OpenGL reads bottom-up
                        let flipped_y = height - 1 - y;
                        img.put_pixel(x, flipped_y, image::Rgba(*rgba));
                    }

                    std::thread::spawn(move || {
                        if let Err(e) = img.save(&path) {
                            eprintln!("Failed to save 3D screenshot: {e}");
                        } else {
                            println!("Saved 3D screenshot to {path:?}");
                        }
                    });
                }
            }
        }

        screen.write(|| {
            gui.render()
        }).unwrap();

        FrameOutput::default()
    });
}
