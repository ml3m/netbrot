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
    let mut app = App::new(&context);

    let mut keys = [false; 4]; // W, A, S, D

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
                        camera = Camera::new_perspective(
                            frame_input.viewport,
                            vec3(-1.0, -5.0, 1.5),
                            vec3(-1.0, 0.0, 0.0),
                            vec3(0.0, 0.0, 1.0),
                            degrees(45.0),
                            0.1,
                            1000.0,
                        );
                        control.target = vec3(-1.0, 0.0, 0.0);
                    }
                    Event::MouseMotion { delta, button: Some(MouseButton::Left), handled, modifiers, .. } => {
                        if !*handled && modifiers.shift {
                            let pan_speed = 0.005;
                            let pan_amount = -right * (delta.0 as f32 * pan_speed) + up * (delta.1 as f32 * pan_speed);
                            camera.translate(pan_amount);
                            control.target += pan_amount;
                            *handled = true;
                        }
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

        let screen = frame_input.screen();
        screen.clear(ClearState::color_and_depth(0.1, 0.1, 0.1, 1.0, 1.0));
            
        if app.is_3d && app.points_generated {
            if let Some(pc) = &app.point_cloud {
                screen.render_with_material(
                    &app.material,
                    &camera,
                    &[pc],
                    &[],
                );
            }
        }

        screen.write(|| {
            gui.render()
        }).unwrap();

        FrameOutput::default()
    });
}
