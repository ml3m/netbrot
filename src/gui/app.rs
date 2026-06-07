use crate::iterate::Netbrot;
use crate::colorschemes::ColorType;
use crate::render::RenderType;
use crate::gui::brot3d::{
    BrotMode, BrotParams, combined_pass_params, generate_points_cpu, generate_points_gpu,
    subsample_points,
};
use crate::gui::colormap3d::{BlendMode3d, ColorMode3d, Colormap3d};
use crate::gui::point_cloud::{PointCloudGeometry, PointCloudMaterial};
use crate::gui::render2d::render_image;
use crate::gui::scene3d::{LayerKind, PointCloudLayer, Scene3D};
use three_d::*;
use egui::TextureHandle;
use serde::{Serialize, Deserialize};
use nalgebra::DMatrix;
use num::complex::{c64, Complex64};
use std::sync::mpsc::{Sender, Receiver, channel};
use std::thread;

pub struct RenderRequest {
    pub render_type: RenderType,
    pub resolution: usize,
    pub bbox: (f64, f64, f64, f64),
    pub brot: Netbrot,
    pub color_type: ColorType,
    pub period: u32,
    pub eps: f64,
    pub egui_ctx: Option<egui::Context>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenMatrixType {
    Fixed,
    Feedforward,
    EqualRow,
}

#[derive(Serialize, Deserialize)]
pub struct Exhibit {
    pub mat: DMatrix<Complex64>,
    pub escape_radius: f64,
    pub upper_left: Complex64,
    pub lower_right: Complex64,
}

pub struct App {
    pub netbrot: Netbrot,
    pub render_type: RenderType,
    pub color_type: ColorType,
    pub period: u32,
    pub eps: f64,
    pub is_3d: bool,
    pub brot_params: BrotParams,
    pub use_gpu: bool,
    
    // 2D state
    pub color_image_texture: Option<egui::TextureHandle>,
    pub render_tx: Option<Sender<RenderRequest>>,
    pub render_rx: Option<Receiver<(egui::ColorImage, (f64, f64, f64, f64))>>,
    
    pub gen_matrix_type: GenMatrixType,
    pub gen_matrix_size: usize,
    pub texture: Option<TextureHandle>,
    pub resolution: usize,
    pub iterations: usize,
    pub escape_radius: f64,
    pub bbox: (f64, f64, f64, f64),
    pub render_bbox: (f64, f64, f64, f64),
    pub zoom_sensitivity: f32,
    pub last_interaction_time: Option<std::time::Instant>,
    pub pending_render_request: bool,
    
    // 3D state — now scene-based
    pub scene: Scene3D,
    pub points_generated: bool,
    pub max_display_points: usize,

    // Per-layer material defaults
    pub bifurcation_opacity: f32,
    pub bifurcation_point_size: f32,
    pub wide_opacity: f32,
    pub wide_point_size: f32,

    // Global material settings
    pub tail_emphasis: f32,
    pub tail_scale: f32,

    // 2D ↔ 3D link
    pub linked_c: Option<(f64, f64)>,
    pub link_enabled: bool,

    // Screenshot
    pub screenshot_requested: bool,

    pub status_msg: Option<String>,
    pub generation_pending: bool,
    
    // UI Panels
    pub show_matrix_editor: bool,
    pub show_view_controls: bool,
    
    // Matrix Editor
    pub matrix_nx: usize,
    pub matrix_ny: usize,
    pub matrix_values: Vec<(f64, f64)>,
    
    pub context: Context,
}

impl App {
    pub fn new(context: &Context) -> Self {
        let mut app = Self {
            // A 1x1 identity matrix z_{n+1} = z_n^2 + c
            netbrot: Netbrot::new(&DMatrix::from_element(1, 1, num::complex::c64(1.0, 0.0)), 100, 4.0),
            render_type: RenderType::Mandelbrot,
            color_type: ColorType::DefaultPalette,
            period: 2,
            eps: 1e-4,
            is_3d: false,
            brot_params: BrotParams::default(),
            use_gpu: true,
            
            color_image_texture: None,
            render_tx: None,
            render_rx: None,
            
            gen_matrix_type: GenMatrixType::Feedforward,
            gen_matrix_size: 2,
            texture: None,
            resolution: 500,
            iterations: 100,
            escape_radius: 4.0,
            bbox: (-2.5, 1.5, -1.5, 1.5),
            render_bbox: (-2.5, 1.5, -1.5, 1.5),
            zoom_sensitivity: 1.0,
            last_interaction_time: None,
            pending_render_request: false,
            
            scene: Scene3D::new(),
            points_generated: false,
            max_display_points: 500_000,

            bifurcation_opacity: 0.12,
            bifurcation_point_size: 1.2,
            wide_opacity: 0.05,
            wide_point_size: 1.0,

            tail_emphasis: 2.5,
            tail_scale: 0.35,

            linked_c: None,
            link_enabled: false,

            screenshot_requested: false,

            status_msg: None,
            generation_pending: false,
            
            show_matrix_editor: false,
            show_view_controls: true,
            
            matrix_nx: 1,
            matrix_ny: 1,
            matrix_values: vec![(1.0, 0.0)],
            
            context: context.clone(),
        };
        
        let (tx, rx_thread) = channel::<RenderRequest>();
        let (tx_thread, rx) = channel::<(egui::ColorImage, (f64, f64, f64, f64))>();
        
        thread::spawn(move || {
            let mut current_params: Option<RenderRequest> = None;
            loop {
                if current_params.is_none() {
                    match rx_thread.recv() {
                        Ok(params) => current_params = Some(params),
                        Err(_) => break, // channel closed
                    }
                }
                
                while let Ok(params) = rx_thread.try_recv() {
                    current_params = Some(params);
                }
                
                if let Some(params) = current_params.take() {
                    let image = render_image(
                        params.render_type,
                        (params.resolution, params.resolution),
                        params.bbox,
                        &params.brot,
                        params.color_type,
                        params.period,
                        params.eps,
                    );
                    let _ = tx_thread.send((image, params.bbox));
                    if let Some(ctx) = params.egui_ctx {
                        ctx.request_repaint();
                    }
                }
            }
        });
        
        app.render_tx = Some(tx);
        app.render_rx = Some(rx);
        
        app
    }

    pub fn draw_gui(&mut self, ctx: &egui::Context) {
        if self.pending_render_request {
            if let Some(last_time) = self.last_interaction_time {
                if last_time.elapsed() > std::time::Duration::from_millis(100) {
                    self.pending_render_request = false;
                    self.request_render_2d(Some(ctx.clone()));
                } else {
                    ctx.request_repaint();
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Netbrot");
                ui.separator();
                if ui.selectable_label(self.show_matrix_editor, "🧮 Matrix Editor").clicked() {
                    self.show_matrix_editor = !self.show_matrix_editor;
                }
                if ui.selectable_label(self.show_view_controls, "👁 View Controls").clicked() {
                    self.show_view_controls = !self.show_view_controls;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("📂 Load Exhibit JSON").clicked() {
                        self.load_json_dialog();
                    }
                    if ui.button("💾 Save Exhibit JSON").clicked() {
                        self.save_json_dialog();
                    }
                    if ui.button("🖼 Save Picture (HQ)").clicked() {
                        self.save_high_quality();
                    }
                });
            });
        });

        if self.show_matrix_editor {
            egui::SidePanel::left("matrix_panel")
                .resizable(true)
                .width_range(150.0..=1000.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::both()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            self.draw_matrix_editor(ui);
                        });
                });
        }

        if self.show_view_controls {
            egui::SidePanel::right("view_panel")
                .resizable(true)
                .width_range(150.0..=600.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::both()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            self.draw_view_controls(ui);
                        });
                });
        }

        let central_frame = if self.is_3d {
            egui::Frame::none().fill(egui::Color32::TRANSPARENT)
        } else {
            egui::Frame::central_panel(&ctx.style())
        };

        egui::CentralPanel::default().frame(central_frame).show(ctx, |ui| {
            if !self.is_3d {
                if let Some(rx) = &self.render_rx {
                    while let Ok((color_image, r_bbox)) = rx.try_recv() {
                        self.texture = Some(ctx.load_texture("fractal", color_image, egui::TextureOptions::LINEAR));
                        self.render_bbox = r_bbox;
                    }
                }
                
                if let Some(texture) = &self.texture {
                    let available = ui.available_size();
                    let aspect = texture.size()[0] as f32 / texture.size()[1] as f32;
                    let size = if available.x / available.y > aspect {
                        egui::vec2(available.y * aspect, available.y)
                    } else {
                        egui::vec2(available.x, available.x / aspect)
                    };
                    
                    let render_width = self.render_bbox.1 - self.render_bbox.0;
                    let render_height = self.render_bbox.3 - self.render_bbox.2;
                    let u0 = (self.bbox.0 - self.render_bbox.0) / render_width;
                    let u1 = (self.bbox.1 - self.render_bbox.0) / render_width;
                    let v0 = (self.render_bbox.3 - self.bbox.3) / render_height;
                    let v1 = (self.render_bbox.3 - self.bbox.2) / render_height;
                    
                    let uv = egui::Rect::from_min_max(
                        egui::pos2(u0 as f32, v0 as f32),
                        egui::pos2(u1 as f32, v1 as f32),
                    );
                    
                    let response = ui.add(egui::Image::new(texture).fit_to_exact_size(size).uv(uv).sense(egui::Sense::click_and_drag()));
                    
                    // 2D → 3D link: click to select c-value
                    if self.link_enabled && response.clicked() {
                        if let Some(pos) = response.interact_pointer_pos() {
                            let x_rel = (pos.x - response.rect.left()) as f64 / size.x as f64;
                            let y_rel = (pos.y - response.rect.top()) as f64 / size.y as f64;
                            let width = self.bbox.1 - self.bbox.0;
                            let height = self.bbox.3 - self.bbox.2;
                            let c_re = self.bbox.0 + x_rel * width;
                            let c_im = self.bbox.3 - y_rel * height;
                            self.linked_c = Some((c_re, c_im));
                            // Update 3D link overlay
                            self.scene.update_link_overlay(&self.context, c_re as f32, c_im as f32);
                        }
                    }

                    if response.dragged() {
                        let delta = response.drag_delta();
                        let width = self.bbox.1 - self.bbox.0;
                        let height = self.bbox.3 - self.bbox.2;
                        let dx = (delta.x as f64 / size.x as f64) * width;
                        let dy = (delta.y as f64 / size.y as f64) * height;
                        
                        self.bbox.0 -= dx;
                        self.bbox.1 -= dx;
                        self.bbox.2 += dy;
                        self.bbox.3 += dy;
                        
                        self.last_interaction_time = Some(std::time::Instant::now());
                        self.pending_render_request = true;
                    }
                    
                    let scroll = ui.input(|i| i.smooth_scroll_delta);
                    if response.hovered() && scroll.y != 0.0 {
                        let diff = (0.2 * self.zoom_sensitivity).clamp(0.01, 0.9) as f64;
                        let zoom_factor = if scroll.y > 0.0 { 1.0 - diff } else { 1.0 + diff };
                        
                        let pointer_pos = response.hover_pos().unwrap_or(response.rect.center());
                        let x_rel = (pointer_pos.x - response.rect.left()) as f64 / size.x as f64;
                        let y_rel = (pointer_pos.y - response.rect.top()) as f64 / size.y as f64;
                        
                        let width = self.bbox.1 - self.bbox.0;
                        let height = self.bbox.3 - self.bbox.2;
                        
                        let center_x = self.bbox.0 + x_rel * width;
                        let center_y = self.bbox.3 - y_rel * height;
                        
                        let new_width = width * zoom_factor;
                        let new_height = height * zoom_factor;
                        
                        self.bbox.0 = center_x - x_rel * new_width;
                        self.bbox.1 = center_x + (1.0 - x_rel) * new_width;
                        self.bbox.3 = center_y + y_rel * new_height;
                        self.bbox.2 = center_y - (1.0 - y_rel) * new_height;
                        
                        self.last_interaction_time = Some(std::time::Instant::now());
                        self.pending_render_request = true;
                    }

                    // Draw linked-c crosshair on the 2D panel
                    if let Some((c_re, c_im)) = self.linked_c {
                        let width = self.bbox.1 - self.bbox.0;
                        let height = self.bbox.3 - self.bbox.2;
                        let x_frac = (c_re - self.bbox.0) / width;
                        let y_frac = (self.bbox.3 - c_im) / height;
                        if x_frac >= 0.0 && x_frac <= 1.0 && y_frac >= 0.0 && y_frac <= 1.0 {
                            let px = response.rect.left() + x_frac as f32 * size.x;
                            let py = response.rect.top() + y_frac as f32 * size.y;
                            let painter = ui.painter();
                            let cross_size = 8.0;
                            let color = egui::Color32::from_rgb(255, 80, 80);
                            painter.line_segment(
                                [egui::pos2(px - cross_size, py), egui::pos2(px + cross_size, py)],
                                egui::Stroke::new(2.0, color),
                            );
                            painter.line_segment(
                                [egui::pos2(px, py - cross_size), egui::pos2(px, py + cross_size)],
                                egui::Stroke::new(2.0, color),
                            );
                            painter.circle_stroke(
                                egui::pos2(px, py),
                                6.0,
                                egui::Stroke::new(1.5, color),
                            );
                            // Label
                            painter.text(
                                egui::pos2(px + 10.0, py - 14.0),
                                egui::Align2::LEFT_BOTTOM,
                                format!("c = {:.4} + {:.4}i", c_re, c_im),
                                egui::FontId::proportional(11.0),
                                egui::Color32::from_rgb(255, 200, 200),
                            );
                        }
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Configure matrix and generate, or load an exhibit JSON.");
                    });
                }
            } else {
                if !self.points_generated {
                    ui.centered_and_justified(|ui| {
                        ui.label("Click 'Generate 3D Points' to render.");
                    });
                }
                // When in 3D, three-d will render to the background
            }
        });
    }

    pub fn draw_matrix_editor(&mut self, ui: &mut egui::Ui) {
        ui.heading("Matrix Editor");
        ui.separator();
        
        egui::CollapsingHeader::new("Generate Random Matrix")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Type:");
                    egui::ComboBox::from_id_source("gen_type")
                        .selected_text(match self.gen_matrix_type {
                            GenMatrixType::Fixed => "Fixed",
                            GenMatrixType::Feedforward => "Feedforward",
                            GenMatrixType::EqualRow => "EqualRow",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.gen_matrix_type, GenMatrixType::Fixed, "Fixed");
                            ui.selectable_value(&mut self.gen_matrix_type, GenMatrixType::Feedforward, "Feedforward");
                            ui.selectable_value(&mut self.gen_matrix_type, GenMatrixType::EqualRow, "EqualRow");
                        });
                });
                ui.add(egui::Slider::new(&mut self.gen_matrix_size, 2..=10).text("Size"));
                if ui.button("Generate & Apply").clicked() {
                    self.generate_random_matrix();
                }
            });
            
        ui.separator();

        ui.horizontal(|ui| {
            if ui.add(egui::DragValue::new(&mut self.matrix_nx).range(1..=10)).changed() {
                self.matrix_ny = self.matrix_nx;
                self.resize_matrix();
            }
            ui.label("Matrix Size (N x N)");
        });
        
        ui.separator();
        
        egui::Grid::new("matrix_grid").striped(true).show(ui, |ui| {
            for y in 0..self.matrix_ny {
                for x in 0..self.matrix_nx {
                    let idx = x * self.matrix_ny + y; // column major
                    if idx < self.matrix_values.len() {
                        let val = &mut self.matrix_values[idx];
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut val.0).speed(0.1));
                            ui.label("+");
                            ui.add(egui::DragValue::new(&mut val.1).speed(0.1));
                            ui.label("i");
                        });
                    }
                }
                ui.end_row();
            }
        });
        
        ui.separator();
        ui.heading("Bounds & Parameters");
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.bbox.0).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.bbox.1).speed(0.1));
            ui.label("X Range");
        });
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut self.bbox.2).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.bbox.3).speed(0.1));
            ui.label("Y Range");
        });
        
        ui.add(egui::DragValue::new(&mut self.escape_radius).speed(0.1).prefix("Escape Radius: "));
        ui.add(egui::DragValue::new(&mut self.iterations).speed(1).prefix("Max Iterations: "));
        ui.add(egui::DragValue::new(&mut self.resolution).speed(10).prefix("Resolution: "));
        
        ui.separator();
        if ui.button("Apply & Generate 2D").clicked() {
            let ctx = ui.ctx().clone();
            self.apply_matrix_and_generate_2d(&ctx);
        }
    }

    fn draw_view_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("View Controls");
        ui.separator();
        
        ui.checkbox(&mut self.is_3d, "3D Mode");
        
        egui::CollapsingHeader::new("2D Generation Settings")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Render Mode:");
                    egui::ComboBox::from_id_source("render_mode")
                        .selected_text(format!("{:?}", self.render_type))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.render_type, RenderType::Mandelbrot, "Mandelbrot");
                            ui.selectable_value(&mut self.render_type, RenderType::Period, "Period");
                            ui.selectable_value(&mut self.render_type, RenderType::Julia, "Julia");
                            ui.selectable_value(&mut self.render_type, RenderType::Attractive, "Attractive Fixed Point");
                        });
                });
                
                ui.horizontal(|ui| {
                    ui.label("Color Scheme:");
                    egui::ComboBox::from_id_source("color_scheme")
                        .selected_text(format!("{:?}", self.color_type))
                        .show_ui(ui, |ui| {
                            let types = [
                                ColorType::DefaultPalette,
                                ColorType::PeriodStack,
                                ColorType::PeriodEndesga,
                                ColorType::PeriodMatlab,
                                ColorType::OrbitBlue,
                                ColorType::OrbitFire,
                            ];
                            for t in types {
                                ui.selectable_value(&mut self.color_type, t, format!("{:?}", t));
                            }
                        });
                });
                
                ui.add(egui::Slider::new(&mut self.zoom_sensitivity, 0.1..=5.0).text("Zoom Sensitivity"));
                
                ui.horizontal(|ui| {
                    if ui.button("Generate 2D").clicked() {
                        let ctx = ui.ctx().clone();
                        self.apply_matrix_and_generate_2d(&ctx);
                    }
                    if ui.button("💾 Save High Quality").clicked() {
                        self.save_high_quality();
                    }
                });
            });
            
        egui::CollapsingHeader::new("3D Visualization Settings")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Mode:");
                    egui::ComboBox::from_id_source("3d_mode")
                        .selected_text(match self.brot_params.mode {
                            BrotMode::Combined => "Combined",
                            BrotMode::Bifurcation => "Bifurcation",
                            BrotMode::WideAttractor => "Wide Attractor",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.brot_params.mode, BrotMode::Combined, "Combined");
                            ui.selectable_value(&mut self.brot_params.mode, BrotMode::Bifurcation, "Bifurcation");
                            ui.selectable_value(&mut self.brot_params.mode, BrotMode::WideAttractor, "Wide Attractor");
                        });
                });
                
                ui.add(egui::Slider::new(&mut self.brot_params.nx, 100..=10000).text("NX"));
                ui.add(egui::Slider::new(&mut self.brot_params.ny, 50..=2000).text("NY (wide)"));
                ui.add(egui::Slider::new(&mut self.brot_params.warmup, 10..=2000).text("Warmup"));
                ui.add(egui::Slider::new(&mut self.brot_params.keep, 10..=5000).text("Keep"));
                ui.add(
                    egui::Slider::new(&mut self.max_display_points, 50_000..=2_000_000)
                        .logarithmic(true)
                        .text("Max display points"),
                );

                ui.checkbox(&mut self.use_gpu, "Use GPU Compute");
                ui.label(
                    "Each bounded c-path records up to Keep orbit points. \
                     Combined ≈ NX×Keep + (NX/4)×(NY/2)×min(Keep,100).",
                );

                ui.separator();
                ui.heading("Layer Controls");

                // Per-layer visibility and material
                if self.brot_params.mode == BrotMode::Combined || self.brot_params.mode == BrotMode::Bifurcation {
                    let mut bif_visible = self.scene.bifurcation.as_ref().map_or(true, |l| l.visible);
                    if ui.checkbox(&mut bif_visible, "Show Bifurcation").changed() {
                        if let Some(l) = &mut self.scene.bifurcation {
                            l.visible = bif_visible;
                        }
                    }
                    ui.add(egui::Slider::new(&mut self.bifurcation_opacity, 0.005..=1.0).text("Bifurcation Opacity"));
                    ui.add(egui::Slider::new(&mut self.bifurcation_point_size, 0.1..=10.0).text("Bifurcation Point Size"));
                }

                if self.brot_params.mode == BrotMode::Combined || self.brot_params.mode == BrotMode::WideAttractor {
                    let mut wide_visible = self.scene.wide.as_ref().map_or(true, |l| l.visible);
                    if ui.checkbox(&mut wide_visible, "Show Wide").changed() {
                        if let Some(l) = &mut self.scene.wide {
                            l.visible = wide_visible;
                        }
                    }
                    ui.add(egui::Slider::new(&mut self.wide_opacity, 0.005..=1.0).text("Wide Opacity"));
                    ui.add(egui::Slider::new(&mut self.wide_point_size, 0.1..=10.0).text("Wide Point Size"));
                }

                ui.separator();
                ui.heading("Visual Controls");

                ui.add(
                    egui::Slider::new(&mut self.tail_emphasis, 0.0..=8.0)
                        .text("Ray emphasis"),
                );
                ui.add(
                    egui::Slider::new(&mut self.tail_scale, 0.05..=2.0)
                        .text("Ray height scale"),
                );
                ui.label("Ray emphasis enlarges and brightens points far from z = 0.");

                // Color mode
                ui.horizontal(|ui| {
                    ui.label("Color Mode:");
                    egui::ComboBox::from_id_source("color_mode_3d")
                        .selected_text(self.scene.color_mode.label())
                        .show_ui(ui, |ui| {
                            for mode in ColorMode3d::ALL {
                                ui.selectable_value(&mut self.scene.color_mode, mode, mode.label());
                            }
                        });
                });

                // Colormap
                ui.horizontal(|ui| {
                    ui.label("Colormap:");
                    egui::ComboBox::from_id_source("colormap_3d")
                        .selected_text(self.scene.colormap.label())
                        .show_ui(ui, |ui| {
                            for cm in Colormap3d::ALL {
                                ui.selectable_value(&mut self.scene.colormap, cm, cm.label());
                            }
                        });
                });

                // Blend mode
                ui.horizontal(|ui| {
                    ui.label("Blend Mode:");
                    egui::ComboBox::from_id_source("blend_mode_3d")
                        .selected_text(self.scene.blend_mode.label())
                        .show_ui(ui, |ui| {
                            for bm in BlendMode3d::ALL {
                                ui.selectable_value(&mut self.scene.blend_mode, bm, bm.label());
                            }
                        });
                });

                ui.separator();
                ui.heading("Clipping");

                ui.add(egui::Slider::new(&mut self.scene.clip.re_min, -5.0..=5.0).text("Clip Re(c) min"));
                ui.add(egui::Slider::new(&mut self.scene.clip.re_max, -5.0..=5.0).text("Clip Re(c) max"));
                ui.add(egui::Slider::new(&mut self.scene.clip.im_min, -5.0..=5.0).text("Clip Im(c) min"));
                ui.add(egui::Slider::new(&mut self.scene.clip.im_max, -5.0..=5.0).text("Clip Im(c) max"));
                ui.add(egui::Slider::new(&mut self.scene.clip.z_min, -5.0..=5.0).text("Clip Re(z) min"));
                ui.add(egui::Slider::new(&mut self.scene.clip.z_max, -5.0..=5.0).text("Clip Re(z) max"));
                if ui.button("Reset Clip to AABB").clicked() {
                    self.scene.reset_clip_to_aabb();
                }

                ui.separator();
                ui.heading("2D ↔ 3D Link");
                ui.checkbox(&mut self.link_enabled, "Enable 2D↔3D link");
                if self.linked_c.is_some() {
                    if ui.button("Clear Link").clicked() {
                        self.linked_c = None;
                        self.scene.clear_link_overlay();
                    }
                }

                ui.separator();
                
                ui.horizontal(|ui| {
                    if ui.button("Generate 3D Points").clicked() {
                        self.status_msg = Some("Generating 3D points...".to_string());
                        self.generation_pending = true;
                    }
                    if ui.button("📸 Save 3D View").clicked() {
                        self.screenshot_requested = true;
                    }
                });
                
                if let Some(msg) = &self.status_msg {
                    ui.label(msg);
                }
            });
    }

    pub fn check_pending_generation(&mut self, ctx: &egui::Context) {
        if self.generation_pending {
            self.generation_pending = false;
            let start = std::time::Instant::now();
            self.generate_3d();
            let elapsed = start.elapsed();
            self.status_msg = Some(format!("Done in {:.2}s", elapsed.as_secs_f32()));
            ctx.request_repaint();
        }
    }

    pub fn generate_3d(&mut self) {
        self.brot_params.re_min = self.bbox.0 as f32;
        self.brot_params.re_max = self.bbox.1 as f32;
        self.brot_params.im_min = self.bbox.2 as f32;
        self.brot_params.im_max = self.bbox.3 as f32;
        self.brot_params.escape_r = self.escape_radius as f32;

        // Extract fields needed by generation to avoid borrowing all of `self`
        let use_gpu = self.use_gpu;
        let context = self.context.clone();
        let mat = self.netbrot.mat.clone();

        let do_generate = |params: &BrotParams| -> (Vec<Vec3>, Vec<Srgba>) {
            if use_gpu {
                match generate_points_gpu(&context, params, &mat) {
                    Ok(res) => res,
                    Err(e) => {
                        eprintln!("GPU err: {e}");
                        generate_points_cpu(params, &mat)
                    }
                }
            } else {
                generate_points_cpu(params, &mat)
            }
        };

        // Generate layers separately
        let mut all_positions = Vec::new();

        if self.brot_params.mode == BrotMode::Combined {
            let (p1, p2) = combined_pass_params(&self.brot_params);

            // Bifurcation layer
            let (pos1, col1) = do_generate(&p1);
            let raw1 = pos1.len();
            let bif_budget = (self.max_display_points as f32 * 0.65) as usize;
            let (pos1, col1) = subsample_points(pos1, col1, bif_budget);
            let n1 = pos1.len();
            all_positions.extend_from_slice(&pos1);
            self.create_or_update_layer(LayerKind::Bifurcation, pos1, col1);

            // Wide layer
            let (pos2, col2) = do_generate(&p2);
            let raw2 = pos2.len();
            let wide_budget = self.max_display_points.saturating_sub(n1);
            let (pos2, col2) = subsample_points(pos2, col2, wide_budget);
            let n2 = pos2.len();
            all_positions.extend_from_slice(&pos2);
            self.create_or_update_layer(LayerKind::Wide, pos2, col2);

            self.status_msg = Some(format!(
                "Showing {n1}+{n2} pts (raw {raw1}+{raw2}, cap {})",
                self.max_display_points,
            ));
        } else {
            let params = self.brot_params.clone();
            let (pos, col) = do_generate(&params);
            let raw_count = pos.len();
            let (pos, col) = subsample_points(pos, col, self.max_display_points);
            let shown = pos.len();
            all_positions.extend_from_slice(&pos);

            let kind = match self.brot_params.mode {
                BrotMode::Bifurcation => LayerKind::Bifurcation,
                _ => LayerKind::Wide,
            };
            self.create_or_update_layer(kind, pos, col);

            // Clear the other layer
            match kind {
                LayerKind::Bifurcation => self.scene.wide = None,
                LayerKind::Wide => self.scene.bifurcation = None,
            }

            self.status_msg = Some(format!(
                "Showing {shown} points (raw {raw_count}, cap {})",
                self.max_display_points,
            ));
        }

        if !all_positions.is_empty() {
            self.scene.compute_aabb_from_positions(&all_positions);
            self.scene.reset_clip_to_aabb();
            self.points_generated = true;
        } else {
            self.status_msg = Some("No bounded orbits found for these parameters.".to_string());
        }
    }

    fn create_or_update_layer(&mut self, kind: LayerKind, positions: Vec<Vec3>, colors: Vec<Srgba>) {
        if positions.is_empty() {
            match kind {
                LayerKind::Bifurcation => self.scene.bifurcation = None,
                LayerKind::Wide => self.scene.wide = None,
            }
            return;
        }

        let count = positions.len();
        let (opacity, point_size) = match kind {
            LayerKind::Bifurcation => (self.bifurcation_opacity, self.bifurcation_point_size),
            LayerKind::Wide => (self.wide_opacity, self.wide_point_size),
        };

        let layer_slot = match kind {
            LayerKind::Bifurcation => &mut self.scene.bifurcation,
            LayerKind::Wide => &mut self.scene.wide,
        };

        if let Some(layer) = layer_slot {
            layer.geometry.update(&positions, &colors);
            layer.count = count;
        } else {
            let mut geom = PointCloudGeometry::new(&self.context);
            geom.update(&positions, &colors);
            *layer_slot = Some(PointCloudLayer {
                kind,
                geometry: geom,
                material: PointCloudMaterial {
                    point_size,
                    opacity,
                    tail_emphasis: self.tail_emphasis,
                    tail_scale: self.tail_scale,
                    is_transparent: true,
                    ..Default::default()
                },
                visible: true,
                count,
            });
        }
    }

    /// Build a `PointCloudMaterial` for a given layer, applying current scene settings.
    pub fn build_material_for_layer(&self, layer: &PointCloudLayer) -> PointCloudMaterial {
        let (opacity, point_size) = match layer.kind {
            LayerKind::Bifurcation => (self.bifurcation_opacity, self.bifurcation_point_size),
            LayerKind::Wide => (self.wide_opacity, self.wide_point_size),
        };

        let scene_min = if self.scene.aabb_is_empty() {
            vec3(-2.5, -1.15, -2.0)
        } else {
            self.scene.aabb.min()
        };
        let scene_max = if self.scene.aabb_is_empty() {
            vec3(0.55, 1.15, 2.0)
        } else {
            self.scene.aabb.max()
        };

        PointCloudMaterial {
            point_size,
            opacity,
            tail_emphasis: self.tail_emphasis,
            tail_scale: self.tail_scale,
            is_transparent: true,
            point_size_reference: 5.0,
            color_mode: self.scene.color_mode,
            colormap: self.scene.colormap,
            blend_mode: self.scene.blend_mode,
            use_shader_coloring: true,
            clip_min: vec3(
                self.scene.clip.re_min,
                self.scene.clip.im_min,
                self.scene.clip.z_min,
            ),
            clip_max: vec3(
                self.scene.clip.re_max,
                self.scene.clip.im_max,
                self.scene.clip.z_max,
            ),
            scene_min,
            scene_max,
        }
    }
    
    pub fn request_render_2d(&self, ctx: Option<egui::Context>) {
        if let Some(tx) = &self.render_tx {
            let req = RenderRequest {
                render_type: self.render_type,
                resolution: self.resolution,
                bbox: self.bbox,
                brot: self.netbrot.clone(),
                color_type: self.color_type,
                period: self.period,
                eps: self.eps,
                egui_ctx: ctx,
            };
            let _ = tx.send(req);
        }
    }
    
    pub fn save_high_quality(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("PNG", &["png"])
            .save_file() {
            
            let req = RenderRequest {
                render_type: self.render_type,
                resolution: 4000,
                bbox: self.bbox,
                brot: self.netbrot.clone(),
                color_type: self.color_type,
                period: self.period,
                eps: self.eps,
                egui_ctx: None,
            };
            
            std::thread::spawn(move || {
                let color_image = render_image(
                    req.render_type,
                    (req.resolution, req.resolution),
                    req.bbox,
                    &req.brot,
                    req.color_type,
                    req.period,
                    req.eps,
                );
                
                let width = color_image.size[0] as u32;
                let height = color_image.size[1] as u32;
                let mut rgb_image = image::RgbImage::new(width, height);
                
                for (i, pixel) in color_image.pixels.iter().enumerate() {
                    let x = (i as u32) % width;
                    let y = (i as u32) / width;
                    rgb_image.put_pixel(x, y, image::Rgb([pixel.r(), pixel.g(), pixel.b()]));
                }
                
                if let Err(e) = rgb_image.save(&path) {
                    eprintln!("Failed to save image to {:?}: {}", path, e);
                } else {
                    println!("Saved high quality image to {:?}", path);
                }
            });
        }
    }

    pub fn generate_2d(&mut self, ctx: &egui::Context) {
        self.request_render_2d(Some(ctx.clone()));
    }
    
    pub fn load_exhibit(&mut self, path: &std::path::Path) {
        use std::fs::File;
        
        if let Ok(file) = File::open(path) {
            if let Ok(exhibit) = serde_json::from_reader::<_, Exhibit>(file) {
                self.netbrot = Netbrot::new(&exhibit.mat, 200, exhibit.escape_radius);
                self.bbox = (exhibit.upper_left.re, exhibit.lower_right.re, exhibit.lower_right.im, exhibit.upper_left.im);
                self.points_generated = false;
                
                // Update matrix editor state
                self.matrix_nx = exhibit.mat.ncols();
                self.matrix_ny = exhibit.mat.nrows();
                self.matrix_values.clear();
                for x in 0..self.matrix_nx {
                    for y in 0..self.matrix_ny {
                        let val = exhibit.mat[(y, x)];
                        self.matrix_values.push((val.re, val.im));
                    }
                }
                self.escape_radius = exhibit.escape_radius;
            }
        }
    }
    
    pub fn load_json_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .pick_file() {
            self.load_exhibit(&path);
        }
    }
    
    pub fn save_json_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .save_file() {
            let exhibit = Exhibit {
                mat: self.netbrot.mat.clone(),
                escape_radius: self.escape_radius,
                upper_left: num::complex::c64(self.bbox.0, self.bbox.3),
                lower_right: num::complex::c64(self.bbox.1, self.bbox.2),
            };
            if let Ok(file) = std::fs::File::create(&path) {
                let _ = serde_json::to_writer_pretty(file, &exhibit);
            }
        }
    }
    
    pub fn resize_matrix(&mut self) {
        let expected_len = self.matrix_nx * self.matrix_ny;
        self.matrix_values.resize(expected_len, (0.0, 0.0));
    }
    
    pub fn apply_matrix_and_generate_2d(&mut self, ctx: &egui::Context) {
        let mut mat = DMatrix::zeros(self.matrix_ny, self.matrix_nx);
        for y in 0..self.matrix_ny {
            for x in 0..self.matrix_nx {
                let idx = x * self.matrix_ny + y;
                if idx < self.matrix_values.len() {
                    let val = self.matrix_values[idx];
                    mat[(y, x)] = num::complex::c64(val.0, val.1);
                }
            }
        }
        self.netbrot = Netbrot::new(&mat, self.iterations, self.escape_radius);
        self.points_generated = false;
        self.generate_2d(ctx);
    }
    
    pub fn generate_random_matrix(&mut self) {
        let size = self.gen_matrix_size;
        
        let mut mat = DMatrix::zeros(size, size);
        
        match self.gen_matrix_type {
            GenMatrixType::Fixed => {
                // Just fallback to the default 2x2
                self.gen_matrix_size = 2;
                mat = DMatrix::zeros(2, 2);
                mat[(0, 0)] = c64(1.0, 0.0);
                mat[(0, 1)] = c64(0.8, 0.0);
                mat[(1, 0)] = c64(1.0, 0.0);
                mat[(1, 1)] = c64(-0.5, 0.0);
            }
            GenMatrixType::Feedforward => {
                for y in 0..size {
                    for x in 0..size {
                        if x <= y {
                            let val: f64 = rand::random::<f64>();
                            mat[(y, x)] = c64(val, 0.0);
                        }
                    }
                }
            }
            GenMatrixType::EqualRow => {
                for y in 0..size {
                    for x in 0..size {
                        let val: f64 = rand::random::<f64>();
                        mat[(y, x)] = c64(val, 0.0);
                    }
                }
                
                let mut row_sums = Vec::new();
                for y in 0..size {
                    let mut sum = 0.0;
                    for x in 0..size {
                        sum += mat[(y, x)].re;
                    }
                    row_sums.push(sum);
                }
                
                let target_sum = row_sums[0];
                for y in 0..size {
                    let scale = target_sum / row_sums[y];
                    for x in 0..size {
                        mat[(y, x)] *= scale;
                    }
                }
            }
        }
        
        // Estimate escape radius using SVD
        let svd = nalgebra::linalg::SVD::new(mat.clone(), true, true);
        let min_sigma = svd.singular_values.min();
        let escape_radius = 2.0 * (self.gen_matrix_size as f64).sqrt() / (min_sigma * min_sigma);
        // Clamp to a reasonable value
        let escape_radius = escape_radius.min(10.0);
        
        // Update App state
        self.matrix_nx = self.gen_matrix_size;
        self.matrix_ny = self.gen_matrix_size;
        self.matrix_values.clear();
        for x in 0..self.matrix_nx {
            for y in 0..self.matrix_ny {
                let val = mat[(y, x)];
                self.matrix_values.push((val.re, val.im));
            }
        }
        self.escape_radius = escape_radius;
        self.points_generated = false;
        self.netbrot = Netbrot::new(&mat, self.iterations, self.escape_radius);
    }
}
