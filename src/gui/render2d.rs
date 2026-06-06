use crate::render::{Renderer, RenderType, MAX_PERIODS};
use crate::iterate::Netbrot;
use crate::colorschemes::ColorType;
use rayon::prelude::*;
use egui::{ColorImage, Color32};

pub fn render_image(
    render_type: RenderType,
    resolution: (usize, usize),
    bbox: (f64, f64, f64, f64),
    brot: &Netbrot,
    color_type: ColorType,
    period: u32,
    eps: f64,
) -> ColorImage {
    let renderer = Renderer::new(
        resolution.1,
        (bbox.0, bbox.1),
        (bbox.2, bbox.3),
        color_type,
        render_type,
    );
    
    let width = renderer.resolution.0;
    let height = renderer.resolution.1;
    
    let mut pixels = vec![Color32::BLACK; width * height];
    
    match render_type {
        RenderType::Mandelbrot => {
            pixels.par_chunks_mut(width).enumerate().for_each(|(row, row_pixels)| {
                let mut local_brot = Netbrot {
                    mat: brot.mat.clone(),
                    z0: brot.z0.clone(),
                    c: brot.c,
                    maxit: brot.maxit,
                    escape_radius_squared: brot.escape_radius_squared,
                };
                let escape_radius = brot.escape_radius_squared.sqrt();
                for (col, pixel) in row_pixels.iter_mut().enumerate() {
                    local_brot.c = renderer.pixel_to_point((col, row));
                    let color = match crate::iterate::netbrot_orbit(&local_brot) {
                        crate::iterate::EscapeResult { iteration: None, .. } => image::Rgb([0, 0, 0]),
                        crate::iterate::EscapeResult { iteration: Some(n), z } => {
                            crate::colorschemes::get_smooth_orbit_color(color_type, n, z.norm(), brot.maxit, escape_radius)
                        }
                    };
                    *pixel = Color32::from_rgb(color[0], color[1], color[2]);
                }
            });
        }
        RenderType::Julia => {
            pixels.par_chunks_mut(width).enumerate().for_each(|(row, row_pixels)| {
                let mut local_brot = Netbrot {
                    mat: brot.mat.clone(),
                    z0: brot.z0.clone(),
                    c: brot.c,
                    maxit: brot.maxit,
                    escape_radius_squared: brot.escape_radius_squared,
                };
                let escape_radius = brot.escape_radius_squared.sqrt();
                for (col, pixel) in row_pixels.iter_mut().enumerate() {
                    local_brot.z0 = crate::iterate::Vector::from_element(brot.z0.len(), renderer.pixel_to_point((col, row)));
                    let color = match crate::iterate::netbrot_orbit(&local_brot) {
                        crate::iterate::EscapeResult { iteration: None, .. } => image::Rgb([0, 0, 0]),
                        crate::iterate::EscapeResult { iteration: Some(n), z } => {
                            crate::colorschemes::get_smooth_orbit_color(color_type, n, z.norm(), brot.maxit, escape_radius)
                        }
                    };
                    *pixel = Color32::from_rgb(color[0], color[1], color[2]);
                }
            });
        }
        RenderType::Period => {
            pixels.par_chunks_mut(width).enumerate().for_each(|(row, row_pixels)| {
                let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());
                for (col, pixel) in row_pixels.iter_mut().enumerate() {
                    local_brot.c = renderer.pixel_to_point((col, row));
                    let color = match crate::iterate::netbrot_orbit_period(&local_brot) {
                        None => image::Rgb([255, 255, 255]),
                        Some(p) => crate::colorschemes::get_period_color(color_type, p % MAX_PERIODS),
                    };
                    *pixel = Color32::from_rgb(color[0], color[1], color[2]);
                }
            });
        }
        RenderType::Attractive => {
            let ndim = brot.z0.len() as u32;
            let nfps = crate::fixedpoints::unique_poly_solutions(ndim, period) as usize;
            pixels.par_chunks_mut(width).enumerate().for_each(|(row, row_pixels)| {
                let mut local_brot = Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());
                for (col, pixel) in row_pixels.iter_mut().enumerate() {
                    local_brot.c = renderer.pixel_to_point((col, row));
                    let fp = crate::fixedpoints::find_fixed_points_by_newton(&local_brot, period, brot.maxit as u32, eps);
                    let color = if fp.len() < nfps {
                        image::Rgb([255, 255, 255])
                    } else {
                        match crate::fixedpoints::fixed_point_type(&local_brot, &fp, period, eps) {
                            crate::fixedpoints::FixedPointType::Attractive { eig, stable } => {
                                crate::colorschemes::get_fixed_point_color(color_type, eig, stable)
                            }
                            crate::fixedpoints::FixedPointType::Repulsive { .. } => image::Rgb([255, 255, 255]),
                        }
                    };
                    *pixel = Color32::from_rgb(color[0], color[1], color[2]);
                }
            });
        }
    }
    
    ColorImage::new([width, height], pixels)
}
