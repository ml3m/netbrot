use crate::colorschemes::ColorType;
use crate::iterate::{
    Netbrot, Vector, netbrot_orbit_escape_2d, netbrot_orbit_escape_ndim, netbrot_orbit_period,
};
use crate::render::{Renderer, RenderType, MAX_PERIODS};

use egui::{Color32, ColorImage};
use rayon::prelude::*;

#[inline]
fn escape_to_color32(
    color_type: ColorType,
    escape: crate::iterate::OrbitEscape,
    maxit: usize,
    escape_radius: f64,
) -> Color32 {
    let color = match escape.iteration {
        None => image::Rgb([0, 0, 0]),
        Some(n) => crate::colorschemes::get_smooth_orbit_color(
            color_type,
            n,
            escape.z_norm,
            maxit,
            escape_radius,
        ),
    };
    Color32::from_rgb(color[0], color[1], color[2])
}

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
            let maxit = brot.maxit;
            let escape_radius = brot.escape_radius_squared.sqrt();
            let escape_r2 = brot.escape_radius_squared;
            let ndim = brot.z0.len();

            if ndim == 2 {
                let m = &brot.mat;
                let a00 = m[(0, 0)];
                let a01 = m[(0, 1)];
                let a10 = m[(1, 0)];
                let a11 = m[(1, 1)];
                let z0 = brot.z0[0];
                let z1 = brot.z0[1];

                pixels
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(row, row_pixels)| {
                        for (col, pixel) in row_pixels.iter_mut().enumerate() {
                            let c = renderer.pixel_to_point((col, row));
                            let escape = netbrot_orbit_escape_2d(
                                a00, a01, a10, a11, z0, z1, c, maxit, escape_r2,
                            );
                            *pixel = escape_to_color32(color_type, escape, maxit, escape_radius);
                        }
                    });
            } else {
                pixels
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(row, row_pixels)| {
                        let mut z = brot.z0.clone();
                        let mut matz = Vector::zeros(ndim);
                        for (col, pixel) in row_pixels.iter_mut().enumerate() {
                            z.copy_from(&brot.z0);
                            let c = renderer.pixel_to_point((col, row));
                            let escape = netbrot_orbit_escape_ndim(
                                &brot.mat, &mut z, c, maxit, escape_r2, &mut matz,
                            );
                            *pixel = escape_to_color32(color_type, escape, maxit, escape_radius);
                        }
                    });
            }
        }
        RenderType::Julia => {
            let maxit = brot.maxit;
            let escape_radius = brot.escape_radius_squared.sqrt();
            let escape_r2 = brot.escape_radius_squared;
            let c = brot.c;
            let ndim = brot.z0.len();

            if ndim == 2 {
                let m = &brot.mat;
                let a00 = m[(0, 0)];
                let a01 = m[(0, 1)];
                let a10 = m[(1, 0)];
                let a11 = m[(1, 1)];

                pixels
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(row, row_pixels)| {
                        for (col, pixel) in row_pixels.iter_mut().enumerate() {
                            let point = renderer.pixel_to_point((col, row));
                            let escape = netbrot_orbit_escape_2d(
                                a00, a01, a10, a11, point, point, c, maxit, escape_r2,
                            );
                            *pixel = escape_to_color32(color_type, escape, maxit, escape_radius);
                        }
                    });
            } else {
                pixels
                    .par_chunks_mut(width)
                    .enumerate()
                    .for_each(|(row, row_pixels)| {
                        let mut z = brot.z0.clone();
                        let mut matz = Vector::zeros(ndim);
                        for (col, pixel) in row_pixels.iter_mut().enumerate() {
                            let point = renderer.pixel_to_point((col, row));
                            z.fill(point);
                            let escape = netbrot_orbit_escape_ndim(
                                &brot.mat, &mut z, c, maxit, escape_r2, &mut matz,
                            );
                            *pixel = escape_to_color32(color_type, escape, maxit, escape_radius);
                        }
                    });
            }
        }
        RenderType::Period => {
            pixels.par_chunks_mut(width).enumerate().for_each(|(row, row_pixels)| {
                let mut local_brot =
                    Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());
                for (col, pixel) in row_pixels.iter_mut().enumerate() {
                    local_brot.c = renderer.pixel_to_point((col, row));
                    let color = match netbrot_orbit_period(&local_brot) {
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
                let mut local_brot =
                    Netbrot::new(&brot.mat, brot.maxit, brot.escape_radius_squared.sqrt());
                for (col, pixel) in row_pixels.iter_mut().enumerate() {
                    local_brot.c = renderer.pixel_to_point((col, row));
                    let fp = crate::fixedpoints::find_fixed_points_by_newton(
                        &local_brot,
                        period,
                        brot.maxit as u32,
                        eps,
                    );
                    let color = if fp.len() < nfps {
                        image::Rgb([255, 255, 255])
                    } else {
                        match crate::fixedpoints::fixed_point_type(&local_brot, &fp, period, eps) {
                            crate::fixedpoints::FixedPointType::Attractive { eig, stable } => {
                                crate::colorschemes::get_fixed_point_color(color_type, eig, stable)
                            }
                            crate::fixedpoints::FixedPointType::Repulsive { .. } => {
                                image::Rgb([255, 255, 255])
                            }
                        }
                    };
                    *pixel = Color32::from_rgb(color[0], color[1], color[2]);
                }
            });
        }
    }

    ColorImage::new([width, height], pixels)
}
