use three_d::*;
use rayon::prelude::*;
use std::f32::consts::PI;
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;
use colors_transform::{Color, Hsl};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BrotMode {
    Bifurcation,
    WideAttractor,
    Combined,
}

#[derive(Clone)]
pub struct BrotParams {
    pub mode: BrotMode,
    pub nx: usize,
    pub ny: usize,
    pub warmup: usize,
    pub keep: usize,
    pub escape_r: f32,
    pub re_min: f32,
    pub re_max: f32,
    pub im_min: f32,
    pub im_max: f32,
}

impl Default for BrotParams {
    fn default() -> Self {
        Self {
            mode: BrotMode::Combined,
            // Interactive defaults — the old Python HQ values (nx=7000, keep=2000)
            // produce tens of millions of points and are not suitable for realtime GL.
            nx: 2000,
            ny: 400,
            warmup: 400,
            keep: 800,
            escape_r: 2.0,
            re_min: -2.5,
            re_max: 0.55,
            im_min: -1.15,
            im_max: 1.15,
        }
    }
}

/// Split Combined mode into a bifurcation slice + a lighter wide-attractor shell.
pub fn combined_pass_params(base: &BrotParams) -> (BrotParams, BrotParams) {
    let mut bifurcation = base.clone();
    bifurcation.mode = BrotMode::Bifurcation;
    bifurcation.ny = 1;

    let mut wide = base.clone();
    wide.mode = BrotMode::WideAttractor;
    wide.nx = (base.nx / 4).clamp(200, 1000);
    wide.ny = (base.ny / 2).clamp(80, 400);
    wide.warmup = base.warmup.min(200);
    wide.keep = base.keep.min(100);

    (bifurcation, wide)
}

/// Downsample for display while keeping high-|z| "ray" points that uniform stride would drop.
pub fn subsample_points(
    positions: Vec<Vec3>,
    colors: Vec<Srgba>,
    max_points: usize,
) -> (Vec<Vec3>, Vec<Srgba>) {
    let n = positions.len();
    if n <= max_points {
        return (positions, colors);
    }

    // Reserve ~35% of the budget for tail points (large |z|, the faint vertical rays).
    let tail_budget = (max_points as f32 * 0.35) as usize;
    let core_budget = max_points.saturating_sub(tail_budget);

    let mut tail_indices: Vec<usize> = positions
        .iter()
        .enumerate()
        .filter(|(_, p)| p.z.abs() > 0.15)
        .map(|(i, _)| i)
        .collect();
    tail_indices.sort_by(|&a, &b| {
        positions[b]
            .z
            .abs()
            .partial_cmp(&positions[a].z.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut picked = vec![false; n];
    let mut pos = Vec::with_capacity(max_points);
    let mut col = Vec::with_capacity(max_points);

    if !tail_indices.is_empty() {
        let tail_stride = tail_indices.len().div_ceil(tail_budget.max(1));
        for (j, &idx) in tail_indices.iter().enumerate() {
            if j % tail_stride == 0 && pos.len() < tail_budget {
                pos.push(positions[idx]);
                col.push(colors[idx]);
                picked[idx] = true;
            }
        }
    }

    let stride = n.div_ceil(core_budget.max(1));
    let mut i = 0;
    while i < n && pos.len() < max_points {
        if !picked[i] {
            pos.push(positions[i]);
            col.push(colors[i]);
        }
        i += stride;
    }

    (pos, col)
}

// Colormaps matching Python's matplotlib
fn magma(t: f32) -> Srgba {
    // A simplified magma/plasma-like colormap (purple to yellow)
    let hue = (280.0 - t * 220.0).clamp(0.0, 360.0);
    let lightness = (10.0 + t * 80.0).clamp(0.0, 100.0);
    let (r, g, b) = Hsl::from(hue, 90.0, lightness).to_rgb().as_tuple();
    Srgba::new(r as u8, g as u8, b as u8, 255)
}

fn cool(t: f32) -> Srgba {
    // Cyan to magenta
    let r = (t * 255.0).clamp(0.0, 255.0) as u8;
    let b = ((1.0 - t) * 255.0).clamp(0.0, 255.0) as u8;
    Srgba::new(r, 255 - r, b, 255)
}

pub fn generate_points_cpu(params: &BrotParams, mat: &DMatrix<Complex64>) -> (Vec<Vec3>, Vec<Srgba>) {
    let mode = params.mode;
    let nx = params.nx;
    let ny = if mode == BrotMode::WideAttractor { params.ny } else { 1 };
    let warmup = params.warmup;
    let keep = params.keep;
    let escape_r2 = (params.escape_r * params.escape_r) as f64;

    let re_min = params.re_min;
    let re_max = params.re_max;
    let im_min = params.im_min;
    let im_max = params.im_max;

    let coords: Vec<(f32, f32)> = if mode == BrotMode::Bifurcation {
        (0..nx).map(|i| {
            let re = re_min + (re_max - re_min) * (i as f32 / (nx - 1) as f32);
            (re, 0.0)
        }).collect()
    } else {
        let mut v = Vec::with_capacity(nx * ny);
        for j in 0..ny {
            let im = im_min + (im_max - im_min) * (j as f32 / (ny - 1) as f32);
            for i in 0..nx {
                let re = re_min + (re_max - re_min) * (i as f32 / (nx - 1) as f32);
                v.push((re, im));
            }
        }
        v
    };
    
    let ndim = mat.nrows();

    let result: Vec<(Vec<Vec3>, Vec<Srgba>)> = coords.par_iter().filter_map(|&(c_re, c_im)| {
        let mut z = DVector::zeros(ndim);
        let c = DVector::from_element(ndim, Complex64::new(c_re as f64, c_im as f64));
        let mut escaped = false;

        for _ in 0..warmup {
            let z_sq = z.map(|v| v * v);
            z = mat * z_sq + &c;

            if z.norm_squared() > escape_r2 {
                escaped = true;
                break;
            }
        }

        if escaped {
            return None;
        }

        let mut pos = Vec::with_capacity(keep);
        let mut col = Vec::with_capacity(keep);

        let color = if mode == BrotMode::Bifurcation {
            let t = (c_re - re_min) / (re_max - re_min);
            let mut c_rgba = magma(t);
            c_rgba.a = (0.12 * 255.0) as u8;
            c_rgba
        } else {
            let phase = c_im.atan2(c_re);
            let t = (phase + PI) / (2.0 * PI);
            let mut c_rgba = cool(t);
            c_rgba.a = (0.05 * 255.0) as u8;
            c_rgba
        };

        for _ in 0..keep {
            let z_sq = z.map(|v| v * v);
            z = mat * z_sq + &c;

            if z.norm_squared() > escape_r2 {
                break;
            }

            pos.push(vec3(c_re, c_im, z[0].re as f32));
            col.push(color);
        }

        Some((pos, col))
    }).collect();

    let mut final_pos = Vec::new();
    let mut final_col = Vec::new();
    for (p, c) in result {
        final_pos.extend(p);
        final_col.extend(c);
    }

    (final_pos, final_col)
}

pub fn generate_points_gpu(context: &Context, params: &BrotParams, mat: &DMatrix<Complex64>) -> Result<(Vec<Vec3>, Vec<Srgba>), String> {
    // Determine OpenGL version
    let mut use_compute = false;
    unsafe {
        use glow::HasContext;
        let version = context.get_parameter_string(glow::VERSION);
        if version.contains("OpenGL 4.3") || version.contains("OpenGL 4.4") || version.contains("OpenGL 4.5") || version.contains("OpenGL 4.6") {
            use_compute = true;
        }
    }

    // Fallback to CPU if matrix is larger than 10x10, or compute not supported
    if !use_compute || mat.nrows() > 10 {
        return Ok(generate_points_cpu(params, mat));
    }

    unsafe {
        use glow::HasContext;
        
        let shader_source = if params.mode == BrotMode::Bifurcation {
            r#"#version 430 core
            layout(local_size_x = 256) in;

            struct Point {
                vec4 pos;
                vec4 color;
            };

            layout(std430, binding = 0) buffer OutputBuffer {
                Point points[];
            };

            layout(binding = 0) uniform atomic_uint counter;

            uniform int nx;
            uniform int warmup;
            uniform int keep;
            uniform float escape_r2;
            uniform float re_min;
            uniform float re_max;

            uniform int ndim;
            uniform vec2 matrix[100]; // flattened N x N matrix (row-major)

            vec4 magma(float t) {
                float hue = clamp(280.0 - t * 220.0, 0.0, 360.0);
                float l = clamp(0.1 + t * 0.8, 0.0, 1.0);
                // Simple hue to RGB
                float c = 1.0 - abs(2.0 * l - 1.0);
                float x = c * (1.0 - abs(mod(hue / 60.0, 2.0) - 1.0));
                float m = l - c / 2.0;
                
                vec3 rgb = vec3(0.0);
                if (hue < 60.0) rgb = vec3(c, x, 0.0);
                else if (hue < 120.0) rgb = vec3(x, c, 0.0);
                else if (hue < 180.0) rgb = vec3(0.0, c, x);
                else if (hue < 240.0) rgb = vec3(0.0, x, c);
                else if (hue < 300.0) rgb = vec3(x, 0.0, c);
                else rgb = vec3(c, 0.0, x);
                
                return vec4(rgb + m, 0.12);
            }

            void main() {
                uint idx = gl_GlobalInvocationID.x;
                if (idx >= nx) return;

                float t = float(idx) / float(nx - 1);
                float c_re = mix(re_min, re_max, t);
                float c_im = 0.0;
                
                vec4 color = magma(t);

                vec2 Z[10];
                for(int i=0; i<ndim; i++) Z[i] = vec2(0.0);

                for (int i = 0; i < warmup; i++) {
                    vec2 Z_sq[10];
                    for(int j=0; j<ndim; j++) {
                        Z_sq[j] = vec2(Z[j].x*Z[j].x - Z[j].y*Z[j].y, 2.0*Z[j].x*Z[j].y);
                    }
                    vec2 new_Z[10];
                    float norm_sq = 0.0;
                    for(int row=0; row<ndim; row++) {
                        vec2 sum = vec2(c_re, c_im);
                        for(int col=0; col<ndim; col++) {
                            vec2 a = matrix[row * ndim + col];
                            vec2 z_sq = Z_sq[col];
                            sum.x += a.x * z_sq.x - a.y * z_sq.y;
                            sum.y += a.x * z_sq.y + a.y * z_sq.x;
                        }
                        new_Z[row] = sum;
                        norm_sq += sum.x*sum.x + sum.y*sum.y;
                    }
                    for(int j=0; j<ndim; j++) Z[j] = new_Z[j];
                    if (norm_sq > escape_r2) return;
                }

                for (int i = 0; i < keep; i++) {
                    vec2 Z_sq[10];
                    for(int j=0; j<ndim; j++) {
                        Z_sq[j] = vec2(Z[j].x*Z[j].x - Z[j].y*Z[j].y, 2.0*Z[j].x*Z[j].y);
                    }
                    vec2 new_Z[10];
                    float norm_sq = 0.0;
                    for(int row=0; row<ndim; row++) {
                        vec2 sum = vec2(c_re, c_im);
                        for(int col=0; col<ndim; col++) {
                            vec2 a = matrix[row * ndim + col];
                            vec2 z_sq = Z_sq[col];
                            sum.x += a.x * z_sq.x - a.y * z_sq.y;
                            sum.y += a.x * z_sq.y + a.y * z_sq.x;
                        }
                        new_Z[row] = sum;
                        norm_sq += sum.x*sum.x + sum.y*sum.y;
                    }
                    for(int j=0; j<ndim; j++) Z[j] = new_Z[j];
                    if (norm_sq > escape_r2) return;

                    uint out_idx = atomicCounterIncrement(counter);
                    points[out_idx].pos = vec4(c_re, c_im, Z[0].x, 0.0);
                    points[out_idx].color = color;
                }
            }
            "#
        } else {
            r#"#version 430 core
            layout(local_size_x = 16, local_size_y = 16) in;

            struct Point {
                vec4 pos;
                vec4 color;
            };

            layout(std430, binding = 0) buffer OutputBuffer {
                Point points[];
            };

            layout(binding = 0) uniform atomic_uint counter;

            uniform int nx;
            uniform int ny;
            uniform int warmup;
            uniform int keep;
            uniform float escape_r2;
            uniform float re_min;
            uniform float re_max;
            uniform float im_min;
            uniform float im_max;

            vec4 cool(float t) {
                t = clamp(t, 0.0, 1.0);
                return vec4(t, 1.0 - t, 1.0, 0.05);
            }

            void main() {
                uint ix = gl_GlobalInvocationID.x;
                uint iy = gl_GlobalInvocationID.y;
                if (ix >= nx || iy >= ny) return;

                float tx = float(ix) / float(nx - 1);
                float ty = float(iy) / float(ny - 1);
                float c_re = mix(re_min, re_max, tx);
                float c_im = mix(im_min, im_max, ty);

                float z_re = 0.0;
                float z_im = 0.0;

                for (int i = 0; i < warmup; i++) {
                    float z_re2 = z_re * z_re - z_im * z_im + c_re;
                    float z_im2 = 2.0 * z_re * z_im + c_im;
                    z_re = z_re2;
                    z_im = z_im2;
                    if (z_re * z_re + z_im * z_im > escape_r2) return;
                }

                float phase = atan(c_im, c_re);
                float tc = (phase + 3.14159265359) / (2.0 * 3.14159265359);
                vec4 color = cool(tc);

                for (int i = 0; i < keep; i++) {
                    float z_re2 = z_re * z_re - z_im * z_im + c_re;
                    float z_im2 = 2.0 * z_re * z_im + c_im;
                    z_re = z_re2;
                    z_im = z_im2;

                    if (z_re * z_re + z_im * z_im > escape_r2) return;

                    uint out_idx = atomicCounterIncrement(counter);
                    points[out_idx].pos = vec4(c_re, c_im, z_re, 0.0);
                    points[out_idx].color = color;
                }
            }
            "#
        };

        let shader = context.create_shader(glow::COMPUTE_SHADER).map_err(|e| e.to_string())?;
        context.shader_source(shader, shader_source);
        context.compile_shader(shader);
        if !context.get_shader_compile_status(shader) {
            return Err(format!("Compute shader compile error: {}", context.get_shader_info_log(shader)));
        }

        let program = context.create_program().map_err(|e| e.to_string())?;
        context.attach_shader(program, shader);
        context.link_program(program);
        if !context.get_program_link_status(program) {
            return Err(format!("Compute shader link error: {}", context.get_program_info_log(program)));
        }

        context.use_program(Some(program));

        // Uniforms
        if let Some(loc) = context.get_uniform_location(program, "nx") { context.uniform_1_i32(Some(&loc), params.nx as i32); }
        if let Some(loc) = context.get_uniform_location(program, "ny") { context.uniform_1_i32(Some(&loc), params.ny as i32); }
        if let Some(loc) = context.get_uniform_location(program, "warmup") { context.uniform_1_i32(Some(&loc), params.warmup as i32); }
        if let Some(loc) = context.get_uniform_location(program, "keep") { context.uniform_1_i32(Some(&loc), params.keep as i32); }
        if let Some(loc) = context.get_uniform_location(program, "escape_r2") { context.uniform_1_f32(Some(&loc), params.escape_r * params.escape_r); }
        if let Some(loc) = context.get_uniform_location(program, "re_min") { context.uniform_1_f32(Some(&loc), params.re_min); }
        if let Some(loc) = context.get_uniform_location(program, "re_max") { context.uniform_1_f32(Some(&loc), params.re_max); }
        if let Some(loc) = context.get_uniform_location(program, "im_min") { context.uniform_1_f32(Some(&loc), params.im_min); }
        if let Some(loc) = context.get_uniform_location(program, "im_max") { context.uniform_1_f32(Some(&loc), params.im_max); }
        if let Some(loc) = context.get_uniform_location(program, "ndim") { context.uniform_1_i32(Some(&loc), mat.nrows() as i32); }
        
        let mut mat_data = Vec::with_capacity(100 * 2);
        for row in 0..mat.nrows() {
            for col in 0..mat.ncols() {
                let val = mat[(row, col)];
                mat_data.push(val.re as f32);
                mat_data.push(val.im as f32);
            }
        }
        if let Some(loc) = context.get_uniform_location(program, "matrix") { context.uniform_2_f32_slice(Some(&loc), &mat_data); }

        // Setup Atomic Counter
        let ac_buffer = context.create_buffer().unwrap();
        context.bind_buffer(glow::ATOMIC_COUNTER_BUFFER, Some(ac_buffer));
        context.buffer_data_u8_slice(glow::ATOMIC_COUNTER_BUFFER, &[0, 0, 0, 0], glow::DYNAMIC_DRAW);
        context.bind_buffer_base(glow::ATOMIC_COUNTER_BUFFER, 0, Some(ac_buffer));

        // Setup SSBO
        let ssbo = context.create_buffer().unwrap();
        context.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(ssbo));
        
        let max_points = params.nx * params.ny * params.keep;
        // vec4 pos (16) + vec4 color (16) = 32 bytes per point
        // Max points * 32 bytes
        // 14M * 32 = ~448MB
        // glow::buffer_data_size requires length in bytes
        context.buffer_data_size(glow::SHADER_STORAGE_BUFFER, (max_points * 32) as i32, glow::STATIC_DRAW);
        context.bind_buffer_base(glow::SHADER_STORAGE_BUFFER, 0, Some(ssbo));

        // Dispatch
        if params.mode == BrotMode::Bifurcation {
            let groups_x = (params.nx as u32 + 255) / 256;
            context.dispatch_compute(groups_x, 1, 1);
        } else {
            let groups_x = (params.nx as u32 + 15) / 16;
            let groups_y = (params.ny as u32 + 15) / 16;
            context.dispatch_compute(groups_x, groups_y, 1);
        }

        context.memory_barrier(glow::SHADER_STORAGE_BARRIER_BIT | glow::ATOMIC_COUNTER_BARRIER_BIT);

        // Read back counter
        context.bind_buffer(glow::ATOMIC_COUNTER_BUFFER, Some(ac_buffer));
        let mut count_data = [0u8; 4];
        
        // Wait, get_buffer_sub_data is not directly taking u8 slice in all glow backends, actually we can just read using get_buffer_sub_data.
        // But since WebGL/OpenGL diffs, let's map or read.
        // To be safer, we can use `get_buffer_sub_data`.
        // Wait, `get_buffer_sub_data` might not exist in `glow::HasContext` depending on backend.
        // Let's use `gl.map_buffer_range`.
        let ptr = context.map_buffer_range(glow::ATOMIC_COUNTER_BUFFER, 0, 4, glow::MAP_READ_BIT);
        std::ptr::copy_nonoverlapping(ptr, count_data.as_mut_ptr(), 4);
        context.unmap_buffer(glow::ATOMIC_COUNTER_BUFFER);
        let actual_points = u32::from_ne_bytes(count_data) as usize;
        
        let actual_points = actual_points.min(max_points);

        // Read back SSBO
        context.bind_buffer(glow::SHADER_STORAGE_BUFFER, Some(ssbo));
        let mut pos = Vec::with_capacity(actual_points);
        let mut col = Vec::with_capacity(actual_points);
        
        if actual_points > 0 {
            let byte_count = actual_points * 32;
            let data_ptr = context.map_buffer_range(glow::SHADER_STORAGE_BUFFER, 0, byte_count as i32, glow::MAP_READ_BIT);
            
            // Reinterpret to struct Point { vec4, vec4 }
            let points_f32 = std::slice::from_raw_parts(data_ptr as *const f32, actual_points * 8);
            
            for i in 0..actual_points {
                let px = points_f32[i * 8 + 0];
                let py = points_f32[i * 8 + 1];
                let pz = points_f32[i * 8 + 2];
                pos.push(vec3(px, py, pz));

                let cr = (points_f32[i * 8 + 4] * 255.0).clamp(0.0, 255.0) as u8;
                let cg = (points_f32[i * 8 + 5] * 255.0).clamp(0.0, 255.0) as u8;
                let cb = (points_f32[i * 8 + 6] * 255.0).clamp(0.0, 255.0) as u8;
                let ca = (points_f32[i * 8 + 7] * 255.0).clamp(0.0, 255.0) as u8;
                col.push(Srgba::new(cr, cg, cb, ca));
            }

            context.unmap_buffer(glow::SHADER_STORAGE_BUFFER);
        }

        context.delete_buffer(ssbo);
        context.delete_buffer(ac_buffer);
        context.delete_program(program);
        context.delete_shader(shader);

        Ok((pos, col))
    }
}
