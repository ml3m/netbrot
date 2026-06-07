use colors_transform::{Color, Hsl};
use three_d::Srgba;

/// How to blend 3D point layers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlendMode3d {
    Transparency,
    Additive,
    Opaque,
}

impl BlendMode3d {
    pub const ALL: [BlendMode3d; 3] = [
        BlendMode3d::Transparency,
        BlendMode3d::Additive,
        BlendMode3d::Opaque,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            BlendMode3d::Transparency => "Transparency",
            BlendMode3d::Additive => "Additive",
            BlendMode3d::Opaque => "Opaque",
        }
    }
}

/// Which scalar drives fragment coloring (selected live in shader).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ColorMode3d {
    /// Color by Re(c) — the x-axis of the parameter plane.
    ReC,
    /// Color by phase(c) = atan2(Im(c), Re(c)).
    PhaseC,
    /// Color by Re(z) — the height axis.
    HeightZ,
}

impl ColorMode3d {
    pub const ALL: [ColorMode3d; 3] = [
        ColorMode3d::ReC,
        ColorMode3d::PhaseC,
        ColorMode3d::HeightZ,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ColorMode3d::ReC => "Re(c)",
            ColorMode3d::PhaseC => "Phase(c)",
            ColorMode3d::HeightZ => "Height(z)",
        }
    }

    /// Integer id sent as a uniform to the fragment shader.
    pub fn uniform_id(&self) -> i32 {
        match self {
            ColorMode3d::ReC => 0,
            ColorMode3d::PhaseC => 1,
            ColorMode3d::HeightZ => 2,
        }
    }
}

/// Which colormap to use in the fragment shader.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Colormap3d {
    Magma,
    Cool,
    Plasma,
    Viridis,
    Fire,
    Gray,
}

impl Colormap3d {
    pub const ALL: [Colormap3d; 6] = [
        Colormap3d::Magma,
        Colormap3d::Cool,
        Colormap3d::Plasma,
        Colormap3d::Viridis,
        Colormap3d::Fire,
        Colormap3d::Gray,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Colormap3d::Magma => "Magma",
            Colormap3d::Cool => "Cool",
            Colormap3d::Plasma => "Plasma",
            Colormap3d::Viridis => "Viridis",
            Colormap3d::Fire => "Fire",
            Colormap3d::Gray => "Gray",
        }
    }

    /// Integer id sent as a uniform to the fragment shader.
    pub fn uniform_id(&self) -> i32 {
        match self {
            Colormap3d::Magma => 0,
            Colormap3d::Cool => 1,
            Colormap3d::Plasma => 2,
            Colormap3d::Viridis => 3,
            Colormap3d::Fire => 4,
            Colormap3d::Gray => 5,
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side colormap helpers (used during generation and for legend display)
// ---------------------------------------------------------------------------

/// Magma-like colormap (purple → orange → yellow).
pub fn magma(t: f32) -> Srgba {
    let t = t.clamp(0.0, 1.0);
    let hue = (280.0 - t * 220.0).clamp(0.0, 360.0);
    let lightness = (10.0 + t * 80.0).clamp(0.0, 100.0);
    let (r, g, b) = Hsl::from(hue, 90.0, lightness).to_rgb().as_tuple();
    Srgba::new(r as u8, g as u8, b as u8, 255)
}

/// Cool colormap (cyan → magenta).
pub fn cool(t: f32) -> Srgba {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 255.0).clamp(0.0, 255.0) as u8;
    let b = ((1.0 - t) * 255.0).clamp(0.0, 255.0) as u8;
    Srgba::new(r, 255 - r, b, 255)
}

/// Plasma-like colormap (purple → pink → yellow).
pub fn plasma(t: f32) -> Srgba {
    let t = t.clamp(0.0, 1.0);
    let hue = (300.0 - t * 260.0).clamp(0.0, 360.0);
    let lightness = (15.0 + t * 70.0).clamp(0.0, 100.0);
    let (r, g, b) = Hsl::from(hue, 95.0, lightness).to_rgb().as_tuple();
    Srgba::new(r as u8, g as u8, b as u8, 255)
}

/// Viridis-like colormap (purple → teal → yellow).
pub fn viridis(t: f32) -> Srgba {
    let t = t.clamp(0.0, 1.0);
    let hue = (280.0 - t * 220.0).clamp(40.0, 280.0);
    let sat = (70.0 + t * 25.0).clamp(0.0, 100.0);
    let lightness = (20.0 + t * 60.0).clamp(0.0, 100.0);
    let (r, g, b) = Hsl::from(hue, sat, lightness).to_rgb().as_tuple();
    Srgba::new(r as u8, g as u8, b as u8, 255)
}

/// Fire colormap (black → red → orange → yellow → white).
pub fn fire(t: f32) -> Srgba {
    let t = t.clamp(0.0, 1.0);
    let r = (t * 3.0).clamp(0.0, 1.0);
    let g = ((t - 0.33) * 3.0).clamp(0.0, 1.0);
    let b = ((t - 0.67) * 3.0).clamp(0.0, 1.0);
    Srgba::new(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
        255,
    )
}

/// Grayscale colormap.
pub fn gray(t: f32) -> Srgba {
    let v = (t.clamp(0.0, 1.0) * 255.0) as u8;
    Srgba::new(v, v, v, 255)
}

/// Evaluate any `Colormap3d` on the CPU.
pub fn eval_colormap(cm: Colormap3d, t: f32) -> Srgba {
    match cm {
        Colormap3d::Magma => magma(t),
        Colormap3d::Cool => cool(t),
        Colormap3d::Plasma => plasma(t),
        Colormap3d::Viridis => viridis(t),
        Colormap3d::Fire => fire(t),
        Colormap3d::Gray => gray(t),
    }
}

// ---------------------------------------------------------------------------
// GLSL snippets — concatenated into the fragment shader by `point_cloud.rs`
// ---------------------------------------------------------------------------

/// All 6 colormap functions in GLSL, plus the dispatcher `applyColormap(int id, float t)`.
pub const COLORMAP_GLSL: &str = r#"
vec3 magma_map(float t) {
    t = clamp(t, 0.0, 1.0);
    float hue = clamp(280.0 - t * 220.0, 0.0, 360.0);
    float l   = clamp(0.1 + t * 0.8, 0.0, 1.0);
    float c   = 1.0 - abs(2.0 * l - 1.0);
    float x   = c * (1.0 - abs(mod(hue / 60.0, 2.0) - 1.0));
    float m   = l - c / 2.0;
    vec3 rgb;
    if      (hue < 60.0)  rgb = vec3(c, x, 0.0);
    else if (hue < 120.0) rgb = vec3(x, c, 0.0);
    else if (hue < 180.0) rgb = vec3(0.0, c, x);
    else if (hue < 240.0) rgb = vec3(0.0, x, c);
    else if (hue < 300.0) rgb = vec3(x, 0.0, c);
    else                  rgb = vec3(c, 0.0, x);
    return rgb + m;
}

vec3 cool_map(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(t, 1.0 - t, 1.0);
}

vec3 plasma_map(float t) {
    t = clamp(t, 0.0, 1.0);
    float hue = clamp(300.0 - t * 260.0, 0.0, 360.0);
    float l   = clamp(0.15 + t * 0.7, 0.0, 1.0);
    float c   = 1.0 - abs(2.0 * l - 1.0);
    float x   = c * (1.0 - abs(mod(hue / 60.0, 2.0) - 1.0));
    float m   = l - c / 2.0;
    vec3 rgb;
    if      (hue < 60.0)  rgb = vec3(c, x, 0.0);
    else if (hue < 120.0) rgb = vec3(x, c, 0.0);
    else if (hue < 180.0) rgb = vec3(0.0, c, x);
    else if (hue < 240.0) rgb = vec3(0.0, x, c);
    else if (hue < 300.0) rgb = vec3(x, 0.0, c);
    else                  rgb = vec3(c, 0.0, x);
    return rgb + m;
}

vec3 viridis_map(float t) {
    t = clamp(t, 0.0, 1.0);
    float hue = clamp(280.0 - t * 220.0, 40.0, 280.0);
    float s   = clamp(0.7 + t * 0.25, 0.0, 1.0);
    float l   = clamp(0.2 + t * 0.6, 0.0, 1.0);
    float c   = (1.0 - abs(2.0 * l - 1.0)) * s;
    float x   = c * (1.0 - abs(mod(hue / 60.0, 2.0) - 1.0));
    float m   = l - c / 2.0;
    vec3 rgb;
    if      (hue < 60.0)  rgb = vec3(c, x, 0.0);
    else if (hue < 120.0) rgb = vec3(x, c, 0.0);
    else if (hue < 180.0) rgb = vec3(0.0, c, x);
    else if (hue < 240.0) rgb = vec3(0.0, x, c);
    else if (hue < 300.0) rgb = vec3(x, 0.0, c);
    else                  rgb = vec3(c, 0.0, x);
    return rgb + m;
}

vec3 fire_map(float t) {
    t = clamp(t, 0.0, 1.0);
    float r = clamp(t * 3.0, 0.0, 1.0);
    float g = clamp((t - 0.333) * 3.0, 0.0, 1.0);
    float b = clamp((t - 0.667) * 3.0, 0.0, 1.0);
    return vec3(r, g, b);
}

vec3 gray_map(float t) {
    t = clamp(t, 0.0, 1.0);
    return vec3(t);
}

vec3 applyColormap(int id, float t) {
    if      (id == 0) return magma_map(t);
    else if (id == 1) return cool_map(t);
    else if (id == 2) return plasma_map(t);
    else if (id == 3) return viridis_map(t);
    else if (id == 4) return fire_map(t);
    else              return gray_map(t);
}
"#;
