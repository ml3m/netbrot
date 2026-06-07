use crate::gui::scene3d::Scene3D;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn export_to_ply(path: &Path, scene: &Scene3D) -> Result<(), std::io::Error> {
    let mut file = BufWriter::new(File::create(path)?);

    // Count total visible points
    let mut total_points = 0;
    for layer in scene.visible_layers() {
        total_points += layer.cpu_positions.len();
    }

    // Write PLY header
    writeln!(file, "ply")?;
    writeln!(file, "format ascii 1.0")?;
    writeln!(file, "element vertex {}", total_points)?;
    writeln!(file, "property float x")?;
    writeln!(file, "property float y")?;
    writeln!(file, "property float z")?;
    writeln!(file, "property uchar red")?;
    writeln!(file, "property uchar green")?;
    writeln!(file, "property uchar blue")?;
    writeln!(file, "end_header")?;

    // Write point data
    for layer in scene.visible_layers() {
        for (pos, col) in layer.cpu_positions.iter().zip(layer.cpu_colors.iter()) {
            writeln!(file, "{} {} {} {} {} {}", pos.x, pos.y, pos.z, col.r, col.g, col.b)?;
        }
    }

    file.flush()?;
    Ok(())
}
