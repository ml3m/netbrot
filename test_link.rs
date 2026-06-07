use three_d::*;
fn main() {
    let mut cpu_mesh = CpuMesh::cylinder(16);
    let transform = Mat4::from_translation(vec3(0.0, 0.0, -2.0))
        * Mat4::from_angle_y(degrees(-90.0))
        * Mat4::from_nonuniform_scale(4.0, 0.02, 0.02);
    cpu_mesh.transform(&transform).unwrap(); // Wait, transform signature takes Mat4 or &Mat4?
}
