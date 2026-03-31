// Redefinition struct Params removed - using inherited uniforms

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y; let pz = id.z;
    if (px >= uniforms.nx || py >= uniforms.ny || pz >= u32(uniforms.p0)) { return; }

    // p0: nz, p1: maxIter, p2: power
    let maxIter = u32(uniforms.p1);
    let power = uniforms.p2;

    // Map voxel to [-1.5, 1.5]³
    let c = vec3<f32>(
        (f32(px) / f32(uniforms.nx) - 0.5) * 3.0,
        (f32(py) / f32(uniforms.ny) - 0.5) * 3.0,
        (f32(pz) / f32(u32(uniforms.p0)) - 0.5) * 3.0
    );

    var z = c;
    var iterations = 0u;

    for (var it = 0u; it < maxIter; it++) {
        let r = length(z);
        if (r > 2.0) { break; }
        iterations = it + 1u;

        // Spherical coordinates for Mandelbulb
        let theta = acos(clamp(z.z / r, -1.0, 1.0));
        let phi = atan2(z.y, z.x);
        let rn = pow(r, power);

        // Mandelbulb transform
        z = vec3<f32>(
            rn * sin(theta * power) * cos(phi * power),
            rn * sin(theta * power) * sin(phi * power),
            rn * cos(theta * power)
        ) + c;
    }

    write_fractal_Now(px, py, pz, f32(iterations) / f32(maxIter));
}
