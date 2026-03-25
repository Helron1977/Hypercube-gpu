struct Params {
    nx: u32, ny: u32, lx: u32, ly: u32,
    t: f32, tick: u32, strideFace: u32, numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32, p4: f32, p5: f32, p6: f32, p7: f32,
    f0: u32, f1: u32, f2: u32, f3: u32, f4: u32, f5: u32, f6: u32, f7: u32,
    f8: u32, f9: u32, f10: u32, f11: u32, f12: u32, f13: u32, f14: u32, f15: u32,
    leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32
};

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    if (px >= params.nx || py >= params.ny) { return; }

    // p0: nz, p1: maxIter, p2: power
    let nz = u32(params.p0);
    let maxIter = u32(params.p1);
    let power = params.p2;
    let lx = params.lx;
    let ly = params.ly;

    // Loop over all Z slices (framework dispatcher is 2D)
    for (var pz = 0u; pz < nz; pz++) {
        let i = ((pz + 1u) * ly + (py + 1u)) * lx + (px + 1u);

        // Map voxel to [-1.5, 1.5]³
        let c = vec3<f32>(
            (f32(px) / f32(params.nx) - 0.5) * 3.0,
            (f32(py) / f32(params.ny) - 0.5) * 3.0,
            (f32(pz) / f32(nz) - 0.5) * 3.0
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

        data[params.f0 * params.strideFace + i] = f32(iterations) / f32(maxIter);
    }
}
