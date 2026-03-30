
// Hypercube D3Q27 REDUCTION BENCHMARK KERNEL (v6.0 Alpha)
// ------------------------------------------------------
// Optimized for Zero-Stall force extraction in 3D.

const DX = array<i32, 27>(0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1);
const DY = array<i32, 27>(0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1, 1, 1,-1,-1, 1, 1,-1,-1);
const DZ = array<i32, 27>(0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1, 1, 1, 1, 1,-1,-1,-1,-1);
const OPP = array<u32, 27>(0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 26, 25, 24, 23, 22, 21, 20, 19);

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny; let nz = u32(uniforms.p0); // p0: nz
    if (id.x >= nx || id.y >= ny || id.z >= nz) { return; }
    let px = id.x; let py = id.y; let pz = id.z;
    
    // Force extraction (Momentum Exchange method)
    if (read_type_Now(px, py, pz) > 0.5) {
        var fx: f32 = 0.0;
        var fy: f32 = 0.0;
        var fz: f32 = 0.0;
        
        for (var d = 1u; d < 27u; d = d + 1u) {
            let n_px = i32(px) + DX[d];
            let n_py = i32(py) + DY[d];
            let n_pz = i32(pz) + DZ[d];
            
            if (n_px >= 0 && n_px < i32(nx) && n_py >= 0 && n_py < i32(ny) && n_pz >= 0 && n_pz < i32(nz)) {
                if (read_type_Now(u32(n_px), u32(n_py), u32(n_pz)) < 0.5) {
                    let f_out = read_f_Now(px, py, pz, d);
                    let f_in  = read_f_Now(u32(n_px), u32(n_py), u32(n_pz), OPP[d]);
                    
                    let momentum = f_out + f_in;
                    fx += momentum * f32(DX[d]);
                    fy += momentum * f32(DY[d]);
                    fz += momentum * f32(DZ[d]);
                }
            }
        }
        
        // v6.0 Atomic Global Workspace for bit-accurate force tracking
        if (fx != 0.0) { atomicAdd_global_forces(0u, fx); }
        if (fy != 0.0) { atomicAdd_global_forces(1u, fy); }
        if (fz != 0.0) { atomicAdd_global_forces(2u, fz); }
    }
}
