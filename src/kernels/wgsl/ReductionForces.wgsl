
// Hypercube D2Q9 MOMENTUM EXCHANGE (MEM) KERNEL
// --------------------------------------------
// Calculates Aerodynamic Drag (Cd) and Lift (Cl).
// v6.0 elite: Direct Global Workspace Accumulation.

const DX = array<i32, 9>(0, 1, 0, -1, 0, 1, -1, -1, 1);
const DY = array<i32, 9>(0, 0, 1, 0, -1, 1, 1, -1, -1);
const OPP = array<u32, 9>(0, 3, 4, 1, 2, 7, 8, 5, 6);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let nx = uniforms.nx; let ny = uniforms.ny;
    if (id.x >= nx || id.y >= ny) { return; }
    let px = id.x; let py = id.y;

    // We only iterate over solid (obstacle) nodes
    if (read_obs_Now(px, py) > 0.5) {
        var fx: f32 = 0.0;
        var fy: f32 = 0.0;
        
        for (var d = 1u; d < 9u; d = d + 1u) {
            let n_px = i32(px) + DX[d];
            let n_py = i32(py) + DY[d];
            
            // Check boundaries
            if (n_px >= 0 && n_px < i32(nx) && n_py >= 0 && n_py < i32(ny)) {
                // If neighbor is fluid
                if (read_obs_Now(u32(n_px), u32(n_py)) < 0.5) {
                    // MEM: Force = Momentum_In + Momentum_Out
                    // f_out is population coming from solid (OPP[d]) towards fluid
                    // f_in is population coming from fluid (d) towards solid
                    let f_out = read_f_Now(px, py, d);
                    let f_in  = read_f_Now(u32(n_px), u32(n_py), OPP[d]);
                    
                    let momentum = f_out + f_in;
                    fx += momentum * f32(DX[d]);
                    fy += momentum * f32(DY[d]);
                }
            }
        }
        
        // p7 is the global precision scale (e.g. 1e9)
        let scale = uniforms.p7;
        
        // Accumulate to Global Workspace (forces)
        if (fx != 0.0) { atomicAdd_global_forces(0u, fx * scale); }
        if (fy != 0.0) { atomicAdd_global_forces(1u, fy * scale); }
    }
}
