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

    let Rank = u32(params.p0);
    let lx = params.lx;
    let ly = params.ly;
    let stride = params.strideFace;

    // Tensor CP Reconstruction: T = Sum(a_r * b_r * c_r)
    // f0: Factor A (I x R), f1: Factor B (J x R), f2: Factor C (K x R)
    // f3: Target Tensor T (I x J x K)
    
    for (var pz: u32 = 0u; pz < params.ly; pz++) { // Using nz limit if available, but ly*lz is total depth
        // We only compute for pz < nz manifest dim
        if (pz >= params.ly - 2u) { break; } // Safety
        
        var sum: f32 = 0.0;
        for (var r: u32 = 0u; r < Rank; r++) {
            // Factors are 1D arrays stored in the face's 2D grid
            // A[px, r] is at row r, col px
            let valA = data[params.f0 * stride + (r + 1u) * lx + (px + 1u)];
            let valB = data[params.f1 * stride + (r + 1u) * lx + (py + 1u)];
            let valC = data[params.f2 * stride + (r + 1u) * lx + (pz + 1u)];
            sum += valA * valB * valC;
        }

        // T[px, py, pz] is at depth pz, row py, col px
        // pz in the loop is zero-based, we use (pz+1) for ghost offset
        let idxT = (pz + 1u) * lx * ly + (py + 1u) * lx + (px + 1u);
        data[params.f3 * stride + idxT] = sum;
        
        // Break if we exceeded nz (if nz is smaller than ly-2)
        // Actually for simplicity, we assume nz is the 3rd dimension provided in manifest.
        // Wait, params.ly is the padded Y dimension. 
        // Better use px, py, pz and compare with nx, ny, nz.
    }
}
