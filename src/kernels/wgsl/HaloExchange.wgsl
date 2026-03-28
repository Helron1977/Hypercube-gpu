/* Standard WGSL header injected by GpuDispatcher handles bindings 0 & 1 */
@group(0) @binding(2) var<storage, read> transfers: array<TransferParams>;

struct TransferParams {
    srcBase: u32,
    dstBase: u32,
    stride: u32,
    lx: u32,
    ly: u32,
    lz: u32,
    numFaces: u32,
    axis: u32,    // 0: X-face, 1: Y-face, 2: Z-face
    srcPos: u32,  // index along axis to read from
    dstPos: u32,  // index along axis to write to
    size1: u32,   // dimension of first orthogonal axis
    size2: u32    // dimension of second orthogonal axis
};

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let t_idx = wg_id.z; // Each Z workgroup slice handles one Transfer operation
    let p = transfers[t_idx];
    
    let u = id.x; // First orthogonal dimension
    let v = id.y; // Second orthogonal dimension
    
    if (u >= p.size1 || v >= p.size2) { return; }
    
    var src_i: u32;
    var dst_i: u32;
    
    if (p.axis == 0u) { // X-Face (U=Y, V=Z)
        src_i = (v * p.ly + u) * p.lx + p.srcPos;
        dst_i = (v * p.ly + u) * p.lx + p.dstPos;
    } else if (p.axis == 1u) { // Y-Face (U=X, V=Z)
        src_i = (v * p.ly + p.srcPos) * p.lx + u;
        dst_i = (v * p.ly + p.dstPos) * p.lx + u;
    } else { // Z-Face (U=X, V=Y)
        src_i = (p.srcPos * p.ly + v) * p.lx + u;
        dst_i = (p.dstPos * p.ly + v) * p.lx + u;
    }
    
    // Copy all faces
    for (var f = 0u; f < p.numFaces; f = f + 1u) {
        data[p.dstBase + f * p.stride + dst_i] = data[p.srcBase + f * p.stride + src_i];
    }
}
