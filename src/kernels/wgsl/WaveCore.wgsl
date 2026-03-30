@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let px = id.x; let py = id.y;
    if (px >= uniforms.nx || py >= uniforms.ny) { return; }

    // p0: alpha (c^2 * dt^2 / dx^2)
    // p5: default_dirichlet_value
    let uNow = read_u_Now(px, py);
    let uOld = read_u_Old(px, py);
    
    var UL = read_u_Now(px - 1u, py);
    var UR = read_u_Now(px + 1u, py);
    var UT = read_u_Now(px, py - 1u);
    var UB = read_u_Now(px, py + 1u);
    
    // Boundary Roles (8: Dirichlet, 9: Neumann/Mirror)
    // Left Boundary (X=0)
    if (px == 0u) {
        if (uniforms.leftRole == 8u) { UL = uniforms.p5; }
        else if (uniforms.leftRole == 9u) { UL = uNow; }
    }
    // Right Boundary (X=nx-1)
    if (px == uniforms.nx - 1u) {
        if (uniforms.rightRole == 8u) { UR = uniforms.p5; }
        else if (uniforms.rightRole == 9u) { UR = uNow; }
    }
    // Top Boundary (Y=0)
    if (py == 0u) {
        if (uniforms.topRole == 8u) { UT = uniforms.p5; }
        else if (uniforms.topRole == 9u) { UT = uNow; }
    }
    // Bottom Boundary (Y=ny-1)
    if (py == uniforms.ny - 1u) {
        if (uniforms.bottomRole == 8u) { UB = uniforms.p5; }
        else if (uniforms.bottomRole == 9u) { UB = uNow; }
    }
    
    let lap = UL + UR + UT + UB - 4.0 * uNow;
    
    // Wave Propagation Equation (Triple Buffering)
    // uNext = 2.0*uNow - uOld + alpha*laplacian
    write_u_Next(px, py, 2.0 * uNow - uOld + uniforms.p0 * lap);
}
