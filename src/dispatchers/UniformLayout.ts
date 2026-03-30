/**
 * UniformLayout.ts
 * Canonical source of truth for the Uniforms buffer layout.
 * Ensures that UniformStagingManager (TypeScript/CPU) and WgslHeaderGenerator (WGSL/GPU)
 * always share the exact same memory structure.
 */

export const UniformLayoutValues = {
    // Basic Dimensions (Index: 0-3)
    NX: 0,
    NY: 1,
    STRIDE_ROW: 2,
    PHYSICAL_NY: 3,

    // Temporal State (Index: 4-7)
    T: 4,
    TICK: 5,
    STRIDE_FACE: 6,
    NUM_FACES: 7,

    // Physical Parameters (Index: 8-15)
    PARAMS_START: 8,
    PARAMS_COUNT: 8,

    // Face Buffer Mappings (Index: 16-79)
    FACES_START: 16,
    FACES_COUNT: 64,

    // Topology & Boundaries (Index: 80-86)
    GHOSTS: 80,
    LEFT_ROLE: 81,
    RIGHT_ROLE: 82,
    TOP_ROLE: 83,
    BOTTOM_ROLE: 84,
    FRONT_ROLE: 85,
    BACK_ROLE: 86
};

/**
 * Returns the WGSL struct definition matching the canonical layout.
 */
export function getUniformsStructWGSL(): string {
    return `
struct Uniforms {
    nx: u32,
    ny: u32,
    strideRow: u32,
    physicalNy: u32,
    t: f32,
    tick: u32,
    strideFace: u32,
    numFaces: u32,
    p0: f32, p1: f32, p2: f32, p3: f32,
    p4: f32, p5: f32, p6: f32, p7: f32,
    faces: array<u32, 64>,
    ghosts: u32,
    leftRole: u32,
    rightRole: u32,
    topRole: u32,
    bottomRole: u32,
    frontRole: u32,
    backRole: u32,
    reserved: u32
};
`;
}
