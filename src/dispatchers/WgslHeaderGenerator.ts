import { FaceMapping, GlobalMapping } from '../DataContract';
import { getUniformsStructWGSL } from './UniformLayout';

/**
 * !!! MANIFESTO: THE GENERATOR IS A SERVANT !!!
 * DO NOT INVENT NOMENCLATURE OR ADD PADDING/SLOTS THAT THE MANIFEST HAS NOT AUTHORIZED.
 * - Standard MODE (v6): Now/Next/Old/Base.
 */

export interface WgslHeaderResult {
    code: string;
    usesAtomics: boolean;
    usesGlobals: boolean;
}

export class WgslHeaderGenerator {
    /**
     * ENCAPSULATED DETECTION LOGIC (v6.0 Zero-Stall)
     * - Discovery Mode (empty source): Returns all true for test/legacy compatibility.
     * - Surgical Mode (provided source): Scans for actual intent keywords.
     */
    public static detectRequirements(kernelSource: string): { usesAtomics: boolean, usesGlobals: boolean } {
        const isDiscovery = kernelSource === "";
        const globRegex = /\b(?!global_invocation_id)global\w*\b|globals\[|atomicAdd_global/;
        
        // Field atomics detection: match general 'atomic' but EXCLUDE 'atomicAdd_global' macros
        const atomRegex = /\batomic(?!Add_global)\w*\b|dataAtomic/;
        
        return {
            usesGlobals: isDiscovery || globRegex.test(kernelSource),
            usesAtomics: isDiscovery || atomRegex.test(kernelSource)
        };
    }

    public static getHeader(mode: string, faces: FaceMapping[], globals: GlobalMapping[] = [], kernelSource: string = ""): WgslHeaderResult {
        const safeFaces = Array.isArray(faces) ? faces : [];
        const safeGlobals = Array.isArray(globals) ? globals : [];

        const req = this.detectRequirements(kernelSource);

        const hasAtomicsInFields = safeFaces.some(f => f.type?.startsWith('atomic')) && req.usesAtomics;
        const hasAtomicsInGlobals = safeGlobals.some(g => g.type.startsWith('atomic'));
        
        const hasAtomicF32 = (safeFaces.some(f => f.type === 'atomic_f32') && req.usesAtomics) ||
                             (safeGlobals.some(g => g.type === 'atomic_f32') && req.usesGlobals);

        // ACTUAL USAGE FLAGS (Definitive source for BindGroup Layout)
        let usesAtomics = req.usesAtomics; // Binding 2
        let usesGlobals = false; // Binding 3

        // 1. CANONICAL RESOURCE BINDINGS
        let h = `@group(0) @binding(0) var<storage, read_write> data: array<f32>;\n`;
        h += `@group(0) @binding(1) var<uniform> uniforms: Uniforms;\n`;
        
        // --- FIELD ATOMICS (@binding 2) ---
        // Only inject if the kernel actually wants atomics AND we have atomic fields
        if (hasAtomicsInFields && req.usesAtomics) {
            h += `@group(0) @binding(2) var<storage, read_write> dataAtomic: array<atomic<i32>>;\n`;
            usesAtomics = true;
        }

        // --- CONTEXTUAL GLOBAL WORKSPACE (Binding 3) ---
        // Only inject if the kernel actually wants globals AND we have globals defined
        if (safeGlobals.length > 0 && req.usesGlobals) {
            const globalType = hasAtomicsInGlobals ? 'atomic<i32>' : 'f32';
            h += `@group(0) @binding(3) var<storage, read_write> globals: array<${globalType}>;\n`;
            usesGlobals = true;
        }
        h += `\n`;

        // 2. CANONICAL UNIFORM STRUCTURE
        h += getUniformsStructWGSL() + `\n`;
        
        h += `// Hypercube SOTA Header (v6.0: Binding 3 Enabled)\n\n`;

        // 3. ATOMIC HELPERS (CAS Loop for Float32)
        const hasAtomicF32Faces = safeFaces.some(f => f.type === 'atomic_f32') && req.usesAtomics;
        const hasAtomicF32Globals = safeGlobals.some(g => g.type === 'atomic_f32') && req.usesGlobals;

        if (hasAtomicF32Faces) {
            h += `fn _hypercube_atomicAddF32(ptr: u32, val: f32) {
    var old_val = atomicLoad(&dataAtomic[ptr]);
    loop {
        let new_val = bitcast<i32>(bitcast<f32>(old_val) + val);
        let res = atomicCompareExchangeWeak(&dataAtomic[ptr], old_val, new_val);
        if (res.exchanged) { break; }
        old_val = res.old_value;
    }
}\n\n`;
        }

        if (hasAtomicF32Globals) {
            h += `fn _hypercube_atomicAddF32_global(ptr: u32, val: f32) {
    var old_val = atomicLoad(&globals[ptr]);
    loop {
        let new_val = bitcast<i32>(bitcast<f32>(old_val) + val);
        let res = atomicCompareExchangeWeak(&globals[ptr], old_val, new_val);
        if (res.exchanged) { break; }
        old_val = res.old_value;
    }
}\n\n`;
        }
        
        // 4. FIELD INDEXING HELPERS
        h += `fn getIndex(x: u32, y: u32) -> u32 {
    let lx = uniforms.nx + uniforms.ghosts * 2u;
    return (y + uniforms.ghosts) * lx + (x + uniforms.ghosts);
}\n\n`;

        h += `fn getIndex3D(x: u32, y: u32, z: u32) -> u32 {
    let lx = uniforms.nx + uniforms.ghosts * 2u;
    let ly = uniforms.ny + uniforms.ghosts * 2u;
    return (z + uniforms.ghosts) * (lx * ly) + (y + uniforms.ghosts) * lx + (x + uniforms.ghosts);
}\n\n`;

        const is3D = mode.toLowerCase().includes('3d') || 
                     safeFaces.some(f => f.type === 'population3D' || f.numComponents === 27 || /^[uvw].*z$/i.test(f.name));

        const indexCall = is3D ? `getIndex3D(x, y, z)` : `getIndex(x, y)`;
        const sig = is3D ? `x: u32, y: u32, z: u32` : `x: u32, y: u32`;

        // 5. FIELD MACROS (Grid-based)
        safeFaces.forEach((f) => {
            const comp = f.numComponents || 1;
            const isAtomic = f.type?.startsWith('atomic');
            const suffixes = ['Now', 'Next', 'Old', 'Base'];
            
            suffixes.forEach((suffix, sIdx) => {
                if (sIdx >= f.numSlots) return;
                const faceIdx = `uniforms.faces[${f.pointerOffset + sIdx}u]`;

                if (isAtomic) {
                    const ptr = `${faceIdx} * uniforms.strideFace + ${indexCall}`;
                    h += `fn read_${f.name}_${suffix}(${sig}) -> f32 { return bitcast<f32>(atomicLoad(&dataAtomic[${ptr}])); }\n`;
                    h += `fn atomicAdd_${f.name}_${suffix}(${sig}, val: f32) { atomicAdd(&dataAtomic[${ptr}], i32(val)); }\n`;
                } else if (comp > 1) {
                    h += `fn read_${f.name}_${suffix}(${sig}, d: u32) -> f32 { return data[${faceIdx} * uniforms.strideFace + d * uniforms.strideFace + ${indexCall}]; }\n`;
                    h += `fn write_${f.name}_${suffix}(${sig}, d: u32, val: f32) { data[${faceIdx} * uniforms.strideFace + d * uniforms.strideFace + ${indexCall}] = val; }\n`;
                } else {
                    h += `fn read_${f.name}_${suffix}(${sig}) -> f32 { return data[${faceIdx} * uniforms.strideFace + ${indexCall}]; }\n`;
                    h += `fn write_${f.name}_${suffix}(${sig}, val: f32) { data[${faceIdx} * uniforms.strideFace + ${indexCall}] = val; }\n`;
                }
            });
        });

        // 6. GLOBAL MACROS (Binding 3 - Contextual scalars)
        if (safeGlobals.length > 0 && req.usesGlobals) {
            h += `// --- GLOBAL WORKSPACE MACROS ---\n`;
            let byteOffset = 0;
            safeGlobals.forEach(g => {
                const idx = byteOffset / 4;
                const isAtomic = g.type.startsWith('atomic');
                const isAtomicF32 = g.type === 'atomic_f32';

                if (isAtomic) {
                    h += `fn read_global_${g.name}(d: u32) -> f32 { return bitcast<f32>(atomicLoad(&globals[${idx}u + d])); }\n`;
                    if (isAtomicF32) {
                        h += `fn atomicAdd_global_${g.name}(d: u32, val: f32) { _hypercube_atomicAddF32_global(${idx}u + d, val); }\n`;
                    } else {
                        h += `fn atomicAdd_global_${g.name}(d: u32, val: f32) { atomicAdd(&globals[${idx}u + d], i32(val)); }\n`;
                    }
                } else {
                    h += `fn read_global_${g.name}(d: u32) -> f32 { return globals[${idx}u + d]; }\n`;
                    h += `fn write_global_${g.name}(d: u32, val: f32) { globals[${idx}u + d] = val; }\n`;
                }
                byteOffset += g.count * 4;
            });
        }

        return {
            code: h,
            usesAtomics,
            usesGlobals
        };
    }
}
