
import { describe, it, expect } from 'vitest';
import { WgslHeaderGenerator } from '../../src/dispatchers/WgslHeaderGenerator';

describe('WgslHeaderGenerator: Resource Detection (v6.0 Zero-Stall)', () => {
    
    it('should return ALL as true in Discovery Mode (empty source) to maintain test compatibility', () => {
        const reqs = WgslHeaderGenerator.detectRequirements("");
        expect(reqs.usesAtomics).toBe(true);
        expect(reqs.usesGlobals).toBe(true);
    });

    it('should identify that ONLY Atomics (@binding(2)) are needed', () => {
        const source = "// Header Logic\n@group(0) @binding(0) var<storage> data: f32;\n@group(0) @binding(2) var<storage> atomicData: atomic<i32>;";
        const reqs = WgslHeaderGenerator.detectRequirements(source);
        expect(reqs.usesAtomics).toBe(true);
        expect(reqs.usesGlobals).toBe(false);
    });

    it('should identify that ONLY Globals (@binding(3)) are needed', () => {
        const source = "@group(0) @binding(3) var<storage> globals: array<f32>;";
        const reqs = WgslHeaderGenerator.detectRequirements(source);
        expect(reqs.usesAtomics).toBe(false);
        expect(reqs.usesGlobals).toBe(true);
    });

    it('should identify that BOTH Atomics and Globals are needed', () => {
        const source = "// Usage: atomic and global\nfn main() { }";
        const reqs = WgslHeaderGenerator.detectRequirements(source);
        expect(reqs.usesAtomics).toBe(true);
        expect(reqs.usesGlobals).toBe(true);
    });

    it('should identify that NO extra resources are needed for simple kernels', () => {
        const source = "@compute @workgroup_size(8,8) fn main() { return; }";
        const reqs = WgslHeaderGenerator.detectRequirements(source);
        expect(reqs.usesAtomics).toBe(false);
        expect(reqs.usesGlobals).toBe(false);
    });
});
