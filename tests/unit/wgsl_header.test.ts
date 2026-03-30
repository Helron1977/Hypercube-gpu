import { describe, it, expect } from 'vitest';
import { WgslHeaderGenerator } from '../../src/dispatchers/WgslHeaderGenerator';

describe('WgslHeaderGenerator: Universal Macro System (v6)', () => {

    it('should generate a complete V6 header with bindings, struct, and macros', () => {
        // PILOT: The contract defines 4 slots for 'rho'
        const faceMappings = [
            { name: 'rho', type: 'scalar', numSlots: 4, numComponents: 1, pointerOffset: 0 },
        ] as any;

        // V6 Call: mode, faceMapping array
        const { code: header } = WgslHeaderGenerator.getHeader('Update', faceMappings);

        // 1. Technical Infrastructure (Bindings)
        expect(header).toContain('@group(0) @binding(0) var<storage, read_write> data: array<f32>;');
        expect(header).toContain('@group(0) @binding(1) var<uniform> uniforms: Uniforms;');

        // 2. Structural Source of Truth (Struct Uniforms)
        expect(header).toContain('struct Uniforms {');
        expect(header).toContain('faces: array<u32, 64>,');

        // 3. Utility Helpers
        expect(header).toContain('fn getIndex(x: u32, y: u32) -> u32');
        expect(header).toContain('fn getIndex3D(x: u32, y: u32, z: u32) -> u32');

        // 4. Temporal Macros (V6 Nomenclature Now/Next/Old/Base)
        expect(header).toContain('fn read_rho_Now(');
        expect(header).toContain('fn write_rho_Next(');
        expect(header).toContain('fn read_rho_Old(');
        expect(header).toContain('fn read_rho_Base(');
    });

    it('should only generate physical slots defined by numSlots', () => {
        // PILOT: Strictly 2 slots for 'vel'
        const faceMappings = [
            { name: 'vel', type: 'vector', numSlots: 2, numComponents: 3, pointerOffset: 0 },
        ] as any;

        const { code: header } = WgslHeaderGenerator.getHeader('Audit', faceMappings);

        expect(header).toContain('fn read_vel_Now(');
        expect(header).toContain('fn write_vel_Next(');
        
        // Should NOT contain Old or Base if numSlots < 3
        expect(header).not.toContain('fn read_vel_Old(');
        expect(header).not.toContain('fn read_vel_Base(');
    });

});
