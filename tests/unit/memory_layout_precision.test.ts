import { describe, it, expect } from 'vitest';
import { MemoryLayout } from '../../src/memory/MemoryLayout';
import { IVirtualGrid } from '../../src/topology/GridAbstractions';

describe('MemoryLayout: Atomic Split Precision', () => {
    
    // Scénario complexe : Standard (1) -> Atomic (1) -> Standard (3 comp, 2 slots) -> Atomic (1)
    const mockGrid = {
        chunks: [
            { id: 'c0', localDimensions: { nx: 32, ny: 32, nz: 1 } },
            { id: 'c1', localDimensions: { nx: 32, ny: 32, nz: 1 } }
        ],
        dataContract: {
            getFaceMappings: () => [
                { name: 'S1', type: 'scalar', numComponents: 1, numSlots: 1 }, // idx 0
                { name: 'A1', type: 'atomic_f32', numComponents: 1, numSlots: 1 }, // idx 1
                { name: 'S2', type: 'vector', numComponents: 3, numSlots: 2 }, // idx 2
                { name: 'A2', type: 'atomic_u32', numComponents: 1, numSlots: 1 }  // idx 3
            ],
            descriptor: {
                requirements: { ghostCells: 1 }
            }
        }
    };

    const layout = new MemoryLayout(mockGrid as unknown as IVirtualGrid);
    // strideRow = 32 + 2 = 34
    // cellsPerFaceRaw = 34 * 34 = 1156
    // bytesPerFaceRaw = 1156 * 4 = 4624
    // aligned to 256: ceil(4624/256)*256 = 4864
    // strideFace = 4864 / 4 = 1216
    const STRIDE = 1216;

    it('should correctly identify bindings for all faces', () => {
        expect(layout.getBinding(0)).toBe(0); // S1
        expect(layout.getBinding(1)).toBe(2); // A1
        expect(layout.getBinding(2)).toBe(0); // S2
        expect(layout.getBinding(3)).toBe(2); // A2
    });

    it('should calculate correct total slots per chunk per binding', () => {
        // Binding 0: S1 (1*1) + S2 (3*2) = 1 + 6 = 7 slots
        expect(layout.totalStandardSlotsPerChunk).toBe(7);
        // Binding 2: A1 (1*1) + A2 (1*1) = 2 slots
        expect(layout.totalAtomicSlotsPerChunk).toBe(2);
    });

    it('should calculate precise intra-buffer offsets for Chunk 0', () => {
        // --- Binding 0 ---
        expect(layout.getFaceOffset(0, 0)).toBe(0); // S1
        expect(layout.getFaceOffset(0, 2, 0)).toBe(1 * STRIDE); // S2 slot 0
        expect(layout.getFaceOffset(0, 2, 1)).toBe(4 * STRIDE); // S2 slot 1 (skip S2.0, S2.1, S2.2)

        // --- Binding 2 ---
        expect(layout.getFaceOffset(0, 1)).toBe(0); // A1 is the FIRST in Binding 2
        expect(layout.getFaceOffset(0, 3)).toBe(1 * STRIDE); // A2 is the SECOND in Binding 2
    });

    it('should handle multi-chunk continuity correctly', () => {
        // Chunk 1, Standard Start = 7 slots * STRIDE
        expect(layout.getFaceOffset(1, 0)).toBe(7 * STRIDE);
        
        // Chunk 1, Atomic Start = 2 slots * STRIDE
        expect(layout.getFaceOffset(1, 1)).toBe(2 * STRIDE);
    });

    it('should map Logical CPU offsets (LogicalFaceOffset) correctly', () => {
        const stdSizeFloats = layout.standardByteLength / 4;
        
        // Expected: 2 chunks * 7 slots * STRIDE = 14 * 1216 = 17024
        expect(stdSizeFloats).toBe(14 * STRIDE);

        // A1 (Chunk 0) logically follows all Standard data
        expect(layout.getLogicalFaceOffset(0, 1)).toBe(stdSizeFloats);
        
        // A2 (Chunk 0, offset 1*STRIDE in Binding 2)
        expect(layout.getLogicalFaceOffset(0, 3)).toBe(stdSizeFloats + STRIDE);

        // A1 (Chunk 1, offset 2*STRIDE in Binding 2)
        expect(layout.getLogicalFaceOffset(1, 1)).toBe(stdSizeFloats + 2 * STRIDE);
    });

    it('should ensure byte lengths are multiples of 256 for WebGPU safety', () => {
        expect(layout.standardByteLength % 256).toBe(0);
        expect(layout.atomicByteLength % 256).toBe(0);
    });
});
