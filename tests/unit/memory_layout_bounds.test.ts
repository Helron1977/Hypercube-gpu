import { describe, it, expect } from 'vitest';
import { MemoryLayout } from '../../src/memory/MemoryLayout';
import { IVirtualGrid } from '../../src/topology/GridAbstractions';

describe('MemoryLayout Bounds', () => {
    const mockGrid = {
        dataContract: {
            getFaceMappings: () => [
                { name: 'phi', numComponents: 1, numSlots: 2, isPingPong: true },
                { name: 'f',   numComponents: 9, numSlots: 1, isPingPong: false }
            ],
            descriptor: {
                requirements: { ghostCells: 1 }
            }
        },
        chunks: [
            { localDimensions: { nx: 10, ny: 10, nz: 1 } },
            { localDimensions: { nx: 10, ny: 10, nz: 1 } }
        ]
    } as any;

    it('should calculate correct offsets for multiple slots (PingPong)', () => {
        const layout = new MemoryLayout(mockGrid);
        
        // Face 0: phi (2 slots)
        // Face 1: f (1 slot)
        
        const offsetSlot0 = layout.getFaceOffset(0, 0, 0);
        const offsetSlot1 = layout.getFaceOffset(0, 0, 1);
        const offsetFace1 = layout.getFaceOffset(0, 1, 0);

        // strideFace depends on 10 + 2*1 = 12 cells in 2D
        // bytesPerFaceRaw = 12 * 12 * 4 = 576 bytes
        // bytesPerFaceAligned = 768 bytes (256 * 3)
        // strideFace = 768 / 4 = 192 floats
        
        expect(offsetSlot1).toBeGreaterThan(offsetSlot0);
        expect(offsetFace1).toBeGreaterThan(offsetSlot1);
        
        // Slot 1 of face 0 should be exactly 1 strideFace after Slot 0
        expect(offsetSlot1 - offsetSlot0).toBe(layout.strideFace);
        
        // Face 1 should start exactly 2 strideFaces after Slot 0 (since face 0 has 2 slots)
        expect(offsetFace1 - offsetSlot0).toBe(2 * layout.strideFace);
    });

    it('should handle multi-chunk indexing correctly', () => {
        const layout = new MemoryLayout(mockGrid);
        const totalSlotsPerChunk = 2 + 9;

        const chunk0Face0 = layout.getFaceOffset(0, 0, 0);
        const chunk1Face0 = layout.getFaceOffset(1, 0, 0);

        // Chunk 1 should start after all slots of Chunk 0
        expect(chunk1Face0 - chunk0Face0).toBe(11 * layout.strideFace);
    });
});
