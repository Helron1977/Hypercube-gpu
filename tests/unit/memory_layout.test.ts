import { describe, it, expect } from 'vitest';
import { MemoryLayout } from '../../src/memory/MemoryLayout';
import { IVirtualGrid } from '../../src/topology/GridAbstractions';

describe('MemoryLayout', () => {
    it('should calculate correct offsets and alignment', () => {
        const mockGrid = {
            chunks: [
                { id: 'c0', localDimensions: { nx: 128, ny: 128, nz: 1 } },
                { id: 'c1', localDimensions: { nx: 128, ny: 128, nz: 1 } }
            ],
            dataContract: {
                getFaceMappings: () => [
                    { name: 'rho', isPingPong: true, numComponents: 1, numSlots: 2 },
                    { name: 'vx', isPingPong: false, numComponents: 1, numSlots: 1 }
                ],
                getGlobalMappings: () => [],
                descriptor: {
                    requirements: { ghostCells: 1 }
                }
            }
        };

        const layout = new MemoryLayout(mockGrid as unknown as IVirtualGrid);

        // (128 + 2) * (128 + 2) = 130 * 130 = 16900
        expect(layout.cellsPerFaceRaw).toBe(16900);

        // In float32 units: 67840 / 4 = 16960
        expect(layout.strideFace).toBe(16960);

        // Face 0: offset 0
        expect(layout.getFaceOffset(0, 0, 0)).toBe(0);
        // Face 1: offset 2 * 16960 (Face 0 is ping-pong, takes 2 slots)
        expect(layout.getFaceOffset(0, 1, 0)).toBe(33920);

        // Chunk 0, Face 0 read (slot 0)
        expect(layout.getFaceOffset(0, 0, 0)).toBe(0);
        // Chunk 0, Face 0 write (slot 1)
        expect(layout.getFaceOffset(0, 0, 1)).toBe(16960);
        // Chunk 0, Face 1 (slot 2)
        expect(layout.getFaceOffset(0, 1, 0)).toBe(2 * 16960);
        
        // Chunk 1, Face 0 (slot 3)
        expect(layout.getFaceOffset(1, 0, 0)).toBe(3 * 16960);
    });
});
