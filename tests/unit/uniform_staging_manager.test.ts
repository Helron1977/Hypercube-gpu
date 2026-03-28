import { describe, it, expect, vi } from 'vitest';
import { UniformStagingManager } from '../../src/dispatchers/UniformStagingManager';

describe('UniformStagingManager', () => {
    const mockGrid = {
        config: { maxRules: 4, params: { p0: 10, p1: 20 }, boundaries: {} },
        chunks: [{ id: 'c0', x: 0, y: 0, localDimensions: { nx: 10, ny: 10, nz: 1 }, joints: [] }],
        chunkLayout: { x: 1, y: 1 },
        dataContract: {
            getFaceMappings: () => [{ name: 'phi', isPingPong: true }],
            descriptor: { requirements: { ghostCells: 1 } }
        }
    } as any;

    const mockBuffer = { 
        strideFace: 192, 
        totalSlotsPerChunk: 2,
        createBuffer: vi.fn().mockReturnValue({ 
            size: 16, 
            destroy: vi.fn(), 
            mapAsync: vi.fn().mockResolvedValue(undefined),
            getMappedRange: vi.fn().mockReturnValue(new Int32Array([1000, 0, 0, 0]).buffer),
            unmap: vi.fn()
        }),
        createBindGroup: vi.fn().mockReturnValue({}),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() }
    } as any;

    const mockParity = {
        currentTick: 5,
        getFaceIndices: vi.fn().mockReturnValue({ read: 0, write: 1, base: 0 })
    } as any;

    it('should initialize staging buffer with correct size', () => {
        const manager = new UniformStagingManager(mockGrid, mockBuffer, mockParity);
        // bytesPerChunkAligned is 512. maxRules is 4. 1 chunk.
        // Size should be 4 * 512 = 2048 bytes = 512 u32.
        expect(manager.stagingBuffer?.length).toBe(512);
    });

    it('should fill uniform data correctly in updateStaging', () => {
        const manager = new UniformStagingManager(mockGrid, mockBuffer, mockParity);
        const rules = [{ type: 'rule1', params: { p0: 99 } }]; // override p0
        const faceMappings = [{ name: 'phi' }];

        manager.updateStaging(1.23, rules, faceMappings);

        const u32 = manager.stagingBuffer!;
        const f32 = new Float32Array(u32.buffer);

        // base = 0 (chunk 0, rule 0)
        expect(u32[0]).toBe(10); // nx
        expect(u32[1]).toBe(10); // ny
        expect(f32[4]).toBeCloseTo(1.23, 5); // time t
        expect(u32[5]).toBe(5); // tick
        
        // Params check
        expect(f32[8]).toBe(99); // p0 (overridden)
        expect(f32[9]).toBe(20); // p1 (from config)
        
        // Face indices check
        expect(u32[16]).toBe(0); // phi index (base or read)
    });
});
