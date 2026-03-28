import { describe, it, expect, vi } from 'vitest';
import { GpuBoundarySynchronizer } from '../../src/topology/GpuBoundarySynchronizer';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('GpuBoundarySynchronizer (Multi-chunk)', () => {
    const mockDevice = {
        createShaderModule: vi.fn().mockReturnValue({}),
        createComputePipeline: vi.fn().mockReturnValue({ getBindGroupLayout: vi.fn() }),
        createBuffer: vi.fn().mockReturnValue({ size: 1024, destroy: vi.fn() }),
        createBindGroup: vi.fn().mockReturnValue({}),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        createCommandEncoder: vi.fn().mockReturnValue({
            beginComputePass: vi.fn().mockReturnValue({
                setPipeline: vi.fn(),
                setBindGroup: vi.fn(),
                dispatchWorkgroups: vi.fn(),
                end: vi.fn()
            }),
            finish: vi.fn()
        })
    } as any;

    vi.spyOn(HypercubeGPUContext, 'isInitialized', 'get').mockReturnValue(true);
    vi.spyOn(HypercubeGPUContext, 'device', 'get').mockReturnValue(mockDevice);

    it('should correctly build sync batches for neighbor chunks', () => {
        const sync = new GpuBoundarySynchronizer();
        
        const mockGrid = {
            dimensions: { nx: 200, ny: 100, nz: 1 },
            chunkLayout: { x: 2, y: 1, z: 1 },
            dataContract: {
                getFaceMappings: () => [{ name: 'phi', requiresSync: true, isPingPong: false }],
                descriptor: { requirements: { ghostCells: 1 } }
            },
            chunks: [
                { id: 'c0', x: 0, y: 0, joints: [{ face: 'right', role: 'joint', neighborId: 'c1' }] },
                { id: 'c1', x: 1, y: 0, joints: [{ face: 'left', role: 'joint', neighborId: 'c0' }] }
            ]
        } as any;

        const mockBuffer = { gpuBuffer: {}, totalSlotsPerChunk: 1, strideFace: 10000 } as any;
        const mockParity = { currentTick: 0, getFaceIndices: () => ({ base: 0 }) } as any;

        // Spy on private method if needed, but here we check if it calls queue.writeBuffer
        sync.syncAll(mockGrid, mockBuffer, mockParity, 'read');

        // It should have found joints and called the GPU
        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
    });
});
