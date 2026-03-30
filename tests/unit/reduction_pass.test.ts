import { describe, it, expect, vi } from 'vitest';
import { ReductionPass } from '../../src/dispatchers/ReductionPass';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('ReductionPass', () => {
    const mockDevice = {
        createBuffer: vi.fn().mockReturnValue({ 
            size: 16, 
            destroy: vi.fn(), 
            mapAsync: vi.fn().mockResolvedValue(undefined),
            getMappedRange: vi.fn().mockReturnValue(new Int32Array([1000, 0, 0, 0]).buffer),
            unmap: vi.fn()
        }),
        createBindGroup: vi.fn().mockReturnValue({}),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        createCommandEncoder: vi.fn().mockReturnValue({
            beginComputePass: vi.fn().mockReturnValue({
                setPipeline: vi.fn(),
                setBindGroup: vi.fn(),
                dispatchWorkgroupCount: vi.fn(),
                dispatchWorkgroups: vi.fn(),
                end: vi.fn()
            }),
            copyBufferToBuffer: vi.fn(),
            finish: vi.fn()
        })
    } as any;

    const mockGrid = {
        config: { maxRules: 1 },
        chunks: [{ localDimensions: { nx: 10, ny: 10, nz: 1 } }],
        dataContract: {
            getFaceMappings: () => [{ name: 'phi', isPingPong: true }],
            descriptor: { requirements: { ghostCells: 1 } }
        }
    } as any;

    const mockBuffer = { strideFace: 192, gpuBuffer: {} } as any;
    const mockParity = { currentTick: 0, getFaceIndices: () => ({ read: 0, write: 1, base: 0 }) } as any;
    const mockPipelineCache = { 
        getPipeline: vi.fn().mockResolvedValue({ 
            pipeline: { getBindGroupLayout: () => ({}) },
            usesAtomics: true,
            usesGlobals: false
        }) 
    } as any;
    const mockStaging = { 
        ensureBuffers: vi.fn(), 
        getMetadata: () => [{ dataOffset: 0, dataSize: 400, vChunk: mockGrid.chunks[0], uniformOffset: 0 }],
        stagingBuffer: new Uint32Array(512),
        uniformBuffer: {},
        getBytesPerChunkAligned: () => 512
    } as any;

    it('should apply precision scale and return correct results', async () => {
        const pass = new ReductionPass();
        const scale = 1000.0;
        
        // Result in mock buffer is 1000. 1000 / scale = 1.0
        const results = await pass.reduce(
            mockDevice, mockGrid, mockBuffer, mockParity, 
            mockPipelineCache, mockStaging, (t) => "header", "source", "phi", 1, scale
        );

        expect(results[0]).toBe(1.0);
        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
    });
});
