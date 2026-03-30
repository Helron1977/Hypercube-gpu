import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

// Global WebGPU Mocks
(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x01, MAP_WRITE: 0x02, COPY_SRC: 0x04, COPY_DST: 0x08,
    INDEX: 0x10, VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80,
};

describe('Structural Alignment: Beacon Temporal Rotation', () => {
    let factory: GpuCoreFactory;
    let mockDevice: any;

    beforeEach(async () => {
        // Persistent Memory Mock
        const bufferStore = new Map<any, ArrayBuffer>();

        mockDevice = {
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
            createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
            createBuffer: (desc: { size: number }) => {
                const buf = { 
                    destroy: () => {}, size: desc.size, 
                    unmap: () => {}, mapAsync: () => Promise.resolve() 
                };
                bufferStore.set(buf, new ArrayBuffer(desc.size));
                (buf as any).getMappedRange = () => bufferStore.get(buf);
                return buf;
            },
            createBindGroup: () => ({}),
            queue: { writeBuffer: () => {}, submit: () => {} },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, dispatchWorkgroups: () => {}, end: () => {} }),
                copyBufferToBuffer: () => {}, finish: () => ({})
            })
        } as unknown as GPUDevice;

        vi.stubGlobal('navigator', {
            gpu: {
                requestAdapter: async () => ({
                    requestDevice: async () => mockDevice
                })
            }
        });

        await HypercubeGPUContext.init();
        factory = new GpuCoreFactory();
    });

    it('should verify that logical Now follows the physical rotation over 4 cycles', async () => {
        const descriptor = {
            name: 'beacon-alignment',
            faces: [
                { name: 'beacon', type: 'scalar', isSynchronized: true, numSlots: 3 }
            ],
            rules: [{ type: 'step', source: '// void', faces: ['beacon'] }],
            requirements: { ghostCells: 0 }
        };

        const config = {
            dimensions: { nx: 4, ny: 4, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'beacon-alignment',
            params: {}
        };

        const engine = await factory.build(config as any, descriptor as any, mockDevice);
        
        // 1. Initialize physical slots with beacons in the HOST buffer
        const data0 = new Float32Array(16).fill(10.0);
        const data1 = new Float32Array(16).fill(20.0);
        const data2 = new Float32Array(16).fill(30.0);

        // Explicit parity bypasses rotation for setup (Writes directly to physical slots 0, 1, 2)
        engine.setFaceData('chunk_0_0_0', 'beacon', data0, false, 0);
        engine.setFaceData('chunk_0_0_0', 'beacon', data1, false, 1);
        engine.setFaceData('chunk_0_0_0', 'beacon', data2, false, 2);

        // 2. Audit Ticks using direct HOST memory access (getFaceData)
        // Note: engine.getFace('beacon') would trigger a destructive mock sync, overwriting host data with 0s.
        
        // TICK 0: Now should be Slot 0 (10.0)
        let now = engine.getFaceData('chunk_0_0_0', 'beacon');
        expect(now[0]).toBe(10.0);

        // TICK 1: Now should be Slot 1 (20.0)
        engine.parityManager.increment();
        now = engine.getFaceData('chunk_0_0_0', 'beacon');
        expect(now[0]).toBe(20.0);

        // TICK 2: Now should be Slot 2 (30.0)
        engine.parityManager.increment();
        now = engine.getFaceData('chunk_0_0_0', 'beacon');
        expect(now[0]).toBe(30.0);

        // TICK 3: Now should be Slot 0 (10.0) - Cycle repeat
        engine.parityManager.increment();
        now = engine.getFaceData('chunk_0_0_0', 'beacon');
        expect(now[0]).toBe(10.0);
    });
});
