
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

// Global WebGPU Mocks
(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x01, MAP_WRITE: 0x02, COPY_SRC: 0x04, COPY_DST: 0x08,
    INDEX: 0x10, VERTEX: 0x20, UNIFORM: 0x40, STORAGE: 0x80,
};
(globalThis as any).GPUMapMode = { READ: 0x01, WRITE: 0x02 };

describe('Hypercube GPU Core v0.2.0 Upgrades (TDD)', () => {
    let factory: GpuCoreFactory;
    let mockDevice: any;

    beforeEach(async () => {
        mockDevice = {
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
            createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
            createBuffer: (desc: { size: number }) => ({ 
                destroy: () => {}, size: desc.size, 
                getMappedRange: () => new ArrayBuffer(desc.size), 
                unmap: () => {}, mapAsync: () => Promise.resolve() 
            }),
            createBindGroup: () => ({}),
            queue: { writeBuffer: () => {}, submit: () => {} },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({ setPipeline: () => {}, setBindGroup: () => {}, setWorkgroupCount: () => {}, dispatchWorkgroups: () => {}, end: () => {} }),
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

    const getTestSetup = () => {
        const descriptor: EngineDescriptor = {
            name: 'v02-test',
            version: '1.0.0',
            faces: [
                { name: 'rho', type: 'scalar', isSynchronized: true, isPingPong: true, numSlots: 3 }
            ],
            rules: [{ type: 'step', source: '// void', params: { modulo: 3 }, faces: ['rho.read', 'rho.write'] }],
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: 4, ny: 4, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {},
            engine: 'v02-test',
            params: { omega: 1.8 }
        };
        return { config, descriptor };
    };

    it('should expose params directly on vGrid (Proxy Test)', async () => {
        const { config, descriptor } = getTestSetup();
        const engine = await factory.build(config, descriptor, mockDevice);
        
        // @ts-ignore - params might not be on interface yet
        expect(engine.vGrid.params).toBeDefined();
        // @ts-ignore
        expect(engine.vGrid.params.omega).toBe(1.8);
        
        // Test reactivity/write-through
        // @ts-ignore
        engine.vGrid.params.omega = 1.9;
        expect(engine.vGrid.config.params.omega).toBe(1.9);
    });

    it('should support Generics for EngineParams and Faces', async () => {
        interface MyParams { omega: number; viscosity: number; }
        interface MyFaces { rho: string; vel: string; }
        
        const { config, descriptor } = getTestSetup();
        // This is a compile-time check mostly
        // @ts-ignore
        const engine = await factory.build<MyParams, MyFaces>(config, descriptor, mockDevice);
        expect(engine).toBeDefined();
    });

    it('should generate WGSL headers with read/write macros', async () => {
        const { config, descriptor } = getTestSetup();
        const engine = await factory.build(config, descriptor, mockDevice);
        
        const header = engine.getWgslHeader('step');
        expect(header).toContain('fn read_rho_Read');
        expect(header).toContain('fn write_rho_Write');
    });

    it('should filter ghost cells with getInnerFaceData()', async () => {
        const { config, descriptor } = getTestSetup();
        const engine = await factory.build(config, descriptor, mockDevice);
        
        // Physical size: (4 + 2*1) * (4 + 2*1) = 6 * 6 = 36
        // Inner size: 4 * 4 = 16
        const fullData = new Float32Array(36).fill(7); 
        engine.setFaceData('chunk_0_0_0', 'rho', fullData);
        
        // @ts-ignore
        const innerData = engine.getInnerFaceData('chunk_0_0_0', 'rho');
        expect(innerData.length).toBe(16);
    });

    it('should support modulo 3 for Triple-Buffering parity', async () => {
        const { config, descriptor } = getTestSetup();
        const engine = await factory.build(config, descriptor, mockDevice);
        
        // Tick 0 -> Read 0, Write 1
        const i0 = engine.parityManager.getFaceIndices('rho', 3);
        expect(i0.read).toBe(0);
        expect(i0.write).toBe(1);

        engine.parityManager.increment(); // Tick 1
        const i1 = engine.parityManager.getFaceIndices('rho', 3);
        expect(i1.read).toBe(1);
        expect(i1.write).toBe(2);

        engine.parityManager.increment(); // Tick 2
        const i2 = engine.parityManager.getFaceIndices('rho', 3);
        expect(i2.read).toBe(2);
        expect(i2.write).toBe(0);
    });

    it('should resolve ready() when engine is fully initialized', async () => {
        const { config, descriptor } = getTestSetup();
        const engine = await factory.build(config, descriptor, mockDevice);
        // ready() now returns void (Promise<void>)
        await expect(engine.ready()).resolves.toBeUndefined();
    });
});
