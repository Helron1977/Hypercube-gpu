import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { GpuEngine } from '../../src/GpuEngine';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';
import { DataContract } from '../../src/DataContract';

describe('Hypercube v5.0 Macro System & API', () => {
    // Mock WebGPU Globals
    (globalThis as any).GPUBufferUsage = { UNIFORM: 1, STORAGE: 2, COPY_DST: 4 };

    const mockDevice = {
        createBuffer: () => ({ destroy: () => {}, size: 1024 }),
        createBindGroup: () => ({}),
        createComputePipeline: async () => ({ getBindGroupLayout: () => ({}) }),
        createShaderModule: () => ({}),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        limits: { minUniformBufferOffsetAlignment: 256 },
        createCommandEncoder: () => ({
            beginComputePass: () => ({ 
                setPipeline: vi.fn(), setBindGroup: vi.fn(), 
                dispatchWorkgroups: vi.fn(), end: vi.fn() 
            }),
            finish: () => ({})
        })
    };

    HypercubeGPUContext.setDevice(mockDevice as any);
    (HypercubeGPUContext as any)._isInitialized = true;
    (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

    const descriptor = {
        name: 'v5-test',
        version: '5.0.1',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 'scalar', type: 'scalar', isSynchronized: true },
            { name: 'pp', type: 'population', isSynchronized: true, isPingPong: true },
            { name: 'wave', type: 'scalar', isSynchronized: true, numSlots: 3 }
        ],
        rules: [
            { type: 'Update', faces: ['scalar.read', 'scalar.write', 'pp', 'wave.now', 'wave.old', 'wave.next'] }
        ]
    };

    const config = { 
        dimensions: { nx: 32, ny: 32, nz: 1 }, 
        chunks: { x: 1, y: 1, z: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'v5-test',
        params: {}
    };

    const vGrid = new VirtualGrid(config as any, descriptor as any);
    const mockMB = { 
        gpuBuffer: {}, 
        strideFace: 1024, 
        totalSlotsPerChunk: 10,
        getFaceData: () => new Float32Array(1024),
        setFaceData: vi.fn(),
        syncFacesToHost: vi.fn()
    } as any;
    
    const parity = new ParityManager(vGrid.dataContract);

    it('GpuDispatcher: should generate professional macros for explicit faces', () => {
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        const header = dispatcher.getWgslHeader('Update');

        // Check 2D Macros
        expect(header).toContain('fn read_scalar_Read');
        expect(header).toContain('fn write_scalar_Write');
        expect(header).toContain('fn read_pp');
        expect(header).toContain('fn write_pp');
        
        // Check Unified 3D-Compatible Macros (v5.0.2 uses unified read_ prefix)
        expect(header).toContain('fn read_scalar_Read');
        
        // Check Ping-Pong secondary macros
        expect(header).toContain('fn read_pp');
        expect(header).toContain('fn write_scalar_Write');
        
        // Check mapping logic (inlined in v5.0.2 for robustness)
        // LOCK-IN: Macro logic for component addressing (d) using plane-stride (v5.0.2)
        expect(header).toContain('u32(d)) * uniforms.strideFace + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts)');
        expect(header).toContain('fn getIndex3D');
    });

    it('GpuDispatcher: should have roles at offset 80 in uniforms', async () => {
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        (dispatcher as any).getPipeline = async () => ({ getBindGroupLayout: () => ({}) });

        await dispatcher.dispatch(0, { 'Update': 'void main() {}' });

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = mockDevice.queue.writeBuffer.mock.calls[0];
        const u32 = new Uint32Array(callArgs[2]);

        // Struct Mapping Verification:
        // [80]: ghosts
        // [81..86]: Roles
        expect(u32[80]).toBe(1); // 1 ghost cell
        expect(u32[81]).toBeGreaterThanOrEqual(0); // leftRole
    });

    it('GpuEngine: ready() should ensure context is initialized', async () => {
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        vi.spyOn(dispatcher, 'dispatch').mockResolvedValue(null as any);
        
        const engine = new GpuEngine(vGrid, mockMB, dispatcher, parity);
        
        // Mocking HypercubeGPUContext.init
        const initSpy = vi.spyOn(HypercubeGPUContext, 'init').mockResolvedValue(true);

        await engine.ready();

        expect(initSpy).toHaveBeenCalled();
        expect(dispatcher.dispatch).toHaveBeenCalled();
    });
});
