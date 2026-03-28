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
        version: '5.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 'scalar', type: 'scalar', isSynchronized: true },
            { name: 'pp', type: 'population', isSynchronized: true, isPingPong: true }
        ],
        rules: [
            { type: 'Update', faces: ['scalar.read', 'scalar.write', 'pp'] }
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
        expect(header).toContain('fn read_scalar_Read(x: u32, y: u32) -> f32');
        expect(header).toContain('fn write_scalar_Write(x: u32, y: u32, val: f32)');
        expect(header).toContain('fn read_pp(x: u32, y: u32) -> f32');
        expect(header).toContain('fn write_pp(x: u32, y: u32, val: f32)');
        
        // Check 3D Macros
        expect(header).toContain('fn read3D_scalar_Read(x: u32, y: u32, z: u32) -> f32');
        
        // Check Ping-Pong secondary macros
        expect(header).toContain('fn read_pp_Read(x: u32, y: u32) -> f32');
        expect(header).toContain('fn write_pp_Write(x: u32, y: u32, val: f32)');
        expect(header).toContain('fn read3D_pp_Read(x: u32, y: u32, z: u32) -> f32');
        expect(header).toContain('fn write3D_pp_Write(x: u32, y: u32, z: u32, val: f32)');

        // Check getIndex with ghosts
        expect(header).toContain('return (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts);');
        expect(header).toContain('fn getIndex3D');
    });

    it('GpuDispatcher: should have roles at offset 80 in uniforms', async () => {
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        (dispatcher as any).getPipeline = async () => ({ getBindGroupLayout: () => ({}) });

        await dispatcher.dispatch(0, { 'Update': 'void main() {}' });

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = mockDevice.queue.writeBuffer.mock.calls[0];
        const u32 = new Uint32Array(callArgs[2]);

        // Role offset 80
        expect(u32[80]).toBeGreaterThanOrEqual(0); // leftRole
        // Ghosts offset 86
        expect(u32[86]).toBe(1); // 1 ghost cell
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
