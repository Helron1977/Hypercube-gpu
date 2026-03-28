import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';
import { EngineDescriptor } from '../../src/types';

describe('Physics: LBM 3D SOTA (D3Q27)', () => {
    // Mock WebGPU Globals
    (globalThis as any).GPUBufferUsage = { UNIFORM: 1, STORAGE: 2, COPY_DST: 4 };

    const mockDevice = {
        createBuffer: vi.fn(() => ({ destroy: () => {}, size: 4096 })),
        createBindGroup: vi.fn(() => ({})),
        createComputePipeline: async () => ({ getBindGroupLayout: () => ({}) }),
        createShaderModule: vi.fn(),
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

    const d3q27_descriptor: EngineDescriptor = {
        name: 'lbm-3d-sota',
        version: '5.0.1',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 'f', type: 'population', numComponents: 27, isPingPong: true, isSynchronized: true },
            { name: 'rho', type: 'scalar', isSynchronized: true },
            { name: 'vel', type: 'vector', numComponents: 3, isSynchronized: true }
        ],
        rules: [
            { type: 'LBM', source: 'LBM_Source', faces: ['f.now', 'f.next'] }
        ]
    };

    const config = { 
        dimensions: { nx: 10, ny: 10, nz: 10 }, 
        chunks: { x: 1, y: 1, z: 1 },
        boundaries: { 
            left: { role: 'inflow' },
            right: { role: 'outflow' },
            top: { role: 'moving_wall' }
        },
        engine: 'lbm-3d-sota',
        params: { p1: 0.1 } // relaxation
    };

    beforeEach(() => {
        vi.clearAllMocks();
        HypercubeGPUContext.setDevice(mockDevice as any);
        (HypercubeGPUContext as any)._isInitialized = true;
        (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;
    });

    it('DataContract: should calculate massive memory for 27 components', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const contract = vGrid.dataContract;
        
        // 10x10x10 with 1 ghost = 12x12x12 = 1728 cells
        // f: 27 components * 2 slots = 54
        // rho: 1 component * 2 slots (ping-pong inherited) = 2
        // vel: 3 components * 2 slots (ping-pong inherited) = 6
        // Total slots = 54 + 2 + 6 = 62
        // Bytes = 1728 * 4 * 62 = 428,544 bytes
        const bytes = contract.calculateChunkBytes(10, 10, 10, 1);
        expect(bytes).toBe(12 * 12 * 12 * 4 * 62);
    });

    it('GpuDispatcher: should generate correct 3D macros for D3Q27', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1728 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const header = dispatcher.getWgslHeader('LBM');

        // Check 3D Multi-Component Macros
        expect(header).toContain('fn read3D_f_Now(x: u32, y: u32, z: u32, d: u32)');
        expect(header).toContain('uniforms.faces[0] + u32(d)'); // f.now is face 0
        expect(header).toContain('getIndex3D(x, y, z)');

        // Check Convenience
        expect(header).toContain('fn read3D_f_Now(x: u32, y: u32, z: u32, d: u32)');
        expect(header).toContain('fn write3D_f_Next(x: u32, y: u32, z: u32, d: u32, val: f32)');
    });

    it('GpuDispatcher: should pack Professional Roles into Uniforms', async () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { gpuBuffer: {}, strideFace: 1728, totalSlotsPerChunk: 58 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        (dispatcher as any).getPipeline = async () => ({ getBindGroupLayout: () => ({}) });

        await dispatcher.dispatch(0, { 'LBM': 'void main() {}' });

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = mockDevice.queue.writeBuffer.mock.calls[0];
        const u32 = new Uint32Array(callArgs[2]);

        // Struct Mapping Verification:
        // [80]: ghosts
        // [81]: leftRole (inflow -> 3)
        // [82]: rightRole (outflow -> 4)
        // [83]: topRole (moving_wall -> 10)
        expect(u32[80]).toBe(1);
        expect(u32[81]).toBe(3);
        expect(u32[82]).toBe(4);
        expect(u32[83]).toBe(10);
    });
});
