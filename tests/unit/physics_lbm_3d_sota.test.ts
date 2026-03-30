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
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
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
        version: '6.0.0-alpha',
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

    it('DataContract: should verify canonical LBM 3D memory (SOTA Sobriety)', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const contract = vGrid.dataContract;
        
        // 10x10x10 with 1 ghost = 12x12x12 = 1728 cells
        // f: 27 components * 2 slots = 54
        // rho: 1 component * 2 slots = 2
        // vel: 3 components * 2 slots = 6
        // Total slots = 54 + 2 + 6 = 62
        // Bytes = 1728 * 4 * 62 = 428,544 bytes
        const bytes = contract.calculateChunkBytes(10, 10, 10, 1);
        expect(bytes).toBe(428544);
    });

    it('GpuDispatcher: should generate V6 Unified Macros for D3Q27', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1728, layout: { totalStandardSlotsPerChunk: 62 } } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const { code: header } = dispatcher.getWgslHeader('LBM');

        // Check 3D Multi-Component Macros (V6 Syntax)
        expect(header).toContain('fn read_f_Now(x: u32, y: u32, z: u32, d: u32)');
        
        // Verification of the pointer resolution (Face 0 Slot 1 for 'Next')
        // In V6: data[uniforms.faces[1u] * uniforms.strideFace + d * uniforms.strideFace + getIndex3D(x, y, z)]
        expect(header).toContain('uniforms.faces[1u] * uniforms.strideFace');
        expect(header).toContain('d * uniforms.strideFace');
        
        // Convenience check for V6 nomenclature
        expect(header).toContain('fn write_f_Next(');
        expect(header).toContain('fn read_rho_Now(');
    });

    it('GpuDispatcher: should pack Professional Roles into Uniforms (V6 Layout)', async () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { gpuBuffer: {}, strideFace: 1728, totalSlotsPerChunk: 31, layout: { totalStandardSlotsPerChunk: 62 } } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);
        (dispatcher as any).getPipeline = async () => ({ getBindGroupLayout: () => ({}) });

        await dispatcher.dispatch(0, { 'LBM': 'void main() {}' });

        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const callArgs = mockDevice.queue.writeBuffer.mock.calls[0];
        const u32 = new Uint32Array(callArgs[2]);

        // Struct Mapping Verification (UniformLayout.ts standards):
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
