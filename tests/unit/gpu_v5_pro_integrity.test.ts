import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { ParityManager } from '../../src/ParityManager';

describe('Hypercube v5.0 Pro: WGSL Integrity Contract', () => {
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

    const d3q27_descriptor = {
        name: 'lbm-3d-integrity',
        version: '5.0.1',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [{ name: 'f', type: 'population', numComponents: 27, isPingPong: true, isSynchronized: true }],
        rules: [{ type: 'LBM', source: 'src', faces: ['f.now', 'f.next'] }]
    };

    const config = { 
        dimensions: { nx: 32, ny: 32, nz: 1 }, 
        chunks: { x: 1, y: 1, z: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'lbm-3d-integrity',
        params: {}
    };

    it('CONTRACT: must declare data binding at (0,0)', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1024 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const header = dispatcher.getWgslHeader('LBM');

        // LOCK-IN: Standard Hypercube Storage Binding
        expect(header).toContain('@group(0) @binding(0) var<storage, read_write> data: array<f32>;');
    });

    it('CONTRACT: must declare uniforms binding at (0,1)', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1024 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const header = dispatcher.getWgslHeader('LBM');

        // LOCK-IN: Standard Hypercube Uniform Binding
        expect(header).toContain('@group(0) @binding(1) var<uniform> uniforms: Uniforms;');
    });

    it('CONTRACT: Uniforms struct must preserve specific member layout', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1024 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const header = dispatcher.getWgslHeader('LBM');

        // LOCK-IN: Exact Struct Layout (Critical for CPU/GPU mapping)
        const expectedStruct = `struct Uniforms {
  nx: u32, ny: u32, strideRow: u32, physicalNy: u32,
  t: f32, tick: u32, strideFace: u32, numFaces: u32,
  p0: f32, p1: f32, p2: f32, p3: f32,
  p4: f32, p5: f32, p6: f32, p7: f32,
  faces: array<u32, 64>,
  ghosts: u32,
  leftRole: u32, rightRole: u32, topRole: u32, bottomRole: u32, frontRole: u32, backRole: u32,
  reserved: u32
};`;
        
        const normalize = (s: string) => s.replace(/\s+/g, '');
        expect(normalize(header)).toContain(normalize(expectedStruct));
    });

    it('CONTRACT: must generate multi-component macros linked to uniforms.faces', () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1024 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        const header = dispatcher.getWgslHeader('LBM');

        // LOCK-IN: Macro logic for component addressing (d)
        // LOCK-IN: Macro logic for component addressing (d) using plane-stride (v5.0.2)
        expect(header).toContain('uniforms.faces[0] + u32(d)) * uniforms.strideFace + (y + uniforms.ghosts) * uniforms.strideRow + (x + uniforms.ghosts)');
    });

    it('CONTRACT: must prepend WGSL header to kernel source before compilation', async () => {
        const vGrid = new VirtualGrid(config as any, d3q27_descriptor as any);
        const parity = new ParityManager(vGrid.dataContract);
        const mockMB = { strideFace: 1024 } as any;
        const dispatcher = new GpuDispatcher(vGrid, mockMB, parity, mockDevice as any);

        // Spy on the static context method
        const spy = vi.spyOn(HypercubeGPUContext, 'createComputePipelineAsync').mockResolvedValue({} as any);

        const kernelSource = 'fn main() { let test = uniforms.nx; }';
        await (dispatcher as any).getPipeline('LBM', kernelSource);

        expect(spy).toHaveBeenCalled();
        const callArgs = spy.mock.calls[0];
        const combinedSource = callArgs[0];

        // Verify header injection
        expect(combinedSource).toContain('struct Uniforms');
        expect(combinedSource).toContain('@group(0) @binding(1) var<uniform> uniforms: Uniforms;');
        expect(combinedSource).toContain(kernelSource);

        spy.mockRestore();
    });
});
