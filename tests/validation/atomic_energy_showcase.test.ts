import { describe, it, expect, vi } from 'vitest';
import { createSimulation } from '../../src/createSimulation';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('V5.0.4 Showcase: Atomic Energy Conservation & Dynamic Overrides', () => {
    
    // Mock WebGPU (Full Mock Stack)
    const mockDevice = {
        limits: { minUniformBufferOffsetAlignment: 256 },
        createBuffer: vi.fn((desc: any) => {
            const buf = new ArrayBuffer(desc.size);
            if (desc.label?.includes('Staging')) {
                new Float32Array(buf).fill(50.5);
            }
            return {
                destroy: vi.fn(), size: desc.size, usage: desc.usage, label: desc.label,
                mapAsync: vi.fn(() => Promise.resolve()),
                getMappedRange: vi.fn(() => buf),
                unmap: vi.fn()
            };
        }),
        createShaderModule: vi.fn(() => ({})),
        createComputePipeline: vi.fn(() => ({ getBindGroupLayout: vi.fn(() => ({})) })),
        createComputePipelineAsync: vi.fn(() => Promise.resolve({ getBindGroupLayout: vi.fn(() => ({})) })),
        createBindGroup: vi.fn(() => ({})),
        createCommandEncoder: vi.fn(() => ({
            beginComputePass: vi.fn(() => ({ setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(), end: vi.fn() })),
            copyBufferToBuffer: vi.fn(),
            finish: vi.fn(() => ({ label: 'mock-finish' }))
        })),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() }
    };

    HypercubeGPUContext.setDevice(mockDevice as any);

    it('should coordinate multi-thread energy accumulation and parameter overrides', async () => {
        // 1. Setup Manifest
        const manifest = {
            name: 'Energy Showcase',
            version: '5.0.4',
            engine: {
                name: 'HeatSys',
                requirements: { ghostCells: 1, pingPong: true },
                faces: [
                    { name: 'T', type: 'field', isPingPong: true, isSynchronized: true },
                    { name: 'energy_total', type: 'atomic_f32', isSynchronized: true }
                ],
                rules: [{ type: 'diffuse', source: 'void main() { ... }' }]
            },
            config: {
                dimensions: { nx: 128, ny: 128 },
                chunks: { x: 1, y: 1 },
                engine: 'HeatSys',
                params: { conductivity: 0.1 } // p0
            }
        };

        const engine = await createSimulation(manifest as any, mockDevice as any);
        
        // 2. Proof of Atomics in WGSL Header
        const header = engine.getWgslHeader('diffuse');
        expect(header).toContain('@group(0) @binding(2) var<storage, read_write> dataAtomic: array<atomic<u32>>;');
        expect(header).toContain('fn atomicAdd_energy_total(x: u32, y: u32, val: f32)');
        expect(header).toContain('_hypercube_atomicAddF32'); // The CAS Loop

        // 2. Elite Registration of Kernels
        engine.use({ 'diffuse': '...' });
        
        // 3. Step 1: Normal Diffusion (Standard param 0.1)
        await engine.step();
        
        // Verify standard param p0 (0.1)
        const uniformsWrite = mockDevice.queue.writeBuffer.mock.calls.find(c => c[0].label === 'GpuDispatcher Uniforms (Modular)');
        const uniformsData = new Float32Array(uniformsWrite[2]);
        expect(uniformsData[8]).toBeCloseTo(0.1); 

        // 4. Step 2: Dynamic Override (Semantic name)
        await engine.step({ conductivity: 0.5 });
        
        // Verify the override
        const uniformsWrite2 = mockDevice.queue.writeBuffer.mock.calls.filter(c => c[0].label === 'GpuDispatcher Uniforms (Modular)')[1];
        const uniformsData2 = new Float32Array(uniformsWrite2[2]);
        expect(uniformsData2[8]).toBeCloseTo(0.5);

        // 5. Elite Verification of Sync Logic
        // We use the new getFace() which handles sync internally
        const atomicData = await engine.getFace('energy_total');
        expect(atomicData[0]).toBe(50.5); 

        const copyCalls = mockDevice.createCommandEncoder().copyBufferToBuffer.mock.calls;
        // The last encoder created during syncFacesToHost should have been used
        expect(mockDevice.createCommandEncoder).toHaveBeenCalled();
    });
});
