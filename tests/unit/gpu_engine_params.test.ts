import { describe, it, expect, vi, beforeEach } from 'vitest';
import { GpuEngine } from '../../src/GpuEngine';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('GpuEngine: Elite Params API (Fluent & Proxy)', () => {
    const mockDevice = {
        createBuffer: () => ({ destroy: () => {}, size: 2048 }),
        createBindGroup: () => ({}),
        createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
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

    const descriptor = {
        name: 'v5-elite',
        version: '1.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [{ name: 'f', type: 'scalar', isSynchronized: true }],
        rules: [{ type: 'Step', faces: ['f'] }]
    };

    const config = {
        dimensions: { nx: 16, ny: 16, nz: 1 },
        chunks: { x: 1, y: 1, z: 1 },
        boundaries: { all: { role: 'wall' } },
        engine: 'v5-elite',
        params: { omega: 1.0, viscosity: 0.1 }
    };

    beforeEach(async () => {
        vi.clearAllMocks();
        HypercubeGPUContext.setDevice(mockDevice as any);
    });

    it('should allow setting parameters via Proxy: engine.params.x = y', async () => {
        const factory = new GpuCoreFactory();
        const engine = await factory.build(config as any, descriptor as any, mockDevice);
        
        // Proxy usage
        engine.params.omega = 1.85;
        
        await engine.use({ 'Step': 'void main() {}' });
        await engine.step(1);
        
        // Check if correct value was written to buffer
        expect(mockDevice.queue.writeBuffer).toHaveBeenCalled();
        const calls = mockDevice.queue.writeBuffer.mock.calls;
        // The first call might be initialization, look for the dispatch call
        const uniformCall = calls.find(c => c[1] === 0); 
        const f32 = new Float32Array(uniformCall[2]);
        
        // Offset for p0 (omega) in v5.0.4 is index 8 (32 bytes)
        expect(f32[8]).toBeCloseTo(1.85, 5);
    });

    it('should allow setting parameters via Fluent API: engine.params.set(x, y)', async () => {
        const factory = new GpuCoreFactory();
        const engine = await factory.build(config as any, descriptor as any, mockDevice);
        
        // Fluent usage
        engine.params
            .set('omega' as any, 1.2)
            .set('viscosity' as any, 0.5);
            
        await engine.use({ 'Step': 'void main() {}' });
        await engine.step(1);
        
        const calls = mockDevice.queue.writeBuffer.mock.calls;
        const uniformCall = calls.find(c => c[1] === 0);
        const f32 = new Float32Array(uniformCall[2]);
        
        expect(f32[8]).toBeCloseTo(1.2, 5);   // p0 (omega)
        expect(f32[9]).toBeCloseTo(0.5, 5);   // p1 (viscosity)
    });

    it('should return the params object from set() for chaining', async () => {
        const factory = new GpuCoreFactory();
        const engine = await factory.build(config as any, descriptor as any, mockDevice);
        
        const p = engine.params.set('omega' as any, 1.1);
        expect(p).toBe(engine.params);
    });
});
