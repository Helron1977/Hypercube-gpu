import { describe, it, expect, vi } from 'vitest';
import { PipelineCache } from '../../src/dispatchers/PipelineCache';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('PipelineCache', () => {
    const mockDevice = {
        createShaderModule: vi.fn().mockReturnValue({}),
        createComputePipeline: vi.fn().mockReturnValue({ getBindGroupLayout: vi.fn() }),
        createBuffer: vi.fn().mockReturnValue({ size: 1024, destroy: vi.fn() }),
        createBindGroup: vi.fn().mockReturnValue({}),
        queue: { writeBuffer: vi.fn(), submit: vi.fn() },
        createCommandEncoder: vi.fn().mockReturnValue({}),
        createComputePipelineAsync: vi.fn().mockResolvedValue({ label: 'MockPipeline' })
    } as any;

    const getHeader = (type: string, source: string) => ({
        code: `header_${type}\n`,
        usesAtomics: false,
        usesGlobals: false,
        toString() { return this.code; }
    });

    it('should generate different hashes for different sources of same length', async () => {
        const spy = vi.spyOn(HypercubeGPUContext, 'createComputePipelineAsync').mockResolvedValue({ label: 'MockPipeline' } as any);
        const cache = new PipelineCache();
        const sourceA = "fn main() { let x = 1; }"; // same length
        const sourceB = "fn main() { let y = 2; }"; // same length
        
        await cache.getPipeline(mockDevice, 'type1', sourceA, getHeader);
        await cache.getPipeline(mockDevice, 'type1', sourceB, getHeader);

        expect(spy).toHaveBeenCalledTimes(2);
        spy.mockRestore();
    });

    it('should hit cache for identical sources', async () => {
        const spy = vi.spyOn(HypercubeGPUContext, 'createComputePipelineAsync').mockResolvedValue({ label: 'MockPipeline' } as any);
        const cache = new PipelineCache();
        const source = "fn main() { }";
        
        await cache.getPipeline(mockDevice, 'type1', source, getHeader);
        await cache.getPipeline(mockDevice, 'type1', source, getHeader);

        expect(spy).toHaveBeenCalledTimes(1);
        spy.mockRestore();
    });

    it('should clear cache when device changes', async () => {
        const spy = vi.spyOn(HypercubeGPUContext, 'createComputePipelineAsync').mockResolvedValue({ label: 'MockPipeline' } as any);
        const cache = new PipelineCache();
        const source = "fn main() { }";
        
        const device1 = { id: 1, ...mockDevice } as any;
        const device2 = { id: 2, ...mockDevice } as any;

        await cache.getPipeline(device1, 'type1', source, getHeader);
        await cache.getPipeline(device2, 'type1', source, getHeader);

        expect(spy).toHaveBeenCalledTimes(2);
        spy.mockRestore();
    });
});
