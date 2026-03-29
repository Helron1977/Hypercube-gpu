import { describe, it, expect, vi, beforeEach } from 'vitest';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

describe('HypercubeGPUContext: Remote Kernel Loader (TDD v5.0.4)', () => {
    
    beforeEach(() => {
        vi.stubGlobal('fetch', vi.fn());
    });

    it('should fetch and return kernel source from a URL', async () => {
        const mockSource = 'fn main() { /* Remote Kernel */ }';
        (fetch as any).mockResolvedValue({
            ok: true,
            text: async () => mockSource
        });

        const source = await HypercubeGPUContext.loadKernel('https://example.com/kernel.wgsl');

        expect(fetch).toHaveBeenCalledWith('https://example.com/kernel.wgsl');
        expect(source).toBe(mockSource);
    });

    it('should throw an error if the fetch fails', async () => {
        (fetch as any).mockResolvedValue({
            ok: false,
            statusText: 'Not Found'
        });

        await expect(HypercubeGPUContext.loadKernel('https://example.com/bad.wgsl'))
            .rejects.toThrow('Failed to load kernel from URL: Not Found');
    });
});
