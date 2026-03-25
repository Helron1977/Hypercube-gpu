import { describe, it, expect, vi } from 'vitest';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import * as fs from 'fs';

describe('SOTA Performance Benchmarking', () => {
    // Instead of mocking WebGPU and producing fake MLUPS, we attempt to use the real GPU.
    // If not available (e.g., in CI without GPU), we skip the test and log clearly.
    
    it('should benchmark 64-chunk LBM dispatch on real GPU', async () => {
        if (HypercubeGPUContext.isMock) {
            console.warn('SKIPPED: Real WebGPU is not available. Skipping performance benchmark.');
            expect(true).toBe(true);
            return;
        }

        const adapter = await (navigator as any).gpu.requestAdapter();
        if (!adapter) {
            console.warn('SKIPPED: Real WebGPU adapter not found.');
            expect(true).toBe(true);
            return;
        }
        
        const device = await adapter.requestDevice();
        (HypercubeGPUContext as any)._device = device;
        (HypercubeGPUContext as any)._isInitialized = true;
        (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

        const config = { 
            dimensions: { nx: 128, ny: 128, nz: 64 }, // Big 1M cells simulation
            chunks: { x: 4, y: 4, z: 4 }, // 64 Chunks
            chunkLayout: { x: 32, y: 32, z: 16 }
        };
        const descriptor = {
            name: 'perf-test',
            requirements: { ghostCells: 1 },
            faces: [{ name: 'a', isSynchronized: true }],
            rules: [{ type: 'Lbm3DCore', faces: { f0: 'a' }, params: {} }]
        };

        const vGrid = new VirtualGrid(config as any, descriptor as any);
        // Create actual buffers to test real memory bandwidth
        const strideFace = 128 * 128 * 64;
        const gpuBuffer = device.createBuffer({
            size: strideFace * 4 * 19, // float32 * 19 pops
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });

        const mockMB = { 
            gpuBuffer: gpuBuffer, 
            strideFace: strideFace, 
            totalSlotsPerChunk: 2,
            getDimensions: () => config.dimensions
        } as any;
        const mockParity = { currentTick: 0, getFaceIndices: () => ({ read: 0, write: 1 }) } as any;

        const dispatcher = new GpuDispatcher(vGrid as any, mockMB, mockParity);
        
        const start = performance.now();
        const iterations = 10; // Keep reasonable for test suites
        
        for(let i = 0; i < iterations; i++) {
            await dispatcher.dispatch(0, { 'Lbm3DCore': 'source' });
        }
        
        // Wait for queue to finish
        await device.queue.onSubmittedWorkDone();
        
        const end = performance.now();
        const avgMs = (end - start) / iterations;
        const totalCells = config.dimensions.nx * config.dimensions.ny * config.dimensions.nz;
        const throughput = (totalCells / (avgMs / 1000)) / 1e6; // MLUPS

        const log = `--- Performance Results (REAL GPU) ---\n` +
                    `Adapter: ${adapter.name || 'Unknown'}\n` +
                    `Grid Size: ${config.dimensions.nx}x${config.dimensions.ny}x${config.dimensions.nz} (${totalCells} cells)\n` +
                    `Chunks: 64\n` +
                    `Avg Dispatch Latency: ${avgMs.toFixed(3)} ms\n` +
                    `Estimated Throughput: ${throughput.toFixed(2)} MLUPS\n`;
        
        console.log(log);
        fs.appendFileSync('perf_log.txt', log + '\n');
        
        gpuBuffer.destroy();
        device.destroy();
        
        expect(avgMs).toBeDefined(); 
    });
});
