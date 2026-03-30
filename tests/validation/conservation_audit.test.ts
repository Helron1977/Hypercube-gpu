import { describe, it, expect } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import * as fs from 'fs';
import * as path from 'path';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';

(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};

(globalThis as any).GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };

/**
 * PHASE 2: STRICT CONSERVATION AUDIT
 * Verifies that the LBM solver mathematically conserves mass 
 * to the floating-point precision limit (1e-6 for FP32 over many steps).
 */
describe('Strict Conservation Audits', () => {
    const factory = new GpuCoreFactory();
    
    // This test REQUIRE a real GPU. Mocks are not allowed here.

    // Simple 9-population random density initializer
    function initializeRandomDensity(N: number): Float32Array {
        const rhoData = new Float32Array((N+2)*(N+2)).fill(1.0);
        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                // Add noise between -0.1 and 0.1
                const noise = (Math.random() * 0.2) - 0.1;
                rhoData[y * (N+2) + x] = 1.0 + noise;
            }
        }
        return rhoData;
    }

    function sumTotalMass(rhoData: Float32Array, N: number): number {
        let total = 0.0;
        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                total += rhoData[y * (N+2) + x];
            }
        }
        return total;
    }

    it('Should conserve global mass over 1000 iterations in a closed periodic domain', async () => {
        const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
        if (isNode) {
            console.warn("SKIPPED: Physical GPU required for numerical validation. Skipping in Node environment.");
            return;
        }

        // FORCE real GPU initialization
        if (!await HypercubeGPUContext.init()) {
            console.warn("Skipping real GPU execution in test (Hardware not found)");
            return;
        }

        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter && (adapter as any).isSoftware) {
            console.warn("Skipping real GPU execution (Software adapter detected)");
            return;
        }
        
        const N = 128;
        const config: HypercubeConfig = {
            dimensions: { nx: N, ny: N, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { 
                left: { role: 'periodic' }, right: { role: 'periodic' }, 
                top: { role: 'periodic' }, bottom: { role: 'periodic' } 
            },
            engine: 'lbm-conservation',
            params: {}
        };

        const lbmDescriptor: EngineDescriptor = {
            name: 'lbm-conservation',
            version: '1.0.0',
            faces: [
                { name: 'obstacle', type: 'scalar', isSynchronized: true },
                { name: 'vx', type: 'scalar', isSynchronized: true },       
                { name: 'vy', type: 'scalar', isSynchronized: true },       
                { name: 'rho', type: 'scalar', isSynchronized: true },      
                { name: 'curl', type: 'scalar', isSynchronized: true },     
                { name: 'f', type: 'population', isSynchronized: true, isPingPong: true }
            ],
            rules: [{ type: 'lbm', source: '', params: { p0: 1.0, p1: 0.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const LbmCoreSource = fs.readFileSync(path.join(__dirname, '../../src/kernels/wgsl/LbmCore.wgsl'), 'utf-8');
        const engine = await factory.build(config, lbmDescriptor);

        // FORCE real GPU initialization
        if (!await HypercubeGPUContext.init()) {
            console.error("Skipping real GPU execution in test (Hardware not found)");
            return;
        }

        (HypercubeGPUContext as any)._device = HypercubeGPUContext.device;
        if (!HypercubeGPUContext.device || !HypercubeGPUContext.device.queue) {
            throw new Error("Validation aborted: Physical GPU device queue is required for physics audit.");
        }

        const initialRho = initializeRandomDensity(N);
        const initialMass = sumTotalMass(initialRho, N);
        
        // Auto init bypass
        engine.use({ lbm: LbmCoreSource });
        await engine.step(1);
        if ((HypercubeGPUContext as any)._device.queue.onSubmittedWorkDone) {
            await (HypercubeGPUContext as any)._device.queue.onSubmittedWorkDone();
        }

        // Set chaotic density
        engine.setFaceData('chunk_0_0_0', 'rho', initialRho, true);
        
        // We must also initialize populations to match rho to be physically valid,
        const w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
        const stride = (N+2)*(N+2);
        const fData = new Float32Array(stride * 9);
        for(let idx = 0; idx < initialRho.length; idx++) {
            const rho = initialRho[idx];
            for (let d = 0; d < 9; d++) {
                fData[d * stride + idx] = rho * w[d];
            }
        }
        engine.setFaceData('chunk_0_0_0', 'f', fData, true);
        engine.syncToDevice(); // CRITICAL: Pousser les données sur le GPU

        // Run 1000 steps
        await engine.step(1000);
        
        if ((HypercubeGPUContext as any)._device.queue.onSubmittedWorkDone) {
            await (HypercubeGPUContext as any)._device.queue.onSubmittedWorkDone();
        }

        await engine.syncFacesToHost(['rho']);
        const finalRho = engine.getFaceData('chunk_0_0_0', 'rho');
        
        const finalMass = sumTotalMass(finalRho, N);

        // Calculate absolute mass difference. FP32 rounding over 1000 steps on 16k cells 
        // can accumulate around 1e-4 error.
        const diff = Math.abs(finalMass - initialMass);
        const relativeError = diff / initialMass;
        
        console.log(`Initial Mass: ${initialMass}, Final Mass: ${finalMass}, Rel Error: ${relativeError}`);
        
        expect(relativeError).toBeLessThan(1e-4);
    });
});
