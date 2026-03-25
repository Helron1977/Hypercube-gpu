import { describe, it, expect } from 'vitest';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import * as fs from 'fs';
import * as path from 'path';

import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { AnalyticalSolutions } from './AnalyticalSolutions';
import { Metrics } from './Metrics';

(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};

(globalThis as any).GPUMapMode = { READ: 0x0001, WRITE: 0x0002 };

describe('Grid Refinement Study (Numerical Verification)', () => {
    const factory = new GpuCoreFactory();
    
    // We only perform this test on an actual GPU as numerical correctness relies on precise shader execution
    // Mock environments cannot validate math.
    const hasGPU = !!navigator && !!(navigator as any).gpu;

    // A helper to initialize and run a TGV case
    async function runTGVCase(N: number, u0: number, nu: number, steps: number): Promise<number> {
        const adapter = await (navigator as any).gpu.requestAdapter();
        const device = await adapter.requestDevice();
        (HypercubeGPUContext as any)._device = device;
        (HypercubeGPUContext as any)._isInitialized = true;
        (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

        const omega = 1.0 / (3.0 * nu + 0.5); // tau = 3*nu + 0.5
        
        const lbmDescriptor: EngineDescriptor = {
            name: `lbm-tgv-${N}`,
            version: '1.0.0',
            faces: [
                { name: 'obstacle', type: 'scalar', isSynchronized: true },
                { name: 'vx', type: 'scalar', isSynchronized: true },       
                { name: 'vy', type: 'scalar', isSynchronized: true },       
                { name: 'rho', type: 'scalar', isSynchronized: true },      
                { name: 'curl', type: 'scalar', isSynchronized: true },     
                { name: 'f0', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f1', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f2', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f3', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f4', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f5', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f6', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f7', type: 'population', isSynchronized: true, isPingPong: true },
                { name: 'f8', type: 'population', isSynchronized: true, isPingPong: true }
            ],
            rules: [{ type: 'lbm', source: '', params: { p0: omega, p1: 0.0 } }], 
            requirements: { ghostCells: 1, pingPong: true }
        };

        const config: HypercubeConfig = {
            dimensions: { nx: N, ny: N, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { 
                left: { role: 'periodic' }, right: { role: 'periodic' }, 
                top: { role: 'periodic' }, bottom: { role: 'periodic' } 
            },
            engine: lbmDescriptor.name,
            params: {}
        };

        const kernelsDir = path.join(__dirname, '../../src/kernels/wgsl');
        const lbmKernelSource = fs.readFileSync(path.join(kernelsDir, 'LbmCore.wgsl'), 'utf-8');

        const engine = await factory.build(config, lbmDescriptor);
        const kernels = { 'lbm': lbmKernelSource };

        // Initialize Velocity Field
        const vxData = new Float32Array((N+2)*(N+2));
        const vyData = new Float32Array((N+2)*(N+2));
        const rhoData = new Float32Array((N+2)*(N+2));

        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                const idx = y * (N+2) + x;
                // LBM internal coords vs macroscopic: center is (x-1, y-1) 
                const exact = AnalyticalSolutions.getTaylorGreenVortex2D(x-1, y-1, 0, u0, N, N, nu);
                vxData[idx] = exact.u;
                vyData[idx] = exact.v;
                rhoData[idx] = exact.rho;
            }
        }
        
        // Push initial fields directly into engine (we bypass LbmCore's automatic init by setting tick = 1 manually, or letting it init and overriding)
        // Wait, LbmCore.wgsl initializes automatically at tick=0 using p1. We can set tick=1 to skip auto-init, 
        // but GpuEngine manages tick. We will just overwrite data at tick 0 before the step.
        // Actually, GpuEngine.step() does the pass. Let's do a trick: we set data, and we don't care if tick 0 overrides it if we rely on it?
        // Wait, LbmCore.wgsl auto-inits at tick==0: if (params.tick == 0u) { ... return; }
        // So step(1) will just run the auto-init and overwrite our data!
        // To fix this without breaking other tests, we should disable auto-init if p1 == -999, or just run 1 step to consume tick 0, then inject our data!
        
        // CONSUME TICK 0
        await engine.step(kernels); 
        if (device.queue.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();

        // INJECT EXACT DATA FOR TGV (at t=0 logically, even though engine is at tick=1)
        engine.setFaceData('chunk_0_0_0', 'vx', vxData);
        engine.setFaceData('chunk_0_0_0', 'vy', vyData);
        engine.setFaceData('chunk_0_0_0', 'rho', rhoData);
        // We also need to init populations 'f' to exact equilibrium! 
        const fData = new Float32Array((N+2)*(N+2) * 9); 
        const DX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
        const DY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
        const W  = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];

        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                const idx = y * (N+2) + x;
                const exact = AnalyticalSolutions.getTaylorGreenVortex2D(x-1, y-1, 0, u0, N, N, nu);
                const u2 = 1.5 * (exact.u*exact.u + exact.v*exact.v);
                
                for(let d=0; d<9; d++) {
                    const cu = 3.0 * (DX[d]*exact.u + DY[d]*exact.v);
                    const feq = W[d] * exact.rho * (1.0 + cu + 0.5*cu*cu - u2);
                    fData[d * (N+2)*(N+2) + idx] = feq;
                }
            }
        }
        engine.setFaceData('chunk_0_0_0', 'f0', fData, true); // true = raw buffer sync
        engine.syncToDevice(); // CRITICAL: Pousser les données sur le GPU

        // RUN STEPS
        for (let i = 0; i < steps; i++) {
            await engine.step(kernels);
        }
        if (device.queue.onSubmittedWorkDone) await device.queue.onSubmittedWorkDone();

        // READ BACK
        await engine.syncFacesToHost(['vx', 'vy']);
        const vxRes = engine.getFaceData('chunk_0_0_0', 'vx');
        const vyRes = engine.getFaceData('chunk_0_0_0', 'vy');

        // COMPUTE ERROR
        const exactVx = new Float32Array(N*N);
        const compVx = new Float32Array(N*N);
        
        let ptr = 0;
        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                const idx = y * (N+2) + x;
                const exact = AnalyticalSolutions.getTaylorGreenVortex2D(x-1, y-1, steps, u0, N, N, nu);
                exactVx[ptr] = exact.u;
                compVx[ptr] = vxRes[idx];
                ptr++;
            }
        }

        return Metrics.computeL2Error(compVx, exactVx, N*N);
    }

    it('should prove O(dx^2) spatial convergence on Taylor-Green Vortex', async () => {
        const isNode = typeof process !== 'undefined' && process.versions && process.versions.node;
        if (isNode) {
            console.warn("SKIPPED: Physical GPU required for numerical validation. Skipping in Node environment.");
            return;
        }

        if (!hasGPU || !await HypercubeGPUContext.init()) {
            console.warn('SKIPPED: Real GPU required for numerical validation. Verification not performed.');
            return;
        }

        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter && (adapter as any).isSoftware) {
            console.warn("Skipping real GPU execution (Software adapter detected)");
            return;
        }

        // Acoustic scaling grid refinement: N -> 2N implies u0 -> u0/2, nu -> nu/2, steps -> 2*steps
        const baseN = 16;
        const baseU0 = 0.05;
        const baseNu = 0.05;
        const baseSteps = 50;

        const error16 = await runTGVCase(baseN, baseU0, baseNu, baseSteps);
        const error32 = await runTGVCase(baseN * 2, baseU0 / 2, baseNu / 2, baseSteps * 2);
        const error64 = await runTGVCase(baseN * 4, baseU0 / 4, baseNu / 4, baseSteps * 4);

        const order32 = Metrics.computeConvergenceOrder(error16, error32, baseN, baseN * 2);
        const order64 = Metrics.computeConvergenceOrder(error32, error64, baseN * 2, baseN * 4);

        const log = `--- LBM SPATIAL CONVERGENCE (TGV) ---\n` +
                    `N=16:  L2 Error = ${error16.toExponential(4)}\n` +
                    `N=32:  L2 Error = ${error32.toExponential(4)} | Order = ${order32.toFixed(2)}\n` +
                    `N=64:  L2 Error = ${error64.toExponential(4)} | Order = ${order64.toFixed(2)}\n`;
        
        console.log(log);
        fs.writeFileSync('convergence_log.txt', log);

        // Theoretical LBM is 2nd order. We expect order to be ~2.0.
        // We allow some tolerance (e.g. > 1.8) because boundary conditions or initialization might be slightly lower order (O(dx)).
        expect(order64).toBeGreaterThan(1.8);
    }, 20000); // 20s timeout for GPU execution
});
