import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { AnalyticalSolutions } from '../validation/AnalyticalSolutions';
import { Metrics } from '../validation/Metrics';

// We import LbmCore directly using Vite's ?raw
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';

const logEl = document.getElementById('log')!;
function print(msg: string) { logEl.innerText += msg + '\n'; }

async function runTGVCase(device: any, N: number, u0: number, nu: number, steps: number): Promise<number> {
    const factory = new GpuCoreFactory();
    const omega = 1.0 / (3.0 * nu + 0.5);

    const lbmDescriptor: EngineDescriptor = {
        name: `lbm-tgv-${N}`,
        version: '1.0.0',
        faces: [
            { name: 'obstacle', type: 'scalar', isSynchronized: true },
            { name: 'vx', type: 'scalar', isSynchronized: true },       
            { name: 'vy', type: 'scalar', isSynchronized: true },       
            { name: 'rho', type: 'scalar', isSynchronized: true },      
            { name: 'curl', type: 'scalar', isSynchronized: true },     
            { name: 'f', type: 'population', isSynchronized: true, isPingPong: true }
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

    const engine = await factory.build(config, lbmDescriptor);
    const strideFace = engine.buffer.strideFace;

    // Initialize Velocity Field
    const vxData = new Float32Array((N+2)*(N+2));
    const vyData = new Float32Array((N+2)*(N+2));
    const rhoData = new Float32Array((N+2)*(N+2));

    for (let y = 1; y <= N; y++) {
        for (let x = 1; x <= N; x++) {
            const idx = y * (N+2) + x;
            const exact = AnalyticalSolutions.getTaylorGreenVortex2D(x-1, y-1, 0, u0, N, N, nu);
            vxData[idx] = exact.u;
            vyData[idx] = exact.v;
            rhoData[idx] = exact.rho;
        }
    }
    
    // CONSUME TICK 0 (LbmCore auto init)
    engine.use({ lbm: LbmCoreSource });
    await engine.step(1); 
    await device.queue.onSubmittedWorkDone();

    // INJECT EXACT DATA FOR TGV 
    engine.setFaceData('chunk_0_0_0', 'vx', vxData);
    engine.setFaceData('chunk_0_0_0', 'vy', vyData);
    engine.setFaceData('chunk_0_0_0', 'rho', rhoData);
    
    const fData = new Float32Array(strideFace * 9);
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
                fData[d * strideFace + idx] = feq;
            }
        }
    }
    engine.setFaceData('chunk_0_0_0', 'f', fData, true);
    
    // Copy JS CPU TypedArrays -> WebGPU Hardware Buffer!
    engine.syncToDevice();

    // RUN 
    await engine.step(steps);
    await device.queue.onSubmittedWorkDone();

    // READ
    await engine.syncFacesToHost(['vx']);
    const vxRes = engine.getFaceData('chunk_0_0_0', 'vx');

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

async function runAll() {
    logEl.innerText = '';
    print("Starting Grid Refinement Study (Taylor-Green Vortex)");
    try {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();
        (HypercubeGPUContext as any)._device = device;
        (HypercubeGPUContext as any)._isInitialized = true;
        (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

        const baseN = 16;
        const baseU0 = 0.05;
        const baseNu = 0.05;
        const baseSteps = 50;

        print("Running Grid 1 (N=16)...");
        const error16 = await runTGVCase(device, baseN, baseU0, baseNu, baseSteps);
        
        print("Running Grid 2 (N=32)...");
        const error32 = await runTGVCase(device, baseN * 2, baseU0 / 2, baseNu / 2, baseSteps * 2);
        
        print("Running Grid 3 (N=64)...");
        const error64 = await runTGVCase(device, baseN * 4, baseU0 / 4, baseNu / 4, baseSteps * 4);

        const order32 = Metrics.computeConvergenceOrder(error16, error32, baseN, baseN * 2);
        const order64 = Metrics.computeConvergenceOrder(error32, error64, baseN * 2, baseN * 4);

        print(`\n--- LBM SPATIAL CONVERGENCE (TGV) ---`);
        print(`N=16: L2 Error = ${error16.toExponential(4)}`);
        print(`N=32: L2 Error = ${error32.toExponential(4)} | Rate = ${order32.toFixed(3)}`);
        print(`N=64: L2 Error = ${error64.toExponential(4)} | Rate = ${order64.toFixed(3)}`);

        if (order64 > 1.8) {
            print("\nRESULT: SUCCESS! Engine converges at expected 2nd order.");
        } else {
            print("\nRESULT: FAILED. Engine sub-optimal convergence.");
        }

    } catch (e: any) {
        print(`\nError: ${e.message}`);
    } finally {
        try {
            await fetch('http://localhost:3000', {
                method: 'POST',
                headers: { 'X-Test-Name': 'grid_refinement' },
                body: logEl.innerText
            });
        } catch (e) {
            // Ignore if headless runner is not listening
        }
        
        const mark = document.createElement('div');
        mark.id = "validation-done";
        document.body.appendChild(mark);
    }
}

runAll();
