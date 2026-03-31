import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';
import ReductionCoreSource from '../../src/kernels/wgsl/ReductionCore.wgsl?raw';

const logEl = document.getElementById('log')!;
function print(msg: string) { logEl.innerText += msg + '\n'; }

function initializeRandomDensity(N: number): Float32Array {
    const rhoData = new Float32Array((N+2)*(N+2)).fill(1.0);
    for (let y = 1; y <= N; y++) {
        for (let x = 1; x <= N; x++) {
            const noise = (Math.random() * 0.2) - 0.1;
            rhoData[y * (N+2) + x] = 1.0 + noise;
        }
    }
    return rhoData;
}

async function runConservationAudit() {
    logEl.innerText = '';
    print("Starting Detailed Conservation Audit (Navier-Stokes Mass Conservation)");
    try {
        const adapter = await navigator.gpu.requestAdapter();
        const device = await adapter!.requestDevice();
        (HypercubeGPUContext as any)._device = device;
        (HypercubeGPUContext as any)._isInitialized = true;
        (HypercubeGPUContext as any).alignToUniform = (n: number) => Math.ceil(n/256)*256;

        const N = 128; // 16,384 cells
        const steps = 1000;
        
        const factory = new GpuCoreFactory();
        const config: HypercubeConfig = {
            dimensions: { nx: N, ny: N, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: { left: { role: 'periodic' }, right: { role: 'periodic' }, top: { role: 'periodic' }, bottom: { role: 'periodic' } },
            engine: 'lbm-conservation', params: {}
        };

        const lbmDescriptor: EngineDescriptor = {
            name: 'lbm-conservation', version: '1.0.0',
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

        const engine = await factory.build(config, lbmDescriptor);

        // --- 1. GPU INITIALIZATION ---
        const initialRho = initializeRandomDensity(N);
        
        // Auto init bypass & Kernel Registration
        await engine.use({ lbm: LbmCoreSource });
        
        // Enclose domain in an impermeable bounce-back border (User Preference)
        const borderData = new Float32Array((N+2)*(N+2));
        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                if (x === 1 || x === N || y === 1 || y === N) {
                    borderData[y * (N+2) + x] = 1.0; 
                }
            }
        }
        engine.setFaceData('chunk_0_0_0', 'obstacle', borderData, true);
        engine.setFaceData('chunk_0_0_0', 'rho', initialRho, true);
        
        // Match populations to initial rho (rest density initialization)
        const w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
        const strideFace = engine.buffer.layout.strideFace;
        const fData = new Float32Array(strideFace * 9);
        for(let idx = 0; idx < initialRho.length; idx++) {
            const rho = initialRho[idx];
            for (let d = 0; d < 9; d++) {
                fData[d * strideFace + idx] = rho * w[d];
            }
        }
        // Use fillAllPingPong=true to sync both buffers at T=0
        engine.setFaceData('chunk_0_0_0', 'f', fData, true);
        engine.syncToDevice();

        // CRITICAL: Step 0 Auto-Init Calibration
        await engine.step(1); 
        await device.queue.onSubmittedWorkDone();

        // --- 2. INITIAL MEASUREMENT (GPU-SIDE) ---
        const m0 = await engine.reduceField('rho', ReductionCoreSource);
        print(`Grid Size: ${N}x${N} (${N*N} computational nodes)`);
        print(`Initial Mass M(t=0) GPU = ${m0.toFixed(6)} kg`);

        // --- 3. SIMULATION ---
        print(`Simulating ${steps} discrete WGSL dispatches...`);
        await engine.step(steps);
        await device.queue.onSubmittedWorkDone();

        // --- 4. FINAL MEASUREMENT (GPU-SIDE) ---
        const m1 = await engine.reduceField('rho', ReductionCoreSource);
        print(`Final Mass M(t=${steps}) GPU = ${m1.toFixed(6)} kg`);
        
        const drift = Math.abs(m1 - m0);
        const relativeError = drift / (m0 + 1e-12);
        print(`Drift Delta: ${drift.toExponential(4)}`);
        print(`Relative Deviation: ${(relativeError * 100).toExponential(4)} %`);

        if (relativeError < 1e-7) {
            print("\nRESULT: SUCCESS! Strict mass conservation proven (Zero-Drift Certified).");
        } else if (relativeError < 1e-3) {
            print("\nRESULT: PASS! Mass conservation within FP32 limits.");
        } else {
            print("\nRESULT: FAILED. Mass leakage detected.");
        }

    } catch (e: any) {
        print(`\nError: ${e.message}`);
    } finally {
        // Pipe result back to the Node CLI Helper
        try {
            await fetch('http://localhost:3000', {
                method: 'POST',
                headers: { 'X-Test-Name': 'strict_conservation' },
                body: logEl.innerText
            });
        } catch (e) {
            // Ignore if helper is not listening
        }
        
        const mark = document.createElement('div');
        mark.id = "validation-done";
        document.body.appendChild(mark);
    }
}

runConservationAudit();
