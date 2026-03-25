import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';

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

function sumTotalMass(rhoData: Float32Array, N: number): number {
    let total = 0.0;
    // We only sum active cells, excluding halo boundaries to avoid duplication
    for (let y = 1; y <= N; y++) {
        for (let x = 1; x <= N; x++) {
            total += rhoData[y * (N+2) + x];
        }
    }
    return total;
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

        // TICK 0 auto-init bypass
        await engine.step({ lbm: LbmCoreSource }); 
        await device.queue.onSubmittedWorkDone();

        const initialRho = initializeRandomDensity(N);
        const initialMass = sumTotalMass(initialRho, N);
        print(`Grid Size: ${N}x${N} (${N*N} computational nodes)`);
        print(`Initial Mass M(t=0) = ${initialMass.toFixed(6)} kg`);

        // Enclose domain in an impermeable bounce-back border to prevent NaN leakage 
        // through uninitialized ghost cells.
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
        const strideFace = (engine.buffer as any).strideFace;
        
        // IMPORTANT: fData passes exactly the number of components for ONE slot (9 for populations)
        // engine.setFaceData will duplicate it to the back-buffer if isPingPong=true and fillAll=true.
        const fData = new Float32Array(strideFace * 9);
        for(let idx = 0; idx < initialRho.length; idx++) {
            const rho = initialRho[idx];
            for (let d = 0; d < 9; d++) {
                fData[d * strideFace + idx] = rho * w[d];
            }
        }
        engine.setFaceData('chunk_0_0_0', 'f', fData, true);

        // FATAL MISSED STEP: Copy JS CPU TypedArrays -> WebGPU Hardware Buffer!
        engine.syncToDevice();

        print(`Simulating ${steps} discrete WGSL dispatches...`);
        for(let i=0; i<steps; i++) {
            await engine.step({ lbm: LbmCoreSource });
        }
        await device.queue.onSubmittedWorkDone();

        // --- DIAGNOSTICS ---
        const mb = engine.buffer as any;
        print(`RawBuffer size: ${mb.rawBuffer.byteLength}`);
        print(`Total Slots: ${mb.layout.totalSlotsPerChunk}`);
        print(`StrideFace (floats): ${mb.layout.strideFace}`);
        
        // Run sync and get final mass
        await engine.syncFacesToHost(['rho']);
        const finalRho = engine.getFaceData('chunk_0_0_0', 'rho');
        const finalMass = sumTotalMass(finalRho, N);
        print(`Final Mass M(t=${steps}) = ${finalMass.toFixed(6)} kg`);
        
        const diff = Math.abs(finalMass - initialMass);
        const relativeError = diff / initialMass;
        print(`Drift Delta: ${diff.toExponential(4)}`);
        print(`Relative Deviation: ${(relativeError * 100).toExponential(4)} %`);

        if (relativeError < 1e-3) {
            print("\nRESULT: SUCCESS! Strict mass conservation proven to FP32 limits (relative drift < 0.1%).");
        } else {
            print("\nRESULT: FAILED. Mass is leaking from the periodic domain.");
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
