import { GpuCoreFactory } from '../../src/GpuCoreFactory';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import LbmBenchmark3DSource from '../kernels/LbmBenchmark3D.wgsl?raw';
import ReductionBenchmark3DSource from '../kernels/ReductionBenchmark3D.wgsl?raw';

const logEl = document.getElementById('log')!;
const statusEl = document.getElementById('status')!;

function log(msg: string) {
    logEl.innerText += msg + '\n';
}

async function runBenchmark() {
    try {
        if (!navigator.gpu) {
            statusEl.innerText = 'WebGPU NOT SUPPORTED';
            statusEl.style.color = '#ef4444';
            statusEl.classList.remove('pulse');
            log('Error: This browser docs not support WebGPU.');
            return;
        }

        log(`[GPU] Requesting Hardware Adapter...`);
        statusEl.innerText = 'Initializing WebGPU...';

        if (!await HypercubeGPUContext.init()) {
            throw new Error('Failed to initialize Hypercube GPU Context');
        }

        const device = HypercubeGPUContext.device;

        device.addEventListener('uncapturederror', (event: any) => {
            logEl.innerText += '\n\nWEBGPU FATAL ERROR:\n' + event.error.message;
        });

        const config = {
            dimensions: { nx: 256, ny: 128, nz: 16 },
            chunks: { x: 1, y: 1, z: 1 },
            chunkLayout: { x: 1, y: 1, z: 1 },
            boundaries: {
                left: { role: 3 }, // 3: Inflow
                right: { role: 4 }, // 4: Outflow 
                top: { role: 1 }, bottom: { role: 1 },
                front: { role: 1 }, back: { role: 1 }
            },
            engine: 'perf-test', params: {}
        };
        const descriptor = {
            name: 'perf-test', version: '1.0.0',
            requirements: { ghostCells: 1, pingPong: true },
            faces: [
                { name: 'rho', type: 'scalar', isPingPong: true },
                { name: 'ux', type: 'scalar', isPingPong: true },
                { name: 'uy', type: 'scalar', isPingPong: true },
                { name: 'uz', type: 'scalar', isPingPong: true },
                { name: "type", type: "scalar" },
                { name: 'f', type: 'population3D', isPingPong: true }
            ],
            globals: [
                { name: 'forces', count: 3, type: 'atomic_f32' }
            ],
            rules: [
                { type: 'Lbm3DCore', params: { p0: 16, p1: 1.85, p2: 0.1, p7: 1000.0 } },
                { type: 'ReductionForces', params: { p0: 16, p7: 1000.0 } }
            ] 
        };

        const factory = new GpuCoreFactory();
        const engine = await factory.build(config as any, descriptor as any);

        log(`[MEM] WebGPU Buffer natively allocated via MasterBuffer: ${Math.round((engine.buffer as any).byteLength / 1024 / 1024)} MB...`);

        // Initialize with Cylinder Obstacle and Constant Inflow
        const strideFace = (engine.buffer as any).strideFace;
        const totalNodes = strideFace;
        const rhoData = new Float32Array(totalNodes);
        const typeData = new Float32Array(totalNodes);
        const fData = new Float32Array(strideFace * 27);
        const W3D = [
            8 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27,
            1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
            1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216, 1 / 216
        ];
        const DX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1];
        const DY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1];
        const DZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1];

        const U = descriptor.rules[0].params.p2 as number;
        const nx = config.dimensions.nx;
        const ny = config.dimensions.ny;
        const nz = config.dimensions.nz;
        const pad = 1;
        const lx = nx + pad * 2;
        const ly = ny + pad * 2;

        for (let z = 0; z < nz + pad * 2; z++) {
            for (let y = 0; y < ly; y++) {
                for (let x = 0; x < lx; x++) {
                    const i = (z * ly + y) * lx + x;
                    if (i >= totalNodes) continue;
                    rhoData[i] = 1.0;
                    const dx = x - (nx / 4 + pad);
                    const dy = y - (ny / 2 + pad);
                    if (dx * dx + dy * dy < 8 * 8) {
                        typeData[i] = 1.0;
                    } else {
                        typeData[i] = 0.0;
                    }
                    const u2 = 1.5 * (U * U);
                    for (let d = 0; d < 27; d++) {
                        const cu = 3.0 * (U * DX[d]);
                        fData[d * strideFace + i] = W3D[d] * 1.0 * (1.0 + cu + 0.5 * cu * cu - u2);
                    }
                }
            }
        }
        engine.setFaceData('chunk_0_0_0', 'rho', rhoData);
        engine.setFaceData('chunk_0_0_0', 'type', typeData);
        engine.setFaceData('chunk_0_0_0', 'ux', new Float32Array(totalNodes).fill(0)); 
        engine.setFaceData('chunk_0_0_0', 'f', fData, true);
        engine.syncToDevice();

        // Warmup
        log(`[RTX] Warming up dynamic pipeline (50 cycles)...`);
        engine.use({ 'Lbm3DCore': LbmBenchmark3DSource, 'ReductionForces': ReductionBenchmark3DSource });
        for (let i = 0; i < 50; i++) await engine.step(1);
        await device.queue.onSubmittedWorkDone();

        log(`[AERO] Stress-test Cylinder Re=100 (D=16) - 5000 iterations...`);
        const t0 = performance.now();
        const iterations = 5000;

        for (let i = 0; i < iterations; i++) {
            await engine.step(1);

            // Extract Forces every 20 steps via Elite Global API
            if (i % 20 === 0) {
                const results = await engine.getGlobal('forces');
                const Fx = results[0];
                const Fy = results[1];
                
                // CD = F / (0.5 * rho * U^2 * D * L)
                const refForce = 0.5 * 1.0 * (U * U) * 16 * 16;
                const Cd = Math.abs(Fx) / refForce;
                const Cl = Fy / refForce;

                if (i % 200 === 0) {
                    log(`[AERO] Step ${i}: Cd=${Cd.toFixed(4)}, Cl=${Cl.toFixed(4)} (Fx=${Fx.toFixed(2)})`);
                }
            }
        }

        // Final verification
        await engine.syncFacesToHost(['rho', 'ux']);
        const t1 = performance.now();

        const validationRhoData = engine.getFaceData('chunk_0_0_0', 'rho');
        const validationUxData = engine.getFaceData('chunk_0_0_0', 'ux');

        let rhoSum = 0;
        for (let i = 0; i < validationRhoData.length; i++) rhoSum += validationRhoData[i];
        const rhoMean = rhoSum / validationRhoData.length;

        let maxUx = 0;
        for (let i = 0; i < validationUxData.length; i++) {
            const v = Math.abs(validationUxData[i]);
            if (v > maxUx) maxUx = v;
        }

        const rawTimeMs = (t1 - t0);
        const avgMs = rawTimeMs / iterations;
        const totalCells = config.dimensions.nx * config.dimensions.ny * config.dimensions.nz;
        const throughput = (totalCells / (avgMs / 1000)) / 1e6; // MLUPS

        statusEl.innerText = 'BENCHMARK COMPLETE';
        statusEl.classList.remove('pulse');
        statusEl.style.color = '#10b981';

        log(`\n--- RÉSULTATS D'ÉVALUATION ---`);
        log(`Grid Size: 256x128x16 (${totalNodes.toLocaleString()} total nodes)`);
        log(`Iterations: ${iterations}`);
        log(`Temps noyau total perçu: ${rawTimeMs.toFixed(2)} ms`);
        log(`Débit Absolu: ${throughput.toFixed(2)} MLUPS\n`);

        log(`[VERIFICATION PHYSIQUE LBM]`);
        log(`rho mean : ${rhoMean.toFixed(6)}`);
        log(`ux max   : ${maxUx.toFixed(6)}`);

        if (maxUx === 0.0 || isNaN(rhoMean) || Math.abs(rhoMean - 1.0) > 0.1) {
            log(`\n❌ ERROR: Calcul physiquement invalide !`);
            throw new Error("Conservation masse violée ou DCE Shader Skip !");
        } else {
            log(`\n✅ CALCULS REELS CONFIRMES: Physique stable.`);
        }

        const mark = document.createElement('div');
        mark.id = "benchmark-done";
        document.body.appendChild(mark);

        // Report to Autonomous Runner
        try {
            await fetch('http://127.0.0.1:3000', {
                method: 'POST',
                headers: { 'X-Test-Name': 'perf-test' },
                body: logEl.innerText
            });
        } catch (err) {
            console.warn("Autonomous Runner not detected on port 3000.");
        }

    } catch (e: any) {
        log('Error: ' + e.message);
        statusEl.innerText = 'FAILED';
        try {
            await fetch('http://127.0.0.1:3000', { method: 'POST', headers: { 'X-Test-Name': 'perf-test' }, body: logEl.innerText + '\n\nBENCHMARK FAILED\n' + e.message });
        } catch (err) { }
    }
}

runBenchmark();
