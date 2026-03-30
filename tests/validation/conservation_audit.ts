
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { ParityManager } from '../../src/ParityManager';
import { GpuEngine } from '../../src/GpuEngine';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

// @ts-ignore
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';
// @ts-ignore
import LbmReductionSource from '../../src/kernels/wgsl/LbmReduction.wgsl?raw';

async function runConservationAudit() {
    console.log(`\n🚀 Starting STRICT Absolute GPU Conservation Audit...`);

    if (!await HypercubeGPUContext.init()) {
        console.error("Failed to initialize Hypercube GPU Context.");
        return;
    }

    const N = 256;
    const config: HypercubeConfig = {
        dimensions: { nx: N, ny: N, nz: 1 },
        chunks: { x: 1, y: 1 },
        engine: 'lbm-scientific-audit',
        boundaries: {
            left: { role: 'periodic' }, right: { role: 'periodic' },
            top: { role: 'periodic' }, bottom: { role: 'periodic' }
        },
        params: { p0: 1.0, p1: 0.1, p2: 0.05, p3: 0.0, p4: 0.0 }
    };

    const descriptor: EngineDescriptor = {
        name: 'lbm-scientific-audit',
        version: '1.0.0',
        faces: [
            { name: 'obs', type: 'mask', isPingPong: false, isSynchronized: false },
            { name: 'ux', type: 'scalar', isPingPong: true, isSynchronized: true },
            { name: 'uy', type: 'scalar', isPingPong: true, isSynchronized: true },
            { name: 'rho', type: 'scalar', isPingPong: true, isSynchronized: true },
            { name: 'curl', type: 'scalar', isPingPong: true, isSynchronized: true },
            { name: 'f', type: 'population', isPingPong: true, isSynchronized: true, numComponents: 9 },
            { name: 'reduction_sum', type: 'scalar', isPingPong: false, isSynchronized: false }
        ],
        requirements: { ghostCells: 1, pingPong: true },
        rules: [
            // FIX: Explicitly list faces required for each rule
            { type: 'LbmCore', source: LbmCoreSource, faces: ['obs', 'ux', 'uy', 'rho', 'curl', 'f'] },
            { type: 'LbmReduction', source: LbmReductionSource, faces: ['rho', 'reduction_sum'] }
        ]
    };

    const grid = new VirtualGrid(config, descriptor);
    const buffer = new MasterBuffer(grid);
    const parity = new ParityManager(grid.dataContract);
    const dispatcher = new GpuDispatcher(grid, buffer, parity);
    const engine = new GpuEngine(grid, buffer, dispatcher, parity);

    const getAbsoluteGlobalInvariants = async () => {
        // High-Precision Atomic Reduction via Dispatcher (30,000 scaling)
        // results[0]=Mass, results[1]=MomX, results[2]=MomY
        const results = await engine.dispatcher.reduce('Reduction', LbmReductionSource, 'rho', 3, 30000);
        return { mass: results[0], momX: results[1], momY: results[2] };
    };

    // 1. Initial Fill
    engine.use({ 'LbmCore': LbmCoreSource });
    await engine.step(1); 

    // 2. Capture Baseline
    const initial = await getAbsoluteGlobalInvariants();
    console.log(`[INITIAL GPU] Mass: ${initial.mass.toFixed(10)}, MomX: ${initial.momX.toFixed(10)}, MomY: ${initial.momY.toFixed(10)}`);

    // 3. Run Simulation
    const iterations = 10000;
    const stepSize = 2000;
    console.log(`[RUN] Simulating ${iterations} steps...`);
    
    for (let s = 0; s < iterations; s += stepSize) {
        await engine.step(stepSize);
        console.log(`Step ${s + stepSize}...`);
    }

    // 4. Capture Result
    const final = await getAbsoluteGlobalInvariants();
    console.log(`[FINAL] Mass: ${final.mass.toFixed(10)}, MomX: ${final.momX.toFixed(10)}, MomY: ${final.momY.toFixed(10)}`);

    const driftM = Math.abs(final.mass - initial.mass) / (Math.abs(initial.mass) || 1.0);
    const driftX = Math.abs(final.momX - initial.momX) / (Math.abs(initial.momX) || 1.0);
    const driftY = Math.abs(final.momY - initial.momY) / (Math.abs(initial.momY) || 1.0);

    console.log(`\n--- Absolute Audit Results ---`);
    console.log(`Relative Mass Drift:     ${driftM.toExponential(4)}`);
    console.log(`Relative Momentum X Drift: ${driftX.toExponential(4)}`);
    console.log(`Relative Momentum Y Drift: ${driftY.toExponential(4)}`);

    const statusM = (driftM < 1e-12) ? 'SUCCESS' : 'FAILED: LEAK';
    const statusX = (driftX < 1e-10) ? 'SUCCESS' : 'FAILED: DRIFT';
    const statusY = (driftY < 1e-10) ? 'SUCCESS' : 'FAILED: DRIFT';

    console.log(`\nAudit Summary: Mass[${statusM}] MomX[${statusX}] MomY[${statusY}]\n`);

    const isStable = statusM === 'SUCCESS' && statusX === 'SUCCESS' && statusY === 'SUCCESS';
    const status = isStable ? 'stable' : 'DRIFT_DETECTED';

    return { initial, final, drifts: { driftM, driftX, driftY }, status };
}

export { runConservationAudit };
