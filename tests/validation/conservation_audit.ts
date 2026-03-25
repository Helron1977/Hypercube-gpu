import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { ParityManager } from '../../src/ParityManager';
import { GpuEngine } from '../../src/GpuEngine';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

// @ts-ignore
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';

async function runConservationAudit() {
    console.log(`\n🚀 Starting Strict Mass & Momentum Conservation Audit...`);

    if (!await HypercubeGPUContext.init()) {
        console.error("Failed to initialize Hypercube GPU Context.");
        return;
    }

    const N = 256;
    const omega = 1.0; // Stability is easy at omega=1
    
    const config: HypercubeConfig = {
        dimensions: { nx: N, ny: N, nz: 1 },
        chunks: { x: 1, y: 1, z: 1 },
        engine: 'lbm-2d-conservation',
        boundaries: {
            left:   { role: 'periodic' },
            right:  { role: 'periodic' },
            top:    { role: 'periodic' },
            bottom: { role: 'periodic' }
        },
        params: {
            p0: omega,
            p1: 0.1,   // Initial ux (uniform)
            p2: 0.05   // Initial uy (uniform)
        }
    };

    const descriptor: EngineDescriptor = {
        name: 'lbm-2d-conserve',
        version: '0.1.0',
        faces: [
            { name: 'type', type: 'mask',       isSynchronized: false, isPingPong: false },
            { name: 'ux',   type: 'scalar',     isSynchronized: true,  isPingPong: false },
            { name: 'uy',   type: 'scalar',     isSynchronized: true,  isPingPong: false },
            { name: 'rho',  type: 'scalar',     isSynchronized: true,  isPingPong: false },
            { name: 'curl', type: 'scalar',     isSynchronized: true,  isPingPong: false },
            { name: 'f',    type: 'population', isSynchronized: true,  isPingPong: true }
        ],
        requirements: { ghostCells: 1, pingPong: true },
        rules: [
            { type: 'LbmCore', source: LbmCoreSource, params: config.params }
        ]
    };

    const vGrid = new VirtualGrid(config, descriptor);
    const buffer = new MasterBuffer(vGrid);
    const parityManager = new ParityManager(vGrid.dataContract);
    const dispatcher = new GpuDispatcher(vGrid, buffer, parityManager);
    const engine = new GpuEngine(vGrid, buffer, dispatcher, parityManager);

    console.log("[RUN] Bypassing auto-init and injecting random state...");
    
    // 1. Run 1 dummy step to let the kernel finish its 'tick 0' auto-init logic.
    // This increments parityManager to 1, ensuring all future steps are normal LBM cycles.
    await engine.step({ 'LbmCore': LbmCoreSource }, 1);

    // 2. Prepare custom state
    const lx = N + 2;
    const ly = N + 2;
    const rhoData = new Float32Array(lx * ly).fill(1.0);
    const ux_val = 0.1;
    const uy_val = 0.05;
    
    for (let i = 0; i < rhoData.length; i++) {
        rhoData[i] = 1.0 + (Math.random() * 0.2 - 0.1);
    }
    
    const uxData = new Float32Array(lx * ly).fill(ux_val);
    const uyData = new Float32Array(lx * ly).fill(uy_val);
    
    // Calculate feq populations for this state
    const w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36];
    const dx = [0, 1, 0, -1, 0, 1, -1, -1, 1];
    const dy = [0, 0, 1, 0, -1, 1, 1, -1, -1];
    const strideFace = (buffer as any).layout.strideFace;
    const fData = new Float32Array(strideFace * 9);
    
    for (let i = 0; i < lx * ly; i++) {
        const r = rhoData[i];
        const u2 = ux_val*ux_val + uy_val*uy_val;
        for (let d = 0; d < 9; d++) {
            const cu = 3.0 * (dx[d] * ux_val + dy[d] * uy_val);
            const feq = r * w[d] * (1.0 + cu + 0.5 * cu * cu - 1.5 * u2);
            fData[d * strideFace + i] = feq;
        }
    }

    // 3. Inject to GPU (both buffers to be safe)
    engine.setFaceData('chunk_0_0_0', 'rho', rhoData, true);
    engine.setFaceData('chunk_0_0_0', 'ux',  uxData,  true);
    engine.setFaceData('chunk_0_0_0', 'uy',  uyData,  true);
    engine.setFaceData('chunk_0_0_0', 'f',   fData,   true);
    engine.syncToDevice();

    const sumQuantities = (rho: Float32Array, ux: Float32Array, uy: Float32Array) => {
        let mass = 0;
        let momX = 0;
        let momY = 0;
        for (let y = 1; y <= N; y++) {
            for (let x = 1; x <= N; x++) {
                const idx = y * lx + x;
                const r = rho[idx];
                mass += r;
                momX += r * ux[idx];
                momY += r * uy[idx];
            }
        }
        return { mass, momX, momY };
    };

    const initial = sumQuantities(rhoData, uxData, uyData);
    console.log(`[INITIAL] Mass: ${initial.mass.toFixed(10)}, MomX: ${initial.momX.toFixed(10)}, MomY: ${initial.momY.toFixed(10)}`);

    const steps = 10000;
    const interval = 2000;
    console.log(`[RUN] Simulating ${steps} steps...`);
    for (let s = 0; s < steps; s += interval) {
        await engine.step({ 'LbmCore': LbmCoreSource }, interval);
        console.log(`Step ${s + interval}...`);
    }

    await engine.syncFacesToHost(['rho', 'ux', 'uy']);
    const finalRho = engine.getFaceData('chunk_0_0_0', 'rho');
    const finalUx = engine.getFaceData('chunk_0_0_0', 'ux');
    const finalUy = engine.getFaceData('chunk_0_0_0', 'uy');
    
    const final = sumQuantities(finalRho, finalUx, finalUy);
    console.log(`[FINAL] Mass: ${final.mass.toFixed(10)}, MomX: ${final.momX.toFixed(10)}, MomY: ${final.momY.toFixed(10)}`);

    const driftMass = Math.abs(final.mass - initial.mass) / initial.mass;
    const driftMomX = Math.abs(final.momX - initial.momX) / Math.max(1, Math.abs(initial.momX));
    const driftMomY = Math.abs(final.momY - initial.momY) / Math.max(1, Math.abs(initial.momY));

    console.log(`\n--- Audit Results ---`);
    console.log(`Relative Mass Drift:     ${driftMass.toExponential(4)}`);
    console.log(`Relative Momentum X Drift: ${driftMomX.toExponential(4)}`);
    console.log(`Relative Momentum Y Drift: ${driftMomY.toExponential(4)}`);

    const status = (driftMass < 1e-7) ? 'stable' : 'failed'; 
    
    const payload = {
        type: 'conservation-audit',
        initial,
        final,
        drifts: { mass: driftMass, momX: driftMomX, momY: driftMomY },
        status
    };

    fetch('http://localhost:3000', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'x-test-name': `conservation-audit`
        },
        body: JSON.stringify(payload)
    }).catch(() => {});

    return payload;
}

export { runConservationAudit };
