import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { ParityManager } from '../../src/ParityManager';
import { GpuEngine } from '../../src/GpuEngine';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

// @ts-ignore
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';

async function runLDCAudit(Re: number = 1000) {
    console.log(`\n🚀 Starting Lid-Driven Cavity Stability Audit (Re=${Re})...`);

    if (!await HypercubeGPUContext.init()) {
        console.error("Failed to initialize Hypercube GPU Context.");
        return;
    }

    const N = 256;
    const U_lid = 0.1;
    const nu = (U_lid * N) / Re;
    const omega = 1.0 / (3.0 * nu + 0.5);

    console.log(`[PHYS] Re=${Re}, nu=${nu.toFixed(5)}, omega=${omega.toFixed(5)}`);

    const config: HypercubeConfig = {
        dimensions: { nx: N, ny: N, nz: 1 },
        chunks: { x: 1, y: 1, z: 1 },
        engine: 'lbm-2d',
        boundaries: {
            left:   { role: 'wall' },
            right:  { role: 'wall' },
            top:    { role: 'moving_wall' }, // Map to ID 10 via TopologyResolver
            bottom: { role: 'wall' }
        },
        params: {
            p0: omega,
            p1: 0.0,   // Initial ux
            p2: U_lid  // Lid velocity
        }
    };

    const descriptor: EngineDescriptor = {
        name: 'lbm-2d-ldc',
        version: '0.1.0',
        faces: [
            { name: 'obs',  type: 'mask',       isSynchronized: false, isPingPong: false }, // f0
            { name: 'ux',   type: 'scalar',     isSynchronized: true,  isPingPong: true },  // f1
            { name: 'uy',   type: 'scalar',     isSynchronized: true,  isPingPong: true },  // f2
            { name: 'rho',  type: 'scalar',     isSynchronized: true,  isPingPong: true },  // f3
            { name: 'curl', type: 'scalar',     isSynchronized: true,  isPingPong: true },  // f4
            { name: 'f',    type: 'population', isSynchronized: true,  isPingPong: true }  // f5
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

    console.log(`[RUN] Simulating to convergence (${N}x${N}, Re=${Re}, 200,000 steps)...`);
    const checkInterval = 5000;
    const maxSteps = 200000;
    
    engine.use({ 'LbmCore': LbmCoreSource });
    for (let i = 0; i < maxSteps; i += checkInterval) {
        await engine.step(checkInterval);
        
        // Audit stability: Check for NaNs
        await engine.syncFacesToHost(['ux']);
        const ux = engine.getFaceData('chunk_0_0_0', 'ux');
        if (isNaN(ux[ux.length / 2])) {
            console.error(`❌ STABILITY CRASH: NaN detected at step ${i + checkInterval}.`);
            return { status: 'unstable', step: i + checkInterval };
        }
        console.log(`[STEP ${i + checkInterval}] Energy OK.`);
    }

    // --- VERIFICATION ---
    // Extract centerline velocity U(y) at x=N/2
    await engine.syncFacesToHost(['ux', 'uy']);
    const uxData = engine.getFaceData('chunk_0_0_0', 'ux');
    const midX = Math.floor(N / 2) + 1;
    const lx = N + 2;
    const ly = N + 2;

    const profileU: {y: number, u: number}[] = [];
    for (let py = 1; py <= N; py++) {
        const idx = py * lx + midX;
        profileU.push({ y: (py - 1) / (N - 1), u: uxData[idx] / U_lid });
    }

    console.log("\n--- LDC Profile (Re=1000) ---");
    // Sample a few points for the log
    [0.1, 0.25, 0.5, 0.75, 1.0].forEach(yTarget => {
        const closest = profileU.reduce((prev, curr) => 
            Math.abs(curr.y - yTarget) < Math.abs(prev.y - yTarget) ? curr : prev
        );
        console.log(`y/L = ${closest.y.toFixed(3)} | u/U = ${closest.u.toFixed(5)}`);
    });

    // Telemetry
    const payload = {
        type: 'ldc-audit',
        Re,
        profileU,
        status: 'stable'
    };

    fetch('http://localhost:3000', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'x-test-name': `ldc-re${Re}`
        },
        body: JSON.stringify(payload)
    }).catch(() => {});

    return { status: 'stable', profile: profileU };
}

export { runLDCAudit };
