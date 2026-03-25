import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { ParityManager } from '../../src/ParityManager';
import { GpuEngine } from '../../src/GpuEngine';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

// @ts-ignore
import LbmCoreSource from '../../src/kernels/wgsl/LbmCore.wgsl?raw';

async function runVonKarmanAudit(onFrame?: (data: Float32Array, nx: number, ny: number) => void) {
    console.log(`\n🚀 Starting Phase 2.1: Von Karman Vortex Street Validation...`);

    if (!await HypercubeGPUContext.init()) {
        console.error("Failed to initialize Hypercube GPU Context.");
        return;
    }

    const NX = 512;
    const NY = 128; // Channel height
    const D = 16;  // Cylinder diameter
    const RE = 100; // Vortex shedding at Re=100
    const U_INF = 0.1;
    
    const nu = (U_INF * D) / RE;
    const omega = 1.0 / (3.0 * nu + 0.5);

    console.log(`[CONFIG] Grid: ${NX}x${NY}, Re: ${RE}, Omega: ${omega.toFixed(4)}`);
    
    const config: HypercubeConfig = {
        dimensions: { nx: NX, ny: NY, nz: 1 },
        chunks: { x: 1, y: 1, z: 1 },
        engine: 'von-karman-lbm',
        boundaries: {
            left:   { role: 'inflow' },   // Zou-He Inflow (ID 3)
            right:  { role: 'outflow' },  // Neumann Outflow (ID 4)
            top:    { role: 'wall' },     // Bounce-back Wall (ID 2)
            bottom: { role: 'wall' }      // Bounce-back Wall (ID 2)
        },
        params: {
            p0: omega,
            p1: U_INF, // Inflow Velocity
            p2: 0.0    // Lateral Velocity
        }
    };

    const descriptor: EngineDescriptor = {
        name: 'von-karman-lbm',
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

    console.log("[INIT] Voxelizing cylinder obstacle with 2-pixel symmetry break...");
    
    const lx = NX + 2;
    const ly = NY + 2;
    const typeData = new Float32Array(lx * ly).fill(0.0);
    const uyData = new Float32Array(lx * ly).fill(0.0);
    
    const cx = NX / 4;
    const cy = NY / 2 + 2; // Offset by 2 pixels to break symmetry
    const radius = D / 2;
    
    for (let y = 1; y <= NY; y++) {
        for (let x = 1; x <= NX; x++) {
            const dx = x - cx;
            const dy = y - cy;
            if (dx*dx + dy*dy <= radius*radius) {
                typeData[y * lx + x] = 1.0; 
            }
            // Add tiny noise to uy to seed the instability
            uyData[y * lx + x] = (Math.random() * 0.0001) - 0.00005;
        }
    }
    
    engine.setFaceData('chunk_0_0_0', 'type', typeData);
    engine.setFaceData('chunk_0_0_0', 'uy', uyData);
    engine.syncToDevice();
    
    console.log("[RUN] Simulating for wake development (20,000 steps, high-freq sampling)...");
    
    const totalSteps = 20000;
    const interval = 100; // Sample every 100 steps to see the waves!
    
    const probeX = Math.floor(cx + D * 4); // 4 diameters downstream
    const probeY = Math.floor(cy + 4); 
    const samples: number[] = [];

    for (let s = 0; s < totalSteps; s += interval) {
        await engine.step({ 'LbmCore': LbmCoreSource }, interval);
        
        await engine.syncFacesToHost(['curl']);
        const curlFace = engine.getFaceData('chunk_0_0_0', 'curl');
        const v = curlFace[probeY * lx + probeX];
        samples.push(v);
        
        if (onFrame) onFrame(curlFace, NX, NY);
        
        if (s % 2000 === 0) console.log(`Step ${s}: Vorticity probe = ${v.toFixed(6)}`);
    }

    // Strouhal Analysis: Count peaks in the last 10,000 steps
    const lateSamples = samples.slice(samples.length / 2);
    let peaks = 0;
    for (let i = 1; i < lateSamples.length - 1; i++) {
        if (lateSamples[i] > lateSamples[i-1] && lateSamples[i] > lateSamples[i+1]) {
            if (lateSamples[i] > 0.001) peaks++; // Signal check
        }
    }

    const duration = (totalSteps / 2);
    const frequency = peaks / duration;
    const St = (frequency * D) / U_INF;

    console.log(`\n--- Audit Results ---`);
    console.log(`Detected Peaks: ${peaks}`);
    console.log(`Calculated Strouhal Number (St): ${St.toFixed(4)}`);
    console.log(`Target St (Re=100): ~0.16`);
    
    // Tolerance check (St is usually 0.14-0.18 depending on boundary proximity)
    const status = (St > 0.12 && St < 0.20) ? 'stable' : 'failed';
    if (status === 'stable') {
        console.log("SUCCESS: Strouhal number within aerodynamic tolerance.");
    } else {
        console.log("FAILED: Frequency is outside physical range.");
    }

    const payload = {
        type: 'von-karman',
        config: { NX, NY, D, RE, U_INF, St, peaks },
        results: { samples },
        status
    };

    fetch('http://localhost:3000', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'x-test-name': `von-karman`
        },
        body: JSON.stringify(payload)
    }).catch(() => {});

    return payload;
}

export { runVonKarmanAudit };
