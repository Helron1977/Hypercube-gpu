import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { ParityManager } from '../../src/ParityManager';
import { GpuEngine } from '../../src/GpuEngine';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

// @ts-ignore
import Lbm3DCoreSource from '../kernels/Lbm3DCore.test.wgsl?raw';
// @ts-ignore
import HaloExchangeSource from '../../src/kernels/wgsl/HaloExchange.wgsl?raw';

async function runMultiChunkValidation() {
    console.log("🚀 Starting Multi-Chunk Continuity Validation...");

    if (!await HypercubeGPUContext.init()) {
        console.error("Failed to initialize Hypercube GPU Context.");
        return;
    }

    const nx = 256;
    const ny = 64;
    const nz = 1;

    const config: HypercubeConfig = {
        dimensions: { nx, ny, nz },
        chunks: { x: 2, y: 1, z: 1 }, // 2x1x1 Split
        engine: 'lbm-3d',
        boundaries: {
            left: { role: 'inflow' },
            right: { role: 'outflow' },
            top: { role: 'wall' },
            bottom: { role: 'wall' },
            front: { role: 'periodic' },
            back: { role: 'periodic' }
        },
        params: {
            p0: nz,   // nz
            p1: 1.85, // omega
            p2: 0.05  // U_inlet
        }
    };

    const descriptor: EngineDescriptor = {
        name: 'lbm-3d-validate',
        version: '0.1.0',
        faces: [
            { name: 'rho', type: 'scalar', isSynchronized: true, isPingPong: false },
            { name: 'ux',  type: 'scalar', isSynchronized: true, isPingPong: false },
            { name: 'uy',  type: 'scalar', isSynchronized: true, isPingPong: false },
            { name: 'uz',  type: 'scalar', isSynchronized: true, isPingPong: false },
            { name: 'type', type: 'mask',   isSynchronized: false, isPingPong: false },
            { name: 'f',    type: 'population3D', isSynchronized: true, isPingPong: true }
        ],
        requirements: { ghostCells: 1, pingPong: true },
        rules: [
            { type: 'Lbm3DCore', source: Lbm3DCoreSource, params: { p0: nz, p1: 1.85, p2: 0.05, p7: 1000.0 } }
        ]
    };

    const vGrid = new VirtualGrid(config, descriptor);
    const buffer = new MasterBuffer(vGrid);
    const parityManager = new ParityManager(vGrid.dataContract);
    const dispatcher = new GpuDispatcher(vGrid, buffer, parityManager);
    const engine = new GpuEngine(vGrid, buffer, dispatcher, parityManager);

    console.log(`[MEM] Multi-Chunk Allocated: ${vGrid.chunks.length} chunks.`);

    // Initialize with rho=1.0 and ux=0.05
    for (const chunk of vGrid.chunks) {
        const dim = chunk.localDimensions;
        const ghosts = descriptor.requirements.ghostCells;
        const lx = dim.nx + 2 * ghosts;
        const ly = dim.ny + 2 * ghosts;
        const lz = (dim.nz && dim.nz > 1) ? dim.nz + 2 * ghosts : 1;
        const size = lx * ly * lz;
        
        const rhoData = new Float32Array(size).fill(1.0);
        engine.setFaceData(chunk.id, 'rho', rhoData, true);
        
        const uxData = new Float32Array(size).fill(0.05);
        engine.setFaceData(chunk.id, 'ux', uxData, true);
    }

    engine.syncToDevice();

    const kernels = { 
        'Lbm3DCore': Lbm3DCoreSource,
        'HaloExchange': HaloExchangeSource
    };

    console.log("[RUN] Simulating 1000 steps across the joint...");
    const t0 = performance.now();
    engine.use(kernels);
    await engine.step(1000);
    const t1 = performance.now();
    console.log(`[PERF] Completed in ${(t1-t0).toFixed(2)}ms.`);

    // --- VERIFICATION ---
    // Read velocity from both chunks at the interface
    // Chunk 0 (left) and Chunk 1 (right)
    await engine.syncFacesToHost(['ux']);
    
    const ux0 = engine.getFaceData('chunk_0_0_0', 'ux');
    const ux1 = engine.getFaceData('chunk_1_0_0', 'ux');

    // Check center value at i=128 (boundary of chunk 0) and i=129 (start of chunk 1)
    const lx0 = vGrid.chunks[0].localDimensions.nx + 2;
    const ly0 = vGrid.chunks[0].localDimensions.ny + 2;
    const midY = Math.floor(ny / 2) + 1;
    const midZ = 0; 

    const valLeft = ux0[(midZ * ly0 + midY) * lx0 + 128]; // Last real cell of Chunk 0
    const valRight = ux1[(midZ * ly0 + midY) * lx0 + 1];  // First real cell of Chunk 1

    console.log(`--- Continuity Report ---`);
    console.log(`Velocity Left (Chunk 0, x=128): ${valLeft.toFixed(6)}`);
    console.log(`Velocity Right (Chunk 1, x=129): ${valRight.toFixed(6)}`);
    
    const diff = Math.abs(valLeft - valRight);
    if (diff < 1e-4) {
        console.log("✅ SUCCESS: Fluid continuity maintained across the GPU Joint.");
    } else {
        console.error(`❌ FAILURE: Velocity jump detected! Diff=${diff.toExponential(4)}`);
    }

    // Telemetry for the suite runner
    const payload = {
        type: 'multichunk-test',
        valLeft, valRight, diff,
        mlups: (nx * ny * nz * 1000) / ((t1 - t0) * 1000)
    };
    
    fetch('http://127.0.0.1:3000', {
        method: 'POST',
        headers: { 
            'Content-Type': 'application/json',
            'x-test-name': 'multichunk-test'
        },
        body: JSON.stringify(payload)
    }).catch(() => {});
}

runMultiChunkValidation();
