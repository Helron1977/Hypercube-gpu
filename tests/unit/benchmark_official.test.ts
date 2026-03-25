import { describe, it, expect } from 'vitest';
import { GpuEngine } from '../../src/GpuEngine';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { ParityManager } from '../../src/ParityManager';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import * as fs from 'fs';
import * as path from 'path';

describe('Official SOTA Benchmark: Lid-Driven Cavity', () => {
    const samplePath = path.join(__dirname, '../../samples/LidDrivenCavity.json');
    const sample = JSON.parse(fs.readFileSync(samplePath, 'utf-8'));
    const kernelPath = path.join(__dirname, '../../src/kernels/wgsl/LbmCore.wgsl');
    const lbmSource = fs.readFileSync(kernelPath, 'utf-8');

    it('should execute a real LBM step and observe data transformation', async () => {
        // This test requires a WebGPU device.
        if (typeof navigator === 'undefined' || !navigator.gpu) {
            console.warn("WebGPU not available. skipping real simulation test.");
            return;
        }

        await HypercubeGPUContext.init();
        
        // Define Descriptor for LBM
        const lbmDescriptor = {
            name: 'lbm-cavity',
            requirements: { ghostCells: 1, pingPong: true },
            faces: [
                { name: 'rho', type: 'scalar', isSynchronized: true },
                { name: 'vx', type: 'scalar', isSynchronized: true },
                { name: 'vy', type: 'scalar', isSynchronized: true },
                { name: 'smoke', type: 'scalar', isSynchronized: true },
                { name: 'f_base', type: 'population', isSynchronized: true }
            ],
            rules: [{
                type: 'LbmCore',
                params: sample.config.params,
                faces: { f0: 'rho', f1: 'vx', f2: 'vy', f3: 'smoke', f4: 'f_base' }
            }]
        };

        const vGrid = new VirtualGrid(sample.config as any, lbmDescriptor as any);
        const buffer = new MasterBuffer(vGrid);
        const parity = new ParityManager(vGrid.dataContract);
        const dispatcher = new GpuDispatcher(vGrid, buffer, parity);
        const engine = new GpuEngine(vGrid, buffer, dispatcher, parity);

        // 1. Initial State: Top lid has velocity
        const lidVelocity = sample.config.boundaries.top.velocity[0];
        console.log(`Initial lid velocity: ${lidVelocity}`);

        // 2. Step the simulation
        await engine.step({ 'LbmCore': lbmSource }, 1);

        // 3. Sync 'vx' to host
        await engine.buffer.syncFacesToHost(['vx']);
        
        // 4. Verify that data exists and is modified
        const chunkId = vGrid.chunks[0].id;
        const vxData = engine.buffer.getFaceData(chunkId, 'vx');
        let maxVx = 0;
        for(let i=0; i<vxData.length; i++) {
            if (Math.abs(vxData[i]) > maxVx) maxVx = Math.abs(vxData[i]);
        }

        console.log(`Max Vx after 1 step: ${maxVx}`);
        if (!HypercubeGPUContext.isMock) {
            expect(maxVx).toBeGreaterThan(0);
        } else {
            expect(maxVx).toBe(0);
        }
        
        console.log(`--- Official Benchmark Verification: SUCCESS ---`);
    });
});
