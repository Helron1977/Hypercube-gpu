import { expect, test, describe, beforeAll } from 'vitest';
import { createSimulation, linkSimulation } from '../../src/createSimulation';
import { SharedMasterBuffer } from '../../src/memory/SharedMasterBuffer';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { HypercubeManifest } from '../../src/types';

describe('MultiPhysicsHub: Shared Memory Verification', () => {
    let mockDevice: any;

    beforeAll(async () => {
        // In a real environment, this would be a real GPU device.
        // For unit tests, we rely on the mock context if defined.
        if (!HypercubeGPUContext.isInitialized) {
            // This assumes vitest setup initializes the context or we are in a headless browser
        }
    });

    test('Should share physical buffer between two engines via linkSimulation', async () => {
        const manifestA: HypercubeManifest = {
            name: 'EngineA',
            version: '1.0.0',
            engine: {
                name: 'CoreA',
                version: '1.0.0',
                faces: [{ name: 'f1', type: 'scalar', isSynchronized: false }],
                rules: [{ type: 'writer', source: 'write_f1_Now(x, y, 1.23);' }],
                requirements: { ghostCells: 0, pingPong: false }
            },
            config: {
                dimensions: { nx: 16, ny: 16, nz: 1 },
                chunks: { x: 1, y: 1 },
                boundaries: {},
                engine: 'CoreA',
                params: {}
            }
        };

        const manifestB: HypercubeManifest = {
            name: 'EngineB',
            version: '1.0.0',
            engine: {
                name: 'CoreB',
                version: '1.0.0',
                faces: [{ name: 'f2', type: 'scalar', isSynchronized: false }],
                rules: [{ type: 'reader', source: 'let v = read_f2_Now(x, y);' }],
                requirements: { ghostCells: 0, pingPong: false }
            },
            config: {
                dimensions: { nx: 16, ny: 16, nz: 1 },
                chunks: { x: 1, y: 1 },
                boundaries: {},
                engine: 'CoreB',
                params: {}
            }
        };

        // 1. Create Shared Pool (Standard: 1MB, Atomic: 1MB)
        const shared = new SharedMasterBuffer(1024 * 1024, 1024 * 1024);

        // 2. Link Engines
        const engineA = await linkSimulation(manifestA, shared);
        const engineB = await linkSimulation(manifestB, shared);

        // 3. Verify Offsets
        expect(engineA.buffer.byteOffset).toBe(0);
        expect(engineB.buffer.byteOffset).toBeGreaterThan(0);
        expect(engineB.buffer.byteOffset % 256).toBe(0); // WebGPU Alignment

        // 4. Verification: Buffer reference integrity
        expect(engineA.buffer.gpuBuffer).toBe(shared.gpuBuffer);
        expect(engineB.buffer.gpuBuffer).toBe(shared.gpuBuffer);

        // 5. Success State
        shared.destroy();
    });
});
