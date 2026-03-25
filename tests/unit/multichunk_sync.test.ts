import { describe, it, expect } from 'vitest';
import { GpuBoundarySynchronizer } from '../../src/topology/GpuBoundarySynchronizer';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { ParityManager } from '../../src/ParityManager';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { IVirtualGrid } from '../../src/topology/GridAbstractions';
import { HypercubeConfig, EngineDescriptor } from '../../src/types';

describe('GpuBoundarySynchronizer SOTA Logic', () => {
    it('should generate sync tasks for faces and diagonals', async () => {
        // Mock GPU Context for the synchronizer constructor
        const mockDevice = {
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
        createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
            createBuffer: () => ({ destroy: () => {} }),
            queue: { writeBuffer: () => {}, submit: () => {} },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({
                    setPipeline: () => {},
                    setBindGroup: () => {},
                    dispatchWorkgroups: () => {},
                    end: () => {}
                }),
                finish: () => ({})
            })
        } as unknown as GPUDevice;
        HypercubeGPUContext.setDevice(mockDevice);

        const config = {
            dimensions: { nx: 32, ny: 32, nz: 1 },
            chunks: { x: 2, y: 2, z: 1 },
            boundaries: { all: { role: 'periodic' } }
        };

        const descriptor = {
            name: 'test',
            requirements: { ghostCells: 1, pingPong: true },
            faces: [{ name: 'rho', type: 'scalar', isSynchronized: true }],
            rules: []
        };

        const vGrid = new VirtualGrid(config as unknown as HypercubeConfig, descriptor as unknown as EngineDescriptor);
        const buffer = {
            gpuBuffer: {} as GPUBuffer,
            totalSlotsPerChunk: 2,
            strideFace: (16 + 2) * (16 + 2) // Local size + ghost
        } as unknown as MasterBuffer;
        const parity = new ParityManager(vGrid.dataContract);
        const sync = new GpuBoundarySynchronizer();

        // Access private dispatchBatch to inspect tasks
        let capturedTasks: { count: number }[] = [];
        (sync as unknown as { dispatchBatch: Function }).dispatchBatch = (_buf: unknown, tasks: { count: number }[]) => {
            capturedTasks = tasks;
        };

        vGrid.chunks.forEach(c => {
            console.log(`Chunk ${c.id} joints:`, c.joints.map(j => j.face));
        });

        sync.syncAll(vGrid, buffer, parity, 'read');

        // Total chunks = 4. Each has 8 neighbors (periodic).
        // Each neighbor syncs 1 face or 1 corner for each synchronized face (rho).
        // For 4 chunks, each having 8 joint-neighbors: 4 * 8 = 32 tasks per face mapping.
        // Since rho is ping-pong, syncFaceOffsets will have 1 index.
        expect(capturedTasks.length).toBeGreaterThanOrEqual(32);

        // Verify a diagonal task (top-left)
        const tlTask = capturedTasks.find(t => t.count === 1); // Corners have count 1
        expect(tlTask).toBeDefined();
        
        // Verify a face task (top/bottom or left/right)
        const faceTask = capturedTasks.find(t => t.count > 1);
        expect(faceTask).toBeDefined();
    });

    it('should generate 3D sync tasks for all 26 directions', async () => {
        const config = {
            dimensions: { nx: 32, ny: 32, nz: 32 },
            chunks: { x: 2, y: 2, z: 2 },
            boundaries: { all: { role: 'periodic' } }
        };

        const descriptor = {
            name: 'test3d',
            requirements: { ghostCells: 1, pingPong: false },
            faces: [{ name: 'f', type: 'scalar', isSynchronized: true }],
            rules: []
        };

        const vGrid = new VirtualGrid(config as any, descriptor as any);
        const buffer = {
            gpuBuffer: {},
            totalSlotsPerChunk: 1,
            strideFace: 18 * 18 * 18
        } as any;
        const parity = new ParityManager(vGrid.dataContract);
        const sync = new GpuBoundarySynchronizer();

        let tasks: any[] = [];
        (sync as any).dispatchBatch = (buf: any, tsk: any[]) => { tasks = tsk; };

        sync.syncAll(vGrid, buffer, parity, 'read');

        // 8 chunks, each with 26 neighbors
        expect(vGrid.chunks.length).toBe(8);
        expect(vGrid.chunks[0].joints.length).toBe(26);
        expect(tasks.length).toBe(1288);
    });
});
