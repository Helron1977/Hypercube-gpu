import { IVirtualGrid, IBoundarySynchronizer } from './GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { ParityManager } from '../ParityManager';
import { HypercubeGPUContext } from '../gpu/HypercubeGPUContext';
import { DataContract } from '../DataContract';

/**
 * Synchronizes boundaries on the GPU using Compute Shaders.
 */
export class GpuBoundarySynchronizer implements IBoundarySynchronizer {
    private device: GPUDevice;
    private pipeline: GPUComputePipeline;
    private batchBuffer?: GPUBuffer;

    constructor() {
        this.device = HypercubeGPUContext.device;
        const wgsl = `
            struct SyncParams { srcOffset: u32, dstOffset: u32, count: u32, stride: u32 };
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @group(0) @binding(2) var<storage, read> batch: array<SyncParams>;
            @compute @workgroup_size(64)
            fn main(@builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) wg_id: vec3<u32>) {
                let p = batch[wg_id.x];
                for (var i = local_id.x; i < p.count; i = i + 64u) {
                    data[p.dstOffset + i * p.stride] = data[p.srcOffset + i * p.stride];
                }
            }
        `;
        const module = this.device.createShaderModule({ code: wgsl });
        this.pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });
    }

    public syncAll(vGrid: IVirtualGrid, buffer: MasterBuffer, parityManager: ParityManager, target: 'read' | 'write'): void {
        const gpuBuffer = buffer.gpuBuffer;
        const dataContract = vGrid.dataContract;
        const padding = dataContract.descriptor.requirements.ghostCells;
        if (padding === 0) return;

        const syncTasks: { srcOffset: number, dstOffset: number, count: number, stride: number }[] = [];
        const tick = parityManager.currentTick;
        const mode = target === 'read' ? tick % 2 : (1 - (tick % 2));

        const nx = Math.floor(vGrid.dimensions.nx / vGrid.chunkLayout.x);
        const ny = Math.floor(vGrid.dimensions.ny / vGrid.chunkLayout.y);
        const nz = vGrid.dimensions.nz > 1 ? Math.floor(vGrid.dimensions.nz / vGrid.chunkLayout.z) : 1;
        const pNx = nx + 2 * padding;
        const pNy = ny + 2 * padding;
        const pNz = nz > 1 ? nz + 2 * padding : 1;
        
        const faceMappings = dataContract.getFaceMappings();
        const syncFaceOffsets: number[] = [];

        let currentOffset = 0;
        for (const face of faceMappings) {
            if (face.requiresSync) syncFaceOffsets.push(face.isPingPong ? currentOffset + mode : currentOffset);
            currentOffset += face.isPingPong ? 2 : 1;
        }

        const facesPerChunk = buffer.totalSlotsPerChunk;
        const strideFace = buffer.strideFace;

        for (let i = 0; i < vGrid.chunks.length; i++) {
            const chunk = vGrid.chunks[i];
            const myBase = i * facesPerChunk * strideFace;
            for (const joint of chunk.joints) {
                if (joint.role !== 'joint' || !joint.neighborId) continue;
                const neighborIdx = vGrid.chunks.findIndex(c => c.id === joint.neighborId);
                const theirBase = neighborIdx * facesPerChunk * strideFace;

                for (const fOS of syncFaceOffsets) {
                    const mF = myBase + fOS * strideFace;
                    const tF = theirBase + fOS * strideFace;

                    // 3D Plane/Line/Point logic:
                    // srcOffset: Where to read from the neighbor (interior)
                    // dstOffset: Where to write to me (ghost)
                    // count: Number of lines or points to transfer
                    // stride: Jump between lines or planes

                    // 26-DIRECTION VOLUMETRIC SYNCHRONIZATION MATRIX
                    switch (joint.face) {
                        // 1. Primary Faces (6)
                        case 'left':
                            for (let z = 0; z < pNz; z++) for (let p = 0; p < padding; p++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (nx + p), dstOffset: mF + z * pNx * pNy + p, count: (ny + 2 * padding), stride: pNx });
                            break;
                        case 'right':
                            for (let z = 0; z < pNz; z++) for (let p = 0; p < padding; p++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (padding + p), dstOffset: mF + z * pNx * pNy + (nx + padding + p), count: (ny + 2 * padding), stride: pNx });
                            break;
                        case 'top':
                            for (let z = 0; z < pNz; z++) for (let p = 0; p < padding; p++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (ny + p) * pNx, dstOffset: mF + z * pNx * pNy + p * pNx, count: (nx + 2 * padding), stride: 1 });
                            break;
                        case 'bottom':
                            for (let z = 0; z < pNz; z++) for (let p = 0; p < padding; p++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (padding + p) * pNx, dstOffset: mF + z * pNx * pNy + (ny + padding + p) * pNx, count: (nx + 2 * padding), stride: 1 });
                            break;
                        case 'front': if (nz > 1) {
                            for (let p = 0; p < padding; p++)
                                syncTasks.push({ srcOffset: tF + (nz + p) * pNx * pNy, dstOffset: mF + p * pNx * pNy, count: pNx * pNy, stride: 1 });
                        } break;
                        case 'back': if (nz > 1) {
                            for (let p = 0; p < padding; p++)
                                syncTasks.push({ srcOffset: tF + (padding + p) * pNx * pNy, dstOffset: mF + (nz + padding + p) * pNx * pNy, count: pNx * pNy, stride: 1 });
                        } break;

                        // 2. Edges (12)
                        case 'top-left':
                            for (let z = 0; z < pNz; z++) for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (ny + p1) * pNx + (nx + p2), dstOffset: mF + z * pNx * pNy + p1 * pNx + p2, count: 1, stride: 0 });
                            break;
                        case 'top-right':
                            for (let z = 0; z < pNz; z++) for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (ny + p1) * pNx + (padding + p2), dstOffset: mF + z * pNx * pNy + p1 * pNx + (nx + padding + p2), count: 1, stride: 0 });
                            break;
                        case 'bottom-left':
                            for (let z = 0; z < pNz; z++) for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (padding + p1) * pNx + (nx + p2), dstOffset: mF + z * pNx * pNy + (ny + padding + p1) * pNx + p2, count: 1, stride: 0 });
                            break;
                        case 'bottom-right':
                            for (let z = 0; z < pNz; z++) for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + z * pNx * pNy + (padding + p1) * pNx + (padding + p2), dstOffset: mF + z * pNx * pNy + (ny + padding + p1) * pNx + (nx + padding + p2), count: 1, stride: 0 });
                            break;

                        // Z-Edges (only if nz > 1)
                        case 'front-left': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (nz + p1) * pNx * pNy + (nx + p2), dstOffset: mF + p1 * pNx * pNy + p2, count: (ny + 2 * padding), stride: pNx });
                        } break;
                        case 'front-right': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (nz + p1) * pNx * pNy + (padding + p2), dstOffset: mF + p1 * pNx * pNy + (nx + padding + p2), count: (ny + 2 * padding), stride: pNx });
                        } break;
                        case 'front-top': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (nz + p1) * pNx * pNy + (ny + p2) * pNx, dstOffset: mF + p1 * pNx * pNy + p2 * pNx, count: (nx + 2 * padding), stride: 1 });
                        } break;
                        case 'back-left': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (padding + p1) * pNx * pNy + (nx + p2), dstOffset: mF + (nz + padding + p1) * pNx * pNy + p2, count: (ny + 2 * padding), stride: pNx });
                        } break;
                        case 'back-right': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (padding + p1) * pNx * pNy + (padding + p2), dstOffset: mF + (nz + padding + p1) * pNx * pNy + (nx + padding + p2), count: (ny + 2 * padding), stride: pNx });
                        } break;
                        case 'back-top': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (padding + p1) * pNx * pNy + (ny + p2) * pNx, dstOffset: mF + (nz + padding + p1) * pNx * pNy + p2 * pNx, count: (nx + 2 * padding), stride: 1 });
                        } break;
                        case 'back-bottom': if (nz > 1) {
                            for (let p1 = 0; p1 < padding; p1++) for (let p2 = 0; p2 < padding; p2++)
                                syncTasks.push({ srcOffset: tF + (padding + p1) * pNx * pNy + (padding + p2) * pNx, dstOffset: mF + (nz + padding + p1) * pNx * pNy + (ny + padding + p2) * pNx, count: (nx + 2 * padding), stride: 1 });
                        } break;

                        // 3. Corners (8)
                        case 'front-top-left': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (nz + pZ) * pNx * pNy + (ny + pY) * pNx + (nx + pX), dstOffset: mF + pZ * pNx * pNy + pY * pNx + pX, count: 1, stride: 0 });
                        } break;
                        case 'front-top-right': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (nz + pZ) * pNx * pNy + (ny + pY) * pNx + (padding + pX), dstOffset: mF + pZ * pNx * pNy + pY * pNx + (nx + padding + pX), count: 1, stride: 0 });
                        } break;
                        case 'front-bottom-left': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (nz + pZ) * pNx * pNy + (padding + pY) * pNx + (nx + pX), dstOffset: mF + pZ * pNx * pNy + (ny + padding + pY) * pNx + pX, count: 1, stride: 0 });
                        } break;
                        case 'front-bottom-right': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (nz + pZ) * pNx * pNy + (padding + pY) * pNx + (padding + pX), dstOffset: mF + pZ * pNx * pNy + (ny + padding + pY) * pNx + (nx + padding + pX), count: 1, stride: 0 });
                        } break;
                        case 'back-top-left': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (padding + pZ) * pNx * pNy + (ny + pY) * pNx + (nx + pX), dstOffset: mF + (nz + padding + pZ) * pNx * pNy + pY * pNx + pX, count: 1, stride: 0 });
                        } break;
                        case 'back-top-right': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (padding + pZ) * pNx * pNy + (ny + pY) * pNx + (padding + pX), dstOffset: mF + (nz + padding + pZ) * pNx * pNy + pY * pNx + (nx + padding + pX), count: 1, stride: 0 });
                        } break;
                        case 'back-bottom-left': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (padding + pZ) * pNx * pNy + (padding + pY) * pNx + (nx + pX), dstOffset: mF + (nz + padding + pZ) * pNx * pNy + (ny + padding + pY) * pNx + pX, count: 1, stride: 0 });
                        } break;
                        case 'back-bottom-right': if (nz > 1) {
                            for (let pZ = 0; pZ < padding; pZ++) for (let pY = 0; pY < padding; pY++) for (let pX = 0; pX < padding; pX++)
                                syncTasks.push({ srcOffset: tF + (padding + pZ) * pNx * pNy + (padding + pY) * pNx + (padding + pX), dstOffset: mF + (nz + padding + pZ) * pNx * pNy + (ny + padding + pY) * pNx + (nx + padding + pX), count: 1, stride: 0 });
                        } break;
                    }
                }
            }
        }
        if (syncTasks.length > 0) this.dispatchBatch(gpuBuffer, syncTasks);
    }

    private dispatchBatch(dataBuffer: GPUBuffer, tasks: any[]) {
        const batchSize = tasks.length * 16;
        if (!this.batchBuffer || this.batchBuffer.size < batchSize) {
            if (this.batchBuffer) this.batchBuffer.destroy();
            this.batchBuffer = this.device.createBuffer({ size: Math.ceil(batchSize/256)*256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        }
        const data = new Uint32Array(tasks.length * 4);
        tasks.forEach((t, i) => { data[i*4+0] = t.srcOffset; data[i*4+1] = t.dstOffset; data[i*4+2] = t.count; data[i*4+3] = t.stride; });
        this.device.queue.writeBuffer(this.batchBuffer!, 0, data);

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: dataBuffer } }, { binding: 2, resource: { buffer: this.batchBuffer! } }] }));
        pass.dispatchWorkgroups(Math.min(65535, tasks.length));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}
