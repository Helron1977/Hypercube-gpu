import { IRasterizer, IVirtualGrid, VirtualChunk } from '../topology/GridAbstractions';
import { MasterBuffer } from '../memory/MasterBuffer';
import { VirtualObject, EngineDescriptor } from '../types';
import { DataContract } from '../DataContract';
import { ParityManager } from '../ParityManager';

/**
 * CPU Rasterizer for baking VirtualObjects into the compute grid.
 */
export class ObjectRasterizer implements IRasterizer {
    constructor(private parityManager?: ParityManager) { }

    rasterizeChunk(vChunk: VirtualChunk, vGrid: IVirtualGrid, buffer: MasterBuffer, t: number, target: 'read' | 'write' = 'write'): void {
        const config = vGrid.config;
        const dataContract = vGrid.dataContract;
        const descriptor = dataContract.descriptor;
        const faceMappings = dataContract.getFaceMappings();

        const chunkObjects: VirtualObject[] = vGrid.getObjectsInChunk(vChunk, t);
        if (chunkObjects.length === 0) return;

        const views = buffer.getChunkViews(vChunk.id).faces;
        const nx = vChunk.localDimensions.nx;
        const ny = vChunk.localDimensions.ny;
        const padding = descriptor.requirements.ghostCells;

        let chunkX0 = 0;
        let chunkY0 = 0;
        for (const c of vGrid.chunks) {
            if (c.y === vChunk.y && c.z === vChunk.z && c.x < vChunk.x) chunkX0 += c.localDimensions.nx;
            if (c.x === vChunk.x && c.z === vChunk.z && c.y < vChunk.y) chunkY0 += c.localDimensions.ny;
        }

        const pNx = nx + 2 * padding;

        for (const obj of chunkObjects) {
            if (obj.isBaked === false) continue;

            let objX = obj.position.x;
            let objY = obj.position.y;
            if (obj.animation?.velocity) {
                objX += obj.animation.velocity.x * t;
                objY += obj.animation.velocity.y * t;
            }

            for (const [propName, propValue] of Object.entries(obj.properties)) {
                const faceIdx = descriptor.faces.findIndex(f => f.name === propName);
                if (faceIdx === -1) continue;

                let bufferIdx: number;
                if (this.parityManager) {
                    const indices = this.parityManager.getFaceIndices(propName);
                    bufferIdx = target === 'read' ? indices.read : indices.write;
                } else {
                    let bIdx = 0;
                    for (let i = 0; i < faceIdx; i++) bIdx += faceMappings[i].isPingPong ? 2 : 1;
                    bufferIdx = bIdx;
                }

                const view = views[bufferIdx];
                this.rasterizeShape(obj, objX, objY, propValue as number, view, chunkX0, chunkY0, nx, ny, pNx, padding);
            }
        }
    }

    private rasterizeShape(obj: VirtualObject, objX: number, objY: number, value: number, view: Float32Array, chunkX0: number, chunkY0: number, nx: number, ny: number, pNx: number, padding: number) {
        const radius = obj.type === 'circle' ? obj.dimensions.w / 2 : 0;
        const influence = obj.influence?.radius || 0;
        const totalR = radius + influence;

        const bX0 = Math.floor(objX - totalR);
        const bX1 = Math.ceil(objX + obj.dimensions.w + totalR);
        const bY0 = Math.floor(objY - totalR);
        const bY1 = Math.ceil(objY + obj.dimensions.h + totalR);

        const sX = Math.max(padding, bX0 - chunkX0 + padding);
        const eX = Math.min(nx + padding, bX1 - chunkX0 + padding);
        const sY = Math.max(padding, bY0 - chunkY0 + padding);
        const eY = Math.min(ny + padding, bY1 - chunkY0 + padding);

        for (let py = sY; py < eY; py++) {
            for (let px = sX; px < eX; px++) {
                const wX = chunkX0 + (px - padding);
                const wY = chunkY0 + (py - padding);
                let factor = 0;

                if (obj.type === 'circle') {
                    const dist = Math.sqrt(Math.pow(wX - (objX + radius), 2) + Math.pow(wY - (objY + radius), 2));
                    factor = dist <= radius ? 1.0 : 0;
                } else if (obj.type === 'rect') {
                    if (wX >= objX && wX < objX + obj.dimensions.w && wY >= objY && wY < objY + obj.dimensions.h) factor = 1.0;
                }

                if (factor > 0) {
                    const idx = py * pNx + px;
                    const v = value * factor;
                    switch (obj.rasterMode) {
                        case 'add': view[idx] += v; break;
                        case 'replace':
                        default: view[idx] = v; break;
                    }
                }
            }
        }
    }
}
