import { IVirtualGrid, VirtualChunk, IMapConstructor } from './GridAbstractions';
import { DataContract } from '../DataContract';
import { EngineDescriptor, HypercubeConfig, VirtualObject, Dimension3D } from '../types';
import { MapConstructor } from './MapConstructor';

/**
 * Orchestrates the virtual arrangement of chunks and their memory requirements.
 */
export class VirtualGrid implements IVirtualGrid {
    public readonly chunks: VirtualChunk[];
    public readonly dataContract: DataContract;

    constructor(
        public readonly config: HypercubeConfig,
        descriptor: EngineDescriptor,
        mapConstructor: IMapConstructor = new MapConstructor()
    ) {
        this.dataContract = new DataContract(descriptor);
        const chunks = config.chunks || { x: 1, y: 1, z: 1 };
        this.chunks = mapConstructor.buildMap(
            config.dimensions,
            chunks,
            config.boundaries
        );
    }

    get dimensions(): Dimension3D {
        return this.config.dimensions;
    }

    get chunkLayout(): { x: number; y: number; z: number } {
        const c = this.config.chunks || { x: 1, y: 1, z: 1 };
        return {
            x: c.x,
            y: c.y,
            z: c.z || 1
        };
    }

    findChunkAt(x: number, y: number, z: number = 0): VirtualChunk | undefined {
        let qx = x, qy = y, qz = z;
        return this.chunks.find(c => c.x === qx && c.y === qy && c.z === qz);
    }

    getObjectsInChunk(chunk: VirtualChunk, t: number = 0): VirtualObject[] {
        if (!this.config.objects) return [];

        let chunkX0 = 0;
        let chunkY0 = 0;
        for (const c of this.chunks) {
            if (c.y === chunk.y && c.z === chunk.z && c.x < chunk.x) chunkX0 += c.localDimensions.nx;
            if (c.x === chunk.x && c.z === chunk.z && c.y < chunk.y) chunkY0 += c.localDimensions.ny;
        }

        const chunkX1 = chunkX0 + chunk.localDimensions.nx;
        const chunkY1 = chunkY0 + chunk.localDimensions.ny;

        return this.config.objects.filter(obj => {
            let posX = obj.position.x;
            let posY = obj.position.y;

            if (obj.animation?.velocity) {
                posX += obj.animation.velocity.x * t;
                posY += obj.animation.velocity.y * t;
            }

            const radius = obj.influence?.radius || 0;
            const objX0 = posX - radius;
            const objX1 = posX + obj.dimensions.w + radius;
            const objY0 = posY - radius;
            const objY1 = posY + obj.dimensions.h + radius;

            return !(objX1 < chunkX0 || objX0 > chunkX1 || objY1 < chunkY0 || objY0 > chunkY1);
        });
    }
}
