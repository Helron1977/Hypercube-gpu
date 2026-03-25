import { EngineDescriptor } from './types';

/**
 * The DataContract describes the memory mapping for an engine's faces.
 */
export interface FaceMapping {
    faceIndex: number;
    name: string;
    type: string;
    requiresSync: boolean;
    isPingPong: boolean;
    numComponents: number;
}

export class DataContract {
    constructor(public readonly descriptor: EngineDescriptor) { }

    getFaceMappings(): FaceMapping[] {
        return this.descriptor.faces.map((face, index) => {
            const isPingPong = face.isPingPong !== undefined
                ? face.isPingPong
                : (face.isSynchronized && this.descriptor.requirements.pingPong && !face.isReadOnly);

            const numComponents = face.type === 'scalar' ? 1 
                                : face.type === 'vector' ? 3 
                                : face.type === 'population' ? 9 
                                : face.type === 'population3D' ? 19 : 1;

            return {
                faceIndex: index,
                name: face.name,
                type: face.type,
                requiresSync: face.isSynchronized,
                isPingPong: !!isPingPong,
                numComponents
            };
        });
    }

    calculateChunkBytes(nx: number, ny: number, nz: number, padding: number): number {
        const floatSize = 4;
        const physicalNx = nx + padding * 2;
        const physicalNy = ny + padding * 2;
        const physicalNz = nz > 1 ? nz + padding * 2 : 1;

        const cellsPerFace = physicalNx * physicalNy * physicalNz;
        const faceMappings = this.getFaceMappings();
        let totalBuffers = 0;
        for (const f of faceMappings) {
            totalBuffers += f.numComponents * (f.isPingPong ? 2 : 1);
        }

        return cellsPerFace * floatSize * totalBuffers;
    }
}
