import { EngineDescriptor } from './types';

/**
 * !!! MANIFESTO: THE MANIFEST IS THE PILOT !!!
 * THE FRAMEWORK IS A SERVANT. DO NOT INDEPENDENTLY UP-SCALE MEMORY OR INJECT 
 * LOGIC NOT COMMANDED BY THE JSON MANIFEST. 
 * - numSlots: Priority 1 = Explicit JSON value. Priority 2 = sync/ping-pong ? 2 : 1.
 * - rotation: Occurs strictly on the physical slots defined by the manifestations.
 * - No implicit Elite *4 up-scaling. Memory is sacred.
 */

/**
 * The DataContract describes the memory mapping for an engine's faces.
 */
export interface FaceMapping {
    faceIndex: number;
    name: string;
    type: string;
    requiresSync: boolean;
    isPingPong: boolean;
    numSlots: number;
    numComponents: number;
    pointerOffset: number;
}

export interface GlobalMapping {
    name: string;
    type: string;
    count: number;
    byteOffset: number;
}

export class DataContract {
    constructor(public readonly descriptor: EngineDescriptor) { }

    getFaceMappings(): FaceMapping[] {
        const rules = this.descriptor.rules || [];
        const faces = this.descriptor.faces || [];
        
        const is3D = rules.some(r => r.type.toLowerCase().includes('3d')) ||
                     faces.some(f => f.type === 'population3D' || /^[uvw].*z$/i.test(f.name));

        let currentOffset = 0;
        return faces.map((face, index) => {
            // PILOT: The manifest's isSynchronized or isPingPong flags are the triggers.
            const isSynchronized = !!face.isSynchronized;
            const isPingPong = !!face.isPingPong;
            const needsRotation = isSynchronized || isPingPong;

            const numComponents = face.numComponents !== undefined 
                                ? face.numComponents 
                                : (face.type === 'scalar' ? 1 
                                : face.type === 'vector' ? 3 
                                : face.type === 'population' ? (is3D ? 27 : 9) 
                                : face.type === 'population3D' ? 27 : 1);

            // STRICTOR: Use JSON numSlots if provided. Default to 2 if rotation is needed. 1 otherwise.
            // DO NOT arbitrary upscale to 4 unless the manifest specifically piloted it.
            const numSlots = face.numSlots || (needsRotation ? 2 : 1);

            const mapping: FaceMapping = {
                faceIndex: index,
                name: face.name,
                type: face.type,
                requiresSync: isSynchronized,
                isPingPong: needsRotation,
                numSlots,
                numComponents,
                pointerOffset: currentOffset
            };

            currentOffset += numSlots * numComponents; // POINTER OFFSET = UNIT ADDRESS
            return mapping;
        });
    }

    getGlobalMappings(): GlobalMapping[] {
        const globals = this.descriptor.globals || [];
        let currentByteOffset = 0;

        return globals.map(g => {
            const mapping: GlobalMapping = {
                name: g.name,
                type: g.type,
                count: g.count,
                byteOffset: currentByteOffset
            };
            currentByteOffset += g.count * 4;
            return mapping;
        });
    }

    calculateGlobalBytes(): number {
        const globals = this.getGlobalMappings();
        if (globals.length === 0) return 0;
        const last = globals[globals.length - 1];
        return last.byteOffset + last.count * 4;
    }

    calculateChunkBytes(nx: number, ny: number, nz: number, padding: number): number {
        const floatSize = 4;
        const physicalNx = nx + padding * 2;
        const physicalNy = ny + padding * 2;
        const physicalNz = nz > 1 ? nz + padding * 2 : 1;

        const cellsPerFace = physicalNx * physicalNy * physicalNz;
        const faceMappings = this.getFaceMappings();
        let totalSlots = 0;
        for (const f of faceMappings) {
            totalSlots += f.numComponents * f.numSlots;
        }

        return cellsPerFace * floatSize * totalSlots;
    }
}
