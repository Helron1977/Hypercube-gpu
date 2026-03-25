import { BoundaryRole, GridBoundaries, Dimension3D, VirtualObject, HypercubeConfig } from '../types';
import { DataContract } from '../DataContract';

/**
 * Descriptor for a joint between two chunks or a world boundary.
 */
export interface JointDescriptor {
    role: BoundaryRole;
    neighborId?: string;
    face: 'left' | 'right' | 'top' | 'bottom' | 'front' | 'back' | 
          'top-left' | 'top-right' | 'bottom-left' | 'bottom-right' |
          'front-left' | 'front-right' | 'front-top' | 'front-bottom' |
          'back-left' | 'back-right' | 'back-top' | 'back-bottom' |
          'front-top-left' | 'front-top-right' | 'front-bottom-left' | 'front-bottom-right' |
          'back-top-left' | 'back-top-right' | 'back-bottom-left' | 'back-bottom-right';
}

export interface IMapConstructor {
    buildMap(dims: Dimension3D, chunks: { x: number; y: number; z?: number }, bounds: GridBoundaries): VirtualChunk[];
}

/**
 * A Virtual Chunk is a declarative slice of the global grid.
 */
export interface VirtualChunk {
    x: number;
    y: number;
    z: number;
    id: string;
    joints: JointDescriptor[];
    localDimensions: Dimension3D;
}

/**
 * Holds the virtual representation of the entire grid.
 */
export interface IVirtualGrid {
    readonly config: HypercubeConfig;
    readonly dataContract: DataContract;
    readonly dimensions: Dimension3D;
    readonly chunkLayout: { x: number; y: number; z: number };
    readonly chunks: VirtualChunk[];
    findChunkAt(x: number, y: number, z?: number): VirtualChunk | undefined;
    getObjectsInChunk(chunk: VirtualChunk, t?: number): VirtualObject[];
}

/**
 * A Physical Chunk is a set of memory views (Float32Array) for a chunk.
 */
export interface IPhysicalChunk {
    readonly id: string;
    readonly faces: Float32Array[];
}

/**
 * The MasterBuffer manages the physical allocation and segmenting of memory on the GPU.
 */
export interface IMasterBuffer {
    readonly byteLength: number;
    readonly totalSlotsPerChunk: number;
    readonly strideFace: number;
    readonly rawBuffer: ArrayBuffer;
    readonly gpuBuffer: GPUBuffer;

    getChunkViews(chunkId: string): IPhysicalChunk;
    syncToDevice(): void;
    setFaceData(chunkId: string, faceName: string, data: Float32Array | number[], fillAllPingPong?: boolean): void;
    getFaceData(chunkId: string, faceName: string): Float32Array;
    syncToHost(): Promise<void>;
    syncFacesToHost(faces: (string | number)[]): Promise<void>;
}

export interface IBoundarySynchronizer {
    syncAll(vGrid: IVirtualGrid, bridge: any, parityManager?: any, mode?: 'read' | 'write'): void;
}

export interface IRasterizer {
    rasterizeChunk(
        vChunk: VirtualChunk,
        vGrid: IVirtualGrid,
        bridge: any,
        t: number,
        parityTarget?: 'read' | 'write'
    ): void;
}
