/**
 * Core type definitions for Hypercube GPU Core.
 * Specialized for WebGPU Zero-Stall execution.
 */

export interface Dimension3D {
    nx: number;
    ny: number;
    nz: number; // 1 for 2D
}

export type BoundaryRole = 'joint' | 'wall' | 'periodic' | 'absorbing' | 'clamped' | 'dirichlet' | 'neumann' | 'symmetry' | 'inflow' | 'outflow' | 'moving_wall';

export const BoundaryCode: Record<BoundaryRole, number> = {
    'joint': 0,
    'periodic': 1,
    'wall': 2,
    'inflow': 3,
    'outflow': 4,
    'symmetry': 5,
    'absorbing': 6,
    'clamped': 7,
    'dirichlet': 8,
    'neumann': 9,
    'moving_wall': 10
};

export interface BoundarySide {
    role: BoundaryRole;
    factor?: number;
    value?: number | number[]; 
}

export interface GridBoundaries {
    left?: BoundarySide;
    right?: BoundarySide;
    top?: BoundarySide;
    bottom?: BoundarySide;
    front?: BoundarySide;
    back?: BoundarySide;
    all?: BoundarySide;
}

export type FaceType = 'scalar' | 'vector' | 'population' | 'macro' | 'field' | 'mask' | 'population3D';

export interface EngineFace {
    name: string;
    type: FaceType;
    isSynchronized: boolean; // Sync ghost cells
    isReadOnly?: boolean;
    isPersistent?: boolean;
    isPingPong?: boolean;
    defaultValue?: number;
}

export interface NumericalScheme {
    type: string;
    source: string;
    destination?: string;
    field?: string;
    params?: Record<string, number | string>;
    faces?: string[];
}

export interface EngineDescriptor<TFaces = any> {
    name: string;
    version: string;
    faces: EngineFace[];
    rules: NumericalScheme[];
    requirements: {
        ghostCells: number;
        pingPong: boolean;
    };
}

export interface VirtualObject {
    id: string;
    type: 'circle' | 'rect' | 'stencil' | 'ellipse' | 'polygon';
    position: { x: number; y: number; z?: number };
    dimensions: { w: number; h: number; d?: number };
    points?: { x: number; y: number }[];
    properties: Record<string, number>;
    influence?: {
        falloff: 'step' | 'linear' | 'gaussian' | 'inverse-square';
        radius: number;
    };
    rasterMode?: 'replace' | 'add' | 'multiply' | 'max' | 'min';
    animation?: {
        velocity?: { x: number; y: number; z?: number };
    };
    isBaked?: boolean; 
}

export interface HypercubeConfig<TParams = any> {
    dimensions: Dimension3D;
    chunks: {
        x: number;
        y: number;
        z?: number;
    };
    boundaries: GridBoundaries;
    engine: string;
    params: TParams;
    objects?: VirtualObject[];
}

export interface HypercubeManifest<TParams = any, TFaces = any> {
    name: string;
    version: string;
    engine: EngineDescriptor<TFaces>;
    config: HypercubeConfig<TParams>;
}
