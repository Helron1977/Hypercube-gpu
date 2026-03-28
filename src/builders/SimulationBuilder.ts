import { 
    HypercubeConfig, 
    EngineDescriptor, 
    HypercubeManifest, 
    EngineFace, 
    NumericalScheme, 
    Dimension3D, 
    FaceType 
} from '../types';

/**
 * Fluent Builder for Hypercube Simulations.
 * Simplifies the creation of manifests with validation and default values.
 */
export class SimulationBuilder<TParams = any, TFaces = any> {
    private _manifest: HypercubeManifest<TParams, TFaces>;

    constructor(name: string = 'Untitled Simulation') {
        this._manifest = {
            name,
            version: '1.0.0',
            engine: {
                name: 'GenericEngine',
                version: '1.0.0',
                faces: [],
                rules: [],
                requirements: { ghostCells: 1, pingPong: true }
            },
            config: {
                dimensions: { nx: 128, ny: 128, nz: 1 },
                chunks: { x: 1, y: 1, z: 1 },
                boundaries: { all: { role: 'wall' } },
                engine: 'GenericEngine',
                params: {} as TParams
            }
        };
    }

    public withDimensions(nx: number, ny: number, nz: number = 1): this {
        this._manifest.config.dimensions = { nx, ny, nz };
        return this;
    }

    public withChunks(x: number, y: number, z: number = 1): this {
        this._manifest.config.chunks = { x, y, z };
        return this;
    }

    public withEngineName(name: string): this {
        this._manifest.engine.name = name;
        this._manifest.config.engine = name;
        return this;
    }

    public withGhostCells(count: number): this {
        this._manifest.engine.requirements.ghostCells = count;
        return this;
    }

    public addFace(name: string, type: FaceType, sync: boolean = true, pingPong?: boolean): this {
        this._manifest.engine.faces.push({
            name,
            type,
            isSynchronized: sync,
            isPingPong: pingPong
        });
        return this;
    }

    public addRule(type: string, source: string, params: Partial<TParams> = {}): this {
        this._manifest.engine.rules.push({
            type,
            source,
            params: params as Record<string, number | string>
        });
        return this;
    }

    public withParams(params: TParams): this {
        this._manifest.config.params = params;
        return this;
    }

    public build(): HypercubeManifest<TParams, TFaces> {
        // Basic validation
        if (this._manifest.engine.faces.length === 0) {
            throw new Error('SimulationBuilder: Engine must have at least one face.');
        }
        if (this._manifest.engine.faces.length > 64) {
            throw new Error('SimulationBuilder: Maximum of 64 faces supported.');
        }
        return this._manifest;
    }
}
