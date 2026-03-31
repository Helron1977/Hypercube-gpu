import { EngineDescriptor, EngineFace, NumericalScheme, EngineGlobal } from '../types';

/**
 * MultiPhysicsHub: A utility to merge multiple physics models into a single simulation.
 * Ensures unique face names and consistent rule orchestration.
 */
export class MultiPhysicsHub {
    private _faces: EngineFace[] = [];
    private _rules: NumericalScheme[] = [];
    private _globals: EngineGlobal[] = [];
    private _requirements = { ghostCells: 0, pingPong: false };

    constructor(public readonly name: string = 'MultiPhysicsHub') {}

    /**
     * Adds a complete physics model to the hub.
     */
    public addModel(descriptor: EngineDescriptor): this {
        // 1. Merge Faces (Prevent duplicates)
        for (const face of descriptor.faces) {
            if (this._faces.some(f => f.name === face.name)) {
                console.warn(`MultiPhysicsHub: Face '${face.name}' already exists. Skipping.`);
                continue;
            }
            this._faces.push(face);
        }

        // 2. Merge Rules
        if (descriptor.rules) {
            this._rules.push(...descriptor.rules);
        }

        // 3. Merge Globals
        if (descriptor.globals) {
            for (const glob of descriptor.globals) {
                if (this._globals.some(g => g.name === glob.name)) {
                    continue;
                }
                this._globals.push(glob);
            }
        }

        // 4. Update Requirements (Take the maximum/strictest)
        this._requirements.ghostCells = Math.max(this._requirements.ghostCells, descriptor.requirements.ghostCells);
        this._requirements.pingPong = this._requirements.pingPong || descriptor.requirements.pingPong;

        return this;
    }

    /**
     * Builds the unified EngineDescriptor.
     */
    public build(): EngineDescriptor {
        return {
            name: this.name,
            version: '1.0.0',
            faces: this._faces,
            rules: this._rules,
            globals: this._globals,
            requirements: this._requirements
        };
    }
}
