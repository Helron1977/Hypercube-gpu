import { EngineDescriptor, EngineFace, NumericalScheme } from '../types';

/**
 * Pre-configured Simulation Presets (Standard Mother-Models).
 * This allows users to start a simulation without manually defining every face and rule.
 */
export const SimulationPacks = {
    /**
     * Standard 2D LBM D2Q9 (Single Population)
     */
    LBM_D2Q9: {
        name: 'LbmCore',
        version: '1.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 'f', type: 'population', isSynchronized: true } as EngineFace
        ],
        rules: [
            { type: 'collision', source: '', params: { omega: 1.0 } } as NumericalScheme
        ]
    } as EngineDescriptor,

    /**
     * Standard 2D Wave Propagation (FDTD)
     */
    WAVE_2D: {
        name: 'WaveCore',
        version: '1.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 'uNow', type: 'field', isSynchronized: true },
            { name: 'uOld', type: 'field', isSynchronized: true },
            { name: 'uNext', type: 'field', isSynchronized: true }
        ] as EngineFace[],
        rules: [
            { type: 'step', source: '', params: { c2: 0.25, dt: 1.0, dx: 1.0 } } as NumericalScheme
        ]
    } as EngineDescriptor,

    /**
     * Standard 2D Diffusion / Heat Equation
     */
    DIFFUSION_2D: {
        name: 'DiffusionCore',
        version: '1.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 't', type: 'field', isSynchronized: true } as EngineFace
        ],
        rules: [
            { type: 'step', source: '', params: { kappa: 0.1, dt: 1.0 } } as NumericalScheme
        ]
    } as EngineDescriptor,

    /**
     * Game of Life / Cellular Automata
     */
    CELLULAR_AUTOMATA: {
        name: 'CellularCore',
        version: '1.0.0',
        requirements: { ghostCells: 1, pingPong: true },
        faces: [
            { name: 's', type: 'scalar', isSynchronized: true } as EngineFace
        ],
        rules: [
            { type: 'step', source: '', params: { birthMin: 3, birthMax: 3, survivalMin: 2, survivalMax: 3 } } as NumericalScheme
        ]
    } as EngineDescriptor
};
