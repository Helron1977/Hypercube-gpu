/**
 * framework_integrity.test.ts
 * 
 * Tests de non-régression qui auraient détecté les 4 bugs du framework.
 * Chaque test correspond à un bug documenté dans implementation_plan.md.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { TopologyResolver, BoundaryRoleID } from '../../src/topology/TopologyResolver';
import { BoundaryCode } from '../../src/types';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { DataContract } from '../../src/DataContract';
import { ParityManager } from '../../src/ParityManager';
import { MemoryLayout } from '../../src/memory/MemoryLayout';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { VirtualChunk, IVirtualGrid } from '../../src/topology/GridAbstractions';
import { EngineDescriptor, HypercubeConfig } from '../../src/types';
import { MasterBuffer } from '../../src/memory/MasterBuffer';

// Mock GPU pour les tests sans WebGPU réel
(globalThis as unknown as { GPUBufferUsage: Record<string, number> }).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};

const mockDevice = {
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), 
    createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
    createBuffer: (desc: { size: number }) => ({ destroy: () => {}, size: desc.size }),
    createBindGroup: () => ({}),
    queue: { writeBuffer: () => {}, submit: () => {} },
    limits: { minUniformBufferOffsetAlignment: 256 },
    createCommandEncoder: () => ({
        beginComputePass: () => ({
            setPipeline: () => {}, setBindGroup: () => {},
            dispatchWorkgroups: () => {}, end: () => {}
        }),
        finish: () => ({})
    })
} as unknown as GPUDevice;

// ============================================================
// BUG 1 : Codes de rôles — TopologyResolver vs BoundaryCode
// ============================================================
describe('Bug 1 : Cohérence des codes de rôles', () => {
    it('TopologyResolver.WALL doit être égal à BoundaryCode.wall', () => {
        expect(BoundaryRoleID.WALL).toBe(BoundaryCode.wall);
    });

    it('TopologyResolver.INFLOW doit être égal à BoundaryCode.inflow', () => {
        expect(BoundaryRoleID.INFLOW).toBe(BoundaryCode.inflow);
    });

    it('TopologyResolver.OUTFLOW doit être égal à BoundaryCode.outflow', () => {
        expect(BoundaryRoleID.OUTFLOW).toBe(BoundaryCode.outflow);
    });

    it('TopologyResolver.CONTINUITY doit être égal à BoundaryCode.periodic', () => {
        expect(BoundaryRoleID.CONTINUITY).toBe(BoundaryCode.periodic);
    });

    it('TopologyResolver.NEUMANN doit être égal à BoundaryCode.neumann', () => {
        expect(BoundaryRoleID.NEUMANN).toBe(BoundaryCode.neumann);
    });

    it('TopologyResolver.ABSORBING doit être égal à BoundaryCode.absorbing', () => {
        expect(BoundaryRoleID.ABSORBING).toBe(BoundaryCode.absorbing);
    });

    it('TopologyResolver.SYMMETRY doit être égal à BoundaryCode.symmetry', () => {
        expect(BoundaryRoleID.SYMMETRY).toBe(BoundaryCode.symmetry);
    });

    it('TopologyResolver.DIRICHLET doit être égal à BoundaryCode.dirichlet', () => {
        expect(BoundaryRoleID.DIRICHLET).toBe(BoundaryCode.dirichlet);
    });

    it('TopologyResolver.CLAMPED doit être égal à BoundaryCode.clamped', () => {
        expect(BoundaryRoleID.CLAMPED).toBe(BoundaryCode.clamped);
    });

    it('resolve() doit retourner les codes compatibles kernel pour LBM tunnel', () => {
        const resolver = new TopologyResolver();
        const vChunk: VirtualChunk = {
            id: 'c0', x: 0, y: 0, z: 0,
            localDimensions: { nx: 128, ny: 128, nz: 1 },
            joints: []
        };
        const boundaries = {
            left: { role: 'inflow' as const },
            right: { role: 'outflow' as const },
            top: { role: 'wall' as const },
            bottom: { role: 'wall' as const }
        };

        const topo = resolver.resolve(vChunk, { x: 1, y: 1 }, boundaries);

        // Le kernel LBM attend ces valeurs EXACTES (de types.ts BoundaryCode)
        expect(topo.leftRole).toBe(3);   // inflow = 3
        expect(topo.rightRole).toBe(4);  // outflow = 4
        expect(topo.topRole).toBe(2);    // wall = 2
        expect(topo.bottomRole).toBe(2); // wall = 2
    });
});

// ============================================================
// BUG 2 : Offsets objets vs rôles dans le uniform buffer
// ============================================================
describe('Bug 2 : Pas d\'écrasement rôles/objets dans les uniforms', () => {
    it('Les objets ne doivent PAS écraser les boundary roles', async () => {
        let capturedData: ArrayBuffer | null = null;
        const spyDevice = {
            ...mockDevice,
            createBuffer: (desc: { size: number }) => ({ destroy: () => {}, size: desc.size }),
            queue: {
                writeBuffer: (_buf: unknown, _off: unknown, data: { buffer: ArrayBuffer }) => {
                    capturedData = data.buffer;
                },
                submit: () => {}
            }
        } as unknown as GPUDevice;
        HypercubeGPUContext.setDevice(spyDevice);

        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: {
                left: { role: 'inflow' },
                right: { role: 'outflow' },
                top: { role: 'wall' },
                bottom: { role: 'wall' }
            },
            engine: 'test',
            params: {},
            objects: [{
                id: 'circle', type: 'circle',
                position: { x: 32, y: 32 },
                dimensions: { w: 10, h: 10 },
                properties: { isObstacle: 1, isSmoke: 0 }
            }]
        };

        const descriptor: EngineDescriptor = {
            name: 'test', version: '1.0.0',
            faces: [{ name: 't', type: 'scalar', isSynchronized: true }],
            rules: [{ type: 'test', source: '' }],
            requirements: { ghostCells: 1, pingPong: true }
        };

        const vGrid = new VirtualGrid(config, descriptor);
        const buffer = { gpuBuffer: {} as GPUBuffer, totalSlotsPerChunk: 2, strideFace: 66 * 66 } as unknown as MasterBuffer;
        const parity = new ParityManager(vGrid.dataContract);

        const dispatcher = new GpuDispatcher(vGrid, buffer, parity);
        await dispatcher.dispatch(0, { 'test': 'void main() {}' });

        expect(capturedData).not.toBeNull();
        if (capturedData) {
            const u32 = new Uint32Array(capturedData);

            // Rôles doivent être intacts aux offsets 32-35
            expect(u32[32]).toBe(BoundaryCode.inflow);   // leftRole
            expect(u32[33]).toBe(BoundaryCode.outflow);  // rightRole
            expect(u32[34]).toBe(BoundaryCode.wall);     // topRole
            expect(u32[35]).toBe(BoundaryCode.wall);     // bottomRole

            // Les objets doivent commencer APRÈS offset 40
            const f32 = new Float32Array(capturedData);
            expect(f32[40]).toBe(32);   // objet.position.x
            expect(f32[41]).toBe(32);   // objet.position.y
        }
    });
});

// ============================================================
// BUG 3 : Taille du uniform buffer vs struct WGSL Params
// ============================================================
describe('Bug 3 : Taille du buffer uniforms >= taille struct Params', () => {
    it('Le buffer uniforms doit faire au moins 416 bytes (struct Params WGSL)', () => {
        HypercubeGPUContext.setDevice(mockDevice);

        // La struct Params WGSL :
        // 4 u32 (nx,ny,lx,ly) + 4 mixed (t,tick,stride,numFaces)
        // + 8 f32 (p0-p7) + 16 u32 (f0-f15) + 6 u32 (roles) + 2 u32 (pad)
        // + 8 GpuObject * 8 u32 = 40 + 64 = 104 u32 = 416 bytes
        const PARAMS_STRUCT_MIN_BYTES = 416;

        const aligned = HypercubeGPUContext.alignToUniform(PARAMS_STRUCT_MIN_BYTES);
        expect(aligned).toBeGreaterThanOrEqual(PARAMS_STRUCT_MIN_BYTES);

        // Le GpuDispatcher doit allouer assez
        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: {},
            engine: 'test', params: {}
        };
        const descriptor: EngineDescriptor = {
            name: 'test', version: '1.0.0',
            faces: [{ name: 't', type: 'scalar', isSynchronized: true }],
            rules: [], requirements: { ghostCells: 1, pingPong: true }
        };

        const vGrid = new VirtualGrid(config, descriptor);
        const buffer = { gpuBuffer: {} as GPUBuffer, totalSlotsPerChunk: 2, strideFace: 66 * 66 } as unknown as MasterBuffer;
        const parity = new ParityManager(vGrid.dataContract);
        const dispatcher = new GpuDispatcher(vGrid, buffer, parity);
        // bytesPerChunkAligned est privé, on le vérifie via la taille du staging
        // Chaque chunk a besoin de >= 416 bytes
        expect((dispatcher as unknown as { bytesPerChunkAligned: number }).bytesPerChunkAligned).toBeGreaterThanOrEqual(PARAMS_STRUCT_MIN_BYTES);
    });
});

// ============================================================
// BUG 4 : strideFace aligné — cohérence MemoryLayout vs kernel
// ============================================================
describe('Bug 4 : strideFace cohérent entre framework et kernel', () => {
    it('strideFace doit être >= cellsPerFaceRaw (lx * ly)', () => {
        const mockGrid = {
            chunks: [{ id: 'c0', localDimensions: { nx: 512, ny: 256, nz: 1 } }],
            dataContract: {
                getFaceMappings: () => [{ name: 'f', isPingPong: true }],
                descriptor: { requirements: { ghostCells: 1 } }
            }
        };

        const layout = new MemoryLayout(mockGrid as unknown as IVirtualGrid);

        // lx = 514, ly = 258, lz = 1 (for 2D sim) → cellsPerFaceRaw = 132,612
        const expectedCells = 514 * 258 * 1;
        expect(layout.cellsPerFaceRaw).toBe(132612);
        expect(layout.strideFace).toBeGreaterThanOrEqual(expectedCells);
    });

    it('strideFace doit être aligné à 256 bytes (64 floats)', () => {
        const mockGrid = {
            chunks: [{ id: 'c0', localDimensions: { nx: 100, ny: 100, nz: 1 } }],
            dataContract: {
                getFaceMappings: () => [{ name: 'f', isPingPong: false }],
                descriptor: { requirements: { ghostCells: 1 } }
            }
        };

        const layout = new MemoryLayout(mockGrid as unknown as IVirtualGrid);
        const bytesPerFace = layout.strideFace * 4;
        expect(bytesPerFace % 256).toBe(0);
    });
});

// ============================================================
// RÉGRESSION : ParityManager doit exposer .base
// ============================================================
describe('ParityManager : base index stable', () => {
    it('getFaceIndices doit retourner un .base qui ne change pas entre ticks', () => {
        const descriptor: EngineDescriptor = {
            name: 'test', version: '1.0.0',
            faces: [
                { name: 'obs', type: 'mask', isSynchronized: false },
                { name: 'f_d0', type: 'population', isSynchronized: true }
            ],
            rules: [], requirements: { ghostCells: 1, pingPong: true }
        };

        const dc = new DataContract(descriptor);
        const pm = new ParityManager(dc);

        const base0 = pm.getFaceIndices('f_d0').base;
        pm.increment();
        const base1 = pm.getFaceIndices('f_d0').base;
        pm.increment();
        const base2 = pm.getFaceIndices('f_d0').base;

        expect(base0).toBe(base1);
        expect(base1).toBe(base2);
        expect(base0).toBe(1); // obs(1 slot) → f_d0 starts at slot 1
    });

    it('read et write doivent alterner mais base reste fixe', () => {
        const descriptor: EngineDescriptor = {
            name: 'test', version: '1.0.0',
            faces: [
                { name: 'f', type: 'population', isSynchronized: true }
            ],
            rules: [], requirements: { ghostCells: 1, pingPong: true }
        };

        const dc = new DataContract(descriptor);
        const pm = new ParityManager(dc);

        const tick0 = pm.getFaceIndices('f');
        expect(tick0.base).toBe(0);
        expect(tick0.read).toBe(0);
        expect(tick0.write).toBe(9);

        pm.increment();
        const tick1 = pm.getFaceIndices('f');
        expect(tick1.base).toBe(0);
        expect(tick1.read).toBe(9);
        expect(tick1.write).toBe(0);
    });
});

// ============================================================
// RÉGRESSION : Layout LbmScientific complet
// ============================================================
describe('LbmScientific : layout mémoire', () => {
    it('14 faces (5 diagnostic + 9 populations PP) = 23 slots', () => {
        const descriptor: EngineDescriptor = {
            name: 'LbmScientific', version: '1.0.0',
            faces: [
                { name: 'obs', type: 'mask', isSynchronized: false },
                { name: 'ux', type: 'macro', isSynchronized: false },
                { name: 'uy', type: 'macro', isSynchronized: false },
                { name: 'rho', type: 'scalar', isSynchronized: false },
                { name: 'curl', type: 'scalar', isSynchronized: false },
                { name: 'f_d0', type: 'scalar', isSynchronized: true },
                { name: 'f_d1', type: 'scalar', isSynchronized: true },
                { name: 'f_d2', type: 'scalar', isSynchronized: true },
                { name: 'f_d3', type: 'scalar', isSynchronized: true },
                { name: 'f_d4', type: 'scalar', isSynchronized: true },
                { name: 'f_d5', type: 'scalar', isSynchronized: true },
                { name: 'f_d6', type: 'scalar', isSynchronized: true },
                { name: 'f_d7', type: 'scalar', isSynchronized: true },
                { name: 'f_d8', type: 'scalar', isSynchronized: true }
            ],
            rules: [],
            requirements: { ghostCells: 1, pingPong: true }
        };

        const dc = new DataContract(descriptor);
        const mappings = dc.getFaceMappings();

        // 5 non-PP + 9 PP
        const totalSlots = mappings.reduce((acc: number, f: any) => acc + (f.isPingPong ? 2 : 1), 0);
        expect(totalSlots).toBe(23); // 5 + 9*2

        // fBase pour le kernel : base de f_d0
        const pm = new ParityManager(dc);
        const fBase = pm.getFaceIndices('f_d0').base;
        expect(fBase).toBe(5); // après les 5 faces diagnostiques

        // Vérifier que fBase + d*2 donne bien le bon slot
        for (let d = 0; d < 9; d++) {
            const slot = pm.getFaceIndices(`f_d${d}`).base;
            expect(slot).toBe(5 + d * 2);
        }
    });
});

// ============================================================
// BUG 5 : config.params (omega, inflow) ignorés par le dispatcher
// ============================================================
describe('Bug 5 : config.params doivent être injectés dans p0-p7', () => {
    it('omega et inflow de config.params doivent arriver dans le uniform buffer', async () => {
        let capturedData: ArrayBuffer | null = null;
        const spyDevice = {
            ...mockDevice,
            createBuffer: (desc: any) => ({ destroy: () => {}, size: desc.size }),
            queue: {
                writeBuffer: (_buf: any, _off: any, data: any) => {
                    capturedData = data.buffer || data;
                },
                submit: () => {}
            }
        };
        HypercubeGPUContext.setDevice(spyDevice);

        const config: HypercubeConfig = {
            dimensions: { nx: 64, ny: 64, nz: 1 },
            chunks: { x: 1, y: 1, z: 1 },
            boundaries: { all: { role: 'wall' } },
            engine: 'test',
            params: { p0: 1.85, p1: 0.12 }
        };

        const descriptor: EngineDescriptor = {
            name: 'test', version: '1.0.0',
            faces: [{ name: 't', type: 'scalar', isSynchronized: false }],
            rules: [{ type: 'test', source: '' }],
            requirements: { ghostCells: 1, pingPong: false }
        };

        const vGrid = new VirtualGrid(config, descriptor);
        const buffer = { gpuBuffer: {} as GPUBuffer, totalSlotsPerChunk: 1, strideFace: 66 * 66 } as unknown as MasterBuffer;
        const parity = new ParityManager(vGrid.dataContract);

        const dispatcher = new GpuDispatcher(vGrid, buffer, parity);
        await dispatcher.dispatch(0, { 'test': 'void main() {}' });

        expect(capturedData).not.toBeNull();
        const f32 = new Float32Array(capturedData!);
        expect(f32[8]).toBeCloseTo(1.85, 2);  // p0 = omega
        expect(f32[9]).toBeCloseTo(0.12, 2);  // p1 = inflow
    });
});
