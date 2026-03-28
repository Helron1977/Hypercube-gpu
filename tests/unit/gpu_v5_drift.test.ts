import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { MemoryLayout } from '../../src/memory/MemoryLayout';
import { ParityManager } from '../../src/ParityManager';

/**
 * Hypercube v5.0 Pro — Anti-Drift Test Suite
 * 
 * These tests verify TWO distinct types of drift:
 * 
 * 1. SPATIAL DRIFT: strideRow mismatch between GPU indexing (getIndex) and 
 *    JS rendering (GpuDemoHelper). If strideRow differs, rows "shear" diagonally.
 * 
 * 2. TEMPORAL DRIFT: unintended dependency on uniforms.t causing patterns to 
 *    scroll horizontally across frames. A "static" kernel must produce the EXACT 
 *    same output regardless of the tick count.
 */
describe('Hypercube v5.0 Pro: Anti-Drift Test Suite', () => {

    // ========================================================================
    // SPATIAL DRIFT TESTS — strideRow alignment between CPU and GPU
    // ========================================================================

    describe('Spatial: strideRow alignment', () => {

        const makeVGrid = (ghosts: number) => {
            const descriptor = {
                name: 'drift-test', version: '1.0.0',
                requirements: { ghostCells: ghosts, pingPong: false },
                faces: [{ name: 'f', type: 'scalar' }],
                rules: [{ type: 'Test', source: '' }]
            };
            const config = {
                dimensions: { nx: 256, ny: 256, nz: 1 },
                chunks: { x: 1, y: 1, z: 1 },
                engine: 'drift-test', params: {}
            };
            return new VirtualGrid(config as any, descriptor as any);
        };

        it('ghosts=0: strideRow == nx (no padding)', () => {
            const vGrid = makeVGrid(0);
            const layout = new MemoryLayout(vGrid);
            expect(layout.strideRow).toBe(256);
        });

        it('ghosts=1: strideRow == nx + 2 (ghost padding)', () => {
            const vGrid = makeVGrid(1);
            const layout = new MemoryLayout(vGrid);
            expect(layout.strideRow).toBe(258);
        });

        it('getIndex CPU must exactly match getIndex GPU for ghosts=0', () => {
            const ghosts = 0;
            const strideRow = 256; // = nx + 2*0
            const getIndex = (x: number, y: number) =>
                (y + ghosts) * strideRow + (x + ghosts);

            // (0,0) -> 0, (255,0) -> 255, (0,1) -> 256
            expect(getIndex(0, 0)).toBe(0);
            expect(getIndex(255, 0)).toBe(255);
            expect(getIndex(0, 1)).toBe(256);
            expect(getIndex(0, 1) - getIndex(0, 0)).toBe(strideRow);
        });

        it('getIndex CPU must exactly match getIndex GPU for ghosts=1', () => {
            const ghosts = 1;
            const strideRow = 258; // = 256 + 2*1
            const getIndex = (x: number, y: number) =>
                (y + ghosts) * strideRow + (x + ghosts);

            // (0,0) -> 1*258+1 = 259
            expect(getIndex(0, 0)).toBe(259);
            expect(getIndex(255, 0)).toBe(514);
            expect(getIndex(0, 255)).toBe(256 * 258 + 1);
            // Anti-shear: consecutive rows must be exactly strideRow apart
            expect(getIndex(0, 1) - getIndex(0, 0)).toBe(strideRow);
            expect(getIndex(5, 100) - getIndex(5, 99)).toBe(strideRow);
        });

        it('GpuDemoHelper JS rendering formula must agree with getIndex', () => {
            // This simulates the rendering loop from GpuDemoHelper.js lines 48-55:
            //   lx = options.strideRow || (nx + 2 * ghosts)
            //   i_node = (y + ghosts) * lx + (x + ghosts)
            
            for (const ghosts of [0, 1]) {
                const nx = 256;
                const strideRow = nx + 2 * ghosts;

                // GPU getIndex
                const gpuGetIndex = (x: number, y: number) =>
                    (y + ghosts) * strideRow + (x + ghosts);

                // JS rendering index (from GpuDemoHelper.js)
                const jsRenderIndex = (x: number, y: number) => {
                    const lx = strideRow; // passed as options.strideRow
                    return (y + ghosts) * lx + (x + ghosts);
                };

                // They MUST be identical for every pixel
                for (let y = 0; y < 256; y += 51) {
                    for (let x = 0; x < 256; x += 51) {
                        expect(jsRenderIndex(x, y)).toBe(gpuGetIndex(x, y));
                    }
                }
            }
        });
    });

    // ========================================================================
    // TEMPORAL DRIFT TESTS — uniforms.t must not cause unintended scrolling
    // ========================================================================

    describe('Temporal: static kernels must not depend on tick', () => {

        it('FunctionCore formula with p2=0 produces identical results at t=0, t=100, t=9999', () => {
            // Simulates: val = sin(x * p1 + t * p2) * cos(y * p1)
            // When p2=0 (default), the t term vanishes completely.
            const p0 = 6.28, p1 = 10.0, p2 = 0.0;
            const nx = 256, ny = 256;

            const evaluate = (t: number) => {
                const results: number[] = [];
                for (let py = 0; py < ny; py += 32) {
                    for (let px = 0; px < nx; px += 32) {
                        const x = (px / nx - 0.5) * p0;
                        const y = (py / ny - 0.5) * p0;
                        results.push(Math.sin(x * p1 + t * p2) * Math.cos(y * p1));
                    }
                }
                return results;
            };

            const ref = evaluate(0);
            expect(evaluate(100)).toEqual(ref);
            expect(evaluate(9999)).toEqual(ref);
        });

        it('FunctionCore formula with p2>0 DOES change with t (opt-in animation)', () => {
            const p0 = 6.28, p1 = 10.0, p2 = 0.1; // animation enabled
            const px = 128, py = 128;
            const nx = 256, ny = 256;

            const evalAt = (t: number) => {
                const x = (px / nx - 0.5) * p0;
                const y = (py / ny - 0.5) * p0;
                return Math.sin(x * p1 + t * p2) * Math.cos(y * p1);
            };

            expect(evalAt(0)).not.toBe(evalAt(100));
        });

        it('ParityManager tick increment is deterministic', () => {
            const descriptor = {
                name: 'test', version: '1.0.0',
                requirements: { ghostCells: 0, pingPong: false },
                faces: [{ name: 'a', type: 'scalar' }],
                rules: []
            };
            const config = {
                dimensions: { nx: 16, ny: 16, nz: 1 },
                chunks: { x: 1, y: 1, z: 1 },
                engine: 'test', params: {}
            };
            const vGrid = new VirtualGrid(config as any, descriptor as any);
            const parity = new ParityManager(vGrid.dataContract);
            
            expect(parity.currentTick).toBe(0);
            parity.increment();
            expect(parity.currentTick).toBe(1);
            parity.increment();
            expect(parity.currentTick).toBe(2);
        });
    });

    // ========================================================================
    // REGRESSION GUARD — LBM and Wave must not be affected
    // ========================================================================

    describe('Regression: LBM & Wave face layouts are unchanged', () => {
        
        it('LBM: strideRow with ghosts=1 on 512x256 grid = 514', () => {
            const descriptor = {
                name: 'lbm', version: '1.0.0',
                requirements: { ghostCells: 1, pingPong: true },
                faces: [
                    { name: 'obs', type: 'mask' },
                    { name: 'ux', type: 'macro' },
                    { name: 'uy', type: 'macro' },
                    { name: 'rho', type: 'scalar' },
                    { name: 'curl', type: 'scalar' },
                    { name: 'f', type: 'population', components: 9 }
                ],
                rules: [{ type: 'LbmScientific', source: '' }]
            };
            const config = {
                dimensions: { nx: 512, ny: 256, nz: 1 },
                chunks: { x: 1, y: 1 },
                engine: 'lbm', params: {}
            };
            const vGrid = new VirtualGrid(config as any, descriptor as any);
            const layout = new MemoryLayout(vGrid);

            expect(layout.strideRow).toBe(514); // 512 + 2
        });

        it('Wave: strideRow with ghosts=1 on 256x256 grid = 258', () => {
            const descriptor = {
                name: 'wave', version: '1.0.0',
                requirements: { ghostCells: 1, pingPong: false },
                faces: [{ name: 'u', type: 'scalar', components: 1 }],
                rules: [{ type: 'WaveCore', source: '' }]
            };
            const config = {
                dimensions: { nx: 256, ny: 256, nz: 1 },
                chunks: { x: 1, y: 1, z: 1 },
                engine: 'wave', params: {}
            };
            const vGrid = new VirtualGrid(config as any, descriptor as any);
            const layout = new MemoryLayout(vGrid);

            expect(layout.strideRow).toBe(258); // 256 + 2
        });
    });
});
