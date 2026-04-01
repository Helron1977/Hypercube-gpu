import { describe, it, expect, vi } from 'vitest';
import { GpuEngine } from '../../src/GpuEngine';
import { WgslHeaderGenerator } from '../../src/dispatchers/WgslHeaderGenerator';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { GpuCoreFactory } from '../../src/GpuCoreFactory';

describe('Hypercube v6.0.5 DX Features', () => {

    it('should generate semantic WGSL aliases for parameters', () => {
        const faces: any[] = [{ name: 'u', type: 'scalar', numSlots: 2, pointerOffset: 0 }];
        const paramNames = ['viscosity', 'gravity', 'forceScale'];
        
        const header = WgslHeaderGenerator.getHeader('2D', faces, [], "", paramNames);
        
        expect(header.code).toContain('struct SimulationParams {');
        expect(header.code).toContain('viscosity: f32,');
        expect(header.code).toContain('gravity: f32,');
        expect(header.code).toContain('forceScale: f32,');
        expect(header.code).toContain('fn get_params() -> SimulationParams');
        expect(header.code).toContain('return SimulationParams(');
        // p3 should NOT be in SimulationParams struct (which has 4 spaces indent)
        expect(header.code).not.toContain('    p3: f32'); 
    });

    it('should prevent GpuEngine construction without initialized context', () => {
        // Force de-initialization for this test
        (HypercubeGPUContext as any)._state.device = null;
        
        expect(() => {
            new GpuEngine(null as any, null as any, null as any, null as any);
        }).toThrow('[GpuEngine] GPU Context was not correctly initialized');
    });

    it('should support custom entry points in NumericalScheme', async () => {
        // Initialize context (Mock)
        await HypercubeGPUContext.init();
        
        const manifest: any = {
            name: 'EntryPointTest',
            version: '1.0.0',
            engine: {
                name: 'Test', version: '1.0',
                requirements: { ghostCells: 1, pingPong: true },
                faces: [{ name: 'u', type: 'scalar', isSynchronized: true, numSlots: 2 }],
                rules: [
                    { type: 'CustomRule', source: 'void_source', entryPoint: 'compute_alt' }
                ]
            },
            config: {
                dimensions: { nx: 16, ny: 16, nz: 1 },
                chunks: { x: 1, y: 1 },
                boundaries: {},
                engine: 'Test',
                params: { viscosity: 0.1 }
            }
        };

        const factory = new GpuCoreFactory();
        const engine = await factory.build(manifest.config, manifest.engine);
        const spy = vi.spyOn(HypercubeGPUContext, 'createComputePipelineAsync');

        await engine.use({ 'CustomRule': '@compute @workgroup_size(8,8,1) fn compute_alt() {}' });
        
        // Verify that the entryPoint was passed to the context
        expect(spy).toHaveBeenCalledWith(expect.any(String), expect.stringContaining('CustomRule'), 'compute_alt');
    });

    it('should execute transient kernels without rotating parity', async () => {
        const manifest: any = {
            name: 'TransientTest',
            version: '1.0.0',
            engine: {
                name: 'Test', version: '1.0',
                requirements: { ghostCells: 0, pingPong: true },
                faces: [{ name: 'u', type: 'scalar', isSynchronized: true, numSlots: 2 }],
                rules: [{ type: 'Step', source: 'void' }]
            },
            config: {
                dimensions: { nx: 16, ny: 16, nz: 1 },
                chunks: { x: 1, y: 1 },
                boundaries: {}, engine: 'Test', params: {}
            }
        };

        const factory = new GpuCoreFactory();
        const engine = await factory.build(manifest.config, manifest.engine);
        const initialTick = engine.parityManager.currentTick;
        
        await engine.executeTransient('Impulse', '@compute @workgroup_size(16,16,1) fn main() {}', { viscosity: 0.5 });
        
        // Parity should NOT have advanced
        expect(engine.parityManager.currentTick).toBe(initialTick);
    });

    it('should inject data into the current active read slot via injectData', async () => {
        const manifest: any = {
            name: 'InjectionTest',
            version: '1.0.0',
            engine: {
                name: 'Test', version: '1.0',
                requirements: { ghostCells: 0, pingPong: true },
                faces: [{ name: 'u', type: 'scalar', isSynchronized: true, numSlots: 2 }],
                rules: [{ type: 'Step', source: 'void' }]
            },
            config: {
                dimensions: { nx: 4, ny: 4, nz: 1 },
                chunks: { x: 1, y: 1 },
                boundaries: {}, engine: 'Test', params: {}
            }
        };

        const factory = new GpuCoreFactory();
        const engine = await factory.build(manifest.config, manifest.engine);
        const spy = vi.spyOn(engine.buffer, 'setFaceData');
        
        const testData = new Float32Array(16).fill(1.0);
        engine.injectData('u', testData);
        
        // Should have targeted the CURRENT read slot (usually 0 at start)
        const currentReadSlot = engine.parityManager.getFaceIndices('u').read;
        expect(spy).toHaveBeenCalledWith('chunk_0_0_0', 'u', testData, false, currentReadSlot);
    });
});
