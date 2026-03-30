import { describe, it, expect } from 'vitest';
import { VirtualGrid } from '../../src/topology/VirtualGrid';
import { DataContract } from '../../src/DataContract';
import { ParityManager } from '../../src/ParityManager';
import { MemoryLayout } from '../../src/memory/MemoryLayout';
import { GpuDispatcher } from '../../src/dispatchers/GpuDispatcher';
import { HypercubeGPUContext } from '../../src/gpu/HypercubeGPUContext';
import { MasterBuffer } from '../../src/memory/MasterBuffer';
import { EngineDescriptor, HypercubeConfig } from '../../src/types';
import * as fs from 'fs';
import * as path from 'path';

// Mock GPU Buffer Usage
(globalThis as any).GPUBufferUsage = {
    MAP_READ: 0x0001, MAP_WRITE: 0x0002, COPY_SRC: 0x0004, COPY_DST: 0x0008,
    INDEX: 0x0010, VERTEX: 0x0020, UNIFORM: 0x0040, STORAGE: 0x0080,
};

describe('Aerodynamics Scientific Sanity', () => {
    
    it('Should correctly map uniforms for the 512x128 grid', async () => {
        let capturedUniforms: ArrayBuffer | null = null;
        
        const mockDevice = {
            createShaderModule: () => ({}),
            createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
            createComputePipelineAsync: async () => ({ getBindGroupLayout: () => ({}) }),
            createBuffer: (desc: any) => ({ destroy: () => {}, size: desc.size }),
            createBindGroup: () => ({}),
            queue: { 
                writeBuffer: (buf: any, off: any, data: any) => {
                    if (off === 0) capturedUniforms = data.buffer || data;
                },
                submit: () => {} 
            },
            limits: { minUniformBufferOffsetAlignment: 256 },
            createCommandEncoder: () => ({
                beginComputePass: () => ({
                    setPipeline: () => {}, setBindGroup: () => {},
                    dispatchWorkgroups: () => {}, end: () => {}
                }),
                finish: () => ({})
            })
        } as any;

        HypercubeGPUContext.setDevice(mockDevice);

        const config: HypercubeConfig = {
            dimensions: { nx: 512, ny: 128, nz: 1 },
            chunks: { x: 1, y: 1 },
            boundaries: {
                left: { role: 'inflow' },
                right: { role: 'outflow' },
                top: { role: 'wall' },
                bottom: { role: 'wall' }
            },
            engine: 'LbmAero',
            params: { p0: 1.85, p1: 0.05 } // omega, U
        };

        const descriptor: EngineDescriptor = {
            name: 'LbmAero',
            version: '1.0.0',
            faces: [
                { name: 'obstacle', type: 'mask', isSynchronized: false },
                { name: 'ux', type: 'macro', isSynchronized: true },
                { name: 'uy', type: 'macro', isSynchronized: true },
                { name: 'rho', type: 'scalar', isSynchronized: true },
                { name: 'curl', type: 'scalar', isSynchronized: true },
                { name: 'f', type: 'population', isSynchronized: true }
            ],
            rules: [
                { type: 'lbm', source: '' },
                { type: 'vor', source: '' }
            ],
            requirements: { ghostCells: 1, pingPong: true }
        };

        const vGrid = new VirtualGrid(config, descriptor);
        const buffer = new MasterBuffer(vGrid);
        const parity = new ParityManager(vGrid.dataContract);
        const dispatcher = new GpuDispatcher(vGrid, buffer, parity);

        // ACT
        await dispatcher.dispatch(0, { 'lbm': '//', 'vor': '//' });

        // ASSERT
        expect(capturedUniforms).not.toBeNull();
        if (capturedUniforms) {
            const u32 = new Uint32Array(capturedUniforms);
            const f32 = new Float32Array(capturedUniforms);
            
            // spatial
            expect(u32[0]).toBe(512); // nx
            expect(u32[1]).toBe(128); // ny
            expect(u32[2]).toBe(514); // lx (strideRow)
            expect(u32[3]).toBe(130); // ly (lyTotal)
            
            // tick & stride
            expect(u32[5]).toBe(0);   // tick
            expect(u32[6]).toBe(66880); // strideFace (aligned)
            
            // params
            expect(f32[8]).toBeCloseTo(1.85, 2); // p0
            expect(f32[9]).toBeCloseTo(0.05, 2); // p1 (U)
            
            // faces (Unified Dynamic Rotation Mapping)
            expect(u32[16]).toBe(0); // obstacle (Base=0)
            expect(u32[17]).toBe(1); // ux_Now (base 1 + offsetNow 0)
            expect(u32[18]).toBe(2); // ux_Next (base 1 + offsetNext 1)
            expect(u32[19]).toBe(3); // uy_Now (base 3 + offsetNow 0)
            expect(u32[20]).toBe(4); // uy_Next (base 3 + offsetNext 1)
            expect(u32[21]).toBe(5); // rho_Now (base 5 + offsetNow 0)
        }
    });
});
