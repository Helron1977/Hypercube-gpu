/**
 * GpuDemoHelper.js
 * Optimized & Robust shared utilities for Hypercube GPU visual verification.
 * Includes Scientific HUD support and Professional (Seismic) Rendering.
 */

export class GpuDemoHelper {
    static async initWebGPU() {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No adapter found");
        const device = await adapter.requestDevice({
            requiredLimits: {
                maxComputeInvocationsPerWorkgroup: 256, // Standard for many GPUs
            }
        });
        return device;
    }

    static createDataBuffer(device, size) {
        return device.createBuffer({
            size: Math.ceil(size / 4) * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
    }

    static async renderToCanvas(device, dataBuffer, canvas, options) {
        const { mode = 'single', nx, ny, nz = 1, pz_layer = 0, faceIdx = -1, faces = [], strideFace, scale = 1.0, offset = 0 } = options;
        const totalSize = dataBuffer.size;
        
        const readBuffer = device.createBuffer({
            size: totalSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });

        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(dataBuffer, 0, readBuffer, 0, totalSize);
        device.queue.submit([encoder.finish()]);

        await device.queue.onSubmittedWorkDone();

        await readBuffer.mapAsync(GPUMapMode.READ);
        const results = new Float32Array(readBuffer.getMappedRange());
        
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(nx, ny);
        const lx = nx + 2;
        const ly = ny + 2;

        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const z_off = (nz > 1) ? (pz_layer + 1) * lx * ly : 0;
                const i_node = z_off + (y + 1) * lx + (x + 1);
                const i4 = (y * nx + x) * 4;
                
                let r = 0, g = 0, b = 0;

                if (mode === 'u_mag') {
                    const vx = results[faces[0] * strideFace + i_node];
                    const vy = results[faces[1] * strideFace + i_node];
                    const mag = Math.sqrt(vx*vx + vy*vy) * scale;
                    // Pro 'Hot' palette
                    r = Math.min(255, mag * 255 * 3.0);
                    g = Math.min(255, mag * 255 * 1.5);
                    b = Math.min(255, mag * 255 * 0.5);
                } else if (mode === 'curl') {
                    const val = results[faceIdx * strideFace + i_node] * scale * 255;
                    if (val > 0) {
                        r = Math.min(255, val * 2.0); g = val; b = 0; // Red/Yellow Glow
                    } else {
                        r = 0; g = -val; b = Math.min(255, -val * 2.0); // Blue Glow
                    }
                } else if (mode === 'bipolar') {
                    const val = results[faceIdx * strideFace + i_node] * scale * 255;
                    const v = val / 255.0; 
                    if (v > 0) { 
                        r = Math.min(255, 40 + v * 215); 
                        g = 40; 
                        b = 40; 
                    } else { 
                        r = 40; 
                        g = 40; 
                        b = Math.min(255, 40 - v * 215); 
                    }
                } else if (mode === 'fractal') {
                    const it = results[faceIdx * strideFace + i_node];
                    r = (Math.sin(it * 10.0 + 0.0) * 0.5 + 0.5) * 255;
                    g = (Math.sin(it * 10.0 + 2.0) * 0.5 + 0.5) * 255;
                    b = (Math.sin(it * 10.0 + 4.0) * 0.5 + 0.5) * 255;
                } else if (mode === 'jfa') {
                    const val = results[faceIdx * strideFace + i_node];
                    if (val > 0) {
                        const seedX = (val & 0xFFFF);
                        const seedY = (val >> 16) & 0xFFFF;
                        // Deterministic color from seed coordinates
                        r = (seedX * 13) % 255;
                        g = (seedY * 17) % 255;
                        b = (seedX * seedY * 7) % 255;
                    } else {
                        r = 0; g = 0; b = 0;
                    }
                } else {
                    const raw = faceIdx !== -1 ? results[faceIdx * strideFace + i_node] : results[i_node];
                    const val = Math.min(255, Math.max(0, (raw - offset) * scale * 255));
                    r = val; g = val; b = val;
                }

                imageData.data[i4] = r;
                imageData.data[i4+1] = g;
                imageData.data[i4+2] = b;
                imageData.data[i4+3] = 255;
            }
        }
        ctx.putImageData(imageData, 0, 0);
        
        const exportedData = {
            results: new Float32Array(results), 
            lx, ly, strideFace
        };

        readBuffer.unmap();
        readBuffer.destroy();

        return exportedData;
    }
}
