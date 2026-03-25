import { GpuEngine } from './GpuEngine';

/**
 * Adapter for exporting Hypercube GPU results to the Model Context Protocol (MCP).
 * This allows external AI agents or tools to "consume" the simulation state.
 */
export class WebMcpAdapter {
    constructor(private engine: GpuEngine) {}

    /**
     * Exports the current state of a face as an MCP-compatible resource.
     */
    public async exportFaceToMcp(chunkId: string, faceName: string): Promise<any> {
        await this.engine.syncFacesToHost([faceName]);
        const data = this.engine.getFaceData(chunkId, faceName);

        return {
            uri: `mcp://hypercube/simulation/${chunkId}/${faceName}`,
            name: `${faceName} in ${chunkId}`,
            mimeType: "application/json",
            blob: {
                data: Array.from(data), // Serialized for MCP transport
                dimensions: this.engine.vGrid.dimensions
            }
        };
    }

    /**
     * Diagnostic shortcut to read face data from the console.
     */
    public async getFaceData(faceName: string, chunkId: string = 'chunk_0_0_0'): Promise<Float32Array> {
        await this.engine.syncFacesToHost([faceName]);
        return this.engine.getFaceData(chunkId, faceName);
    }

    /**
     * Diagnostic shortcut to write face data from the console.
     */
    public setFaceData(faceName: string, data: number[] | Float32Array, chunkId: string = 'chunk_0_0_0'): void {
        this.engine.setFaceData(chunkId, faceName, data);
        this.engine.syncToDevice();
    }

    /**
     * Provides a summary of the engine's current health and configuration.
     */
    public async exportDiagnosticsToMcp(): Promise<any> {
        return {
            engine: this.engine.vGrid.dataContract.descriptor.name,
            topology: this.engine.vGrid.chunkLayout,
            status: "Running",
            memoryUsage: `${(this.engine.buffer.byteLength / 1024 / 1024).toFixed(2)} MB`
        };
    }
}
