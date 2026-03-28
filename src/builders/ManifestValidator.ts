import { HypercubeManifest } from '../types';

/**
 * Validates a Hypercube Manifest for common errors before runtime.
 */
export class ManifestValidator {
    /**
     * Validates a manifest and throws an error if issues are found.
     */
    public static validate(manifest: HypercubeManifest): void {
        const { engine, config } = manifest;

        // 1. Linking Check
        if (config.engine !== engine.name) {
            throw new Error(`[ManifestValidator] Linkage Error: config.engine ('${config.engine}') does not match engine.name ('${engine.name}').`);
        }

        // 2. Face Integrity
        const faceNames = new Set<string>();
        engine.faces.forEach(face => {
            if (faceNames.has(face.name)) {
                throw new Error(`[ManifestValidator] Duplicate Face Error: Face '${face.name}' is defined multiple times.`);
            }
            faceNames.add(face.name);
        });

        // 3. Rule Consistency
        if (engine.rules.length === 0) {
            console.warn("[ManifestValidator] Warning: No rules defined. The simulation will not compute anything.");
        }

        engine.rules.forEach(rule => {
            if (rule.faces) {
                rule.faces.forEach(fName => {
                    const baseName = fName.split('.')[0];
                    if (!faceNames.has(baseName)) {
                        throw new Error(`[ManifestValidator] Missing Face Error: Rule '${rule.type}' references unknown face '${baseName}' (full name: '${fName}').`);
                    }
                });
            }
        });

        // 4. Dimensions
        if (config.dimensions.nx <= 0 || config.dimensions.ny <= 0) {
            throw new Error(`[ManifestValidator] Dimension Error: nx and ny must be greater than zero.`);
        }

        // 5. Chunks
        if (config.chunks.x <= 0 || config.chunks.y <= 0) {
            throw new Error(`[ManifestValidator] Chunk Error: chunks.x and chunks.y must be greater than zero.`);
        }

        console.info(`[ManifestValidator] Manifest '${manifest.name}' validated successfully.`);
    }
}
