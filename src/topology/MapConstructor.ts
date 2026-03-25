import { IMapConstructor, VirtualChunk, JointDescriptor } from './GridAbstractions';
import { Dimension3D, GridBoundaries } from '../types';

export class MapConstructor implements IMapConstructor {
    buildMap(
        dims: Dimension3D,
        chunks: { x: number; y: number; z?: number },
        globalBoundaries: GridBoundaries
    ): VirtualChunk[] {
        const virtualChunks: VirtualChunk[] = [];
        const numZ = chunks.z || 1;

        const getLocalSize = (total: number, count: number, index: number) => {
            const base = Math.floor(total / count);
            const remainder = total % count;
            return index < remainder ? base + 1 : base;
        };

        for (let cz = 0; cz < numZ; cz++) {
            const localNz = getLocalSize(dims.nz || 1, numZ, cz);
            for (let cy = 0; cy < chunks.y; cy++) {
                const localNy = getLocalSize(dims.ny, chunks.y, cy);
                for (let cx = 0; cx < chunks.x; cx++) {
                    const localNx = getLocalSize(dims.nx, chunks.x, cx);
                    const chunkId = `chunk_${cx}_${cy}_${cz}`;
                    const joints: JointDescriptor[] = [];

                    const faces: JointDescriptor['face'][] = [
                        'left', 'right', 'top', 'bottom', 'front', 'back',
                        'top-left', 'top-right', 'bottom-left', 'bottom-right',
                        'front-left', 'front-right', 'front-top', 'front-bottom',
                        'back-left', 'back-right', 'back-top', 'back-bottom',
                        'front-top-left', 'front-top-right', 'front-bottom-left', 'front-bottom-right',
                        'back-top-left', 'back-top-right', 'back-bottom-left', 'back-bottom-right'
                    ];

                    faces.forEach(f => {
                        const joint = this.deduceJoint(cx, cy, cz, f, chunks, globalBoundaries);
                        if (joint) joints.push(joint);
                    });

                    virtualChunks.push({
                        x: cx, y: cy, z: cz, id: chunkId, joints,
                        localDimensions: { nx: localNx, ny: localNy, nz: localNz }
                    });
                }
            }
        }
        return virtualChunks;
    }

    private deduceJoint(
        cx: number, cy: number, cz: number,
        face: JointDescriptor['face'] | any,
        chunks: { x: number; y: number; z?: number },
        globalBoundaries: GridBoundaries
    ): JointDescriptor | undefined {
        let dx = 0; let dy = 0; let dz = 0;
        switch (face) {
            case 'left': dx = -1; break;
            case 'right': dx = 1; break;
            case 'top': dy = -1; break;
            case 'bottom': dy = 1; break;
            case 'front': dz = -1; break;
            case 'back': dz = 1; break;

            case 'top-left': dx = -1; dy = -1; break;
            case 'top-right': dx = 1; dy = -1; break;
            case 'bottom-left': dx = -1; dy = 1; break;
            case 'bottom-right': dx = 1; dy = 1; break;

            case 'front-left': dx = -1; dz = -1; break;
            case 'front-right': dx = 1; dz = -1; break;
            case 'front-top': dy = -1; dz = -1; break;
            case 'front-bottom': dy = 1; dz = -1; break;

            case 'back-left': dx = -1; dz = 1; break;
            case 'back-right': dx = 1; dz = 1; break;
            case 'back-top': dy = -1; dz = 1; break;
            case 'back-bottom': dy = 1; dz = 1; break;

            case 'front-top-left': dx = -1; dy = -1; dz = -1; break;
            case 'front-top-right': dx = 1; dy = -1; dz = -1; break;
            case 'front-bottom-left': dx = -1; dy = 1; dz = -1; break;
            case 'front-bottom-right': dx = 1; dy = 1; dz = -1; break;

            case 'back-top-left': dx = -1; dy = -1; dz = 1; break;
            case 'back-top-right': dx = 1; dy = -1; dz = 1; break;
            case 'back-bottom-left': dx = -1; dy = 1; dz = 1; break;
            case 'back-bottom-right': dx = 1; dy = 1; dz = 1; break;
        }

        let nx = cx + dx;
        let ny = cy + dy;
        let nz = cz + dz;
        const numZ = chunks.z || 1;

        const wrap = (val: number, max: number) => (val + max) % max;
        const isOutX = nx < 0 || nx >= chunks.x;
        const isOutY = ny < 0 || ny >= chunks.y;
        const isOutZ = nz < 0 || nz >= numZ;

        if (!isOutX && !isOutY && !isOutZ) {
            return { role: 'joint', face, neighborId: `chunk_${nx}_${ny}_${nz}` };
        }

        const bounds = globalBoundaries || {};
        const isPeriodic = (dirRole: any) => dirRole === 'periodic' || bounds.all?.role === 'periodic';

        const isXPeriodic = (dx === -1 && isPeriodic(bounds.left?.role)) || (dx === 1 && isPeriodic(bounds.right?.role));
        const isYPeriodic = (dy === -1 && isPeriodic(bounds.top?.role)) || (dy === 1 && isPeriodic(bounds.bottom?.role));
        const isZPeriodic = (dz === -1 && isPeriodic(bounds.front?.role)) || (dz === 1 && isPeriodic(bounds.back?.role));

        if ((isOutX && !isXPeriodic) || (isOutY && !isYPeriodic) || (isOutZ && !isZPeriodic)) {
            if (dx !== 0 && dy === 0 && dz === 0 && isOutX) return { role: (dx === -1 ? bounds.left : bounds.right)?.role || 'wall', face };
            if (dy !== 0 && dx === 0 && dz === 0 && isOutY) return { role: (dy === -1 ? bounds.top : bounds.bottom)?.role || 'wall', face };
            if (dz !== 0 && dx === 0 && dy === 0 && isOutZ) return { role: (dz === -1 ? bounds.front : bounds.back)?.role || 'wall', face };
            return undefined;
        }

        return { 
            role: 'joint', face, 
            neighborId: `chunk_${isOutX ? wrap(nx, chunks.x) : nx}_${isOutY ? wrap(ny, chunks.y) : ny}_${isOutZ ? wrap(nz, numZ) : nz}` 
        };
    }
}
