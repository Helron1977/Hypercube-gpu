import { VirtualChunk } from './GridAbstractions';
import { GridBoundaries } from '../types';

export enum BoundaryRoleID {
    JOINT = 0,
    CONTINUITY = 1,
    WALL = 2,
    INFLOW = 3,
    OUTFLOW = 4,
    SYMMETRY = 5,
    ABSORBING = 6,
    CLAMPED = 7,
    DIRICHLET = 8,
    NEUMANN = 9,
    MOVING_WALL = 10
}

export interface ResolvedTopology {
    leftRole: number;
    rightRole: number;
    topRole: number;
    bottomRole: number;
    frontRole: number;
    backRole: number;
}

export class TopologyResolver {
    resolve(
        vChunk: VirtualChunk,
        chunkLayout: { x: number; y: number; z?: number },
        globalBoundaries: GridBoundaries
    ): ResolvedTopology {
        const topology: ResolvedTopology = {
            leftRole: 0, rightRole: 0, topRole: 0, bottomRole: 0, frontRole: 0, backRole: 0
        };

        const faces: Array<'left' | 'right' | 'top' | 'bottom' | 'front' | 'back'> = 
            ['left', 'right', 'top', 'bottom', 'front', 'back'];

        for (const face of faces) {
            const joint = vChunk.joints.find(j => j.face === face);
            let role = 0;

            if (joint) {
                role = (joint.role === 'joint') ? BoundaryRoleID.CONTINUITY : this.mapRoleToID(joint.role);
            } else {
                const boundaries = globalBoundaries as Record<string, any>;
                const boundarySide = boundaries[face] || globalBoundaries.all || { role: 'wall' };
                role = this.mapRoleToID(boundarySide.role);
            }

            switch (face) {
                case 'left': topology.leftRole = role; break;
                case 'right': topology.rightRole = role; break;
                case 'top': topology.topRole = role; break;
                case 'bottom': topology.bottomRole = role; break;
                case 'front': topology.frontRole = role; break;
                case 'back': topology.backRole = role; break;
            }
        }
        return topology;
    }

    private mapRoleToID(role: string): number {
        switch (role) {
            case 'wall': return BoundaryRoleID.WALL;
            case 'inflow': return BoundaryRoleID.INFLOW;
            case 'outflow': return BoundaryRoleID.OUTFLOW;
            case 'periodic': return BoundaryRoleID.CONTINUITY;
            case 'joint': return BoundaryRoleID.CONTINUITY;
            case 'symmetry': return BoundaryRoleID.SYMMETRY;
            case 'absorbing': return BoundaryRoleID.ABSORBING;
            case 'dirichlet': return BoundaryRoleID.DIRICHLET;
            case 'neumann': return BoundaryRoleID.NEUMANN;
            case 'clamped': return BoundaryRoleID.CLAMPED;
            case 'moving_wall': return BoundaryRoleID.MOVING_WALL;
            default: return BoundaryRoleID.WALL;
        }
    }
}
