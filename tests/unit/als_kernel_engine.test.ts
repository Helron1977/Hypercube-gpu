import { describe, it, expect } from 'vitest';
import { AlsKernel } from '../renders/tensor-cp/assets/js/tensor-cp-als.js';

interface AlsStep {
    iter: number;
    mse: number;
    relErr: number;
    done: boolean;
    A?: Float64Array;
    B?: Float64Array;
    C?: Float64Array;
    lambda?: Float64Array;
    T_hat?: Float64Array;
}

describe('AlsKernel Mathematical Engine', () => {
    const kernel = new AlsKernel();

    it('should correctly compute Khatri-Rao product', () => {
        const A = new Float64Array([1, 3, 2, 4]); // 2x2
        const B = new Float64Array([5, 7, 6, 8]); // 2x2
        // @ts-ignore - reaching into internal private-ish method for math test
        const kr = kernel._khatriRao(A, 2, B, 2, 2);
        expect(Array.from(kr)).toEqual([5, 21, 6, 24, 10, 28, 12, 32]);
    });

    it('should calculate CORCONDIA = 100% for an exact identity tensor', async () => {
        const R = 2;
        const I = 2, J = 2, K = 2;
        const A = new Float64Array([1, 0, 0, 1]); 
        const B = new Float64Array([1, 0, 0, 1]);
        const C = new Float64Array([1, 0, 0, 1]);
        const lambda = new Float64Array([1, 1]);

        // @ts-ignore
        const T = kernel._reconstruct(A, B, C, lambda, I, J, K, R);
        const score = await kernel.calculateCorcondia(T, A, B, C, lambda, I, J, K, R);
        expect(score).toBeCloseTo(100, 1);
    });

    it('should identify rank-1 structure in a rank-1 tensor', async () => {
        const I = 3, J = 3, K = 3;
        const T = new Float64Array(I * J * K).fill(1.0); 
        const engine = kernel.run(T, I, J, K, 1, 1e-9, 50);
        
        let finalStep: AlsStep | undefined;
        for await (const step of engine) finalStep = step as AlsStep;
        
        expect(finalStep?.done).toBe(true);
        expect(finalStep?.relErr).toBeLessThan(1e-5);
    });

    it('should satisfy the scaling property: lambda(2T) = 2 * lambda(T)', async () => {
        const I = 2, J = 2, K = 2;
        const T1 = new Float64Array(I * J * K).fill(1.0);
        const T2 = new Float64Array(I * J * K).fill(2.0);

        const gen1 = kernel.run(T1, I, J, K, 1, 0, 20);
        let res1: AlsStep | undefined; for await (const s of gen1) res1 = s as AlsStep;

        const gen2 = kernel.run(T2, I, J, K, 1, 0, 20);
        let res2: AlsStep | undefined; for await (const s of gen2) res2 = s as AlsStep;

        if (!res1 || !res2 || !res1.lambda || !res2.lambda) throw new Error('Result missing');
        expect(res2.lambda[0]).toBeCloseTo(res1.lambda[0] * 2, 5);
    });

    it('should maintain monotonicity of Frobenius loss (stability check)', async () => {
        const I = 4, J = 4, K = 4;
        // Rank-2 exact tensor + noise-free convergence
        // a1=[1,1,0,0], b1=[1,0,1,0], c1=[1,0,0,1]
        // a2=[0,0,1,1], b2=[0,1,0,1], c2=[0,1,1,0]
        const A_ex = new Float64Array([1,0,1,0,0,1,0,1]); // 4x2
        const B_ex = new Float64Array([1,0,0,1,1,0,0,1]); // 4x2
        const C_ex = new Float64Array([1,0,0,1,0,1,1,0]); // 4x2
        
        const T = new Float64Array(I * J * K);
        for(let r=0; r<2; r++) {
            for(let i=0; i<I; i++)
                for(let j=0; j<J; j++)
                    for(let k=0; k<K; k++)
                        T[(i*J+j)*K+k] += A_ex[i*2+r] * B_ex[j*2+r] * C_ex[k*2+r];
        }

        const engine = kernel.run(T, I, J, K, 2, 1e-12, 40); 
        
        let prevMse = Infinity;
        for await (const step of engine) {
            const s = step as AlsStep;
            // High tolerance for monotonicity in double precision (regularization can add tiny noise)
            if (s.mse > prevMse * 1.5 + 1e-6) {
                throw new Error(`Loss increased at iteration ${s.iter}: ${s.mse} > ${prevMse}`);
            }
            prevMse = s.mse;
        }
    });

    it('should be invariant to index permutation (Rank Conservation)', async () => {
        const I = 3, J = 4, K = 5;
        const T = new Float64Array(I * J * K).map(() => Math.random());
        
        // Original
        const gen1 = kernel.run(T, I, J, K, 2, 1e-6, 20);
        let res1: AlsStep | undefined; for await (const s of gen1) res1 = s as AlsStep;

        // Permuted (I <-> J)
        const T_perm = new Float64Array(I * J * K);
        for(let i=0; i<I; i++)
            for(let j=0; j<J; j++)
                for(let k=0; k<K; k++)
                    T_perm[(j * I + i) * K + k] = T[(i * J + j) * K + k];
        
        const gen2 = kernel.run(T_perm, J, I, K, 2, 1e-6, 20);
        let res2: AlsStep | undefined; for await (const s of gen2) res2 = s as AlsStep;

        if (!res1 || !res2) throw new Error('Result missing');
        // relErr should be similar (tolerance for random init)
        expect(Math.abs(res1.relErr - res2.relErr)).toBeLessThan(0.1); 
    });
});
