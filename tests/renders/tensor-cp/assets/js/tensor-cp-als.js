/**
 * AlsKernel — Tensor CP Decomposition via Alternating Least Squares
 * 
 * Pure math module. No UI dependencies, no DOM, no Chart.js.
 * All async methods yield control via setTimeout(0) to keep the
 * browser responsive — but the caller decides when to update UI.
 *
 * Public API:
 *   async *run(T_obs, I, J, K, R, lambda_reg, maxIter, sparseEntries?)
 *   async calculateCorcondia(T_obs, A, B, C, lambda, I, J, K, R, sparseEntries?)
 */
export class AlsKernel {

    // ─── MAIN ALS LOOP ────────────────────────────────────────────────────────

    async *run(T_obs, I, J, K, R, lambda_reg, maxIter, sparseEntries = null) {
        // Random init
        let A = new Float64Array(I * R).map(() => Math.random() * 2 - 1);
        let B = new Float64Array(J * R).map(() => Math.random() * 2 - 1);
        let C = new Float64Array(K * R).map(() => Math.random() * 2 - 1);
        let lambda = new Float64Array(R).fill(1);

        const normTSq = sparseEntries
            ? sparseEntries.reduce((s, e) => s + e.v * e.v, 0)
            : this._frobSq(T_obs, I * J * K);
        const normT = Math.sqrt(normTSq);

        // Precompute unfoldings once (dense path only)
        let T1, T2, T3;
        if (!sparseEntries) {
            T1 = this._unfold(T_obs, I, J, K, 0);
            T2 = this._unfold(T_obs, I, J, K, 1);
            T3 = this._unfold(T_obs, I, J, K, 2);
        }

        for (let iter = 0; iter < maxIter; iter++) {

            // ── Update A ──
            {
                const V = this._hadamard(this._gram(B, J, R), this._gram(C, K, R), R);
                this._addReg(V, R, lambda_reg);
                const M = sparseEntries
                    ? this._mttkrpSparse(sparseEntries, B, C, I, R, 0)
                    : this._mttkrp(T1, this._khatriRao(B, J, C, K, R), I, J * K, R);
                A.set(this._solveRHS(M, V, I, R));
            }

            // ── Update B ──
            // KR for mode-1: C ⊗ A (C outer, A inner — matches T_(2) column indexing)
            {
                const V = this._hadamard(this._gram(A, I, R), this._gram(C, K, R), R);
                this._addReg(V, R, lambda_reg);
                const M = sparseEntries
                    ? this._mttkrpSparse(sparseEntries, A, C, J, R, 1)
                    : this._mttkrp(T2, this._khatriRao(C, K, A, I, R), J, I * K, R);
                B.set(this._solveRHS(M, V, J, R));
            }

            // ── Update C ──
            {
                const V = this._hadamard(this._gram(A, I, R), this._gram(B, J, R), R);
                this._addReg(V, R, lambda_reg);
                const M = sparseEntries
                    ? this._mttkrpSparse(sparseEntries, A, B, K, R, 2)
                    : this._mttkrp(T3, this._khatriRao(A, I, B, J, R), K, I * J, R);
                C.set(this._solveRHS(M, V, K, R));
            }

            // ── Normalize — lambda = product of all three column norms ──
            const lA = this._normalizeColumns(A, I, R);
            const lB = this._normalizeColumns(B, J, R);
            const lC = this._normalizeColumns(C, K, R);
            for (let r = 0; r < R; r++) lambda[r] = lA[r] * lB[r] * lC[r];

            // ── Relative error ──
            const errSq = sparseEntries
                ? Math.max(0, normTSq + this._frobSqFactors(A, B, C, lambda, I, J, K, R)
                    - 2 * this._dotSparse(sparseEntries, A, B, C, lambda, R))
                : this._errSqDense(T_obs, A, B, C, lambda, I, J, K, R);

            const relErr = Math.sqrt(errSq) / (normT + 1e-12);

            // Yield progress — no T_hat during iterations (too expensive every step)
            yield { iter, mse: errSq / (I * J * K), relErr, done: false };

            // Yield control to browser every ~16ms
            if (iter % 3 === 0) await new Promise(r => setTimeout(r, 0));
        }

        // ── Final yield with full reconstruction ──
        const T_hat = this._reconstruct(A, B, C, lambda, I, J, K, R);
        const finalErrSq = sparseEntries
            ? Math.max(0, normTSq + this._frobSqFactors(A, B, C, lambda, I, J, K, R)
                - 2 * this._dotSparse(sparseEntries, A, B, C, lambda, R))
            : this._errSqDense(T_obs, A, B, C, lambda, I, J, K, R);
        const finalRelErr = Math.sqrt(finalErrSq) / (normT + 1e-12);

        yield {
            iter: maxIter,
            mse: finalErrSq / (I * J * K),
            relErr: finalRelErr,
            A, B, C, lambda, T_hat,
            done: true
        };
    }

    // ─── CORCONDIA ────────────────────────────────────────────────────────────

    async calculateCorcondia(T_obs, A, B, C, lambda, I, J, K, R, sparseEntries = null) {
        const pinvA = this._pseudoInverse(A, I, R);
        const pinvB = this._pseudoInverse(B, J, R);
        const pinvC = this._pseudoInverse(C, K, R);

        // Step 1: T' = T ×₁ pinv(A)  →  shape R × J × K
        const Tp = new Float64Array(R * J * K);
        if (sparseEntries) {
            for (const e of sparseEntries) {
                for (let r = 0; r < R; r++) {
                    Tp[r * J * K + e.j * K + e.k] += e.v * pinvA[r * I + e.i];
                }
            }
        } else {
            for (let r = 0; r < R; r++) {
                for (let j = 0; j < J; j++) {
                    for (let k = 0; k < K; k++) {
                        let s = 0;
                        for (let i = 0; i < I; i++) s += T_obs[i * J * K + j * K + k] * pinvA[r * I + i];
                        Tp[r * J * K + j * K + k] = s;
                    }
                }
            }
        }
        await new Promise(r => setTimeout(r, 0));

        // Step 2: T'' = T' ×₂ pinv(B)  →  shape R × R × K
        const Tpp = new Float64Array(R * R * K);
        for (let r1 = 0; r1 < R; r1++) {
            for (let r2 = 0; r2 < R; r2++) {
                for (let k = 0; k < K; k++) {
                    let s = 0;
                    for (let j = 0; j < J; j++) s += Tp[r1 * J * K + j * K + k] * pinvB[r2 * J + j];
                    Tpp[r1 * R * K + r2 * K + k] = s;
                }
            }
        }
        await new Promise(r => setTimeout(r, 0));

        // Step 3: G = T'' ×₃ pinv(C)  →  R × R × R
        // Compare G to super-diagonal scaled by lambda
        let diffSq = 0;
        let normTargetSq = 0;
        for (let r1 = 0; r1 < R; r1++) {
            for (let r2 = 0; r2 < R; r2++) {
                for (let r3 = 0; r3 < R; r3++) {
                    let g = 0;
                    for (let k = 0; k < K; k++) g += Tpp[r1 * R * K + r2 * K + k] * pinvC[r3 * K + k];
                    const target = (r1 === r2 && r2 === r3) ? lambda[r1] : 0.0;
                    diffSq += (g - target) ** 2;
                    normTargetSq += target * target;
                }
            }
        }

        // CORCONDIA score: 100% = perfect super-diagonal core
        return Math.max(0, 100 * (1 - diffSq / (normTargetSq + 1e-12)));
    }

    // ─── PRIVATE MATH ─────────────────────────────────────────────────────────

    _unfold(T, I, J, K, mode) {
        if (mode === 0) {
            // T_(1): rows=I, cols=J*K — row-major copy (no-op for row-major tensor)
            return Float32Array.from(T);
        } else if (mode === 1) {
            // T_(2): rows=J, cols=I*K — column ordering: (i,k) for each j
            const M = new Float32Array(J * I * K);
            for (let j = 0; j < J; j++)
                for (let i = 0; i < I; i++)
                    for (let k = 0; k < K; k++)
                        M[j * I * K + i * K + k] = T[i * J * K + j * K + k];
            return M;
        } else {
            // T_(3): rows=K, cols=I*J
            const M = new Float32Array(K * I * J);
            for (let k = 0; k < K; k++)
                for (let i = 0; i < I; i++)
                    for (let j = 0; j < J; j++)
                        M[k * I * J + i * J + j] = T[i * J * K + j * K + k];
            return M;
        }
    }

    // Khatri-Rao product: A ⊗ B, result shape (m*n) × R
    _khatriRao(A, m, B, n, R) {
        const KR = new Float64Array(m * n * R);
        for (let i = 0; i < m; i++)
            for (let j = 0; j < n; j++)
                for (let r = 0; r < R; r++)
                    KR[(i * n + j) * R + r] = A[i * R + r] * B[j * R + r];
        return KR;
    }

    // Gram matrix: G[r1,r2] = Σᵢ A[i,r1] * A[i,r2]
    _gram(A, rows, R) {
        const G = new Float64Array(R * R);
        for (let r1 = 0; r1 < R; r1++)
            for (let r2 = 0; r2 < R; r2++) {
                let s = 0;
                for (let i = 0; i < rows; i++) s += A[i * R + r1] * A[i * R + r2];
                G[r1 * R + r2] = s;
            }
        return G;
    }

    // Element-wise product of two R×R matrices
    _hadamard(G1, G2, R) {
        const H = new Float64Array(R * R);
        for (let i = 0; i < R * R; i++) H[i] = G1[i] * G2[i];
        return H;
    }

    // Add regularization to diagonal in-place
    _addReg(V, R, reg) {
        // Essential floor for numerical stability even if reg is 0
        const eps = Math.max(reg, 1e-12);
        for (let r = 0; r < R; r++) V[r * R + r] += eps;
    }

    // Gauss-Jordan inversion of R×R matrix
    _invertR(V, R) {
        const aug = new Float64Array(R * 2 * R);
        for (let i = 0; i < R; i++) {
            for (let j = 0; j < R; j++) aug[i * (2 * R) + j] = V[i * R + j];
            aug[i * (2 * R) + R + i] = 1.0;
        }
        for (let col = 0; col < R; col++) {
            // Partial pivoting
            let maxVal = Math.abs(aug[col * (2 * R) + col]), maxRow = col;
            for (let row = col + 1; row < R; row++) {
                const v = Math.abs(aug[row * (2 * R) + col]);
                if (v > maxVal) { maxVal = v; maxRow = row; }
            }
            if (maxRow !== col) {
                for (let c = 0; c < 2 * R; c++) {
                    const t = aug[col * (2 * R) + c];
                    aug[col * (2 * R) + c] = aug[maxRow * (2 * R) + c];
                    aug[maxRow * (2 * R) + c] = t;
                }
            }
            const pivot = aug[col * (2 * R) + col];
            if (Math.abs(pivot) < 1e-12) continue; // singular column, skip
            for (let c = 0; c < 2 * R; c++) aug[col * (2 * R) + c] /= pivot;
            for (let row = 0; row < R; row++) {
                if (row === col) continue;
                const f = aug[row * (2 * R) + col];
                for (let c = 0; c < 2 * R; c++) aug[row * (2 * R) + c] -= f * aug[col * (2 * R) + c];
            }
        }
        const inv = new Float64Array(R * R);
        for (let i = 0; i < R; i++)
            for (let j = 0; j < R; j++)
                inv[i * R + j] = aug[i * (2 * R) + R + j];
        return inv;
    }

    // MTTKRP (dense): M = T_(n) × KR
    _mttkrp(Tunfold, KR, dimN, colsKR, R) {
        const M = new Float64Array(dimN * R);
        for (let i = 0; i < dimN; i++)
            for (let r = 0; r < R; r++) {
                let s = 0;
                for (let c = 0; c < colsKR; c++) {
                    const val = Tunfold[i * colsKR + c];
                    if (val !== 0) s += val * KR[c * R + r];
                }
                M[i * R + r] = s;
            }
        return M;
    }

    // MTTKRP (sparse): avoids materializing the unfolded tensor
    // mode 0: update A → M[i] += v * B[j] * C[k]
    // mode 1: update B → M[j] += v * A[i] * C[k]
    // mode 2: update C → M[k] += v * A[i] * B[j]
    _mttkrpSparse(entries, M1, M2, dimN, R, mode) {
        const M = new Float64Array(dimN * R);
        for (const e of entries) {
            const [n, p, q] = mode === 0 ? [e.i, e.j, e.k]
                : mode === 1 ? [e.j, e.i, e.k]
                    : [e.k, e.i, e.j];
            for (let r = 0; r < R; r++)
                M[n * R + r] += e.v * M1[p * R + r] * M2[q * R + r];
        }
        return M;
    }

    // Solve X = M × V⁻¹
    _solveRHS(M, V, numRows, R) {
        const Vinv = this._invertR(V, R);
        const X = new Float64Array(numRows * R);
        for (let i = 0; i < numRows; i++)
            for (let r = 0; r < R; r++) {
                let s = 0;
                for (let c = 0; c < R; c++) s += M[i * R + c] * Vinv[c * R + r];
                X[i * R + r] = s;
            }
        return X;
    }

    // Normalize columns of A to unit norm, return norms
    _normalizeColumns(A, rows, R) {
        const norms = new Float64Array(R);
        for (let r = 0; r < R; r++) {
            let norm = 0;
            for (let i = 0; i < rows; i++) norm += A[i * R + r] ** 2;
            norms[r] = Math.sqrt(norm) || 1e-12;
            for (let i = 0; i < rows; i++) A[i * R + r] /= norms[r];
        }
        return norms;
    }

    // Pseudoinverse: pinv(M) = (MᵀM + εI)⁻¹ Mᵀ
    _pseudoInverse(M, rows, cols) {
        const MtM = this._gram(M, rows, cols);
        for (let r = 0; r < cols; r++) MtM[r * cols + r] += 1e-10; // regularize
        const invMtM = this._invertR(MtM, cols);
        const pinv = new Float64Array(cols * rows);
        for (let r = 0; r < cols; r++)
            for (let i = 0; i < rows; i++) {
                let s = 0;
                for (let c = 0; c < cols; c++) s += invMtM[r * cols + c] * M[i * cols + c];
                pinv[r * rows + i] = s;
            }
        return pinv;
    }

    // Reconstruct dense tensor from factors
    _reconstruct(A, B, C, lambda, I, J, K, R) {
        const T = new Float64Array(I * J * K);
        for (let i = 0; i < I; i++)
            for (let j = 0; j < J; j++)
                for (let k = 0; k < K; k++) {
                    let s = 0;
                    for (let r = 0; r < R; r++) s += lambda[r] * A[i * R + r] * B[j * R + r] * C[k * R + r];
                    T[i * J * K + j * K + k] = s;
                }
        return T;
    }

    _errSqDense(T_obs, A, B, C, lambda, I, J, K, R) {
        const T_hat = this._reconstruct(A, B, C, lambda, I, J, K, R);
        let s = 0;
        for (let x = 0; x < I * J * K; x++) {
            const d = T_hat[x] - T_obs[x];
            s += d * d;
        }
        return s;
    }

    // Dot product <T, X_CP> for dense T
    _dotDense(T1, A, B, C, lambda, I, J, K, R) {
        const KR = this._khatriRao(B, J, C, K, R);
        const M = this._mttkrp(T1, KR, I, J * K, R);
        let s = 0;
        for (let i = 0; i < I; i++)
            for (let r = 0; r < R; r++)
                s += lambda[r] * A[i * R + r] * M[i * R + r];
        return s;
    }

    // ‖X_CP‖² = Σᵣₛ λᵣλₛ (aᵣ·aₛ)(bᵣ·bₛ)(cᵣ·cₛ)
    _frobSqFactors(A, B, C, lambda, I, J, K, R) {
        const gA = this._gram(A, I, R);
        const gB = this._gram(B, J, R);
        const gC = this._gram(C, K, R);
        let s = 0;
        for (let r = 0; r < R; r++)
            for (let sc = 0; sc < R; sc++)
                s += lambda[r] * lambda[sc] * gA[r * R + sc] * gB[r * R + sc] * gC[r * R + sc];
        return s;
    }

    // ⟨T, X_CP⟩ over sparse entries
    _dotSparse(entries, A, B, C, lambda, R) {
        let s = 0;
        for (const e of entries) {
            let tr = 0;
            for (let r = 0; r < R; r++) tr += lambda[r] * A[e.i * R + r] * B[e.j * R + r] * C[e.k * R + r];
            s += e.v * tr;
        }
        return s;
    }

    _frobSq(T, n) {
        let s = 0;
        for (let i = 0; i < n; i++) s += T[i] * T[i];
        return s;
    }
}
