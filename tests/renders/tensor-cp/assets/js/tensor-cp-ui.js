/**
 * tensor-cp-ui.js
 * UI Controller for Tensor CP Decomposition Dashboard
 *
 * Architecture:
 *   DataManager   — parsing, sizing, state
 *   ChartManager  — chart init, reset, update (two modes: history / scan)
 *   HeatmapRenderer — canvas drawing
 *   UiController  — orchestrates the above, handles events
 */

import { AlsKernel } from './tensor-cp-als.js';

// ─── DATA MANAGER ─────────────────────────────────────────────────────────────

class DataManager {
    constructor() {
        this.reset();
    }

    reset() {
        this.T = null;   // Float32Array, dense tensor
        this.entries = null;   // sparse entries [{i,j,k,v}]
        this.I = this.J = this.K = 0;
        this.label = '';
    }

    get cells() { return this.I * this.J * this.K; }
    get isLoaded() { return this.T !== null; }

    // Returns suggested { R, label } based on tensor size
    autoSize() {
        const cells = this.cells;
        // Budget: keep one ALS iteration under ~50ms
        // Cost per iter ≈ cells * R * 4 ops, at ~500k ops/ms
        const rMax = Math.floor(50 * 500_000 / (cells * 4 + 1));
        const R = Math.max(3, Math.min(15, rMax));
        if (cells < 10_000) return { R, label: 'rapide <1ms/iter' };
        if (cells < 100_000) return { R, label: 'interactif ~10ms/iter' };
        if (cells < 500_000) return { R, label: 'lent ~50ms/iter' };
        return { R: Math.min(R, 3), label: '⚠ très large — R limité' };
    }

    loadFromText(text, sourceName) {
        const isTns = sourceName.toLowerCase().endsWith('.tns');
        const lines = text.trim().split('\n');
        let maxI = 0, maxJ = 0, maxK = 0;
        const entries = [];

        for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed[0] === '%' || trimmed[0] === '#') continue;
            const cols = trimmed.includes(',') ? trimmed.split(',') : trimmed.split(/\s+/);
            if (cols.length < 4) continue;

            let i = parseInt(cols[0]);
            let j = parseInt(cols[1]);
            let k = parseInt(cols[2]);
            const v = parseFloat(cols[3]);

            if (isNaN(i) || isNaN(j) || isNaN(k) || isNaN(v)) continue;
            if (isTns) { i--; j--; k--; }
            if (i < 0 || j < 0 || k < 0) continue;

            maxI = Math.max(maxI, i + 1);
            maxJ = Math.max(maxJ, j + 1);
            maxK = Math.max(maxK, k + 1);
            entries.push({ i, j, k, v });
        }

        if (maxI === 0 || maxJ === 0 || maxK === 0)
            throw new Error(`Tenseur invalide (${maxI}×${maxJ}×${maxK})`);

        this.I = maxI; this.J = maxJ; this.K = maxK;
        this.entries = entries;
        this.T = new Float32Array(maxI * maxJ * maxK);
        for (const { i, j, k, v } of entries)
            this.T[i * maxJ * maxK + j * maxK + k] = v;
        this.label = sourceName;
        return this;
    }

    loadSynthetic(I = 20, J = 20, K = 20) {
        this.I = I; this.J = J; this.K = K;
        const T = new Float32Array(I * J * K);
        const entries = [];
        for (let i = 0; i < I; i++)
            for (let j = 0; j < J; j++)
                for (let k = 0; k < K; k++) {
                    const v =
                        Math.sin(Math.PI * i / I) * Math.cos(Math.PI * j / J) * Math.exp(-k / K) +
                        0.5 * Math.cos(2 * Math.PI * i / I) * Math.sin(2 * Math.PI * j / J) * Math.cos(Math.PI * k / K) +
                        (Math.random() - 0.5) * 0.05;
                    T[i * J * K + j * K + k] = v;
                    entries.push({ i, j, k, v });
                }
        this.T = T;
        this.entries = entries;
        this.label = `Synthétique ${I}×${J}×${K}`;
        return this;
    }
}

// ─── CHART MANAGER ────────────────────────────────────────────────────────────

class ChartManager {
    constructor(canvasId) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        this._chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], datasets: [
                    {
                        label: 'CORCONDIA (%)',
                        data: [],
                        borderColor: '#00e5cc',
                        backgroundColor: 'rgba(0,229,204,0.08)',
                        yAxisID: 'yLeft',
                        tension: 0.3,
                        pointRadius: 3,
                    },
                    {
                        label: 'Loss (MSE)',
                        data: [],
                        borderColor: '#7b5ea7',
                        borderDash: [4, 4],
                        yAxisID: 'yRight',
                        tension: 0.3,
                        pointRadius: 2,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,        // disable for performance
                scales: {
                    yLeft: {
                        type: 'linear', position: 'left',
                        min: 0, max: 105,
                        title: { display: true, text: 'CORCONDIA %', color: '#00e5cc' },
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#71768a' }
                    },
                    yRight: {
                        type: 'logarithmic', position: 'right',
                        title: { display: true, text: 'MSE', color: '#7b5ea7' },
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#71768a' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#71768a' }
                    }
                },
                plugins: {
                    legend: { labels: { color: '#71768a', font: { size: 10 } } }
                }
            }
        });
    }

    // Reset chart data and configure for a given mode
    reset(mode = 'history') {
        const d = this._chart.data;
        d.labels = [];
        d.datasets[1].data = [];
        // Only clear CORCONDIA if explicitly changing mode or for a fresh scan
        if (mode === 'scan') d.datasets[0].data = []; 
        else d.datasets[0].data = new Array(d.labels.length).fill(null);

        const opts = this._chart.options.scales;
        
        // Reset scale bounds to force recalculation
        opts.yLeft.min = 0; opts.yLeft.max = 105;
        opts.yRight.min = undefined; 
        opts.yRight.max = undefined;
        opts.x.min = undefined; 
        opts.x.max = undefined;

        if (mode === 'history') {
            d.datasets[0].label = 'CORCONDIA (%)';
            d.datasets[1].label = 'Loss history (MSE)';
            opts.x.title = { display: true, text: 'Itération', color: '#71768a' };
            opts.yLeft.display = false; 
        } else { // scan mode
            d.datasets[0].label = 'CORCONDIA (%)';
            d.datasets[1].label = 'MSE final par rang';
            opts.x.title = { display: true, text: 'Rang R', color: '#71768a' };
            opts.yLeft.display = true;
        }
        this._chart.update('none');
    }

    // Push a history step (single run mode)
    pushHistory(iter, mse) {
        const d = this._chart.data;
        d.labels.push(iter);
        d.datasets[1].data.push(mse);
        // Throttle redraws
        if (iter % 5 === 0) this._chart.update('none');
    }

    // Push a scan result (rank scan mode)
    pushScan(R, corcondia, mse) {
        const d = this._chart.data;
        d.labels.push(R);
        d.datasets[0].data.push(corcondia);
        d.datasets[1].data.push(mse);
        this._chart.update('none');
    }

    // Set corcondia value on the last history point
    finalizeHistory(corcondia) {
        const d = this._chart.data;
        // Add a single corcondia point at end of run
        this._chart.options.scales.yLeft.display = true;
        d.datasets[0].data = new Array(d.labels.length).fill(null);
        d.datasets[0].data[d.labels.length - 1] = corcondia;
        this._chart.update('none');
    }

    onClickRank(callback) {
        this._chart.options.onClick = (e, items) => {
            if (items.length > 0) {
                const R = this._chart.data.labels[items[0].index];
                if (typeof R === 'number') callback(R);
            }
        };
    }
}

// ─── HEATMAP RENDERER ─────────────────────────────────────────────────────────

class HeatmapRenderer {
    static draw(canvas, data, I, J, K, k) {
        const ctx = canvas.getContext('2d');
        const W = 256, H = 256;
        canvas.width = W;
        canvas.height = H;
        const img = ctx.createImageData(W, H);

        // Compute min/max on current slice only
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < I; i++)
            for (let j = 0; j < J; j++) {
                const v = data[i * J * K + j * K + k];
                if (v < min) min = v;
                if (v > max) max = v;
            }

        const range = max - min + 1e-12;
        const stepI = I / H;
        const stepJ = J / W;

        for (let y = 0; y < H; y++) {
            const iLo = Math.floor(y * stepI);
            const iHi = Math.max(iLo + 1, Math.floor((y + 1) * stepI));
            for (let x = 0; x < W; x++) {
                const jLo = Math.floor(x * stepJ);
                const jHi = Math.max(jLo + 1, Math.floor((x + 1) * stepJ));

                // Max-pooling over the bin (preserves sparse points)
                let peak = -Infinity;
                for (let i = iLo; i < iHi; i++)
                    for (let j = jLo; j < jHi; j++) {
                        const v = data[i * J * K + j * K + k];
                        if (v > peak) peak = v;
                    }

                const t = (peak - min) / range;
                const p = (y * W + x) * 4;
                // Viridis-inspired: dark purple → teal → yellow
                img.data[p] = Math.round(68 + t * 187);   // R
                img.data[p + 1] = Math.round(1 + t * 229);   // G
                img.data[p + 2] = Math.round(84 - t * 32);    // B
                img.data[p + 3] = 255;
            }
        }
        ctx.putImageData(img, 0, 0);
    }
}

// ─── UI CONTROLLER ────────────────────────────────────────────────────────────

class UiController {
    constructor() {
        this._kernel = new AlsKernel();
        this._data = new DataManager();
        this._chart = new ChartManager('canvasDiagnostic');

        // Running state
        this._running = false;
        this._stopFlag = false;
        this._lastFactors = null;

        this._bindDom();
        this._bindEvents();

        // Allow clicking on scan chart to pick rank
        this._chart.onClickRank(R => {
            this._dom.paramR.value = R;
            this._dom.valR.textContent = R;
            this._log(`Rang sélectionné depuis graphique : R=${R}`);
        });
    }

    // ── DOM binding ──

    _bindDom() {
        const $ = id => document.getElementById(id);
        this._dom = {
            paramR: $('paramR'), valR: $('valR'),
            paramIter: $('paramIter'), valIter: $('valIter'),
            paramReg: $('paramReg'), valReg: $('valReg'),
            scanMin: $('scanMin'), scanMax: $('scanMax'),
            btnRun: $('runProcess'),
            btnStop: $('stopProcess'),
            btnScan: $('btnAutoScan'),
            btnUpload: $('uploadAction'),
            btnBench: $('benchAction'),
            btnDemo: $('btnDemo'),
            fileInput: $('fileInput'),
            logStream: $('logStream'),
            frobLoss: $('frobLoss'),
            corcondiaVal: $('corcondiaVal'),
            currentIter: $('currentIter'),
            relErr: $('relErr'),
            relErrBox: $('relErr')?.parentElement,
            sysStatus: $('sysStatus'),
            currentTask: $('currentTask'),
            sliceK: $('sliceK'),
            canvasObs: $('canvasObs'),
            canvasRec: $('canvasRec'),
            exportBtn: $('exportData'),
            
            // View switcher
            viewSim: $('viewSim'),
            viewDiag: $('viewDiag'),
            panelObs: $('panelObs'),
            panelRec: $('panelRec'),
            panelDiag: $('diagnosticPanel'),
            visGrid: document.querySelector('.visualization-grid'),
        };
    }

    _bindEvents() {
        const d = this._dom;

        // View Switcher
        d.viewSim.onclick = () => this._setView('simulation');
        d.viewDiag.onclick = () => this._setView('diagnostic');

        // Sliders — live display
        d.paramR.oninput = () => { d.valR.textContent = d.paramR.value; };
        d.paramIter.oninput = () => { d.valIter.textContent = d.paramIter.value; };
        d.paramReg.oninput = () => {
            d.valReg.textContent = Math.pow(10, parseFloat(d.paramReg.value)).toExponential(0);
        };

        // Data sources
        d.btnUpload.onclick = () => d.fileInput.click();
        d.fileInput.onchange = e => this._handleUpload(e);
        d.btnBench.onclick = () => this._loadBenchmark();
        d.btnDemo.onclick = () => this._loadSynthetic();

        // Compute
        d.btnRun.onclick = () => this._runAls();
        d.btnStop.onclick = () => { this._stopFlag = true; };
        d.btnScan.onclick = () => this._runScan();

        // Visualization
        d.sliceK.oninput = () => this._renderSlice();

        // Export
        d.exportBtn.onclick = () => this._exportFactors();
    }

    // ── Logging / status ──

    _log(msg, level = 'info') {
        const div = document.createElement('div');
        const color = level === 'err' ? '#ff6b6b' : level === 'warn' ? '#ffcc44' : '#71768a';
        const time = new Date().toLocaleTimeString();
        div.innerHTML = `<span style="color:${color}">[${time}]</span> ${msg}`;
        this._dom.logStream.appendChild(div);
        this._dom.logStream.scrollTop = this._dom.logStream.scrollHeight;
    }

    _setStatus(text, color = '#00e5cc') {
        this._dom.sysStatus.textContent = text;
        this._dom.sysStatus.style.color = color;
    }

    // ── Data loading ──

    async _handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;
        this._log(`Chargement : ${file.name}`);
        try {
            const text = await file.text();
            this._loadData(text, file.name);
        } catch (err) {
            this._log(`Erreur : ${err.message}`, 'err');
        }
    }

    async _loadBenchmark() {
        try {
            this._log('Chargement MovieLens Tags...');
            const res = await fetch('/tests/manifests/tensor_movielens_tags.csv');
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            this._loadData(await res.text(), 'MOVIELENS_TAGS');
        } catch (err) {
            this._log(`Benchmark indisponible (${err.message}) — chargement synthétique`, 'warn');
            this._loadSynthetic();
        }
    }

    _loadSynthetic() {
        this._data.loadSynthetic(20, 20, 20);
        this._onDataLoaded();
        this._log(`Tenseur synthétique 20×20×20 généré`);
    }

    _loadData(text, name) {
        this._data.loadFromText(text, name);
        this._onDataLoaded();
    }

    _onDataLoaded() {
        const { I, J, K } = this._data;
        const { R, label } = this._data.autoSize();
        this._dom.currentTask.textContent = this._data.label.toUpperCase();
        this._dom.sliceK.max = K - 1;
        this._dom.sliceK.value = 0;
        this._dom.paramR.value = R;
        this._dom.valR.textContent = R;
        this._log(`✓ ${I}×${J}×${K} — ${this._data.entries.length} entrées | R suggéré : ${R} (${label})`);
        this._lastFactors = null;
        this._renderSlice();
        this._setStatus('PRÊT');
    }

    // ── ALS single run ──

    async _runAls() {
        if (!this._data.isLoaded) { this._log('Aucune donnée chargée', 'err'); return; }
        if (this._running) return;

        this._running = true;
        this._stopFlag = false;
        this._lastFactors = null;
        this._toggleBusy(true);
        this._setStatus('CALCUL...', '#ffcc44');

        const R = parseInt(this._dom.paramR.value);
        const maxIter = parseInt(this._dom.paramIter.value);
        const reg = Math.pow(10, parseFloat(this._dom.paramReg.value));

        this._log(`ALS — R=${R}, iter=${maxIter}, reg=${reg.toExponential(1)}`);
        this._setView('simulation'); // Auto-switch to heatmap
        this._chart.reset('history');

        const { T, entries, I, J, K } = this._data;
        const gen = this._kernel.run(T, I, J, K, R, reg, maxIter, entries);

        for await (const step of gen) {
            if (this._stopFlag) { this._log('Arrêté par l\'utilisateur.'); break; }
            this._onStep(step);
        }

        // Compute CORCONDIA on final factors
        if (this._lastFactors) {
            this._log('Calcul CORCONDIA...');
            try {
                const { A, B, C, lambda } = this._lastFactors;
                const corcondia = await this._kernel.calculateCorcondia(
                    T, A, B, C, lambda, I, J, K, R, entries
                );
                this._dom.corcondiaVal.textContent = corcondia.toFixed(1) + '%';
                this._chart.finalizeHistory(corcondia);
                this._log(`CORCONDIA : ${corcondia.toFixed(2)}%`, corcondia > 80 ? 'info' : 'warn');
            } catch (err) {
                this._log(`CORCONDIA échoué : ${err.message}`, 'err');
            }
        }

        this._toggleBusy(false);
        this._setStatus('TERMINÉ', '#00e5cc');
        this._running = false;
    }

    // ── Rank scan ──

    async _runScan() {
        if (!this._data.isLoaded) return;
        if (this._running) return;

        this._running = true;
        this._stopFlag = false;
        this._toggleBusy(true);
        this._setStatus('SCAN...', '#7b5ea7');

        const Rmin = parseInt(this._dom.scanMin.value);
        const Rmax = parseInt(this._dom.scanMax.value);
        const maxIter = Math.min(50, parseInt(this._dom.paramIter.value));
        const reg = Math.pow(10, parseFloat(this._dom.paramReg.value));

        this._log(`Rank scan R=${Rmin}→${Rmax}...`);
        this._setView('diagnostic'); // Auto-switch to graph
        this._chart.reset('scan');

        const { T, entries, I, J, K } = this._data;

        for (let R = Rmin; R <= Rmax; R++) {
            if (this._stopFlag) break;
            this._log(`Scan R=${R}...`);

            const gen = this._kernel.run(T, I, J, K, R, reg, maxIter, entries);
            let last = null;
            for await (const step of gen) last = step;

            if (!last) continue;

            const corcondia = await this._kernel.calculateCorcondia(
                T, last.A, last.B, last.C, last.lambda, I, J, K, R, entries
            );
            this._chart.pushScan(R, corcondia, last.mse);
            this._log(`R=${R} → CORCONDIA=${corcondia.toFixed(1)}%, MSE=${last.mse.toExponential(2)}`);
        }

        this._log('Scan terminé. Cliquer sur le graphe pour choisir un rang.');
        this._toggleBusy(false);
        this._setStatus('SCAN TERMINÉ', '#7b5ea7');
        this._running = false;
    }

    // ── Per-step update ──

    _onStep(step) {
        this._lastFactors = step;

        // Metrics display
        this._dom.frobLoss.textContent = step.mse.toExponential(3);
        this._dom.currentIter.textContent = step.iter;
        const pct = (step.relErr * 100).toFixed(2);
        this._dom.relErr.textContent = pct;

        // CSS indicator — fix: use step.relErr not this.state.relErr
        if (this._dom.relErrBox) {
            this._dom.relErrBox.className =
                step.relErr < 0.05 ? 'accuracy-indicator optimal' :
                    step.relErr > 0.20 ? 'accuracy-indicator warning' :
                        'accuracy-indicator';
        }

        // Chart
        if (!step.done) this._chart.pushHistory(step.iter, step.mse);

        // Heatmap only when reconstruction is available (final step)
        if (step.T_hat) this._renderSlice(step.T_hat);
    }

    // ── Visualization ──

    _renderSlice(T_hat = null) {
        if (!this._data.isLoaded) return;
        const { T, I, J, K } = this._data;
        const k = parseInt(this._dom.sliceK.value);
        HeatmapRenderer.draw(this._dom.canvasObs, T, I, J, K, k);
        const rec = T_hat || this._lastFactors?.T_hat;
        if (rec) HeatmapRenderer.draw(this._dom.canvasRec, rec, I, J, K, k);
    }

    // ── Export ──

    _exportFactors() {
        if (!this._lastFactors) { this._log('Aucun résultat à exporter', 'warn'); return; }
        const f = this._lastFactors;
        const R = parseInt(this._dom.paramR.value);
        const payload = JSON.stringify({
            rank: R,
            iterations: f.iter,
            finalMse: f.mse,
            A: Array.from(f.A),
            B: Array.from(f.B),
            C: Array.from(f.C),
            lambda: Array.from(f.lambda),
        }, null, 2);
        const a = document.createElement('a');
        a.href = URL.createObjectURL(new Blob([payload], { type: 'application/json' }));
        a.download = `tensor_cp_R${R}.json`;
        a.click();
    }

    // ── UI helpers ──

    _toggleBusy(busy) {
        this._dom.btnRun.style.display = busy ? 'none' : 'block';
        this._dom.btnStop.style.display = busy ? 'block' : 'none';
        this._dom.btnScan.disabled = busy;
        this._dom.btnUpload.disabled = busy;
        this._dom.btnBench.disabled = busy;
        this._dom.btnDemo.disabled = busy;
    }

    _setView(view) {
        const d = this._dom;
        if (view === 'simulation') {
            d.viewSim.classList.add('active');
            d.viewDiag.classList.remove('active');
            d.panelObs.classList.remove('view-hidden');
            d.panelRec.classList.remove('view-hidden');
            d.visGrid.classList.remove('diagnostic-mode');
            this._log('Vue : Simulation');
        } else {
            d.viewSim.classList.remove('active');
            d.viewDiag.classList.add('active');
            d.panelObs.classList.add('view-hidden');
            d.panelRec.classList.add('view-hidden');
            d.visGrid.classList.add('diagnostic-mode');
            this._log('Vue : Diagnostic');
        }
        // Force chart update on view change if visible
        if (view === 'diagnostic') {
            this._chart._chart.resize();
            this._chart._chart.update('none');
        }
    }
}

// ─── BOOT ─────────────────────────────────────────────────────────────────────

window.addEventListener('DOMContentLoaded', () => {
    window.controller = new UiController();
});