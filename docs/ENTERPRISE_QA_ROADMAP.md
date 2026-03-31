# Enterprise QA Roadmap — v5.0.2 Baseline

Ce document définit les standards de qualité et de validation pour le framework Hypercube GPU Core. La version **5.0.2** marque la stabilisation complète de l'architecture de calcul déporté (standalone).

## Statut de Validation (v5.0.2)
Toutes les briques fondamentales du framework ont été validées via la suite d'audit scientifique (`tests/renders/`).

### 1. Stabilité des Macros WGSL [COMPLÉTÉ]
- [x] **Dédoublonnage** : Aucune erreur de redéclaration sur les faces multiples (FDTD/Noise).
- [x] **Suffix Handling** : Support automatique des suffixes `.read`, `.write`, `.now`, `.next`.
- [x] **Type Safety** : Signature différenciée pour les faces `scalar` (2D/3D) et `population` (D2Q9).

### 2. Topologie & Indexation [COMPLÉTÉ]
- [x] **Alignement 3D** : Calcul de `physicalNy` corrigé pour les grilles volumétriques (Mandelbulb 64³).
- [x] **Component Stride** : Indexation par plans de stride pour les populations LBM.
- [x] **Anti-Drift** : Alignement bit-à-bit vérifié entre le MasterBuffer GPU et le rendu JS.

### 3. Suite de Simulation Certifiée [COMPLÉTÉ]
Les algorithmes suivants sont certifiés stables et fonctionnels dans la version 5.0.2 :
*   **CFD (LBM D2Q9)** : Écoulement et Vorticités stabilisés.
*   **Électromagnétisme (FDTD Maxwell)** : Propagation d'ondes TE/TM.
*   **Fractales 3D (Mandelbulb)** : Rendu volumétrique 64³.
*   **Automates Cellulaires (Life)** : Jeu de la vie stable.
*   **Diffusion Thermique (Heat)** : Dissipation layout-agnostic.
*   **Équation de Poisson** : Solveur de pression opérationnel.

---

## Prochaines Étapes (v5.1+)
- [x] Support des textures WebGPU en sortie native (Zero-Copy Display).
- [x] Optimisation des Multi-Chunks (MPI-style ghost cell sync).
- [x] Intégration de kernels de calcul tensoriel (ML-ready).
