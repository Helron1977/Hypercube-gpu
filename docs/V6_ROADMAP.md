# Hypercube GPU Core v6.0 - Shared Memory Architecture

Le cycle v6.0 fait passer Hypercube d'un framework mono-moteur à un **Hub Multi-Physique** capable d'exécuter des simulations couplées (ex: LBM + Thermique + Contraintes) sur un segment VRAM unifié.

## 1. Le Problème : Les Silos Mémoire
En v5.0.4, chaque `GpuEngine` possède son propre `MasterBuffer`. Coupler deux moteurs nécessite actuellement :
1. `engineA.syncFacesToHost()` (GPU -> CPU)
2. `engineB.setFaceData()` (CPU -> GPU)
C'est un goulot d'étranglement qui annule les avantages de performance de WebGPU.

## 2. La Solution : SharedMasterBuffer
La v6 introduit le concept de **Persistent Memory Slices**. 

### Layout Mémoire v6 :
- **Binding 0** : `GlobalSharedBuffer` (Contient toutes les faces scalaires/champs de tous les moteurs).
- **Binding 2** : `GlobalAtomicBuffer` (Registres d'accumulation partagés).

### Changements Architecturaux :
1. **Découplage MasterBuffer** : 
   - Le `MasterBuffer` ne créera plus systématiquement son propre buffer matériel. 
   - Il supportera un mode `externalBuffer` où il mappe ses faces à un `offset` spécifique dans un pool géant.
2. **MultiPhysicsHub** :
   - Un orchestrateur qui alloue le `GlobalSharedBuffer`.
   - Distribue des "Slices" aux moteurs spécialisés.
   - Gère l'ordre des `dispatch` pour assurer la causalité (ex: l'étape LBM doit finir avant que l'étape Thermique ne lise la vitesse).

## 3. Feuille de Route d'Implémentation

### Phase 6.1 : Injection de Buffer
- Modifier le constructeur de `MasterBuffer` pour accepter un `GPUBuffer` optionnel et un `baseOffset`.
- Mettre à jour `MemoryLayout` pour calculer des offsets relatifs.

### Phase 6.2 : Orchestrateur Hub
- Création de la classe `MultiPhysicsHub`.
- Support de `hub.attach(engineA, "offset0")`, `hub.attach(engineB, "offset1")`.

### Phase 6.3 : Couplage "Zéro-Copie"
- Faire tourner un LBM et un champ Thermique sur le MÊME buffer physique.
- Le kernel thermique lira la vitesse LBM directement via `data[lbm_u_offset]`.

---

> [!IMPORTANT]
> **Objectif Performance** : Atteindre 100% de couplage "Zero-Readback" pour les scénarios multiphysiques complexes.

> [!WARNING]
> **V6 Breaking Change** : La configuration des `faces` dans le manifest devra probablement inclure un champ `globalOffset` ou `partition` pour aider le Hub à organiser la mémoire.
