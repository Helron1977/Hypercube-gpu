# User Quickstart : Utiliser les Kernels Existants (@hypercube/gpu-core)

Ce guide est destiné aux développeurs qui souhaitent intégrer le cœur GPU pour exécuter des simulations physiques.

## 1. Installation

```bash
npm install @hypercube/gpu-core
```

## 2. Initialisation

```typescript
import { HypercubeGPUContext } from '@hypercube/gpu-core';

await HypercubeGPUContext.init();
```

## 3. Utilisation de l'Engine

```typescript
import { SimulationPacks, GpuCoreFactory } from 'hypercube-gpu-core';

const factory = new GpuCoreFactory();
const engine = await factory.build(config, descriptor);

// Boucle de calcul
async function run() {
    // 1. Génération automatique du Header (Indexation, Uniforms)
    const header = engine.getWgslHeader('MyRule');
    const kernels = {
        'MyRule': header + `/* Votre code WGSL */`
    };
    
    // 2. Calcul principal
    await engine.step(kernels, 1);
    
    // 3. Mise à jour dynamique des paramètres (Optionnel)
    // syncParams permet de changer p0-p7 sans écraser les données du GPU
    engine.vGrid.config.params.p0 = 1.23;
    await engine.syncParams(kernels);
    
    requestAnimationFrame(run);
}
```

## 5. Performance & Readback (ABK) ⚠️

> [!CAUTION]
> **Ne lisez pas les données à chaque frame !** L'appel à `getFaceData()` ou `syncToHost()` est extrêmement lent car il force le GPU à s'arrêter pour parler au CPU. 

### La règle d'or :
- **Pour le Rendu** : Utilisez `engine.buffer.gpuBuffer` directement dans votre moteur 3D (Zéro-Readback).
- **Pour les Diagnostics** : Utilisez `syncFacesToHost()` de manière asynchrone et uniquement pour les faces nécessaires.

```typescript
// Exemple de synchronisation asynchrone sécurisée
async function updateHUD() {
    // 1. Demande de sync asynchrone (ne bloque pas le rendu)
    await engine.syncFacesToHost(['rho']);
    
    // 2. Lecture des données filtrées (sans ghost cells)
    const data = engine.getInnerFaceData('chunk_0_0_0', 'rho');
    console.log("Densité moyenne:", data[0]);
    
    // 3. Replanifier (ex: toutes les secondes, pas à chaque frame)
    setTimeout(updateHUD, 1000);
}
```
