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
import { GpuCoreFactory } from '@hypercube/gpu-core';

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

## 4. Points Clés
- **`syncParams()`** : À utiliser pour les interactions en temps réel.
- **`getWgslHeader()`** : Supprime le besoin de calculer les offsets mémoire à la main.
- **64 Slots** : Le moteur supporte désormais jusqu'à 64 variables par kernel.
