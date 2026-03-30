# User Quickstart : Utiliser les Kernels Existants (@hypercube/gpu-core)

Ce guide est destiné aux développeurs qui souhaitent intégrer le cœur GPU pour exécuter des simulations physiques.

## 1. Installation

```bash
npm install hypercube-gpu-core
```

## 2. Initialisation

```typescript
import { HypercubeGPUContext } from 'hypercube-gpu-core';

await HypercubeGPUContext.init();
```

## 3. Utilisation de l'Engine

```typescript
import { SimulationPacks, GpuCoreFactory } from 'hypercube-gpu-core';

const factory = new GpuCoreFactory();
const engine = await factory.build(config, descriptor);

// Boucle de calcul
async function run() {
    // 1. Initialisation unique
    await engine.ready();

    // 2. Génération automatique du Header (Indexation, Macros)
    const header = engine.getWgslHeader('MyRule');
    const kernels = {
        'MyRule': header + `
        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let x = id.x; let y = id.y;
            if (x >= uniforms.nx || y >= uniforms.ny) { return; }
            
            // Usage des Macros Professionnelles (v5.0)
            let val = read_Density(x, y); 
            write_Density(x, y, val + 1.0);
        }`
    };
    
    // 3. Calcul principal
    // Nouveau v5.0.4 : Définition fluide des paramètres
    engine.params
        .set('viscosity', 0.02)
        .set('gravity', 0.001);

    // Nouveau v5.0.4 : Enregistrement persistant des kernels
    engine.use(kernels);
    
    // Step sans avoir besoin de repasser les sources ou paramètres
    await engine.step(1);
    
    requestAnimationFrame(run);
}
```

## 4. Pilotage Direct (Proxy)

Vous pouvez également modifier les paramètres directement comme des propriétés de l'objet `params`, ce qui est idéal pour le binding avec des interfaces UI (dat.GUI, Tweakpane).

```typescript
engine.params.viscosity = 0.05;
await engine.step(1); // La nouvelle valeur est envoyée au GPU
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
