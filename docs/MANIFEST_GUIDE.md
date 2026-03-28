# Hypercube Manifest & Configuration Guide

Ce guide explique comment configurer une simulation Hypercube et comment écrire vos propres kernels WGSL.

## 1. Structure du Manifest (`hypercube.schema.json`)

Le manifest est le "cerveau" de votre simulation. Il définit la mémoire (Faces), la physique (Rules) et la topologie (Config).

### Concepts Clés :
- **Ghost Cells** : Cellules de bordure (padding) servant aux échanges entre chunks. (Généralement 1 pour LBM/Wave).
- **Ping-Pong** : Double-buffering automatique. Si activé, chaque face aura un état `Read` et un état `Write`.
- **Faces** : Emplacements mémoire (slots). Le moteur supporte jusqu'à 64 faces par simulation.

---

## 2. Écrire des Kernels WGSL 🚀

Hypercube simplifie l'écriture WGSL en injectant automatiquement un header.

### Ressources WGSL :
- [**WGSL Specification (W3C)**](https://www.w3.org/TR/WGSL/) : La référence technique complète.
- [**WebGPU Fundamentals**](https://webgpufundamentals.org/) : Excellent tutoriel pour comprendre le fonctionnement des pipelines GPU.
- [**Hypercube Mother-Models**](file:///c:/Users/rolan/OneDrive/Desktop/hypercube-gpu-core/src/kernels/) : Étudiez nos implémentations certifiées (LBM, Waves).

### Cheat Sheet WGSL Rapide :
```wgsl
// Types de base
let a: f32 = 1.0;
let b: u32 = 1u;
let pos: vec2<f32> = vec2<f32>(x, y);

// Fonctions utiles pour la physique
let d = distance(p1, p2);
let s = step(threshold, value);
let m = min(a, b);
let n = normalize(v);
```

### Macros Hypercube (Auto-injectées) :
Grâce à `getWgslHeader()`, vous n'avez plus à calculer les index. Le moteur injecte automatiquement un struct `Uniforms` et des helpers :

- **Accès Uniformes** : Utilisez l'objet global `uniforms` (e.g. `uniforms.nx`, `uniforms.t`, `uniforms.p0..p7`).
- **Lecture/Écriture Scalaire** : `read_faceName(x, y)` et `write_faceName(x, y, value)`.
- **Multi-Composants (LBM/Vector)** : `read_faceName(x, y, d)` où `d` est l'indice du composant (0..26 pour D3Q27).
- **Ping-Pong Dynamique** : `read_faceName_Now(x, y)` et `write_faceName_Next(x, y, val)`.
- **Indexation Linéaire** : `getIndex(x, y)` (2D) et `getIndex3D(x, y, z)` (3D).

---

## 3. Configuration des Boundary Roles
Hypercube v5.0 Pro supporte des comportements de bordure avancés injectés dans les uniformes:
- `leftRole`, `rightRole`, etc.
- Valeurs : `1: Periodic`, `2: Wall (Bounce-Back)`, `3: Inflow`, `4: Outflow`.

## 3. Utilisation des Simulation Packs

Pour éviter de tout configurer à la main, utilisez nos presets :

```typescript
import { SimulationPacks, GpuCoreFactory } from '@hypercube/gpu-core';

const descriptor = SimulationPacks.LBM_D2Q9;
const config = { ... }; 
const engine = await factory.build(config, descriptor);
```

## 4. Validation Automatique

Si vous utilisez VS Code, ajoutez ceci en haut de votre fichier `.json` pour bénéficier de l'autocomplétion :

```json
{
  "$schema": "../schemas/hypercube.schema.json",
  "name": "Ma Simulation",
  ...
}
```
