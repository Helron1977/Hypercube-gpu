# Cookbook : Lattice Boltzmann (LBM) with Hypercube

Ce guide détaille les meilleures pratiques pour implémenter des simulations de fluides performantes sur le framework.

## 1. La Règle d'Injection de Masse ⚠️

**Problème** : Si vous modifiez uniquement la face `rho` (densité) pour ajouter un obstacle ou un fluide, le kernel LBM écrasera vos changements à l'itération suivante car il se base sur les distributions `f0-f8`.

**Solution** : Modifiez toujours les distributions d'équilibre.
- Pour ajouter de la masse, augmentez proportionnellement `f0-f8` selon les poids du réseau (weights).
- Ou utilisez le futur `PhysicsBridge` (en cours de développement).

## 2. Filtrage pour le Rendu (Ghost Cells)

N'envoyez pas le MasterBuffer complet à votre moteur de rendu ! Utilisez la nouvelle méthode de filtrage :

```typescript
const displayData = engine.getInnerFaceData('chunk_0_0_0', 'rho');
// 'displayData' ne contient que les nx * ny cellules réelles, sans ghost cells.
```

## 3. Macros WGSL pour LBM

Utilisez `getWgslHeader('LbmStep')` pour obtenir des accès simplifiés :

```wgsl
// Dans votre shader
let rho = read_rho(x, y);
let feq = calculateEquilibrium(rho, u, v);
write_f0(x, y, feq[0]);
```

## 4. Diagnostics de Stabilité
Surveillez `p7` (souvent utilisé pour l'échelle de réduction) et assurez-vous que la viscosité calculée ne tombe pas sous la limite de stabilité ($1/\omega > 0.5$).
