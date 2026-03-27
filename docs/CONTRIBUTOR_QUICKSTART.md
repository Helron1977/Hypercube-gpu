# Contributor Quickstart : Implémenter de nouveaux Kernels

## 1. Data Contract & Faces
Définissez vos faces dans le descripteur. Le moteur supporte jusqu'à 64 faces.

```typescript
{
    name: 'MyPhysics',
    faces: [{ name: 'rho', type: 'scalar', isPingPong: true }],
    requirements: { ghostCells: 1, pingPong: true }
}
```

## 2. Triple-Buffering (Modulo 3)
Pour les équations d'ondes de 2nd ordre, vous pouvez spécifier un modulo 3 dans les paramètres du kernel :

```typescript
rules: [{ 
    type: 'WaveStep', 
    params: { modulo: 3 } 
}]
```

Dans votre shader, utilisez `engine.getWgslHeader()` pour récupérer les index corrects (`rho_Read`, `rho_Write`).

## 3. Utilisation du Header Generator
Ne calculez plus vos index manuellement. Utilisez :
```typescript
const header = engine.getWgslHeader('MyStep');
```
Cela générera pour vous :
```wgsl
const rho_Read: u32 = uniforms.faces[0];
const rho_Write: u32 = uniforms.faces[1];
fn getIndex(x: u32, y: u32) -> u32 { ... }
```
