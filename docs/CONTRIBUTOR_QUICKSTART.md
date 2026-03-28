# Contributor Quickstart : Implémenter de nouveaux Kernels (v5.0.2)

## 1. Définition des Faces
Définissez vos faces dans le descripteur de l'engine. Le framework gère automatiquement l'allocation mémoire et le ping-pong.

```typescript
{
    name: 'MyPhysics',
    faces: [
        { name: 'phi', type: 'scalar', isPingPong: true },
        { name: 'velocity', type: 'vector' } // 3 components (u, v, w)
    ],
    requirements: { ghostCells: 1, pingPong: true }
}
```

## 2. Accès Mémoire (Macros)
Ne calculez plus jamais d'index manuellement. Le `GpuDispatcher` génère des macros WGSL de haut niveau pour vous.

### Utilisation dans le Kernel :
```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; let y = id.y;
    
    // Lecture automatique du slot courant (parity handled by host)
    let val = read_phi(x, y);
    
    // Écriture vers le slot suivant (ping-pong)
    write_phi(x, y, val * 0.99);
}
```

## 3. Support Multi-Composantes (LBM)
Pour les faces de type `population` (ex: D2Q9), la macro accepte un argument `d` supplémentaire :

```wgsl
// Lecture de la direction 'd' pour la face 'f'
let f_in = read_f(x, y, d);
```

## 4. Support Volumétrique (3D)
Si votre grille est 3D, utilisez les macros suffixées `3D` :

```wgsl
let voxel = read_fractal(x, y, z);
write_fractal(x, y, z, 1.0);
```

### Intégration côté Host :
Récupérez toujours le header généré avant de compiler votre shader :
```typescript
const wgslHeader = engine.dispatcher.getWgslHeader('MyKernelName');
const finalShader = wgslHeader + kernelBody;
```
