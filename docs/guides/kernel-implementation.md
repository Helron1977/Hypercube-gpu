# Guide Step-by-Step : Implémentation Kernel Hypercube (v6.0.6)

Ce guide détaille le passage rigoureux d'une équation physique à un solveur GPU. 

## GLOSSAIRE TECHNIQUE (Avec Références)
- **Face** : Une couche de données en VRAM. Types valides : `scalar`, `field`, `vector`, `population`, `mask`. 
  - *Ref : Section 3 du "DX Guide" (`docs/index.html`).*
- **Ping-Pong** : Technique de double-buffering (Isolation Temporelle). On lit dans l'état "A" (Now) et on écrit dans l'état "B" (Next). Obligatoire pour toute physique où une cellule interagit avec ses voisins.
  - *Ref : Section 4 du "DX Guide" (`docs/index.html`) — Macros Now/Next.*
- **Ghost Cells** : Zone de recouvrement (Halo de 1 pixel) entre blocs physiques de mémoire. Elle permet à un processeur GPU de lire les valeurs du voisin sans conflit. 
  - *Ref : Section 5 du "Agent Onboarding" (`agents/README.md`).*
- **DataContract** : Le "plan de montage" de la mémoire généré à partir de votre manifeste.
- **Ordre Numérique (Familles)** :
  - **Ordre 0** (Simple) : 1 slot. Pas de voisinage.
  - **Ordre 1** (Standard) : 2 slots (Ping-Pong). Voisinage direct (n +/- 1). **C'est notre mode (Diffusion).**
  - **Ordre 2** (Accélération) : 3 slots. Historique pour ondes 2nd ordre.
  - *Ref : `docs/CORE_PRINCIPLES.md`.*

---

## ÉTAPE 0 : Analyse du Problème Physique
Problème : Simulation de la Diffusion de la Chaleur 2D.
L'équation discrétisée (Euler) est : 
`[T_next] = [T_now] + k * dt * Laplacian(T)`

### Décomposition Technico-Physique :
1. **[T_next] et [T_now]** : L'existence de deux états temporels impose le mécanisme **`isPingPong: true`** (Double Buffering) pour éviter que l'écriture du futur n'écrase la lecture du présent (Race Condition).
2. **Laplacian(T)** : Le calcul nécessite de lire les 4 voisins directs du point (N, S, E, O). Cela impose **`ghostCells: 1`** afin que le GPU puisse lire "au-delà" des limites de son Chunk mémoire sans conflit de bordure.
3. **Coefficients k et dt** : Ce sont les paramètres de contrôle physique. Ils sont accessibles via le block **`params`** (Binding 1).

---

## ÉTAPE 1 : Définition du Contrat de Données (Manifeste)
Le manifeste est le document qui lie la physique (le "quoi") au matériel (le "comment"). 

### 1. Pillarisation Architecturale du Manifeste
| PILIER | BLOC JSON | FONCTION | RÉFÉRENCE |
| :--- | :--- | :--- | :--- |
| **DONNÉES** | `engine.faces` | Définit la nature de la simulation (Champ vs Masque). | `docs/index.html > Sec 3` |
| **LOGIQUE** | `engine.rules` | Enregistre le nom de votre kernel de calcul. | `docs/index.html > Sec 1` |
| **TOPOLOGIE** | `config` | Définit l'échelle spatiale (nx, ny), le partitionnement et les params. | `agents/README.md > Sec 5` |

### 2. Taxonomie Sémantique des Types de Faces (Immutables)
Ces types sont définis dans le `DataContract` du Core. Ils dictent l'allocation binaire du `MasterBuffer`.
| Type | Composants | Usage Sémantique |
| :--- | :--- | :--- |
| **`field`** | 1 (f32) | Champ continu **synchronisé** (Bénéficie des Ghost Cells). |
| **`mask`** | 1 (f32) | Données statiques (Obstacles, Limites). Optimisé pour la lecture. |
| **`scalar`** | 1 (f32) | Donnée scalaire simple non-synchronisée. |
| **`vector`** | 3 (f32) | Vecteurs physiques (ex : Vitesse U). Accès via index `0,1,2`. |
| **`population`**| N (f32) | Multi-états (ex : D2Q9). Stockage optimisé par plans. |
| **`atomic_*`** | 1 (u32/f32)| Accumulateurs sécurisés (Binding 2). |

### 3. Choix des Faces pour le Problème Diffusion
Le choix Face : `field` vs `scalar`
Bien que les deux utilisent des f32, on choisit **`field`** car c'est le type sémantique conçu pour les champs continus (Diffusion/Fluides). Le type **`mask`** est choisi pour la face `source` pour indiquer son immobilité (Read-Only).
- **Face `temp`** : Type `field`. On choisit `field` pour la synchronisation automatique des Halos.
- **Face `source`** : Type `mask`. On choisit `mask` pour son immobilité (`isReadOnly: true`).

### 4. Manifeste v6.0.6 Résultant :
```json
{
  "name": "HeatEngine_Exhaustive",
  "engine": {
    "name": "DiffusionV1",
    "faces": [
      { "name": "temp",   "type": "field", "isPingPong": true }, // Stockage T (Now/Next)
      { "name": "source", "type": "mask",  "isReadOnly": true }   // Sources imposées
    ],
    "rules": [
      { "type": "step_diffusion", "source": "", "entryPoint": "main" }
    ],
    "requirements": { "ghostCells": 1 } // Voisinage direct (Laplacien)
  },
  "config": {
    "dimensions": { "nx": 256, "ny": 256, "nz": 1 },
    "params": { 
        "k": 0.1,  // Paramètre p0
        "dt": 0.01 // Paramètre p1
    }
  }
}
```

---

## ÉTAPE 2 : Le Moteur de Virtualisation (Header WGSL)
Hypercube ne se contente pas d'envoyer votre code au GPU. Il construit un Header dynamique qui contient les mappings exacts de votre mémoire. 
Il virtualise l'accès à la VRAM.

### 1. La Triple Transformation du Header (Abstraction)
Le header effectue des calculs complexes pour vous permettre d'utiliser une syntaxe de haut niveau :
- **Linéarisation (Stride Mapping)** : Conversion `(x, y)` $\rightarrow$ Index à plat via `strideRow` et `strideFace`.
- **Compensation de Bordure** : Gestion automatique de l'offset des `ghostCells`.
- **Commutation de Parité** : Redirection dynamique `Now/Next` selon l'état du moteur.
**Bénéfice** : Le développeur ne doit jamais se préoccuper de ces calculs. Le framework garantit l'intégrité mémoire sous le capot.

### 2. Aliasing Sémantique vs Uniforms Bruts
- **Uniforms Bruts** : Zones mémoire `Read-Only` accessibles via des indices anonymes (`uniforms.p0`, `uniforms.p1`). 
- **Aliasing Sémantique (`SimulationParams`)** : Introduit en v6.0.6, ce mécanisme crée une structure nommée basée sur votre manifeste.
- **Preuve Source** : `src/dispatchers/WgslHeaderGenerator.ts` (Lignes 72-99).
- **Utilisation** : `let p = get_params(); let k = p.k;` $\rightarrow$ Le code devient lisible et auto-documenté.

### 3. Les Macros d'Accès de Données (Stencil)
Le framework injecte des macros basées sur vos noms de face. Grace aux **`ghostCells: 1`**, ces accès sont sécurisés :
| Macro | Slot | Rôle |
| :--- | :--- | :--- |
| **`read_temp_Now(x, y)`** | Slot Actif | Etat actuel de la chaleur. |
| **`read_temp_Now(x+1, y)`**| Slot Actif | Voisin Est (Stencil). |
| **`write_temp_Next(x, y, v)`**| **Slot Futur** | Sauvegarde (Ping-Pong). |

### 4. Squelette du Noyau (Kernel Shell)
```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    // A. Récupération des paramètres config.params
    let p = get_params();
    let alpha = p.k;
    let delta_t = p.dt;

    // B. Guard de dimension (nx/ny injectés par les Uniforms)
    if (x < uniforms.nx && y < uniforms.ny) {
        // [Logique Étape 3 ici]
    }
}
```

---

## ÉTAPE 3 : Logique Physique et Équations de Diffusion
Nous transcrirons l'équation discrétisée en un algorithme parallèle.

### 1. La Discrétisation Spatiale (Laplacien 2D)
On utilise le schéma de Différences Finies au centre. L'opérateur Laplacien se calcule par la différence avec les voisins cardinaux (Von Neumann).
`L = T(x+1, y) + T(x-1, y) + T(x, y+1) + T(x, y-1) - (4.0 * t_now)`

### 2. Condition de Stabilité (CFL Condition)
Pour garantir la convergence numérique, le produit `k * dt` doit rester inférieur à **0.25**. (Ref : `CORE_PRINCIPLES.md`).

### 3. Implémentation Final (Code Source WGSL)
```wgsl
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;
    if (x < uniforms.nx && y < uniforms.ny) {
        let p = get_params();
        let t_now = read_temp_Now(x, y);
        let s_now = read_source_Now(x, y);

        // Somme des voisins directs via macros automatiques
        let l = read_temp_Now(x + 1, y) + 
                read_temp_Now(x - 1, y) + 
                read_temp_Now(x, y + 1) + 
                read_temp_Now(x, y - 1) - (4.0 * t_now);

        var t_next = t_now + (p.k * p.dt * l);

        // Application du Masquage (Boundary Constraint)
        if (s_now > 0.0) {
            t_next = s_now;
        }

        write_temp_Next(x, y, t_next);
    }
}
```

---

## CONCLUSION : Synthèse et Instrumentation Host (TypeScript)

### 1. Vision Juxtaposée (Contrat vs Simulation)
| Manifeste (JSON) | Kernel (WGSL) |
| :--- | :--- |
| `field temp { isPingPong: true }` | `read_temp_Now` / `write_temp_Next` |
| `params { k, dt }` | `let p = get_params(); p.k * p.dt` |
| `mask source` | `read_source_Now` |

### 2. Code de Lancement (Orchestration Host)
Voici le code permettant de charger le manifeste, d'injecter votre kernel WGSL et de démarrer la boucle de calcul.

```typescript
import { HypercubeGPUContext, createSimulation } from 'hypercube-gpu-core';

// 1. Initialiser le contexte WebGPU
const context = new HypercubeGPUContext();
await context.initialize();

// 2. Créer l'instance à partir du manifeste
const engine = await createSimulation(myManifestJSON, context);

// 3. Enregistrer le Kernel WGSL (lié par le nom 'step_diffusion' dans le manifeste)
engine.use('step_diffusion', myWgslSourceCode);

// 4. Boucle de simulation
async function simulate() {
  await engine.step(100); // 100 Ticks GPU
  engine.params.k = 0.15; // Modification dynamique sémantique
  requestAnimationFrame(simulate);
}
simulate();
```

---
*Processus d'Implémentation Kernel v6.0.6 — Document Maître Exhaustif.*
