# 📐 Guide : Physique Numérique et Résolution d'Équations

Ce répertoire regroupe les modèles permettant de résoudre des équations différentielles fondamentales utilisées dans toutes les branches de l'ingénierie.

---

## 1. Poisson / Laplace (Champs de Potentiel)

### C'est quoi ?
L'équation de Poisson est utilisée pour trouver un équilibre. Par exemple :
- La température finale d'une plaque chauffée.
- La pression de l'eau dans un tuyau.
- Le potentiel électrique autour d'une charge.

### Dans Hypercube
Le framework utilise un algorithme itératif (Jacobi) sur GPU. À chaque frame, l'ordinateur essaie de lisser les différences entre les voisins jusqu'à ce que le champ soit stable.

---

## 2. L'Équation d'Onde 2D

### C'est quoi ?
C'est le modèle mathématique derrière une membrane de tambour ou la surface de l'eau. Contrairement au LBM (qui simule les particules), ici on simule la **vibrations d'une surface**.

---

## 3. Fractals : Mandelbulb 3D

### C'est quoi ?
C'est la démonstration de la capacité du framework à gérer des calculs récursifs lourds. On utilise le **Ray-marching** (un rayon part de vos yeux et rebondit sur la mathématique) pour dessiner ces structures infiniment complexes.

C'est l'outil idéal pour tester la robustesse des kernels GPU d'Hypercube sur des algorithmes "orientés calculs purs".
