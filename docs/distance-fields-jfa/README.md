# 🗺️ Guide : Champs de Distance & Voronoi (Jump Flooding Algorithm - JFA)

Ce guide explique comment Hypercube calcule des distances par rapport à des points en un temps record grâce à l'algorithme **JFA**.

---

## 1. C'est quoi un Champ de Distance ?

Un champ de distance est une image où chaque pixel connaît sa distance par rapport à l'objet le plus proche.
- Si le pixel est sur un point, sa valeur est **0**.
- Plus il s'éloigne, plus sa valeur augmente.

**Pourquoi c'est utile ?**
Cela permet de créer des **Diagrammes de Voronoi** (découper l'espace en zones d'influence), de lisser des polices de caractères (SDF) ou de calculer des trajectoires d'évitement pour des robots.

---

## 2. Le concept du "Jump Flooding" (JFA)

Calculer la distance pour chaque pixel par rapport à chaque point de façon classique est extrêmement lent ($O(N \times M)$).
L'algorithme **JFA** est une astuce GPU qui divise le problème par "sauts" successifs :
1. On regarde à une distance de **128 pixels**.
2. Puis **64 pixels**.
3. Puis **32**, **16**, **8**, **4**, **2**, **1**.

En seulement **8 étapes**, chaque pixel a trouvé son voisin le plus proche sur une grille de 256x256. C'est d'une efficacité redoutable sur GPU.

---

## 3. Observer le Showcase JFA

Dans l'interface Hypercube, vous voyez des zones de couleurs vives.
- **Les Points** : Ce sont les "graines" (seeds).
- **Les Zones de Couleur** : Chaque couleur représente le territoire d'une graine (Voronoi).
- **Le dégradé** : C'est la distance pure.

---

## 4. Utilisation et Paramètres

Dans le framework, le JFA est souvent utilisé comme une étape de préparation pour d'autres simulations (ex: trouver l'obstacle le plus proche pour un flux d'air).

**Astuce visuelle** : Le mode de rendu `jfa` dans Hypercube utilise les coordonnées de la graine pour générer une couleur unique et stable, ce qui permet de voir les frontières bouger en temps réel si vous déplacez les points dans le manifest.
