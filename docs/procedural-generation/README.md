# 🎲 Guide : Génération Procédurale & Bruit (Noise GPU)

Ce guide explique comment Hypercube génère des textures, des terrains et des mondes infinis à partir de fonctions mathématiques.

---

## 1. C'est quoi le Bruit Procédural ? (Simplex / Perlin)

Contrairement au "bruit blanc" (une télé sans signal qui grésille), le **bruit procédural** est un chaos organisé. Il crée des variations douces et cohérentes, comme des collines, des nuages ou les veines d'un marbre.

**Les types de bruit dans Hypercube :**
- **Simplex Noise** : Un algorithme très rapide sur GPU qui évite les lignes droites visibles.
- **Perlin Noise** : Le standard réputé pour son aspect naturel.

---

## 2. Pourquoi le GPU est-il indispensable ?

Générer un terrain pour un jeu vidéo ou une simulation demande de calculer des millions de valeurs.
Si vous le faites sur le processeur (CPU), le chargement est long. 
Sur Hypercube, le moteur **WebGPU** génère des textures de 1024x1024 en une fraction de seconde, permettant de créer des mondes qui se construisent sous vos yeux au fur et à mesure que vous avancez.

---

## 3. Observer le Showcase Noise

Le test `noise_procedural_verify.html` affiche une texture mouvante.
- **Fréquence** : Si elle est basse, vous voyez de grandes collines douces. Si elle est haute, vous voyez des micro-détails (gravier).
- **Amplitude** : Détermine le contraste entre les zones noires et blanches.

---

## 4. Applications Réelles

### Jeux Vidéo & Metaverse
Génération de paysages, de forêts ou de grottes sans avoir à stocker des Gigaoctets d'images sur le disque.

### Simulation de Matériaux
Créer des textures de matériaux composites ou de tissus pour des analyses de résistance.

### Astronomie
Simuler la distribution de la matière sombre ou des nébuleuses dans l'espace.
