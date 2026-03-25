# 🌿 Guide : Biologie Synthétique & Vie Cellulaire (Life Nebula)

Ce guide explore comment Hypercube simule des comportements émergents complexes à partir de règles très simples, inspirées du **Jeu de la Vie**.

---

## 1. L'Émergence : De l'ordre dans le chaos

Dans ce modèle, chaque pixel est une "cellule" qui peut être **vivante** ou **morte**.
Le destin d'une cellule dépend uniquement de ses 8 voisines directes.

**Les règles standard (Conway) :**
- Une cellule vivante avec 2 ou 3 voisines survit.
- Une cellule morte avec exactement 3 voisines naît.
- Sinon, elle meurt (isolement ou surpopulation).

**Pourquoi Hypercube ?**
Sur un processeur classique, simuler des millions de cellules demande beaucoup de boucles. Sur Hypercube, le moteur GPU traite chaque cellule en parallèle, permettant des simulations massives (millions d'agents) à 60 images par seconde.

---

## 2. Au-delà du Jeu de la Vie (Life Nebula)

Le showcase **Cellular Life** d'Hypercube permet d'aller plus loin en testant des variantes :
- **Règles personnalisées** : Changer les seuils de survie.
- **États multiples** : Des cellules qui ont des niveaux d'énergie ou de nutriments.

---

## 3. Applications Réelles

Bien que ça ressemble à un jeu, les Automates Cellulaires sont utilisés pour :
- **La simulation de foules** (évacuation d'urgence).
- **La croissance de cristaux** ou de tissus biologiques.
- **La propagation d'épidémies** ou de feux de forêt.

---

## 4. Comment l'utiliser ?

1. Lancez le test `cellular_life_verify.html`.
2. Observez les structures stables (blocs) et les "gliders" (vaisseaux qui voyagent).
3. Modifiez le manifest pour changer la densité initiale et observez si la "vie" finit par s'éteindre ou si elle colonise tout l'espace.
