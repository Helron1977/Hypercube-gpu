# Hypercube Agent Onboarding Guide

Ce dépôt utilise une architecture tripartite (Données/Logique/Topologie). Pour tout développement de kernel ou modification du Core, suivez impérativement cet ordre de lecture pour garantir l'intégrité scientifique du simulateur.

## Ordre de Lecture Mandataire
1.  **[Kernel Implementation Guide](file:///docs/guides/kernel-implementation.md)**  
    *Objectif : Comprendre le flux du problème physique vers le GPU (Manifeste, Header Engine, Ping-Pong).*
2.  **[API Reference (DX Hub)](file:///docs/index.html)**  
    *Objectif : Référence complète des types de faces, macros WGSL et méthodes d'orchestration v6.0.6.*
3.  **[Exemple Certifié (LBM Core)](file:///src/kernels/wgsl/LbmCore.wgsl)**  
    *Objectif : Étude d'un noyau de production utilisant le stencil D2Q9 et l'aliasing sémantique.*
4.  **[Pattern d'Usage Complet](file:///tests/renders/conservation.ts)**  
    *Objectif : Voir comment le moteur est instancié et audité pour la conservation de masse/énergie.*

---
*Note : Aucune modification du Core (MasterBuffer, GpuDispatcher) ne doit être tentée sans lecture préalable de ces quatre piliers.*
