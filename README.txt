# Topological Social Choice ML
**Topological Machine Learning for Real-World Elections via Borda Embeddings + Persistent Homology**  
Homology-separated **Polar Persistence Distance (PPD)** kernel + **SVM** for election classification (Condorcet-cycle detection).

Repo: https://github.com/nikossampanis/topological-social-choice-ML

---

## Overview
This project implements an end-to-end pipeline:
1. **Parse elections** (PrefLib-style strict orders, e.g., `.soi`)
2. **Embed votes** as **Borda vectors** → election becomes a point cloud in \(\mathbb{R}^m\)
3. Compute **Vietoris–Rips persistent homology** in degrees \(H_0, H_1, H_2\)
4. Compare persistence diagrams via a **Polar Persistence Distance (PPD)** (persistence-weighted, diagonal-suppressing matching)
5. Build a **homology-separated Gaussian (RBF) kernel**
6. Train/test **SVM** using the precomputed kernel matrix
7. Export **CSV** + **APA-style LaTeX tables** for papers

> Design note: the code is intentionally self-contained and does **not** require external TDA libraries (e.g., ripser/gudhi).

---

## Repository structure
