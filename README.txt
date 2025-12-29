# Topological Social Choice Machine Learning

This repository accompanies the research paper:

**Topological Machine Learning for Real-World Elections via Borda Embeddings**  
*A Homology-Separated Polar Persistence Kernel for SVM Classification*

Author: Nikolaos Sampanis

---

## Overview

This project proposes a complete **topological machine learning pipeline**
for the analysis and classification of elections with ordinal preference profiles.

Each election is treated as a geometric and topological object:
- individual rankings are embedded into Euclidean space via **Borda vectors**,
- elections become point clouds,
- **persistent homology** (H₀, H₁, H₂) is computed using Vietoris–Rips filtrations,
- persistence diagrams are compared using a novel **Polar Persistence Distance (PPD)**,
- a **homology-separated Gaussian kernel** enables **SVM classification of elections**
  without diagram vectorization.

The methodology is designed to be:
- interpretable (homology separation),
- stable (diagram-level and profile-level guarantees),
- suitable for real-world election data.

---

## Repository Structure


---

## Main Script

- `classification_pipeline.py`

Implements the full pipeline:
1. parsing election files (PrefLib-style strict orders),
2. Borda embedding and point cloud construction,
3. persistent homology computation,
4. Polar Persistence Distance (PPD),
5. kernel matrix construction,
6. SVM training and evaluation.

The script is modular and can be extended to:
- cross-validation,
- homology ablation,
- alternative kernels or classifiers.

---

## Data

- `data/tda_pref_profile_results.csv`  
  Contains processed per-election topological summaries used for reporting.

- `data/tda_pref_profile_results_table.tex`  
  LaTeX-ready subset used directly in the manuscript tables.

Raw election data originate from **PrefLib**:
http://www.preflib.org

---

## Dependencies

Minimal Python dependencies:

numpy
scipy
pandas
scikit-learn


Install with:

```bash
pip install numpy scipy pandas scikit-learn
