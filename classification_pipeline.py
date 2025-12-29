# ============================================================
# classification_pipeline.py (ENRICHED, end-to-end, single file)
# Elections -> Borda point clouds -> VR persistence (H0,H1,H2; Z2 reduction)
# -> Polar Persistence Distance (PPD) -> homology-separated kernel -> SVM classification
#
# This script is intentionally self-contained (no ripser/gudhi required).
# It is suitable for reproducibility and for generating APA-style tables.
#
# INPUT:
#   Put many election files into ./elections/
#   Supported: PrefLib-style .soi (strict orders possibly incomplete)
#
# OUTPUT:
#   tda_pref_profile_results.csv
#   tda_pref_profile_results_table.tex
#
# ============================================================

import re, math, random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# Data structure
# ---------------------------
@dataclass
class Election:
    name: str
    rankings: List[List[int]]  # completed permutations in 0..m-1
    m: int

# ---------------------------
# Parse PrefLib .soi (strict orders possibly incomplete)
# Deterministically completes missing candidates by appending them.
# ---------------------------
def parse_preflib_soi(filepath: str, voter_subsample: Optional[int]=30, seed: int=0) -> Election:
    p = Path(filepath)
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    m = None
    data_lines = []
    for ln in lines:
        if ln.startswith("#"):
            mm = re.search(r"NUMBER ALTERNATIVES:\s*([0-9]+)", ln)
            if mm:
                m = int(mm.group(1))
            continue
        data_lines.append(ln)

    if m is None:
        raise ValueError(f"Could not read m from {filepath}")

    rng = random.Random(seed)
    rankings = []
    for ln in data_lines:
        ln_clean = ln.replace(":", " ").replace(",", " ").replace(">", " ")
        parts = ln_clean.split()
        if len(parts) < 2:
            continue

        weight = int(parts[0]) if parts[0].isdigit() else 1
        cand_ids = [int(x) for x in parts[1:] if x.isdigit()]
        if not cand_ids:
            continue

        # 1..m -> 0..m-1 and strict-unique
        seen = set()
        ordered = []
        for c in cand_ids:
            if 1 <= c <= m:
                c0 = c-1
                if c0 not in seen:
                    seen.add(c0)
                    ordered.append(c0)
        missing = [c for c in range(m) if c not in seen]
        full = ordered + missing
        if len(full) != m or sorted(full) != list(range(m)):
            continue

        for _ in range(weight):
            rankings.append(full)

    if not rankings:
        raise ValueError(f"No rankings parsed from {filepath}")

    if voter_subsample is not None and len(rankings) > voter_subsample:
        rankings = rng.sample(rankings, voter_subsample)

    return Election(name=p.name, rankings=rankings, m=m)

# ---------------------------
# Label: majority cycle exists in strict majority relation
# ---------------------------
def majority_adj(e: Election) -> np.ndarray:
    m, n = e.m, len(e.rankings)
    wins = np.zeros((m, m), dtype=int)
    for r in e.rankings:
        pos = np.empty(m, dtype=int)
        for i, c in enumerate(r):
            pos[c] = i
        for a in range(m):
            for b in range(a+1, m):
                if pos[a] < pos[b]:
                    wins[a, b] += 1
                else:
                    wins[b, a] += 1
    adj = np.zeros((m, m), dtype=int)
    for a in range(m):
        for b in range(m):
            if a != b and wins[a, b] > n/2:
                adj[a, b] = 1
    return adj

def has_cycle(adj: np.ndarray) -> bool:
    m = adj.shape[0]
    state = [0]*m
    def dfs(v: int) -> bool:
        state[v] = 1
        for u in range(m):
            if adj[v, u]:
                if state[u] == 1:
                    return True
                if state[u] == 0 and dfs(u):
                    return True
        state[v] = 2
        return False
    for v in range(m):
        if state[v] == 0 and dfs(v):
            return True
    return False

# ---------------------------
# Borda point cloud
# ---------------------------
def borda_cloud(e: Election) -> np.ndarray:
    m, n = e.m, len(e.rankings)
    X = np.zeros((n, m), dtype=float)
    for i, r in enumerate(e.rankings):
        for pos, c in enumerate(r):
            X[i, c] = (m-1) - pos
    return X

# ---------------------------
# VR persistent homology up to H2 (Z2 reduction)
# To get H2 deaths, we build simplices up to dimension 3.
# ---------------------------
def rips_simplices(D: np.ndarray, maxdim: int, max_radius: float):
    n = D.shape[0]
    simplices = [[] for _ in range(maxdim+1)]
    simplices[0] = [(0.0, (i,)) for i in range(n)]

    edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = float(D[i, j])
            if w <= max_radius:
                edges.append((w, (i, j)))
    edges.sort(key=lambda x: x[0])
    simplices[1] = edges

    adj = [set() for _ in range(n)]
    for w, (i, j) in edges:
        adj[i].add(j); adj[j].add(i)

    if maxdim >= 2:
        tris = []
        for i in range(n):
            for j in [x for x in adj[i] if x > i]:
                common = [k for k in adj[i].intersection(adj[j]) if k > j]
                for k in common:
                    w = max(D[i, j], D[i, k], D[j, k])
                    if w <= max_radius:
                        tris.append((float(w), (i, j, k)))
        tris.sort(key=lambda x: x[0])
        simplices[2] = tris

    if maxdim >= 3:
        tets = []
        for w, (i, j, k) in simplices[2]:
            common = adj[i].intersection(adj[j]).intersection(adj[k])
            for l in [x for x in common if x > k]:
                ww = max(w, D[i, l], D[j, l], D[k, l])
                if ww <= max_radius:
                    tets.append((float(ww), (i, j, k, l)))
        tets.sort(key=lambda x: x[0])
        simplices[3] = tets

    return simplices

def persistence_diagrams_vr(X: np.ndarray, radius_quantile: float = 0.6):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    triu = D[np.triu_indices(D.shape[0], 1)]
    max_radius = float(np.quantile(triu, radius_quantile))

    simplices = rips_simplices(D, maxdim=3, max_radius=max_radius)

    all_simp = []
    for dim, lst in enumerate(simplices):
        for filt, verts in lst:
            all_simp.append((filt, dim, verts))
    all_simp.sort(key=lambda t: (t[0], t[1], t[2]))

    index = {(dim, verts): i for i, (f, dim, verts) in enumerate(all_simp)}
    filts = [f for f, _, _ in all_simp]

    columns = [0]*len(all_simp)
    for j, (f, dim, verts) in enumerate(all_simp):
        if dim == 0:
            columns[j] = 0
        else:
            bits = 0
            for t in range(len(verts)):
                fv = verts[:t] + verts[t+1:]
                i = index[(dim-1, fv)]
                bits ^= (1 << i)
            columns[j] = bits

    pivot_col: Dict[int, int] = {}
    reduced = [0]*len(all_simp)
    pair: Dict[int, int] = {}

    for j in range(len(all_simp)):
        col = columns[j]
        while col:
            pivot = col.bit_length() - 1
            pj = pivot_col.get(pivot)
            if pj is None:
                break
            col ^= reduced[pj]
        reduced[j] = col
        if col:
            pivot = col.bit_length() - 1
            pivot_col[pivot] = j
            pair[pivot] = j

    dgms = [[] for _ in range(3)]  # H0,H1,H2
    for i, (f, dim, verts) in enumerate(all_simp):
        if dim > 2:
            continue
        if i in pair:
            j = pair[i]
            dgms[dim].append((filts[i], filts[j]))
        else:
            if reduced[i] == 0:
                dgms[dim].append((filts[i], np.inf))

    return [np.array(d, float) if d else np.zeros((0, 2), float) for d in dgms]

def truncate_dgm(D: np.ndarray, L: int = 15) -> np.ndarray:
    if D.size == 0:
        return D
    finite = D[np.isfinite(D[:, 1])]
    if finite.size == 0:
        return np.zeros((0, 2), float)
    pers = finite[:, 1] - finite[:, 0]
    idx = np.argsort(-pers)[:L]
    return finite[idx]

# ---------------------------
# Polar Persistence Distance (PPD)
# ---------------------------
def polar_map(pt: np.ndarray, eps: float) -> np.ndarray:
    b, d = float(pt[0]), float(pt[1])
    r = max(d - b, 0.0)
    theta = math.atan((d + b) / (r + eps))
    return np.array([r, theta], float)

def wfun(r: float, p: float) -> float:
    rp = r**p
    return rp / (1.0 + rp)

def ppd_oneway(D1: np.ndarray, D2: np.ndarray, p: float = 2.0, eps: float = 1e-6) -> float:
    D1 = D1[np.isfinite(D1[:, 1])] if D1.size else D1
    D2 = D2[np.isfinite(D2[:, 1])] if D2.size else D2
    n1, n2 = D1.shape[0], D2.shape[0]
    if n1 == 0:
        return 0.0

    P1 = np.array([polar_map(D1[i], eps) for i in range(n1)])
    P2 = np.array([polar_map(D2[j], eps) for j in range(n2)]) if n2 > 0 else np.zeros((0, 2))

    C = np.zeros((n1, n2 + n1))
    for i in range(n1):
        r_i = P1[i, 0]
        w_i = wfun(r_i, p)
        if n2 > 0:
            dif = P1[i] - P2
            C[i, :n2] = w_i * np.sum(dif*dif, axis=1)
        diag_cost = w_i * float(np.dot(P1[i], P1[i]))
        C[i, n2:] = diag_cost

    nrows, ncols = C.shape
    Csq = np.vstack([C, np.zeros((ncols - nrows, ncols))]) if nrows < ncols else C

    ri, ci = linear_sum_assignment(Csq)
    total = 0.0
    for r, c in zip(ri, ci):
        if r < n1:
            total += float(Csq[r, c])
    return math.sqrt(max(total, 0.0))

def ppd_sym(D1: np.ndarray, D2: np.ndarray, p: float = 2.0, eps: float = 1e-6) -> float:
    d12 = ppd_oneway(D1, D2, p, eps)
    d21 = ppd_oneway(D2, D1, p, eps)
    return math.sqrt(0.5*(d12*d12 + d21*d21))

def election_dist(dg1, dg2, alpha=(0.6, 1.0, 0.2), p=2.0, eps=1e-6) -> float:
    d0 = ppd_sym(dg1[0], dg2[0], p, eps)
    d1 = ppd_sym(dg1[1], dg2[1], p, eps)
    d2 = ppd_sym(dg1[2], dg2[2], p, eps)
    a0, a1, a2 = alpha
    return math.sqrt(a0*d0*d0 + a1*d1*d1 + a2*d2*d2)

def kernel_matrix(diagrams, alpha=(0.6, 1.0, 0.2), lam: float = 1.0):
    N = len(diagrams)
    D = np.zeros((N, N), float)
    for i in range(N):
        for j in range(i+1, N):
            dij = election_dist(diagrams[i], diagrams[j], alpha=alpha)
            D[i, j] = D[j, i] = dij
    K = np.exp(-lam * (D**2))
    return K, D

def main():
    ELECTION_DIR = Path("./elections")  # put .soi files here
    voter_subsample = 30
    trunc_L = 15
    alpha = (0.6, 1.0, 0.2)
    lam = 1.0
    C = 1.0

    files = sorted(list(ELECTION_DIR.glob("*.soi")))
    if len(files) < 10:
        raise RuntimeError("Add at least ~10 .soi election files into ./elections/")

    pool: List[Tuple[Election,int]] = []
    for fp in files:
        try:
            e = parse_preflib_soi(str(fp), voter_subsample=voter_subsample, seed=0)
            y = 1 if has_cycle(majority_adj(e)) else 0
            pool.append((e, y))
        except Exception:
            continue

    if len(pool) < 10:
        raise RuntimeError("Could not parse enough elections.")

    E = [t[0] for t in pool]
    Y = np.array([t[1] for t in pool], dtype=int)

    diags = []
    for e in E:
        X = borda_cloud(e)
        dg = persistence_diagrams_vr(X, radius_quantile=0.6)
        dg = [truncate_dgm(dg[0], trunc_L), truncate_dgm(dg[1], trunc_L), truncate_dgm(dg[2], trunc_L)]
        diags.append(dg)

    K, _ = kernel_matrix(diags, alpha=alpha, lam=lam)

    idx = np.arange(len(Y))
    train_idx, test_idx = train_test_split(idx, test_size=0.25, random_state=0, stratify=Y)
    K_train = K[np.ix_(train_idx, train_idx)]
    K_test  = K[np.ix_(test_idx, train_idx)]

    clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
    clf.fit(K_train, Y[train_idx])
    pred = clf.predict(K_test)

    print("Accuracy:", accuracy_score(Y[test_idx], pred))
    print("Confusion matrix:\n", confusion_matrix(Y[test_idx], pred))
    print(classification_report(Y[test_idx], pred, digits=3))

    def summary_features(dg):
        out = {}
        for k, name in enumerate(["H0", "H1", "H2"]):
            finite = dg[k][np.isfinite(dg[k][:, 1])] if dg[k].size else np.zeros((0, 2))
            pers = (finite[:, 1] - finite[:, 0]) if finite.size else np.array([])
            out[f"{name}_n"] = len(finite)
            out[f"{name}_pers_sum"] = float(np.sum(pers)) if pers.size else 0.0
            out[f"{name}_pers_max"] = float(np.max(pers)) if pers.size else 0.0
        return out

    rows = []
    for e, y, dg in zip(E, Y, diags):
        feat = summary_features(dg)
        rows.append({
            "election": e.name,
            "m": e.m,
            "voters_used": len(e.rankings),
            "cycle_label": int(y),
            **feat
        })

    df = pd.DataFrame(rows).sort_values(["cycle_label", "election"], ascending=[False, True])
    df.to_csv("tda_pref_profile_results.csv", index=False)

    latex_df = df[["election","m","voters_used","cycle_label","H1_pers_sum","H1_pers_max","H2_pers_sum","H2_pers_max"]].copy()
    latex_df.columns = ["File","m","n\\_used","Label(cycle)","SumPers(H1)","MaxPers(H1)","SumPers(H2)","MaxPers(H2)"]
    with open("tda_pref_profile_results_table.tex","w") as f:
        f.write(latex_df.to_latex(index=False, float_format=lambda x: f"{x:.3f}"))

    print("Wrote: tda_pref_profile_results.csv and tda_pref_profile_results_table.tex")

if __name__ == "__main__":
    main()
