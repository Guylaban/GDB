"""Inter-rater reliability (reproduces Table 3).

Computes Cohen's quadratic-weighted kappa between A1 and A2 on the full
4134-response corpus for GF / FA / DS, and Gwet's AC1 for the binary
DS>=2 outcome. Then recomputes external validation kappas (R1 vs R2,
etc.) on the 191-item subset and prints the diff against the precomputed
values in kappa_results.json.

Note on binary agreement: we deliberately do not report Cohen's binary
kappa for the DS>=2 outcome. The two authors apply slightly different
thresholds on that boundary (A1 marginal 8.1%, A2 marginal 14.6%);
Cohen's binary kappa is depressed under that marginal asymmetry by a
known artifact even though percent agreement is high. Gwet's AC1 is the
prevalence-stable index used throughout the manuscript (AC1 = 0.896).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA, EXT  # noqa: E402


def quadratic_weighted_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Cohen's quadratic-weighted kappa from scratch."""
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    cats = np.union1d(np.unique(y1), np.unique(y2))
    cat_idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    if k == 1:
        return 1.0
    O = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        O[cat_idx[a], cat_idx[b]] += 1
    n = O.sum()
    row = O.sum(axis=1)
    col = O.sum(axis=0)
    E = np.outer(row, col) / n
    W = np.zeros((k, k), dtype=float)
    denom = (k - 1) ** 2
    for i in range(k):
        for j in range(k):
            W[i, j] = ((cats[i] - cats[j]) ** 2) / denom
    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 1.0


def gwet_ac1_binary(y1: np.ndarray, y2: np.ndarray) -> float:
    """Gwet's AC1 for two raters on binary labels."""
    y1 = np.asarray(y1).astype(int)
    y2 = np.asarray(y2).astype(int)
    po = (y1 == y2).mean()
    # Marginal proportions for category 1, averaged across raters.
    pi1 = ((y1.sum() + y2.sum()) / (2.0 * len(y1)))
    pe = 2.0 * pi1 * (1.0 - pi1)
    return (po - pe) / (1.0 - pe) if pe < 1 else 1.0


def _safe_pair(df: pd.DataFrame, c1: str, c2: str) -> pd.DataFrame:
    sub = df[[c1, c2]].dropna()
    return sub


def main() -> None:
    df = pd.read_csv(DATA)

    print("Author IRR (A1 vs A2) on full corpus")
    print("-" * 60)
    for dim in ["gf", "fa", "ds"]:
        if dim == "fa":
            # FA is coded as -1 (N/A) on F0 and F3 scenarios where no binding
            # constraint is present. Exclude those rows before computing kappa.
            sub = df[(df["fa_a1"] != -1) & (df["fa_a2"] != -1)][
                ["fa_a1", "fa_a2"]
            ].dropna()
        else:
            sub = _safe_pair(df, f"{dim}_a1", f"{dim}_a2")
        kw = quadratic_weighted_kappa(sub.iloc[:, 0].values, sub.iloc[:, 1].values)
        print(f"  {dim.upper():3s} quadratic-weighted kappa = {kw:.3f}  (n={len(sub)})")

    # Prevalence-stable agreement index on the binary headline outcome.
    sub = _safe_pair(df, "ds_a1", "ds_a2")
    b1 = (sub["ds_a1"] >= 2).astype(int).values
    b2 = (sub["ds_a2"] >= 2).astype(int).values
    print(f"  DS>=2 Gwet's AC1           = {gwet_ac1_binary(b1, b2):.3f}"
          f"  [paper reports 0.896]")

    print()
    print("External validation (191-item subset)")
    print("-" * 60)
    ext = pd.read_csv(EXT / "all_raters_scores.csv")

    pairs = {
        "A1 vs A2 (author IRR)": [("gf_a1", "gf_a2"), ("fa_a1", "fa_a2"),
                                  ("ds_a1", "ds_a2")],
        "R1 vs R2": [("gf_r1", "gf_r2"), ("fa_r1", "fa_r2"),
                     ("ds_r1", "ds_r2")],
        "R1 vs Author": [("gf_r1", "gf_author"), ("fa_r1", "fa_author"),
                         ("ds_r1", "ds_author")],
        "R2 vs Author": [("gf_r2", "gf_author"), ("fa_r2", "fa_author"),
                         ("ds_r2", "ds_author")],
    }

    recomputed: dict[str, dict[str, dict[str, float | int]]] = {}
    for label, dims in pairs.items():
        recomputed[label] = {}
        print(f"\n  {label}")
        for c1, c2 in dims:
            dim = c1.split("_")[0]
            sub = _safe_pair(ext, c1, c2)
            sub = sub[(sub[c1] != -1) & (sub[c2] != -1)]
            if len(sub) == 0:
                recomputed[label][dim] = {"kappa": None, "n": 0}
                print(f"    {dim.upper():3s}  (no overlap)")
                continue
            kw = quadratic_weighted_kappa(sub[c1].values, sub[c2].values)
            recomputed[label][dim] = {"kappa": round(kw, 3), "n": len(sub)}
            print(f"    {dim.upper():3s} kappa = {kw:.3f}  (n={len(sub)})")

    # Verify against the stored kappa_results.json TOTAL block.
    stored_path = EXT / "kappa_results.json"
    if stored_path.exists():
        stored = json.loads(stored_path.read_text())
        total = stored.get("TOTAL", {})
        print()
        print("Diff vs kappa_results.json (TOTAL block):")
        print("-" * 60)
        any_diff = False
        for label, dims in recomputed.items():
            for dim, val in dims.items():
                if val["kappa"] is None:
                    continue
                expected = total.get(label, {}).get(dim, {}).get("kappa")
                if expected is None:
                    continue
                delta = abs(val["kappa"] - expected)
                if delta > 0.005:
                    any_diff = True
                    print(f"  {label:<25s} {dim.upper():3s} "
                          f"recomputed={val['kappa']:.3f} stored={expected:.3f} "
                          f"diff={delta:.3f}")
        if not any_diff:
            print("  All recomputed kappas match stored values within 0.005.")

    external_vs_judges(ext)
    external_gold_tier_check(ext)


# ---------------------------------------------------------------------------
# External raters vs LLM judges, six-rater Krippendorff alpha, and the
# external-gold tier check (Table 1 rows 3 and 10; Section 5; Appendix H).
# These need the per-item model identity, released in the ``model`` column of
# all_raters_scores.csv.
# ---------------------------------------------------------------------------

def _load_judges() -> pd.DataFrame:
    import glob
    from _paths import LLM
    frames = []
    for tag, pattern in [("g", "judge_scal_gpt41_*.csv"), ("k", "judge_scal_kimi_k2_*.csv")]:
        parts = [pd.read_csv(f) for f in sorted(glob.glob(str(LLM / pattern)))]
        j = pd.concat(parts, ignore_index=True).drop_duplicates(["scenario_id", "response_type"])
        j = j.rename(columns={"response_type": "model", "goal_fidelity": f"gf_{tag}",
                              "feasibility_ack": f"fa_{tag}", "displacement_severity": f"ds_{tag}"})
        frames.append(j[["scenario_id", "model", f"gf_{tag}", f"fa_{tag}", f"ds_{tag}"]])
    return frames[0].merge(frames[1], on=["scenario_id", "model"], how="inner")


def _rint_avg(a, b) -> np.ndarray:
    """Average two ordinal scores and round half-to-even to an integer."""
    return np.rint((np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0).astype(int)


def _fa_avg(a, b) -> np.ndarray:
    """FA average for a pair of raters or judges.

    FA is -1 when a rater or judge marked the dimension not applicable. If
    both scores are valid the half-to-even rounded mean is used; if only one
    is valid that score is used; if neither is valid the result is -1 and the
    item is excluded from the FA comparison.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = np.where(np.isnan(a), -1.0, a); b = np.where(np.isnan(b), -1.0, b)
    both = (a >= 0) & (b >= 0)
    out = np.where(both, np.rint((a + b) / 2.0), np.where(a >= 0, a, np.where(b >= 0, b, -1.0)))
    return out.astype(int)


def _balanced_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
    tpr = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else float("nan")
    tnr = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else float("nan")
    return (tpr + tnr) / 2.0


def krippendorff_alpha_ordinal(M: np.ndarray) -> float:
    """Krippendorff's alpha with the ordinal metric.

    ``M`` is items x raters with NaN for missing scores. The coincidence
    matrix pairs each score only with the *other* raters' scores on the same
    item (self-pairs excluded), as in Krippendorff (2011). Matches the
    ``krippendorff`` package to three decimals.
    """
    R = np.asarray(M, dtype=float)
    vals = sorted({int(v) for v in R[~np.isnan(R)]})
    idx = {v: i for i, v in enumerate(vals)}
    K = len(vals)
    o = np.zeros((K, K))
    for row in R:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[idx[int(r[i])], idx[int(r[j])]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    delta = np.zeros((K, K))
    for c in range(K):
        for k in range(K):
            if c != k:
                lo, hi = min(c, k), max(c, k)
                s = n_c[lo:hi + 1].sum() - (n_c[c] + n_c[k]) / 2.0
                delta[c, k] = s * s
    Do = (o * delta).sum()
    De = sum(n_c[c] * n_c[k] * delta[c, k] for c in range(K) for k in range(K)) / (n - 1)
    return 1.0 - Do / De if De > 0 else float("nan")


def external_vs_judges(ext: pd.DataFrame) -> None:
    if "model" not in ext.columns:
        print("\n(external_vs_judges skipped: no model column in all_raters_scores.csv)")
        return
    j = _load_judges()
    d = ext.merge(j, on=["scenario_id", "model"], how="inner")
    print()
    print("External-Avg (R1,R2) vs Judge-Avg (GPT-4.1, Kimi-K2)  [Table 1, row 3]")
    print("-" * 60)
    print(f"  items joined: {len(d)} of {len(ext)}")
    gf_e, gf_j = _rint_avg(d["gf_r1"], d["gf_r2"]), _rint_avg(d["gf_g"], d["gf_k"])
    ds_e, ds_j = _rint_avg(d["ds_r1"], d["ds_r2"]), _rint_avg(d["ds_g"], d["ds_k"])
    fa_e, fa_j = _fa_avg(d["fa_r1"], d["fa_r2"]), _fa_avg(d["fa_g"], d["fa_k"])
    fam = (fa_e >= 0) & (fa_j >= 0)
    print(f"  GF kappa_w = {quadratic_weighted_kappa(gf_e, gf_j):.3f}  (n={len(d)})  [paper 0.591]")
    print(f"  FA kappa_w = {quadratic_weighted_kappa(fa_e[fam], fa_j[fam]):.3f}  (n={int(fam.sum())})  [paper 0.720]")
    print(f"  DS kappa_w = {quadratic_weighted_kappa(ds_e, ds_j):.3f}  (n={len(d)})  [paper 0.493]")
    ba = _balanced_accuracy((ds_e >= 2).astype(int), (ds_j >= 2).astype(int))
    print(f"  DS>=2 balanced accuracy (human side as reference) = {ba:.3f}  [paper 0.669]")

    print()
    print("Krippendorff's alpha (ordinal) across R1, R2, A1, A2, GPT-4.1, Kimi-K2  [Table 1, last row]")
    print("-" * 60)
    for dim, paper in [("gf", 0.715), ("fa", 0.737), ("ds", 0.648)]:
        cols = [f"{dim}_r1", f"{dim}_r2", f"{dim}_a1", f"{dim}_a2", f"{dim}_g", f"{dim}_k"]
        M = d[cols].to_numpy(dtype=float)
        if dim == "fa":
            M = np.where(M < 0, np.nan, M)
        M = M[(~np.isnan(M)).sum(axis=1) >= 2]
        print(f"  {dim.upper()} alpha_ord = {krippendorff_alpha_ordinal(M):.3f}  (items={len(M)})  [paper {paper:.3f}]")


def external_gold_tier_check(ext: pd.DataFrame, n_perm: int = 100000, seed: int = 0) -> None:
    if "model" not in ext.columns:
        return
    from headline_table import TIER
    from scipy import stats
    d = ext.copy()
    d["tier"] = d["model"].map(TIER)
    d["ext"] = (((d["ds_r1"] + d["ds_r2"]) / 2.0) >= 2).astype(int)
    d["auth"] = (((d["ds_a1"] + d["ds_a2"]) / 2.0) >= 2).astype(int)
    order = ["Frontier", "Mid-tier", "OW-large", "OW-small"]
    print()
    print("External-gold tier check on the 191-item subset  [Section 5; Appendix H]")
    print("-" * 60)
    for label, fams in [("Trap (F1+F2)", ["F1", "F2"]), ("F0", ["F0"]), ("F3", ["F3"])]:
        sub = d[d["family"].isin(fams)]
        g = sub.groupby("tier").agg(n=("ext", "size"), author=("auth", "mean"), external=("ext", "mean")).reindex(order)
        print(f"  {label}:")
        for t, r in g.iterrows():
            print(f"    {t:<9s} n={int(r['n']):3d}  author {100*r['author']:5.1f}%  external {100*r['external']:5.1f}%")
    trap = d[d["family"].isin(["F1", "F2"])]
    g = trap.groupby("model").agg(n=("ext", "size"), ext=("ext", "mean"), auth=("auth", "mean"))
    rng = np.random.default_rng(seed)
    for label, minn in [(">= 5 trap items", 5), ("all models", 1)]:
        s = g[g["n"] >= minn]
        rho, _ = stats.spearmanr(s["ext"], s["auth"])
        x = s["ext"].rank().to_numpy(); y = s["auth"].rank().to_numpy()
        obs = abs(stats.spearmanr(x, y)[0]); hits = 0
        for _ in range(n_perm):
            hits += abs(stats.spearmanr(x, rng.permutation(y))[0]) >= obs - 1e-12
        print(f"  per-model author vs external gold, {label}: n={len(s)}, Spearman rho={rho:.3f}, "
              f"permutation p={(hits + 1) / (n_perm + 1):.4f}")


if __name__ == "__main__":
    main()
