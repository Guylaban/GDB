"""Arena Elo vs trap displacement correlation (reproduces Figure 2 numbers).

Hardcodes a small dict of paper-time Chatbot Arena Elo values for 15 of
the 18 models (3 had no public Elo at submission time). Loads the
unified dataset, computes per-model trap displacement (mean of F1+F2
binary outcomes), and prints Spearman rho, Pearson r, n, and the
leave-one-out range for Spearman.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DATA  # noqa: E402
from headline_table import DISPLAY, avg_rater_displaced  # noqa: E402

# Chatbot Arena (LMArena) Elo values as used for Figure 2 of the paper
# (leaderboard snapshot taken at submission time, spring 2026). With these
# values and Trap = mean(F1, F2) the script reproduces rho = -0.706, p = 0.003
# (n = 15). Updating the values to a current leaderboard will change rho.
ELO = {
    "gemini-2.5-pro": 1467,
    "gemini-3.1-pro-preview": 1490,
    "gpt-5.4": 1484,
    "claude-sonnet-4-6": 1468,
    "deepseek-chat": 1334,
    "gpt-4o": 1340,
    "gemini-2.5-flash": 1415,
    "gpt-4o-mini": 1288,
    "ollama:mistral": 1097,
    "ollama:llama3": 1193,
    "ollama:llama3.1:8b": 1195,
    "ollama:llama3.1:70b": 1310,
    "ollama:qwen2.5:32b": 1272,
    "ollama:qwen2.5:72b": 1310,
    "ollama:nemotron": 1250,
}

# Models without public Elo at submission time (kept for documentation):
#   deepseek-reasoner, claude-haiku-4-5-20251001, gpt-5.4-mini-2026-03-17


def main() -> None:
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["ds_a1", "ds_a2"]).copy()
    df["disp"] = avg_rater_displaced(df)

    trap = df[df["trap_family"].isin([1, 2])]
    per_model = trap.groupby("model")["disp"].mean()

    rows = []
    for m, elo in ELO.items():
        if m in per_model.index:
            rows.append((m, elo, per_model[m]))
    paired = pd.DataFrame(rows, columns=["model", "elo", "trap_rate"])

    rho, p_rho = stats.spearmanr(paired["elo"], paired["trap_rate"])
    r_p, p_r = stats.pearsonr(paired["elo"], paired["trap_rate"])

    print(f"n = {len(paired)} (of 18 models; 3 lack public Elo)")
    print(f"Spearman rho = {rho:.3f}, p = {p_rho:.3e}"
          f"  [paper: rho = -0.706, p = 0.003, with the Elo snapshot stored in ELO]")
    print(f"Pearson r    = {r_p:.3f}, p = {p_r:.3e}")

    # Leave-one-out range for Spearman.
    los: list[float] = []
    for i in range(len(paired)):
        sub = paired.drop(paired.index[i])
        rho_i, _ = stats.spearmanr(sub["elo"], sub["trap_rate"])
        los.append(rho_i)
    print(f"LOO Spearman rho range: [{min(los):.3f}, {max(los):.3f}]")

    print()
    print("Per-model values used:")
    paired_sorted = paired.sort_values("elo")
    for _, r in paired_sorted.iterrows():
        name = DISPLAY.get(r["model"], r["model"])
        print(f"  {name:<22s} Elo {int(r['elo']):>5d}   "
              f"trap {r['trap_rate'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
