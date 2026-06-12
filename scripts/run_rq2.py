import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (ATTENTION_SUBSET_SEED, DEFAULT_DATA_SEED,
                        DEFAULT_SPLIT_SEED, DEVICE, LOG_MONEYNESS,
                        N_SAMPLES, PARAM_NAMES, PARAM_TEX, TAB_DIR,
                        TAU_DAYS, TRAIN_SEEDS, surface_values_to_grid)
from src.data import load_or_generate, make_splits, transform_labels
from src.interp import (attention_entropy, collect_pma_attention,
                        effective_cells, mean_attention_maps,
                        pairwise_cosine, seed_ablation_table,
                        top_frac_mass)
from src.runs import load_trained_model
from src.utils import banner, savefig


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


CONCENTRATION_THRESHOLD = 0.40
DISTINCT_COSINE = 0.80
CAUSAL_RATIO = 1.5


def plot_attention(maps, train_seed):
    fig, axes = plt.subplots(1, maps.shape[0], figsize=(3.1 * maps.shape[0], 3.4))
    grids = surface_values_to_grid(maps)
    vmax = grids.max()
    for s, ax in enumerate(np.atleast_1d(axes)):
        im = ax.imshow(grids[s], aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax,
                       extent=[-0.5, len(TAU_DAYS) - 0.5, LOG_MONEYNESS[0], LOG_MONEYNESS[-1]])
        ax.set_xticks(range(len(TAU_DAYS)))
        ax.set_xticklabels(TAU_DAYS, fontsize=6, rotation=45)
        ax.set_title(f"seed vector {s}")
        ax.set_xlabel("maturity (days)")
        ax.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("log-moneyness")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean attention")
    fig.suptitle(f"RQ2 — PMA seed attention, SAT training seed {train_seed} "
                 f"(mean over test surfaces)")
    savefig(plt, f"rq2_attention_seed{train_seed}.png")


banner(f"RQ2 SEED SPECIALIZATION  SAT seeds={list(TRAIN_SEEDS)}")
N_ATTN = 1000
test_idx = bundle["splits"]["test"]
rng = np.random.default_rng(ATTENTION_SUBSET_SEED)
attn_idx = test_idx[rng.choice(len(test_idx), size=min(N_ATTN, len(test_idx)),
                               replace=False)]
X_attn = bundle["X"][attn_idx]
X_abl, y_abl_t = bundle["X"][test_idx], bundle["y_t"][test_idx]

stat_rows, abl_rows = [], []
concentrated_per_seed, distinct_ok_per_seed, causal_hits_per_seed = [], [], []

for ts in TRAIN_SEEDS:
    model = load_trained_model("sat", ts, DEVICE)
    attn = collect_pma_attention(model, X_attn, device=DEVICE)
    maps = mean_attention_maps(attn)
    plot_attention(maps, ts)

    mass = top_frac_mass(maps)
    ent = attention_entropy(maps)
    eff = effective_cells(maps)
    cos = pairwise_cosine(maps)
    off_diag = cos[~np.eye(len(cos), dtype=bool)]
    for s in range(len(maps)):
        stat_rows.append({"train_seed": ts, "pma_seed": s,
                          "top20_mass": mass[s], "entropy": ent[s],
                          "effective_cells": eff[s],
                          "max_offdiag_cosine": float(np.max(np.delete(cos[s], s)))})
    n_conc = int((mass >= CONCENTRATION_THRESHOLD).sum())
    n_distinct_pairs = int((off_diag < DISTINCT_COSINE).sum() // 2)
    concentrated_per_seed.append(n_conc)
    distinct_ok_per_seed.append(n_distinct_pairs)
    print(f"  [sat/seed{ts}] top-20% mass per PMA seed: "
          f"{np.array2string(mass, precision=3)}  -> {n_conc}/4 concentrated; "
          f"{n_distinct_pairs}/6 pairs distinct")

    tab = seed_ablation_table(model, X_abl, y_abl_t, device=DEVICE)
    delta = tab["delta_mae"]
    for s in range(delta.shape[0]):
        row = {"train_seed": ts, "pma_seed": s}
        for j, p in enumerate(PARAM_NAMES):
            row[f"dmae_{p}"] = delta[s, j]
        abl_rows.append(row)
    hits = 0
    for j in range(delta.shape[1]):
        col = delta[:, j]
        top, med = float(np.max(col)), float(np.median(col))
        if top > 0 and (med <= 0 or top >= CAUSAL_RATIO * max(med, 1e-12)):
            hits += 1
    causal_hits_per_seed.append(hits)
    print(f"  [sat/seed{ts}] params with a clearly dominant seed (>= {CAUSAL_RATIO}x median ΔMAE): {hits}/5")

stats_df = pd.DataFrame(stat_rows)
abl_df = pd.DataFrame(abl_rows)
stats_df.to_csv(TAB_DIR / "rq2_attention_stats.csv", index=False)
abl_df.to_csv(TAB_DIR / "rq2_seed_ablation.csv", index=False)

mean_delta = (abl_df.groupby("pma_seed")[[f"dmae_{p}" for p in PARAM_NAMES]]
              .mean().values)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
im = ax.imshow(mean_delta, cmap="Reds", aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels([PARAM_TEX[p] for p in PARAM_NAMES])
ax.set_yticks(range(mean_delta.shape[0]))
ax.set_yticklabels([f"seed {s}" for s in range(mean_delta.shape[0])])
for i in range(mean_delta.shape[0]):
    for j in range(mean_delta.shape[1]):
        ax.text(j, i, f"{mean_delta[i, j]:.3f}", ha="center", va="center", fontsize=7)
ax.grid(False)
fig.colorbar(im, label="ΔMAE (ablated − baseline)")
ax.set_title("RQ2 — zero-ablation ΔMAE (mean over training seeds)")
savefig(plt, "rq2_ablation_heatmap.png")

banner("RQ2 AUTOMATIC VERDICT vs PREREGISTRATION.md")
c1 = all(n >= 3 for n in concentrated_per_seed)
c2 = all(n >= 4 for n in distinct_ok_per_seed)
c3 = all(h >= 2 for h in causal_hits_per_seed)
print(f"  P1 concentration  (>=3/4 seeds with top-20% mass >= {CONCENTRATION_THRESHOLD}, "
      f"every training seed): {'PASS' if c1 else 'FAIL'}  {concentrated_per_seed}")
print(f"  P2 distinctness   (>=4/6 pairs with cosine < {DISTINCT_COSINE}): "
      f"{'PASS' if c2 else 'FAIL'}  {distinct_ok_per_seed}")
print(f"  P3 causal mapping (>=2/5 params with a dominant seed): "
      f"{'PASS' if c3 else 'FAIL'}  {causal_hits_per_seed}")
if c1 and c2 and c3:
    print("  => Specialization hypothesis SUPPORTED under the preregistered criteria.")
else:
    print("  => Specialization hypothesis NOT supported; per PREREGISTRATION.md "
          "§falsification the write-up drops the seed-specialization claim and "
          "reports these numbers as-is.")
