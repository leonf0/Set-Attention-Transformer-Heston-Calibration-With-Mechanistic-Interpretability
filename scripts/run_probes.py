import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED, DEVICE,
                        N_SAMPLES, PARAM_NAMES, PARAM_TEX, TAB_DIR,
                        TRAIN_SEEDS)
from src.data import load_or_generate, make_splits, transform_labels
from src.interp import collect_internals
from src.probes import probe_layers
from src.runs import load_trained_model
from src.utils import banner, savefig


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


PROBE_LAYERS = ["input", "input_flat", "embed", "sab1", "sab2", "pooled"]
N_PROBE_TRAIN = 8000
N_PROBE_TEST = 5000

banner(f"LINEAR PROBES  layers={PROBE_LAYERS}")
tr_idx = bundle["splits"]["train"][:N_PROBE_TRAIN]
te_idx = bundle["splits"]["test"][:N_PROBE_TEST]
X_tr, X_te = bundle["X"][tr_idx], bundle["X"][te_idx]
Y_tr, Y_te = bundle["y_t"][tr_idx], bundle["y_t"][te_idx]


def gather_probe_features(model, Xs):
    acts = collect_internals(model, Xs, device=DEVICE,
                             keys=("embed", "sab1", "sab2", "pooled"))
    feats = {"input": Xs.mean(axis=1),
             "input_flat": Xs.reshape(len(Xs), -1)}
    feats.update(acts)
    return feats


probe_rows = []
for ts in TRAIN_SEEDS:
    model = load_trained_model("sat", ts, DEVICE)
    f_tr = gather_probe_features(model, X_tr)
    f_te = gather_probe_features(model, X_te)
    r2 = probe_layers(f_tr, f_te, Y_tr, Y_te, alpha=1.0)
    for layer in PROBE_LAYERS:
        if layer not in r2:
            continue
        row = {"train_seed": ts, "layer": layer}
        for j, p in enumerate(PARAM_NAMES):
            row[f"r2_{p}"] = float(r2[layer][j])
        probe_rows.append(row)
    print(f"  [sat/seed{ts}] probe R² (mean over params): "
          + ", ".join(f"{l}={np.mean(r2[l]):.3f}" for l in PROBE_LAYERS if l in r2))

probe_df = pd.DataFrame(probe_rows)
probe_df.to_csv(TAB_DIR / "probe_r2.csv", index=False)

mean_r2 = (probe_df.groupby("layer")[[f"r2_{p}" for p in PARAM_NAMES]]
           .mean().reindex(PROBE_LAYERS))
fig, ax = plt.subplots(figsize=(6.2, 3.8))
im = ax.imshow(mean_r2.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(5)); ax.set_xticklabels([PARAM_TEX[p] for p in PARAM_NAMES])
ax.set_yticks(range(len(PROBE_LAYERS))); ax.set_yticklabels(PROBE_LAYERS)
for i in range(mean_r2.shape[0]):
    for j in range(mean_r2.shape[1]):
        v = mean_r2.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                color="white" if v < 0.6 else "black")
ax.grid(False)
fig.colorbar(im, label="test R² (ridge probe)")
ax.set_title("Per-parameter linear decodability across depth (mean over seeds)")
savefig(plt, "probe_r2_heatmap.png")
