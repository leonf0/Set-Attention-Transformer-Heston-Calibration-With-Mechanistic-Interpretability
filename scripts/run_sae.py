import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED, DEVICE,
                        LOG_MONEYNESS, N_GRID, N_SAMPLES, TAB_DIR,
                        TAU_DAYS, TRAIN_SEEDS, surface_values_to_grid)
from src.data import load_or_generate, make_splits, transform_labels
from src.interp import collect_token_activations
from src.runs import load_trained_model
from src.sae import (feature_property_correlations, sae_features,
                     surface_properties, train_sae)
from src.utils import banner, save_json, savefig


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


SAE_SITE, SAE_EXPANSION, SAE_L1 = "sab2", 8, 5e-3
SAE_EPOCHS = 25
SAE_MODEL_SEED = TRAIN_SEEDS[0]
N_SAE_TRAIN = 8000
N_CHAR = 2000

banner(f"SAE  site={SAE_SITE}  expansion={SAE_EXPANSION}x  λ={SAE_L1}")
model = load_trained_model("sat", SAE_MODEL_SEED, DEVICE)

tr_idx = bundle["splits"]["train"][:N_SAE_TRAIN]
print(f"Collecting {len(tr_idx)} × {N_GRID} training activations at '{SAE_SITE}'...")
acts_tr = collect_token_activations(model, bundle["X"][tr_idx], site=SAE_SITE, device=DEVICE)
flat_tr = acts_tr.reshape(-1, acts_tr.shape[-1])
print(f"  SAE training set: {flat_tr.shape}")

sae, sae_info = train_sae(flat_tr, expansion=SAE_EXPANSION, l1=SAE_L1,
                          epochs=SAE_EPOCHS, device=DEVICE, seed=SAE_MODEL_SEED)
save_json(TAB_DIR / "sae_summary.json",
          {k: v for k, v in sae_info.items() if k != "history"} |
          {"site": SAE_SITE, "model_seed": SAE_MODEL_SEED,
           "n_train_acts": int(len(flat_tr))})

te_idx = bundle["splits"]["test"][:N_CHAR]
X_char = bundle["X"][te_idx]
print(f"Characterizing on {len(te_idx)} held-out surfaces...")
acts_te = collect_token_activations(model, X_char, site=SAE_SITE, device=DEVICE)
B, N, d = acts_te.shape
feats = sae_features(sae, acts_te.reshape(-1, d), sae_info["norm_scale"], device=DEVICE)
feats = feats.reshape(B, N, -1)
feat_per_surface = feats.mean(axis=1)

props = surface_properties(X_char[..., 2].astype(np.float64), bundle["y"][te_idx])
cp = feature_property_correlations(feat_per_surface, props)
corr, prop_names = cp["corr"], cp["prop_names"]

sae_rows, top_features = [], {}
for i, name in enumerate(prop_names):
    r = corr[i]
    order = np.argsort(-np.abs(np.nan_to_num(r)))
    for rank in range(3):
        f = int(order[rank])
        sae_rows.append({"property": name, "rank": rank, "feature": f,
                         "pearson_r": float(r[f])})
    top_features[name] = int(order[0])
sae_df = pd.DataFrame(sae_rows)
sae_df.to_csv(TAB_DIR / "sae_top_features.csv", index=False)
print("Top |r| per property:")
print(sae_df[sae_df["rank"] == 0].set_index("property")[["feature", "pearson_r"]]
      .round(3).to_string())

fig, ax = plt.subplots(figsize=(6.4, 3.6))
maxr = [abs(sae_df[(sae_df.property == n) & (sae_df["rank"] == 0)]["pearson_r"].iloc[0])
        for n in prop_names]
ax.bar(range(len(prop_names)), maxr, color="#2ca02c")
ax.set_xticks(range(len(prop_names))); ax.set_xticklabels(prop_names, rotation=45, ha="right")
ax.set_ylabel("max |Pearson r| over features")
ax.set_ylim(0, 1)
ax.set_title(f"SAE feature ↔ surface-property alignment "
             f"(FVU {sae_info['fvu']:.2f}, L0 {sae_info['mean_l0']:.0f}, "
             f"dead {sae_info['dead_features']}/{sae_info['n_features']})")
savefig(plt, "sae_property_corr.png")

uniq = sorted(set(top_features.values()))
maps = feats[:, :, uniq].mean(axis=0).T
grids = surface_values_to_grid(maps)
ncol = min(5, len(uniq))
nrow = int(np.ceil(len(uniq) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.9 * nrow), squeeze=False)
inv = {f: [n for n, ff in top_features.items() if ff == f] for f in uniq}
for k, f in enumerate(uniq):
    ax = axes[k // ncol][k % ncol]
    im = ax.imshow(grids[k], aspect="auto", origin="lower", cmap="magma",
                   extent=[-0.5, len(TAU_DAYS) - 0.5, LOG_MONEYNESS[0], LOG_MONEYNESS[-1]])
    ax.set_title(f"feat {f}\n({', '.join(inv[f])})", fontsize=7)
    ax.set_xticks(range(0, len(TAU_DAYS), 2))
    ax.set_xticklabels(TAU_DAYS[::2], fontsize=6)
    ax.grid(False)
for k in range(len(uniq), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")
fig.suptitle("Mean token activation of the top SAE feature per property")
savefig(plt, "sae_feature_maps.png")
