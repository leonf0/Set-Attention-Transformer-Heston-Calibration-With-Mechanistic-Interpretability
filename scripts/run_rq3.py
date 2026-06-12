import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED, DEVICE,
                        N_GRID, N_SAMPLES, NOISE_LEVELS, NOISE_SEED_BASE,
                        PARAM_NAMES, PERMUTATION_SEED, TAB_DIR,
                        TRAIN_SEEDS)
from src.data import (add_iv_noise, load_or_generate, make_splits,
                      transform_labels)
from src.models import SET_BASED
from src.runs import load_trained_model
from src.training import compute_metrics, predict
from src.utils import arch_color, banner, savefig


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


banner("RQ3  (a) PERMUTATION INVARIANCE")
N_PERM_SURFACES, N_PERMS = 200, 10
test_idx = bundle["splits"]["test"]
Xp = bundle["X"][test_idx[:N_PERM_SURFACES]]

rng = np.random.default_rng(PERMUTATION_SEED)
perms = [rng.permutation(N_GRID) for _ in range(N_PERMS)]

perm_rows, sat_max_dev = [], 0.0
for arch in ("sat", "mlp"):
    for ts in TRAIN_SEEDS:
        model = load_trained_model(arch, ts, DEVICE)
        base = predict(model, Xp, device=DEVICE)
        max_dev = 0.0
        for perm in perms:
            dev = np.abs(predict(model, Xp[:, perm, :], device=DEVICE) - base).max()
            max_dev = max(max_dev, float(dev))
        perm_rows.append({"arch": arch, "seed": ts, "max_abs_dev": max_dev,
                          "set_based": arch in SET_BASED})
        print(f"  [{arch}/seed{ts}] max |Δpred| over {N_PERMS} permutations: {max_dev:.2e}")
        if arch == "sat":
            sat_max_dev = max(sat_max_dev, max_dev)
pd.DataFrame(perm_rows).to_csv(TAB_DIR / "rq3_permutation.csv", index=False)
tol = 1e-3
status = "PASS" if sat_max_dev < tol else "FAIL"
print(f"  SAT permutation-invariance assertion (preregistered < {tol:.0e}): "
      f"{status} (actual {sat_max_dev:.2e})")
assert sat_max_dev < tol, "SAT failed the permutation-invariance assertion"

banner("RQ3  (b) NOISE ROBUSTNESS")
X_test = bundle["X"][test_idx]
y_test_t = bundle["y_t"][test_idx]

noise_rows = []
levels = (0.0,) + tuple(NOISE_LEVELS)
for li, sig in enumerate(levels):
    if sig == 0.0:
        Xn = X_test
    else:
        rng_n = np.random.default_rng(NOISE_SEED_BASE + li)
        Xn = add_iv_noise(X_test, sig, rng_n)
    for arch in ("sat", "mlp"):
        for ts in TRAIN_SEEDS:
            model = load_trained_model(arch, ts, DEVICE)
            m = compute_metrics(predict(model, Xn, device=DEVICE), y_test_t)
            row = {"arch": arch, "seed": ts, "noise": sig,
                   "mae_mean": m["mae_mean"], "r2_mean": m["r2_mean"]}
            for p in PARAM_NAMES:
                row[f"mae_{p}"] = m["mae"][p]
                row[f"r2_{p}"] = m["r2"][p]
            noise_rows.append(row)
    print(f"  σ={sig:.0%}: evaluated {2 * len(TRAIN_SEEDS)} runs", flush=True)
noise_df = pd.DataFrame(noise_rows)
noise_df.to_csv(TAB_DIR / "rq3_noise.csv", index=False)

fig, ax = plt.subplots(figsize=(5.6, 3.8))
for arch, g in noise_df.groupby("arch"):
    stats = g.groupby("noise")["mae_mean"].agg(["mean", "std"]).reset_index()
    ax.plot(stats["noise"], stats["mean"], marker="o", color=arch_color(arch), label=arch)
    ax.fill_between(stats["noise"], stats["mean"] - stats["std"].fillna(0),
                    stats["mean"] + stats["std"].fillna(0),
                    color=arch_color(arch), alpha=0.15)
ax.set_xlabel("multiplicative IV noise σ")
ax.set_ylabel("mean MAE (original scale)")
ax.set_title("RQ3 — robustness to test-time IV noise (trained clean)")
ax.legend()
savefig(plt, "rq3_noise_mae.png")

print("\nMean-MAE by noise level:")
print(noise_df.groupby(["arch", "noise"])["mae_mean"].mean().unstack("noise").round(4).to_string())
