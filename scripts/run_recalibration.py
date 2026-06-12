import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED, DEVICE,
                        N_SAMPLES, TAB_DIR, TRAIN_SEEDS)
from src.data import (inverse_transform_labels, load_or_generate,
                      make_splits, transform_labels)
from src.recalibration import (recalibration_rmse, reprice_surface,
                               summarize_recalibration)
from src.runs import load_trained_model
from src.training import predict
from src.utils import arch_color, banner, savefig


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


N_RECAL = 1000
SELF_CHECK_N = 25

banner(f"RECALIBRATION  ({N_RECAL} test surfaces)")
idx = bundle["splits"]["test"][:N_RECAL]
Xr = bundle["X"][idx]
yr = bundle["y"][idx]
true_iv = Xr[..., 2].astype(np.float64)

print("Self-consistency: re-pricing TRUE parameters...")
sc = []
for i in range(min(SELF_CHECK_N, len(idx))):
    iv_hat, _ = reprice_surface(yr[i])
    v = np.isfinite(iv_hat)
    sc.append(np.sqrt(np.mean((iv_hat[v] - true_iv[i, v]) ** 2)))
print(f"  TRUE-params IV RMSE: median {np.median(sc):.2e} "
      f"(expected ≈ bisection tolerance; sanity bound < 1e-4)")
assert np.median(sc) < 1e-4, "pricing pipeline not self-consistent"

recal_rows, per_surface = [], {}
for arch in ("sat", "mlp"):
    for ts in TRAIN_SEEDS:
        model = load_trained_model(arch, ts, DEVICE)
        pred_t = predict(model, Xr, device=DEVICE)
        pred = inverse_transform_labels(pred_t)
        print(f"  [{arch}/seed{ts}] re-pricing predictions...")
        rmse, failures = recalibration_rmse(pred, true_iv, verbose_every=500)
        summ = summarize_recalibration(rmse, failures)
        summ.update({"arch": arch, "seed": ts})
        param_mae = np.mean(np.abs(pred - yr), axis=1)
        valid = np.isfinite(rmse)
        summ["corr_param_mae_vs_rmse"] = float(
            np.corrcoef(param_mae[valid], rmse[valid])[0, 1])
        recal_rows.append(summ)
        per_surface[(arch, ts)] = (param_mae, rmse)
        print(f"    median IV RMSE {summ['median_iv_rmse_bps']:.1f} bps | "
              f"failure rate {summ['point_failure_rate']:.2%} | "
              f"corr(param MAE, RMSE) {summ['corr_param_mae_vs_rmse']:.3f}")

pd.DataFrame(recal_rows).to_csv(TAB_DIR / "recalibration.csv", index=False)

fig, ax = plt.subplots(figsize=(5.6, 3.8))
s0 = TRAIN_SEEDS[0]
for arch in ("sat", "mlp"):
    _, rmse = per_surface[(arch, s0)]
    ax.hist(rmse[np.isfinite(rmse)] * 1e4, bins=60, alpha=0.5,
            color=arch_color(arch), label=f"{arch} (seed {s0})")
ax.set_xlabel("per-surface IV RMSE (bps)")
ax.set_ylabel("count")
ax.set_title("Recalibration error distribution")
ax.legend()
savefig(plt, "recal_hist.png")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), squeeze=False)
for ax, arch in zip(axes[0], ("sat", "mlp")):
    pm, rmse = per_surface[(arch, s0)]
    v = np.isfinite(rmse)
    ax.scatter(pm[v], rmse[v] * 1e4, s=4, alpha=0.3, color=arch_color(arch))
    ax.set_xlabel("per-surface parameter MAE")
    ax.set_ylabel("IV RMSE (bps)")
    ax.set_title(f"{arch} (seed {s0})")
fig.suptitle("Parameter error vs surface reconstruction error")
savefig(plt, "recal_scatter.png")
