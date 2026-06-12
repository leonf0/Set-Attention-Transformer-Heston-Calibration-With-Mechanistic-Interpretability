import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED, DEVICE,
                        N_DRAWS, N_SAMPLES, PARAM_NAMES, PARAM_TEX,
                        RUN_AUG, SPARSITY_LEVELS, TAB_DIR, TRAIN_SEEDS)
from src.data import load_or_generate, make_splits, transform_labels
from src.models import SET_BASED
from src.runs import load_trained_model, run_is_complete
from src.sparsity import make_keep_mask, mask_seed, mlp_impute, sat_inputs
from src.training import compute_metrics, predict
from src.utils import arch_color, banner, savefig


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


def evaluate_sparsity(arch_runs, X_test, y_test_t, levels, n_draws, device):
    records = []
    B = len(X_test)
    for li, frac in enumerate(levels):
        for draw in range(n_draws if frac > 0 else 1):
            keep = make_keep_mask(B, frac, seed=mask_seed(li, draw))
            X_set = sat_inputs(X_test, keep) if frac > 0 else X_test
            X_imp = mlp_impute(X_test, keep) if frac > 0 else X_test
            for (run_label, arch, seed, model) in arch_runs:
                Xin = X_set if arch in SET_BASED else X_imp
                m = compute_metrics(predict(model, Xin, device=device), y_test_t)
                rec = {"run": run_label, "arch": arch, "seed": seed,
                       "sparsity": frac, "draw": draw,
                       "mae_mean": m["mae_mean"], "r2_mean": m["r2_mean"]}
                for p in PARAM_NAMES:
                    rec[f"mae_{p}"] = m["mae"][p]
                    rec[f"r2_{p}"] = m["r2"][p]
                records.append(rec)
            print(f"  level {frac:.0%} draw {draw}: evaluated {len(arch_runs)} runs", flush=True)
    return pd.DataFrame.from_records(records)


def plot_rq1_metric(df, metric_prefix, fname, ylabel):
    panels = PARAM_NAMES + ["mean"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, p in zip(axes.ravel(), panels):
        col = f"{metric_prefix}_{p}" if p != "mean" else f"{metric_prefix}_mean"
        for run_label, g in df.groupby("run"):
            per_seed = g.groupby(["seed", "sparsity"])[col].mean().reset_index()
            stats = per_seed.groupby("sparsity")[col].agg(["mean", "std"]).reset_index()
            base_arch = g["arch"].iloc[0]
            ls = "--" if run_label.endswith("_aug") else "-"
            ax.plot(stats["sparsity"], stats["mean"], marker="o", ls=ls,
                    color=arch_color(base_arch), label=run_label)
            ax.fill_between(stats["sparsity"],
                            stats["mean"] - stats["std"].fillna(0),
                            stats["mean"] + stats["std"].fillna(0),
                            color=arch_color(base_arch), alpha=0.15)
        ax.set_title(PARAM_TEX.get(p, "mean over params"))
        if metric_prefix == "r2":
            ax.set_ylim(min(0.0, df[col].min()) - 0.02, 1.01)
    for ax in axes[-1]:
        ax.set_xlabel("fraction of points dropped at test time")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("RQ1 — degradation under test-time sparsity (paired masks; "
                 "bands = ±1 std over training seeds)")
    savefig(plt, fname)


def run_rq1(include_aug=False, n_draws=None):
    n_draws = n_draws or N_DRAWS
    banner(f"RQ1 SPARSITY SWEEP  levels={SPARSITY_LEVELS} draws={n_draws} aug={include_aug}")
    X_test = bundle["X"][bundle["splits"]["test"]]
    y_test_t = bundle["y_t"][bundle["splits"]["test"]]

    arch_runs = []
    for arch, seed in itertools.product(("sat", "mlp"), TRAIN_SEEDS):
        for aug in ([False, True] if include_aug else [False]):
            if not run_is_complete(arch, seed, aug):
                if aug:
                    continue   # aug runs are optional
                raise RuntimeError(f"missing trained run {arch}/seed{seed}; run the training cell first")
            label = f"{arch}_aug" if aug else arch
            arch_runs.append((label, arch, seed, load_trained_model(arch, seed, DEVICE, aug=aug)))
    print(f"Loaded {len(arch_runs)} trained runs")

    df = evaluate_sparsity(arch_runs, X_test, y_test_t, SPARSITY_LEVELS, n_draws, DEVICE)
    df.to_csv(TAB_DIR / "rq1_sparsity.csv", index=False)
    plot_rq1_metric(df, "mae", "rq1_mae_vs_sparsity.png", "MAE (original scale)")
    plot_rq1_metric(df, "r2", "rq1_r2_vs_sparsity.png", "R²")

    pivot = (df[~df["run"].str.endswith("_aug")]
             .groupby(["run", "sparsity"])["mae_mean"].mean().unstack("sparsity"))
    print("\nMean-MAE by sparsity level (rows=model):")
    print(pivot.round(4).to_string())
    if {"sat", "mlp"} <= set(pivot.index):
        for run in ("sat", "mlp"):
            base = pivot.loc[run, 0.0]
            infl50 = (pivot.loc[run, 0.50] - base) / base * 100
            print(f"  {run}: mean-MAE inflation at 50% sparsity = {infl50:+.1f}% "
                  f"(preregistered: SAT <= +25%, MLP >= +100%)")
    return df


rq1_df = run_rq1(include_aug=RUN_AUG)
