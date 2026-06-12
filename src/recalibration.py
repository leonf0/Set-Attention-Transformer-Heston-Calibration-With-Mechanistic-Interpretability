import numpy as np

from .config import config
from .heston import build_iv_surface


def reprice_surface(params: np.ndarray):
    surface, n_fail = build_iv_surface(params)
    return surface[:, 2], n_fail


def recalibration_rmse(pred_params: np.ndarray, true_iv: np.ndarray,
                       verbose_every: int = 200):
    B = len(pred_params)
    rmse = np.full(B, np.nan)
    failures = np.zeros(B, dtype=int)
    for i in range(B):
        iv_hat, _ = reprice_surface(pred_params[i])
        valid = np.isfinite(iv_hat)
        failures[i] = int(np.sum(~valid))
        if valid.any():
            d = iv_hat[valid] - true_iv[i, valid]
            rmse[i] = float(np.sqrt(np.mean(d ** 2)))
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    re-priced {i + 1}/{B} surfaces "
                  f"(median RMSE so far {np.nanmedian(rmse[:i + 1]) * 1e4:.1f} bps)",
                  flush=True)
    return rmse, failures


def summarize_recalibration(rmse: np.ndarray, failures: np.ndarray) -> dict:
    valid = np.isfinite(rmse)
    return {
        "n_surfaces": int(len(rmse)),
        "mean_iv_rmse": float(np.nanmean(rmse)),
        "median_iv_rmse": float(np.nanmedian(rmse)),
        "p90_iv_rmse": float(np.nanpercentile(rmse[valid], 90)) if valid.any() else float("nan"),
        "mean_iv_rmse_bps": float(np.nanmean(rmse) * 1e4),
        "median_iv_rmse_bps": float(np.nanmedian(rmse) * 1e4),
        "point_failure_rate": float(failures.sum() / (len(rmse) * config.N_GRID)),
        "surfaces_with_any_failure": int(np.sum(failures > 0)),
    }
