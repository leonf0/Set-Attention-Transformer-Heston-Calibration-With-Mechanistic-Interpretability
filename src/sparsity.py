import numpy as np
import torch

from .config import config


def make_keep_mask(
    n_surfaces: int,
    drop_frac: float,
    n_points: int = config.N_GRID,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng(seed)
    n_drop = int(round(drop_frac * n_points))
    n_drop = min(max(n_drop, 0), n_points - 1)
    if n_drop == 0:
        return np.ones((n_surfaces, n_points), dtype=bool)

    order = np.argsort(rng.random((n_surfaces, n_points)), axis=1)
    keep = np.ones((n_surfaces, n_points), dtype=bool)
    np.put_along_axis(keep, order[:, :n_drop], False, axis=1)
    return keep


def mask_seed(level_idx: int, draw: int) -> int:
    return config.SPARSITY_MASK_SEED_BASE + 97 * draw + level_idx


def kept_indices(keep: np.ndarray) -> np.ndarray:
    keep = np.asarray(keep, dtype=bool)
    counts = keep.sum(axis=1)
    k = int(counts[0])
    if not np.all(counts == k):
        raise ValueError("kept_indices requires an equal keep-count per row")
    idx = np.argsort(~keep, axis=1, kind="stable")[:, :k]
    return idx.astype(np.int64)


def sat_inputs(X: np.ndarray, keep: np.ndarray) -> np.ndarray:
    idx = kept_indices(keep)
    return np.take_along_axis(X, idx[:, :, None], axis=1)


def mlp_impute(X: np.ndarray, keep: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    keep = np.asarray(keep, dtype=bool)
    iv = X[..., 2]
    counts = keep.sum(axis=1, keepdims=True).astype(iv.dtype)
    mean_obs = (iv * keep).sum(axis=1, keepdims=True) / np.maximum(counts, 1.0)
    iv_imp = np.where(keep, iv, mean_obs)
    out = X.copy()
    out[..., 2] = iv_imp
    return out

def make_train_augmenter(set_mode: bool, run_seed: int, levels=config.SPARSITY_LEVELS):
    n_points = config.N_GRID
    levels = tuple(float(f) for f in levels)
    gen_cpu = torch.Generator(device="cpu").manual_seed(int(run_seed) + 555_000)

    def aug_fn(xb: "torch.Tensor") -> "torch.Tensor":
        B = xb.shape[0]
        lvl = levels[int(torch.randint(len(levels), (1,), generator=gen_cpu).item())]
        n_drop = int(round(lvl * n_points))
        if n_drop <= 0:
            return xb
        n_keep = n_points - n_drop
        scores = torch.rand(B, n_points, generator=gen_cpu).to(xb.device)
        order = torch.argsort(scores, dim=1)
        keep_idx, _ = torch.sort(order[:, :n_keep], dim=1)
        if set_mode:
            return torch.gather(xb, 1, keep_idx.unsqueeze(-1).expand(-1, -1, xb.shape[-1]))
        keep = torch.zeros(B, n_points, dtype=torch.bool, device=xb.device)
        keep.scatter_(1, keep_idx, True)
        iv = xb[..., 2]
        mean_obs = (iv * keep).sum(dim=1, keepdim=True) / n_keep
        out = xb.clone()
        out[..., 2] = torch.where(keep, iv, mean_obs)
        return out

    return aug_fn
