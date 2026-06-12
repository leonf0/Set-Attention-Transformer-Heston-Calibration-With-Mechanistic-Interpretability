import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import qmc
from torch.utils.data import DataLoader, Dataset

from .config import (BOUNDS_HI, BOUNDS_LO, DATA_DIR, DEFAULT_DATA_SEED,
                     DEFAULT_SPLIT_SEED, GRID_LOGM, GRID_SQRT_TAU, N_GRID,
                     TEST_FRAC, VAL_FRAC)
from .heston import build_iv_surface


_CLIP_LO, _CLIP_HI = -12.0, 8.0


def transform_labels(params: np.ndarray) -> np.ndarray:
    t = np.empty_like(params, dtype=float)
    t[:, 0] = np.log(params[:, 0])
    t[:, 1] = np.log(params[:, 1])
    t[:, 2] = np.log(params[:, 2])
    t[:, 3] = np.log(params[:, 3])
    t[:, 4] = np.arctanh(params[:, 4])
    return t


def inverse_transform_labels(transformed: np.ndarray, clip: bool = True) -> np.ndarray:
    t = np.asarray(transformed, dtype=float)
    if clip:
        t = np.clip(t, _CLIP_LO, _CLIP_HI)
    p = np.empty_like(t)
    p[:, 0] = np.exp(t[:, 0])
    p[:, 1] = np.exp(t[:, 1])
    p[:, 2] = np.exp(t[:, 2])
    p[:, 3] = np.exp(t[:, 3])
    p[:, 4] = np.tanh(t[:, 4])
    return p


def sample_params(n_samples: int, seed: int = 42, reject_feller: bool = True):
    if reject_feller:
        oversample_factor = 3
        sampler = qmc.LatinHypercube(d=5, seed=seed)
        raw = sampler.random(n=n_samples * oversample_factor)
        scaled = qmc.scale(raw, BOUNDS_LO, BOUNDS_HI)

        kappa, theta, sigma = scaled[:, 1], scaled[:, 2], scaled[:, 3]
        feller_mask = (2 * kappa * theta) > (sigma**2)
        params = scaled[feller_mask][:n_samples]
        if len(params) < n_samples:
            raise ValueError(
                f"Only {len(params)} Feller-satisfying samples from "
                f"{n_samples * oversample_factor} draws. Increase oversample_factor."
            )
        return params, np.ones(len(params), dtype=bool)

    sampler = qmc.LatinHypercube(d=5, seed=seed)
    scaled = qmc.scale(sampler.random(n=n_samples), BOUNDS_LO, BOUNDS_HI)
    kappa, theta, sigma = scaled[:, 1], scaled[:, 2], scaled[:, 3]
    feller_flags = (2 * kappa * theta) > (sigma**2)
    return scaled, feller_flags


def generate_dataset(n_samples: int = 50_000, seed: int = 42, verbose: bool = True):
    params_arr, _ = sample_params(n_samples, seed=seed)

    X_list, y_list = [], []
    skipped = 0
    t0 = time.time()
    for i, params in enumerate(params_arr):
        report_every = max(200, n_samples // 20)
        if verbose and i > 0 and i % report_every == 0:
            rate = i / max(time.time() - t0, 1e-9)
            eta = (n_samples - i) / max(rate, 1e-9)
            print(f"Generating sample {i}/{n_samples}  "
                  f"({rate:.0f}/s, ETA {eta/60:.1f} min)", flush=True)

        surface, _ = build_iv_surface(params)
        ivs = surface[:, 2]
        if np.any(np.isnan(ivs)) or np.any(ivs < 0.01) or np.any(ivs > 3.0):
            skipped += 1
            continue
        X_list.append(surface)
        y_list.append(params)

    if verbose:
        print(f"Skipped {skipped} samples ({100 * skipped / n_samples:.1f}%) | "
              f"total {time.time() - t0:.1f}s", flush=True)

    X = np.asarray(X_list)
    y = np.asarray(y_list)

    assert X.shape[1] == N_GRID
    assert np.allclose(X[0, :, 0], GRID_LOGM) and np.allclose(X[0, :, 1], GRID_SQRT_TAU)
    meta = {"n_requested": n_samples, "n_kept": len(X), "skipped": skipped,
            "data_seed": seed}
    return X, y, meta


def add_iv_noise(X: np.ndarray, noise_std: float, rng: np.random.Generator):
    Xn = X.copy()
    noise = rng.normal(0.0, noise_std, size=Xn.shape[:2])
    Xn[:, :, 2] = np.maximum(Xn[:, :, 2] * (1.0 + noise), 1e-4)
    return Xn


def dataset_path(n_samples: int, data_seed: int) -> Path:
    return Path(DATA_DIR) / f"surfaces_n{n_samples}_dataseed{data_seed}.npz"


def load_or_generate(n_samples: int, data_seed: int = DEFAULT_DATA_SEED,
                     verbose: bool = True):
    path = dataset_path(n_samples, data_seed)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            X, y = z["X"], z["y"]
            meta = json.loads(str(z["meta_json"]))
        if verbose:
            print(f"Loaded cached dataset {path.name}: X{X.shape} y{y.shape}")
        return X, y, meta

    if verbose:
        print(f"No cache at {path.name}; generating {n_samples} surfaces ...")
    X, y, meta = generate_dataset(n_samples=n_samples, seed=data_seed, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32), y=y,
                        meta_json=json.dumps(meta))
    if verbose:
        print(f"Cached dataset to {path}")
    return X.astype(np.float32), y, meta


def make_splits(n: int, split_seed: int = DEFAULT_SPLIT_SEED,
                val_frac: float = VAL_FRAC, test_frac: float = TEST_FRAC):
    path = Path(DATA_DIR) / f"splits_n{n}_seed{split_seed}.npz"
    if path.exists():
        with np.load(path) as z:
            return {k: z[k] for k in ("train", "val", "test")}
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    splits = {
        "test": np.sort(perm[:n_test]),
        "val": np.sort(perm[n_test:n_test + n_val]),
        "train": np.sort(perm[n_test + n_val:]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **splits)
    return splits


def load_arrays_and_splits(n_samples: int, data_seed: int = DEFAULT_DATA_SEED,
                           split_seed: int = DEFAULT_SPLIT_SEED, verbose: bool = True):
    X, y, meta = load_or_generate(n_samples, data_seed, verbose=verbose)
    X = X.astype(np.float32)
    y_t = transform_labels(y).astype(np.float32)
    splits = make_splits(len(X), split_seed)
    if verbose:
        print(f"Split — train: {len(splits['train'])}  val: {len(splits['val'])}  "
              f"test: {len(splits['test'])}")
    return X, y, y_t, splits, meta


def build_loaders(X: np.ndarray, y_t: np.ndarray, splits: dict,
                  batch_size: int, train_seed: int, num_workers: int = 0):

    class HestonSurfaceDataset(Dataset):
        def __init__(self, X_, y_):
            self.X = torch.from_numpy(np.ascontiguousarray(X_)).float()
            self.y = torch.from_numpy(np.ascontiguousarray(y_)).float()

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    pin = torch.cuda.is_available()
    gen = torch.Generator().manual_seed(train_seed)
    mk = lambda idx: HestonSurfaceDataset(X[idx], y_t[idx])
    train_loader = DataLoader(mk(splits["train"]), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, generator=gen)
    val_loader = DataLoader(mk(splits["val"]), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    test_loader = DataLoader(mk(splits["test"]), batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, test_loader
