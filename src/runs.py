from pathlib import Path

import torch

from .config import config
from .data import build_loaders
from .models import SET_BASED, build_model, count_params
from .sparsity import make_train_augmenter
from .training import compute_metrics, predict, run_training
from .utils import load_json, save_json, set_seed


def run_name(arch: str, aug: bool) -> str:
    return f"{arch}_aug" if aug else arch


def get_run_dir(arch: str, seed: int, aug: bool = False) -> Path:
    return config.RUNS_DIR / run_name(arch, aug) / f"seed{seed}"


def run_is_complete(arch: str, seed: int, aug: bool = False) -> bool:
    d = get_run_dir(arch, seed, aug)
    return (d / "best.pt").exists() and (d / "test_metrics.json").exists()


def load_trained_model(arch: str, seed: int, device, aug: bool = False):
    """Rebuild the architecture and load its best checkpoint (weights_only)."""
    d = get_run_dir(arch, seed, aug)
    model = build_model(arch)
    state = torch.load(d / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def train_one(arch: str, seed: int, arrays: dict, device, aug: bool = False,
              max_epochs=None, batch_size=None, force: bool = False) -> dict:
    run_dir = get_run_dir(arch, seed, aug)
    if run_is_complete(arch, seed, aug) and not force:
        print(f"  [{run_name(arch, aug)}/seed{seed}] already complete - skipping")
        return load_json(run_dir / "test_metrics.json")

    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    model = build_model(arch)
    n_params = count_params(model)
    print(f"  [{run_name(arch, aug)}/seed{seed}] {n_params:,} params on {device}")

    X, y_t, splits = arrays["X"], arrays["y_t"], arrays["splits"]
    train_loader, val_loader, _ = build_loaders(
        X, y_t, splits,
        batch_size=batch_size or config.TRAIN_DEFAULTS["batch_size"],
        train_seed=seed)
    aug_fn = make_train_augmenter(set_mode=arch in SET_BASED, run_seed=seed) if aug else None

    model, history = run_training(model, train_loader, val_loader, device,
                                  max_epochs=max_epochs, aug_fn=aug_fn)

    pred_t = predict(model, X[splits["test"]], device=device)
    metrics = compute_metrics(pred_t, y_t[splits["test"]])
    metrics["n_params"] = n_params
    metrics["arch"] = arch
    metrics["seed"] = seed
    metrics["aug"] = bool(aug)
    metrics["best_val_loss"] = history["best_val_loss"]
    metrics["best_epoch"] = history["best_epoch"]
    metrics["wall_seconds"] = history["wall_seconds"]

    torch.save(model.state_dict(), run_dir / "best.pt")
    save_json(run_dir / "config.json", {
        "arch": arch, "seed": seed, "aug": bool(aug), "n_params": n_params,
        "train_defaults": config.TRAIN_DEFAULTS,
        "data_seed": config.DEFAULT_DATA_SEED, "split_seed": config.DEFAULT_SPLIT_SEED,
        "n_train": int(len(splits["train"])), "n_val": int(len(splits["val"])),
        "n_test": int(len(splits["test"])),
    })
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "test_metrics.json", metrics)
    print(f"    test mae_mean {metrics['mae_mean']:.4f} | r2_mean {metrics['r2_mean']:.4f}")
    return metrics


def train_many(archs, seeds, arrays, device, aug: bool = False,
               max_epochs=None, batch_size=None) -> list[dict]:
    results = []
    for arch in archs:
        for seed in seeds:
            results.append(train_one(arch, seed, arrays, device, aug=aug,
                                     max_epochs=max_epochs, batch_size=batch_size))
    return results


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
