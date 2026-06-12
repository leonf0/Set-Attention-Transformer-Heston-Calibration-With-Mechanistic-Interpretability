import copy
import time

import numpy as np
import torch
import torch.nn as nn

from .config import config
from .data import inverse_transform_labels


def train_one_epoch(model, loader, optimizer, device, grad_clip: float, aug_fn=None) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        if aug_fn is not None:
            xb = aug_fn(xb)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total, n_elems = 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        total += loss_fn(model(xb), yb).item()
        n_elems += yb.numel()
    return total / max(n_elems, 1)


def run_training(model, train_loader, val_loader, device,
                 max_epochs=None, lr=None, weight_decay=None, patience=None,
                 scheduler_patience=None, scheduler_factor=None, grad_clip=None,
                 aug_fn=None, verbose: bool = True):
    d = config.TRAIN_DEFAULTS
    max_epochs = max_epochs or d["max_epochs"]
    lr = lr or d["lr"]
    weight_decay = d["weight_decay"] if weight_decay is None else weight_decay
    patience = patience or d["patience"]
    scheduler_patience = scheduler_patience or d["scheduler_patience"]
    scheduler_factor = scheduler_factor or d["scheduler_factor"]
    grad_clip = d["grad_clip"] if grad_clip is None else grad_clip

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience)

    best_val, best_state, best_epoch = float("inf"), None, -1
    history = {"train_loss": [], "val_loss": [], "lr": []}
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(max_epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device, grad_clip, aug_fn)
        va = evaluate(model, val_loader, device)
        scheduler.step(va)
        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        history["lr"].append(cur_lr)

        if va < best_val - 1e-7:
            best_val, best_epoch, bad_epochs = va, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1

        if verbose and (epoch % 5 == 0 or bad_epochs >= patience or epoch == max_epochs - 1):
            print(f"    epoch {epoch:3d} | train {tr:.5f} | val {va:.5f} | "
                  f"lr {cur_lr:.1e} | best {best_val:.5f}@{best_epoch} | "
                  f"{time.time() - t0:.0f}s", flush=True)
        if bad_epochs >= patience:
            if verbose:
                print(f"    early stop at epoch {epoch} (patience {patience})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["best_epoch"] = best_epoch
    history["wall_seconds"] = time.time() - t0
    return model, history


def predict(model, X: np.ndarray, device=None, batch_size: int = 2048,
            forward=None) -> np.ndarray:

    if device is None:
        device = next(model.parameters()).device
    fwd = forward if forward is not None else model
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(fwd(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def compute_metrics(pred_t: np.ndarray, true_t: np.ndarray) -> dict:
    pred = inverse_transform_labels(pred_t)
    true = inverse_transform_labels(true_t)
    mae, r2 = {}, {}
    for j, name in enumerate(config.PARAM_NAMES):
        err = pred[:, j] - true[:, j]
        mae[name] = float(np.mean(np.abs(err)))
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((true[:, j] - true[:, j].mean()) ** 2))
        r2[name] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "mae": mae,
        "r2": r2,
        "mae_mean": float(np.mean(list(mae.values()))),
        "r2_mean": float(np.mean(list(r2.values()))),
    }
