import numpy as np
import torch

from .config import config
from .training import compute_metrics, predict


def collect_internals(model, X: np.ndarray, device=None, batch_size: int = 1024,
                      keys=("embed", "sab1", "sab2", "pooled")) -> dict:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    acc = {k: [] for k in keys}
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            _, internals = model.forward_with_internals(xb)
            for k in keys:
                if k in internals:
                    acc[k].append(internals[k].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in acc.items() if v}


def collect_token_activations(model, X: np.ndarray, site: str = "sab2",
                              device=None, batch_size: int = 1024) -> np.ndarray:
    return collect_internals(model, X, device, batch_size, keys=(site,))[site]


def collect_pma_attention(model, X: np.ndarray, device=None,
                          batch_size: int = 1024) -> np.ndarray:
    attn = collect_internals(model, X, device, batch_size, keys=("pma_attn",))["pma_attn"]
    return attn.mean(axis=1)


def mean_attention_maps(attn: np.ndarray) -> np.ndarray:
    return np.asarray(attn).mean(axis=0)


def top_frac_mass(maps: np.ndarray, frac: float = 0.20) -> np.ndarray:
    maps = np.asarray(maps)
    k = max(1, int(round(frac * maps.shape[1])))
    sorted_desc = np.sort(maps, axis=1)[:, ::-1]
    return sorted_desc[:, :k].sum(axis=1) / maps.sum(axis=1)


def attention_entropy(maps: np.ndarray) -> np.ndarray:
    p = np.asarray(maps)
    p = p / p.sum(axis=1, keepdims=True)
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1)


def effective_cells(maps: np.ndarray) -> np.ndarray:
    return np.exp(attention_entropy(maps))


def pairwise_cosine(maps: np.ndarray) -> np.ndarray:
    m = np.asarray(maps, dtype=np.float64)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return m @ m.T


def seed_ablation_table(model, X: np.ndarray, y_t: np.ndarray, device=None) -> dict:
    base_pred = predict(model, X, device=device,
                        forward=lambda xb: model.forward_seed_ablated(xb, None))
    base = compute_metrics(base_pred, y_t)
    n_seeds = model.n_seeds
    ablated, delta = [], np.zeros((n_seeds, len(config.PARAM_NAMES)))
    for s in range(n_seeds):
        pred = predict(model, X, device=device,
                       forward=lambda xb, s=s: model.forward_seed_ablated(xb, s))
        m = compute_metrics(pred, y_t)
        ablated.append(m)
        for j, name in enumerate(config.PARAM_NAMES):
            delta[s, j] = m["mae"][name] - base["mae"][name]
    return {"baseline": base, "ablated": ablated, "delta_mae": delta}
