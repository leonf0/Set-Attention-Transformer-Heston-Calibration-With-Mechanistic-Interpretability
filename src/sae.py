import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import config
from .data import transform_labels


def _short_tau_slice(iv_flat: np.ndarray) -> np.ndarray:
    return iv_flat[:, : config.N_LOGM]


def _fit_quadratic_coeffs(y: np.ndarray, x: np.ndarray):
    A = np.stack([np.ones_like(x), x, x ** 2], axis=1)
    pinv = np.linalg.pinv(A)
    coefs = y @ pinv.T
    return coefs[:, 1], coefs[:, 2]


def surface_properties(iv_flat: np.ndarray, y_orig: np.ndarray) -> dict[str, np.ndarray]:
    iv_flat = np.asarray(iv_flat)
    props: dict[str, np.ndarray] = {}

    y_t = transform_labels(np.asarray(y_orig))
    for j, name in enumerate(config.PARAM_NAMES):
        props[f"t_{name}"] = y_t[:, j]

    atm = config.ATM_LOGM_IDX
    short = _short_tau_slice(iv_flat)
    long_rows = iv_flat[:, (config.N_TAU - 1) * config.N_LOGM:]
    props["atm_short"] = short[:, atm]
    props["term_slope"] = long_rows[:, atm] - short[:, atm]

    slope, curve = _fit_quadratic_coeffs(short, config.LOG_MONEYNESS)
    props["skew_short"] = slope
    props["curve_short"] = 2.0 * curve
    return props


def make_sae(d_in: int, expansion: int = 8):

    class SparseAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            n_feat = d_in * expansion
            self.n_feat = n_feat
            self.W_enc = nn.Parameter(torch.empty(d_in, n_feat))
            self.b_enc = nn.Parameter(torch.zeros(n_feat))
            self.W_dec = nn.Parameter(torch.empty(n_feat, d_in))
            self.b_dec = nn.Parameter(torch.zeros(d_in))
            nn.init.kaiming_uniform_(self.W_enc, a=5 ** 0.5)
            with torch.no_grad():
                self.W_dec.copy_(self.W_enc.t())
                self._normalize_decoder()

        @torch.no_grad()
        def _normalize_decoder(self):
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
            self.W_dec.div_(norms)

        def encode(self, x):
            return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

        def forward(self, x):
            f = self.encode(x)
            return f @ self.W_dec + self.b_dec, f

    return SparseAutoencoder()


def train_sae(acts: np.ndarray, expansion: int = 8, l1: float = 5e-3,
              lr: float = 1e-3, batch_size: int = 4096, epochs: int = 25,
              device=None, seed: int = 0, verbose: bool = True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    acts = np.asarray(acts, dtype=np.float32)
    norm_scale = float(np.linalg.norm(acts, axis=1).mean())
    X = torch.from_numpy(acts / norm_scale).to(device)

    sae = make_sae(acts.shape[1], expansion).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    M = len(X)
    history = []
    for epoch in range(epochs):
        perm = torch.from_numpy(rng.permutation(M)).to(device)
        tot_rec, tot_l1, n = 0.0, 0.0, 0
        for i in range(0, M, batch_size):
            xb = X[perm[i:i + batch_size]]
            opt.zero_grad(set_to_none=True)
            x_hat, f = sae(xb)
            rec = ((x_hat - xb) ** 2).sum(dim=1).mean()
            sp = f.abs().sum(dim=1).mean()
            (rec + l1 * sp).backward()
            opt.step()
            sae._normalize_decoder()
            tot_rec += rec.item() * len(xb)
            tot_l1 += sp.item() * len(xb)
            n += len(xb)
        history.append({"epoch": epoch, "rec": tot_rec / n, "l1": tot_l1 / n})
        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            print(f"    SAE epoch {epoch:2d} | rec {tot_rec / n:.4f} | "
                  f"L1 {tot_l1 / n:.3f}", flush=True)

    with torch.no_grad():
        idx = torch.from_numpy(rng.choice(M, size=min(M, 50_000), replace=False)).to(device)
        xb = X[idx]
        x_hat, f = sae(xb)
        resid_var = ((x_hat - xb) ** 2).sum().item()
        total_var = ((xb - xb.mean(dim=0)) ** 2).sum().item()
        fvu = resid_var / max(total_var, 1e-12)
        l0 = (f > 0).float().sum(dim=1).mean().item()
        dead = int((f.max(dim=0).values <= 0).sum().item())
    info = {"norm_scale": norm_scale, "history": history, "fvu": fvu,
            "mean_l0": l0, "dead_features": dead, "n_features": sae.n_feat,
            "expansion": expansion, "l1": l1, "epochs": epochs}
    if verbose:
        print(f"    SAE final: FVU {fvu:.3f} | mean L0 {l0:.1f} | dead {dead}/{sae.n_feat}")
    return sae, info


def sae_features(sae, acts: np.ndarray, norm_scale: float,
                 device=None, batch_size: int = 8192) -> np.ndarray:
    if device is None:
        device = next(sae.parameters()).device
    outs = []
    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            xb = torch.as_tensor(acts[i:i + batch_size] / norm_scale,
                                 dtype=torch.float32, device=device)
            outs.append(sae.encode(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def feature_property_correlations(feat_per_surface: np.ndarray,
                                  props: dict[str, np.ndarray]) -> dict:
    F = np.asarray(feat_per_surface, dtype=np.float64)
    Fc = F - F.mean(axis=0)
    Fs = F.std(axis=0)
    names = list(props)
    corr = np.full((len(names), F.shape[1]), np.nan)
    for i, name in enumerate(names):
        p = np.asarray(props[name], dtype=np.float64)
        pc = p - p.mean()
        ps = p.std()
        if ps < 1e-12:
            continue
        valid = Fs > 1e-12
        corr[i, valid] = (pc @ Fc[:, valid]) / (len(p) * ps * Fs[valid])
    return {"corr": corr, "prop_names": names}
