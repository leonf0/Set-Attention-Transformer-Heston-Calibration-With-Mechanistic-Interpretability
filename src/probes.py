import numpy as np


def ridge_fit_predict(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray,
                      alpha: float = 1.0):
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    y_mu = Y_tr.mean(axis=0)
    Yc = Y_tr - y_mu
    d = Z_tr.shape[1]
    A = Z_tr.T @ Z_tr + alpha * np.eye(d)
    W = np.linalg.solve(A, Z_tr.T @ Yc)
    return Z_te @ W + y_mu


def ridge_r2(X_tr, Y_tr, X_te, Y_te, alpha: float = 1.0) -> np.ndarray:
    pred = ridge_fit_predict(np.asarray(X_tr, dtype=np.float64),
                             np.asarray(Y_tr, dtype=np.float64),
                             np.asarray(X_te, dtype=np.float64), alpha)
    Y_te = np.asarray(Y_te, dtype=np.float64)
    ss_res = ((Y_te - pred) ** 2).sum(axis=0)
    ss_tot = ((Y_te - Y_te.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def pool_layer(act: np.ndarray) -> np.ndarray:
    act = np.asarray(act)
    if act.ndim == 3:
        return act.reshape(act.shape[0], -1) if act.shape[1] <= 8 else act.mean(axis=1)
    return act


def probe_layers(features_tr: dict, features_te: dict,
                 Y_tr: np.ndarray, Y_te: np.ndarray, alpha: float = 1.0) -> dict:
    out = {}
    for layer in features_tr:
        out[layer] = ridge_r2(pool_layer(features_tr[layer]),
                              Y_tr,
                              pool_layer(features_te[layer]),
                              Y_te, alpha=alpha)
    return out
