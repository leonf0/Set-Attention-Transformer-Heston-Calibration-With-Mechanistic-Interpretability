import numpy as np
from scipy.special import ndtr

from .config import LOG_MONEYNESS, TAUS


VOL_LO = 1e-6
VOL_HI = 5.0


def bs_price(S, K, tau, r, sigma, is_call):
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    F = S * np.exp(r * tau)
    srt = sigma * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * tau) / srt
    d2 = d1 - srt
    disc = np.exp(-r * tau)
    call = disc * (F * ndtr(d1) - K * ndtr(d2))
    put = disc * (K * ndtr(-d2) - F * ndtr(-d1))
    return np.where(is_call, call, put)


def bs_call(S, K, tau, r, sigma):
    return bs_price(S, K, tau, r, sigma, is_call=True)


def bs_put(S, K, tau, r, sigma):
    return bs_price(S, K, tau, r, sigma, is_call=False)


def implied_vol_bisection(S, K, tau, r, price, is_call,
                          lo=VOL_LO, hi=VOL_HI, n_iter=100):
    K, tau, price, is_call = np.broadcast_arrays(
        np.asarray(K, float), np.asarray(tau, float),
        np.asarray(price, float), np.asarray(is_call, bool),
    )
    lo_arr = np.full(price.shape, lo, dtype=float)
    hi_arr = np.full(price.shape, hi, dtype=float)

    disc_K = K * np.exp(-r * tau)
    intrinsic = np.where(is_call, np.maximum(S - disc_K, 0.0),
                         np.maximum(disc_K - S, 0.0))
    invalid = price < intrinsic - 1e-8
    invalid |= price <= 0.0
    invalid |= price < bs_price(S, K, tau, r, lo_arr, is_call)
    invalid |= price > bs_price(S, K, tau, r, hi_arr, is_call)

    for _ in range(n_iter):
        mid = 0.5 * (lo_arr + hi_arr)
        go_up = bs_price(S, K, tau, r, mid, is_call) < price
        lo_arr = np.where(go_up, mid, lo_arr)
        hi_arr = np.where(go_up, hi_arr, mid)

    iv = 0.5 * (lo_arr + hi_arr)
    iv = np.where(invalid, np.nan, iv)
    return iv


def heston_cf(u, tau, v0, kappa, theta, sigma, rho, r=0.0, q=0.0):
    xi = kappa - rho * sigma * 1j * u
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j * u))
    g2 = (xi - d) / (xi + d)
    exp_m_d_tau = np.exp(-d * tau)

    C = (
        (r - q) * 1j * u * tau
        + (kappa * theta / sigma**2)
        * ((xi - d) * tau - 2.0 * np.log((1.0 - g2 * exp_m_d_tau) / (1.0 - g2)))
    )
    D = ((xi - d) / sigma**2) * ((1.0 - exp_m_d_tau) / (1.0 - g2 * exp_m_d_tau))
    return np.exp(C + D * v0)


def _gl_panels(u_max: float, n_panels: int, nodes_per_panel: int,
               n_fine: int = 10):
    key = (float(u_max), int(n_panels), int(nodes_per_panel), int(n_fine))
    cached = _GL_CACHE.get(key)
    if cached is not None:
        return cached
    x, w = np.polynomial.legendre.leggauss(nodes_per_panel)   # on [-1, 1]
    width = u_max / n_panels
    edges = np.unique(np.r_[np.linspace(0.0, width, n_fine + 1),
                            np.arange(1, n_panels + 1) * width])
    a, b = edges[:-1, None], edges[1:, None]
    u = (0.5 * (b - a) * x[None, :] + 0.5 * (a + b)).ravel()
    wq = (0.5 * (b - a) * w[None, :]).ravel()
    _GL_CACHE[key] = (u, wq)
    return u, wq


_GL_CACHE: dict = {}


def carr_madan_call_price(S, K_arr, tau, v0, kappa, theta, sigma, rho,
                          r=0.0, q=0.0, alpha=1.5,
                          u_max=1500.0, n_panels=60, nodes_per_panel=32,
                          cf=None):
    u, wq = _gl_panels(u_max, n_panels, nodes_per_panel)

    cf_fn = heston_cf if cf is None else cf
    cf_arg = u - (alpha + 1) * 1j
    phi = np.exp(1j * cf_arg * np.log(S)) * cf_fn(
        cf_arg, tau, v0, kappa, theta, sigma, rho, r, q)
    denominator = alpha**2 + alpha - u**2 + 1j * u * (2.0 * alpha + 1.0)
    psi = np.exp(-r * tau) * phi / denominator            # (n_u,)

    k = np.log(np.asarray(K_arr, dtype=float))            # (n_K,)
    kernel = np.exp(-1j * np.outer(k, u))                 # (n_K, n_u)
    integral = kernel @ (psi * wq)                        # (n_K,)
    call_prices = np.exp(-alpha * k) / np.pi * integral.real
    return np.maximum(call_prices, 0.0)


def build_iv_surface(params, S: float = 100.0, r: float = 0.0,
                     log_moneyness_grid=LOG_MONEYNESS, tau_arr=TAUS):
    v0, kappa, theta, sigma, rho = params
    log_m = np.asarray(log_moneyness_grid, dtype=float)
    K_slice = S * np.exp(log_m)
    is_call_slice = log_m >= 0
    n_logm, n_tau = len(log_m), len(tau_arr)

    prices = np.empty((n_tau, n_logm))
    for j, tau in enumerate(tau_arr):
        cp = carr_madan_call_price(S, K_slice, tau, v0, kappa, theta, sigma, rho, r=r)
        disc_K = K_slice * np.exp(-r * tau)
        put_p = np.maximum(cp - S + disc_K, 0.0)
        prices[j] = np.where(is_call_slice, cp, put_p)

    K_full = np.tile(K_slice, n_tau)
    tau_full = np.repeat(np.asarray(tau_arr, float), n_logm)
    is_call_full = np.tile(is_call_slice, n_tau)
    iv = implied_vol_bisection(S, K_full, tau_full, r, prices.ravel(), is_call_full)

    surface = np.stack(
        [np.tile(log_m, n_tau), np.sqrt(tau_full), iv], axis=1
    )
    return surface, int(np.isnan(iv).sum())
