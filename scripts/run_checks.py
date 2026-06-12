import numpy as np

from src.config import GRID_LOGM, GRID_SQRT_TAU, LOG_MONEYNESS, TAUS
from src.data import (inverse_transform_labels, sample_params,
                      transform_labels)
from src.heston import (bs_call, bs_price, build_iv_surface,
                        carr_madan_call_price, heston_cf)


def _bs_cf_adapter(u, tau, v0, kappa, theta, sigma, rho, r=0.0, q=0.0):
    return np.exp(-0.5 * v0 * tau * (u**2 + 1j * u))

_sig, _K = 0.2, 100.0 * np.exp(LOG_MONEYNESS)
_is_call = LOG_MONEYNESS >= 0
for _tau in (TAUS[0], TAUS[3], TAUS[-1]):
    _cp = carr_madan_call_price(100.0, _K, _tau, _sig**2, 0, 0, 0, 0, cf=_bs_cf_adapter)
    _bs = bs_call(100.0, _K, _tau, 0.0, _sig)
    assert np.abs(_cp - _bs).max() < 1e-10
    _otm = np.where(_is_call, _cp, _cp - 100.0 + _K)
    _bso = bs_price(100.0, _K, _tau, 0.0, _sig, _is_call)
    _m = _bso > 1e-9
    assert np.abs(_otm[_m] / _bso[_m] - 1.0).max() < 1e-4
print("PASS  inversion exact (abs < 1e-10; parity wings rel < 1e-4)")

_p50, _ = sample_params(50, seed=5)
for _p in _p50:
    for _tau in (TAUS[0], 1.0):
        assert abs(heston_cf(np.array([-1j]), _tau, *_p)[0] - 1.0) < 1e-10
print("PASS  martingale identity over 50 LHS draws")

_surf, _ = build_iv_surface(np.array([_sig**2, 3.0, _sig**2, 1e-6, -0.5]))
_pbs = bs_price(100.0, 100.0 * np.exp(GRID_LOGM), GRID_SQRT_TAU**2, 0.0, _sig,
                GRID_LOGM >= 0)
_ident = _pbs > 1e-3
assert not np.any(np.isnan(_surf[_ident, 2]))
assert np.abs(_surf[_ident, 2] - _sig).max() < 2e-3
print(f"PASS  flat-vol surface recovers IV=0.20 on {_ident.sum()}/120 identifiable points")

_pp, _ = sample_params(200, seed=1)
assert np.allclose(inverse_transform_labels(transform_labels(_pp)), _pp, atol=1e-10)
print("PASS  label-transform round-trip")
