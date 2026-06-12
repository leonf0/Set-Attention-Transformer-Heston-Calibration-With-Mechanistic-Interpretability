import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device: {DEVICE}")


RUN_AUG = False

N_SAMPLES   = 50_000
TRAIN_SEEDS = (0, 1, 2)
MAX_EPOCHS  = 200

N_DRAWS     = 3
BATCH_SIZE  = 512

REPO_ROOT = Path.cwd()
DATA_DIR = REPO_ROOT / "data"

RUN_TAG = f"n{N_SAMPLES}"
RUNS_DIR = REPO_ROOT / "runs" / RUN_TAG
RESULTS_DIR = REPO_ROOT / "results" / RUN_TAG
FIG_DIR = RESULTS_DIR / "figures"
TAB_DIR = RESULTS_DIR / "tables"
for _d in (DATA_DIR, RUNS_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

PARAM_NAMES = ["v0", "kappa", "theta", "sigma", "rho"]
PARAM_TEX = {"v0": r"$v_0$", "kappa": r"$\kappa$", "theta": r"$\theta$",
             "sigma": r"$\xi$", "rho": r"$\rho$"}

BOUNDS_LO = np.array([0.01, 0.5, 0.01, 0.1, -0.95])
BOUNDS_HI = np.array([0.25, 8.0, 0.25, 1.0, -0.10])

TAUS = np.array([7, 14, 30, 60, 91, 182, 273, 365], dtype=float) / 365.0
TAU_DAYS = np.array([7, 14, 30, 60, 91, 182, 273, 365])
LOG_MONEYNESS = np.linspace(-0.4, 0.2, 15)

N_LOGM = len(LOG_MONEYNESS)
N_TAU = len(TAUS)
N_GRID = N_LOGM * N_TAU

TAU_IDX_OF_ROW = np.repeat(np.arange(N_TAU), N_LOGM)
LOGM_IDX_OF_ROW = np.tile(np.arange(N_LOGM), N_TAU)
GRID_LOGM = LOG_MONEYNESS[LOGM_IDX_OF_ROW]
GRID_SQRT_TAU = np.sqrt(TAUS)[TAU_IDX_OF_ROW]
ATM_LOGM_IDX = int(np.argmin(np.abs(LOG_MONEYNESS)))


def surface_values_to_grid(values):
    values = np.asarray(values)
    out = values.reshape(*values.shape[:-1], N_TAU, N_LOGM)
    return np.swapaxes(out, -1, -2)


def grid_to_surface_values(grid):
    grid = np.asarray(grid)
    return np.swapaxes(grid, -1, -2).reshape(*grid.shape[:-2], N_GRID)

DEFAULT_N_SAMPLES = N_SAMPLES
DEFAULT_DATA_SEED = 42
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_SEEDS = TRAIN_SEEDS

VAL_FRAC, TEST_FRAC = 0.10, 0.10

SPARSITY_LEVELS = (0.0, 0.10, 0.25, 0.50, 0.75)
NOISE_LEVELS = (0.01, 0.03, 0.05)

SPARSITY_MASK_SEED_BASE = 7_000
NOISE_SEED_BASE = 31_337
PERMUTATION_SEED = 12_345
ATTENTION_SUBSET_SEED = 2_024

TRAIN_DEFAULTS = dict(lr=3e-4, weight_decay=1e-4, batch_size=BATCH_SIZE,
                      max_epochs=200, patience=10, scheduler_patience=5,
                      scheduler_factor=0.5, grad_clip=1.0)

config = SimpleNamespace(
    **{k: v for k, v in list(globals().items())
       if not k.startswith("_") and k.upper() == k and k not in ("In", "Out")},
    surface_values_to_grid=surface_values_to_grid,
    grid_to_surface_values=grid_to_surface_values,
)

print(f"FULL EXPERIMENT  n_samples={N_SAMPLES}  train_seeds={TRAIN_SEEDS}  "
      f"batch={BATCH_SIZE}  max_epochs={MAX_EPOCHS} (early stop, patience "
      f"{TRAIN_DEFAULTS['patience']})  aug={RUN_AUG}")
print(f"runs -> {RUNS_DIR.relative_to(REPO_ROOT)}   "
      f"results -> {RESULTS_DIR.relative_to(REPO_ROOT)}")
