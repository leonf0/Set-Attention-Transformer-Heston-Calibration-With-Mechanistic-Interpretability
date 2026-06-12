import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import FIG_DIR, REPO_ROOT


def banner(title):
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}", flush=True)


def savefig(plt_, name):
    path = FIG_DIR / name
    plt_.savefig(path)
    plt_.show()
    plt_.close()
    print(f"  figure -> {path.relative_to(REPO_ROOT)}", flush=True)
    return path


ARCH_COLORS = {"sat": "#d62728", "mlp": "#1f77b4", "sat_nopma": "#ff9896",
               "sat_nosab": "#e377c2", "transformer_pe": "#2ca02c", "cnn2d": "#9467bd"}


def arch_color(arch):
    return ARCH_COLORS.get(arch.replace("_aug", ""), "#7f7f7f")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _jsonable(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"Object of type {type(o)} is not JSON serialisable")

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def load_json(path):
    with open(path) as f:
        return json.load(f)
