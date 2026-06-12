import matplotlib.pyplot as plt
import numpy as np

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED,
                        LOG_MONEYNESS, N_SAMPLES, PARAM_NAMES,
                        surface_values_to_grid)
from src.data import load_or_generate, make_splits, transform_labels
from src.utils import banner, savefig


banner(f"GENERATE DATA  (n={N_SAMPLES}, data_seed={DEFAULT_DATA_SEED})")
X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}

print(f"Dataset: X {X.shape}  y {y.shape}")
print(f"Splits : train {len(splits['train'])} | val {len(splits['val'])} "
      f"| test {len(splits['test'])}")
print(f"Meta   : {meta}")

fig, ax = plt.subplots(figsize=(5.6, 3.4))
for i in range(4):
    ax.plot(LOG_MONEYNESS, surface_values_to_grid(X[i, :, 2])[:, 0], marker=".",
            label=", ".join(f"{n}={v:.2f}" for n, v in zip(PARAM_NAMES, y[i])))
ax.set_xlabel("log-moneyness"); ax.set_ylabel("IV"); ax.legend(fontsize=6)
ax.set_title("Example 7-day smiles from the generated dataset")
savefig(plt, "data_example_smiles.png")
