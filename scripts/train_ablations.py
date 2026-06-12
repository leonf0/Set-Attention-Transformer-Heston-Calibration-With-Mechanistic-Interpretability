import numpy as np

from src.config import (DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED, DEVICE,
                        MAX_EPOCHS, N_SAMPLES, TRAIN_SEEDS)
from src.data import load_or_generate, make_splits, transform_labels
from src.models import ABLATION_ARCHS
from src.runs import train_many
from src.utils import banner


X, y, meta = load_or_generate(N_SAMPLES, DEFAULT_DATA_SEED)
X = X.astype(np.float32)
y_t = transform_labels(y).astype(np.float32)
splits = make_splits(len(X), DEFAULT_SPLIT_SEED)
bundle = {"X": X, "y": y, "y_t": y_t, "splits": splits, "meta": meta}


banner(f"TRAIN ABLATIONS  {ABLATION_ARCHS} seeds={list(TRAIN_SEEDS)}")
abl_metrics = train_many(ABLATION_ARCHS, TRAIN_SEEDS, bundle, DEVICE, max_epochs=MAX_EPOCHS)
