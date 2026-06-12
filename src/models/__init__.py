from .set_attention import SAB, PMA, HestonSetAttention
from .baselines import (HestonMLP, sinusoidal_pe, HestonTransformerPE,
                        tokens_to_image, HestonCNN2D)


MODEL_BUILDERS = {
    "sat": lambda: HestonSetAttention(pooling="pma", n_sab_layers=2),
    "sat_nopma": lambda: HestonSetAttention(pooling="mean", n_sab_layers=2),
    "sat_nosab": lambda: HestonSetAttention(pooling="pma", n_sab_layers=0),
    "mlp": lambda: HestonMLP(),
    "transformer_pe": lambda: HestonTransformerPE(),
    "cnn2d": lambda: HestonCNN2D(),
}

ARCH_LABELS = {
    "sat": "SAT (PMA, 2 SAB)",
    "sat_nopma": "SAT w/o PMA (mean pool)",
    "sat_nosab": "SAT w/o SAB (embed->PMA)",
    "mlp": "MLP (v1 baseline)",
    "transformer_pe": "Transformer + pos. enc.",
    "cnn2d": "2D CNN (15x8 grid)",
}

SET_BASED = {"sat", "sat_nopma", "sat_nosab"}

MAIN_ARCHS = ("sat", "mlp")
ABLATION_ARCHS = ("sat_nopma", "sat_nosab", "transformer_pe", "cnn2d")


def build_model(arch: str):
    if arch not in MODEL_BUILDERS:
        raise KeyError(f"unknown architecture '{arch}'; known: {sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[arch]()


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
