# Heston SAT — Mechanistic Interpretability

This codebase is a direct re-organisation of `HestonSAT_MechInterp.ipynb` into
`src/` (definitions) and `scripts/` (the notebook's executable cells). All
functions, classes, and constants are copied verbatim from the notebook; the
only additions are the import statements each module needs, and the analysis
scripts repeat the notebook's own five `bundle`-construction lines (from the
data-generation cell) so each script is standalone. The unused
`from IPython.display import display, Markdown` import was dropped.

Dependencies: numpy, scipy, pandas, torch, matplotlib.

## Layout (notebook cell → file)

| Cell | Contents | File |
|---|---|---|
| 0–1 | environment, paths, grid, experiment config | `src/config.py` |
| 2 | banner / savefig / seeding / json helpers | `src/utils.py` |
| 3, 4, 6 (first half) | Black–Scholes, Heston CF, Carr–Madan, `build_iv_surface` | `src/heston.py` |
| 5, 6 (second half), 7 | label transforms, LHS sampling, dataset generation/caching, splits, loaders | `src/data.py` |
| 8 | sparsity masks, MLP imputation, train-time augmenter | `src/sparsity.py` |
| 11 | SAB, PMA, `HestonSetAttention` | `src/models/set_attention.py` |
| 12 | MLP, Transformer+PE, 2D CNN baselines | `src/models/baselines.py` |
| 13 | model registry (`MODEL_BUILDERS`, `build_model`, …) | `src/models/__init__.py` |
| 14 | train/evaluate/`run_training`/`predict`/`compute_metrics` | `src/training.py` |
| 15 | run dirs, checkpointing, `train_one` / `train_many` | `src/runs.py` |
| 19 | internals collection, attention stats, seed ablation | `src/interp.py` |
| 22 | re-pricing / recalibration RMSE | `src/recalibration.py` |
| 24 | ridge probe utilities | `src/probes.py` |
| 26 | surface properties + sparse autoencoder | `src/sae.py` |
| 9 | pricing/inversion sanity checks | `scripts/run_checks.py` |
| 10 | data generation + example smiles figure | `scripts/generate_data.py` |
| 16 | train SAT + MLP (and optional aug arm) | `scripts/train_main.py` |
| 17 | train ablation architectures | `scripts/train_ablations.py` |
| 18 | RQ1 sparsity sweep | `scripts/run_rq1.py` |
| 20 | RQ2 PMA seed specialization | `scripts/run_rq2.py` |
| 21 | RQ3 permutation invariance + noise robustness | `scripts/run_rq3.py` |
| 23 | recalibration of predicted parameters | `scripts/run_recalibration.py` |
| 25 | linear probes across depth | `scripts/run_probes.py` |
| 27 | SAE training + feature characterisation | `scripts/run_sae.py` |

## Running

Run everything from the repo root (`REPO_ROOT = Path.cwd()`, exactly as in the
notebook — `data/`, `runs/`, and `results/` are created in the working
directory), in the notebook's cell order:

```bash
python -m scripts.run_checks
python -m scripts.generate_data
python -m scripts.train_main
python -m scripts.train_ablations
python -m scripts.run_rq1
python -m scripts.run_rq2
python -m scripts.run_rq3
python -m scripts.run_recalibration
python -m scripts.run_probes
python -m scripts.run_sae
```

Datasets and trained runs are cached on disk (`data/`, `runs/`), so the
analysis scripts reload identical state rather than recomputing it — the same
mechanism the notebook itself uses.
