# Heston Calibration Set Attention Transformer + Mechanistic Interpretability

Dependencies: numpy, scipy, pandas, torch, matplotlib.

## Repo Structure

```
heston_sat_mechinterp/
├── README.md                        
├── src/
│   ├── __init__.py                  
│   ├── config.py                    # paths, IV grid, experiment constants
│   ├── utils.py                     # banner, savefig, seeding, JSON helpers
│   ├── heston.py                    # Black–Scholes, Heston CF, Carr–Madan, build_iv_surface
│   ├── data.py                      # label transforms, LHS sampling, generation/caching, splits, loaders
│   ├── sparsity.py                  # keep-masks, MLP imputation, train-time augmenter
│   ├── models/
│   │   ├── __init__.py              # registry (MODEL_BUILDERS, build_model, ARCH_LABELS)
│   │   ├── set_attention.py         # SAB, PMA, HestonSetAttention
│   │   └── baselines.py             # HestonMLP, HestonTransformerPE, HestonCNN2D
│   ├── training.py                  # train/evaluate, run_training, predict, compute_metrics
│   ├── runs.py                      # run dirs, checkpointing, train_one / train_many
│   ├── interp.py                    # internals collection, attention stats, seed ablation
│   ├── recalibration.py             # re-pricing, recalibration RMSE
│   ├── probes.py                    # ridge probe utilities
│   └── sae.py                       # surface properties + sparse autoencoder
├── scripts/                         # numbers = execution order
│   ├── __init__.py
│   ├── run_checks.py                # (1)  pricing/inversion sanity checks
│   ├── generate_data.py             # (2)  dataset generation + example smiles figure
│   ├── train_main.py                # (3)  train SAT + MLP (optional aug arm)
│   ├── train_ablations.py           # (4)  train ablation architectures
│   ├── run_rq1.py                   # (5)  sparsity sweep
│   ├── run_rq2.py                   # (6)  PMA seed specialization
│   ├── run_rq3.py                   # (7)  permutation invariance + noise robustness
│   ├── run_recalibration.py         # (8)  re-price predicted parameters
│   ├── run_probes.py                # (9)  linear probes across depth
│   └── run_sae.py                   # (10) SAE training + feature characterisation
├── data/                            # created at runtime: cached surfaces + splits (.npz)
├── runs/n50000/                     # created at runtime: checkpoints, configs, histories per arch/seed
└── results/n50000/                  # created at runtime: figures/ and tables/ 
```
## Running

Run everything from the repo root (`REPO_ROOT = Path.cwd()`, :

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
