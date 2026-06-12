from .config import (config, DEVICE, RUN_AUG, N_SAMPLES, TRAIN_SEEDS, MAX_EPOCHS,
                     N_DRAWS, BATCH_SIZE, REPO_ROOT, DATA_DIR, RUNS_DIR,
                     RESULTS_DIR, FIG_DIR, TAB_DIR, PARAM_NAMES, PARAM_TEX,
                     BOUNDS_LO, BOUNDS_HI, TAUS, TAU_DAYS, LOG_MONEYNESS,
                     N_LOGM, N_TAU, N_GRID, TAU_IDX_OF_ROW, LOGM_IDX_OF_ROW,
                     GRID_LOGM, GRID_SQRT_TAU, ATM_LOGM_IDX,
                     DEFAULT_N_SAMPLES, DEFAULT_DATA_SEED, DEFAULT_SPLIT_SEED,
                     DEFAULT_TRAIN_SEEDS, VAL_FRAC, TEST_FRAC,
                     SPARSITY_LEVELS, NOISE_LEVELS, SPARSITY_MASK_SEED_BASE,
                     NOISE_SEED_BASE, PERMUTATION_SEED, ATTENTION_SUBSET_SEED,
                     TRAIN_DEFAULTS, surface_values_to_grid,
                     grid_to_surface_values)
from .utils import (banner, savefig, ARCH_COLORS, arch_color, set_seed,
                    save_json, load_json)
from .heston import (VOL_LO, VOL_HI, bs_price, bs_call, bs_put,
                     implied_vol_bisection, heston_cf, carr_madan_call_price,
                     build_iv_surface)
from .data import (transform_labels, inverse_transform_labels, sample_params,
                   generate_dataset, add_iv_noise, dataset_path,
                   load_or_generate, make_splits, load_arrays_and_splits,
                   build_loaders)
from .sparsity import (make_keep_mask, mask_seed, kept_indices, sat_inputs,
                       mlp_impute, make_train_augmenter)
from .models import (SAB, PMA, HestonSetAttention, HestonMLP, sinusoidal_pe,
                     HestonTransformerPE, tokens_to_image, HestonCNN2D,
                     MODEL_BUILDERS, ARCH_LABELS, SET_BASED, MAIN_ARCHS,
                     ABLATION_ARCHS, build_model, count_params)
from .training import (train_one_epoch, evaluate, run_training, predict,
                       compute_metrics)
from .runs import (run_name, get_run_dir, run_is_complete, load_trained_model,
                   train_one, train_many, pick_device)
from .interp import (collect_internals, collect_token_activations,
                     collect_pma_attention, mean_attention_maps,
                     top_frac_mass, attention_entropy, effective_cells,
                     pairwise_cosine, seed_ablation_table)
from .recalibration import (reprice_surface, recalibration_rmse,
                            summarize_recalibration)
from .probes import ridge_fit_predict, ridge_r2, pool_layer, probe_layers
from .sae import (surface_properties, make_sae, train_sae, sae_features,
                  feature_property_correlations)
