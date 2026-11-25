# CFB solids-holdup prediction — reviewer reproducibility materials

**Manuscript:** *Machine-learning Characterization of Axial–radial Heterogeneity in Solids Holdup Across Particle Systems Within a Circulating Fluidized Bed Riser*  
**Manuscript ID:** POWTEC-D-26-05045  
**Status:** Research reproducibility materials prepared for public access. Reuse licenses remain pending rights-holder confirmation. See `LICENSE_PENDING.md`.

## Contents

- `data/cfb_864.csv`: **864 compiled, time-averaged local solids-holdup observations** (16 material/operating-condition fields × 54 locations), in the original modeling row order; not raw optical-probe voltage or 100 kHz time series.
- `data/split_ids/`: the original pointwise 778 training / 86 test row IDs (`order`).
- `src/cfb_repro/`, `scripts/train_model.py`, `scripts/predict_model.py`: preprocessing, model architecture, training, evaluation and inference for the four model configurations, **DNE-ANN, OHE-ANN, DNE-RF, OHE-RF**.
- `scripts/optuna_search.py`: available Optuna search implementation, with bounds covering the fixed model configurations in `docs/HYPERPARAMETERS.md`. The *full historical 200-trial Optuna study was not rerun or recovered*; this code uses a **new grouped-validation objective**, not the original pointwise CV underlying manuscript Table 8.
- `trained_models/`: four **independently retrained** models (two PyTorch `state_dict` files, two RF `joblib` bundles), plus their necessary preprocessors; **not** the historical checkpoints that generated the originally submitted Tables 7 and 8.
- `verification/`: reload consistency checks, checkpoint SHA-256 hashes and model test metrics.
- `docs/`: data provenance, train/evaluation limitations, data source and release review.

**Scope:** Training on the original historical pointwise split produces comparable, not identical, predictive performance. This does **not** close the reviewer's separate requirement for field-disjoint (GroupKFold) validation or resolve the provenance of all Table 8 means/SDs.

## Environment and tests

Recommended Python 3.10+; last verified locally on Python 3.13, CPU PyTorch 2.10.0, NumPy 2.3.5 and scikit-learn 1.8.0. Dependencies are in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

## Data format and particle encoding

The CSV contains 864 rows with the fields `order, Materials, A, B, C, D, SMD, LP, H, E, F, Ug, G, I, J, Gs, εs`. The machine-learning code uses only `order`, `Materials`, `SMD`, `A`–`D`, `LP`, `H`, `Ug`, `Gs`, and `εs`; the five additional historical indicator columns `E,F,G,I,J` are retained for raw-compilation fidelity but are **not used as model inputs**. The original `εs` column contains time-averaged calibrated local solids holdup, not normalized targets. Any input/target Min–Max scaling is fitted **only on each training subset**, and saved with the checkpoints.

- `Materials`: 0 = FCC, 1/2/3 = glass-bead grades 1/2/3, 4 = sand.
- `SMD`: particle Sauter mean diameter in μm; `LP`: normalized lateral coordinate y/Y; `H`: elevation in m; `Ug`: superficial gas velocity in m/s; `Gs`: solids circulation flux in kg/(m²·s).
- **DNE-ANN/DNE-RF: 5 input features**, `[SMD, LP, H, Ug, Gs]`.
- **OHE-ANN/OHE-RF: 8 input features**, `[A, B, C, D, LP, H, Ug, Gs]` (four dummy indicators for grades 1–4; FCC is the reference/all-zero class). This matches the verified historical source implementation. **The author has indicated that the revised manuscript will be updated to 8 input dimensions.** It does not use five one-hot category columns.

See `data/README.md` for source and rights description. A publicly accessible thesis is not automatically an open data license; see `docs/DATA_SOURCE.md` and `LICENSE_PENDING.md` for the source attribution and current reuse terms.

## Retrain the four models

Run ANNs as separate processes. The commands below use the included modeling data and the exact historical split identifiers; newly trained artifacts go to `outputs/` and do not replace the provided checkpoints.

```bash
python scripts/train_model.py --data data/cfb_864.csv --train-ids data/split_ids/original_train_ids.csv --test-ids data/split_ids/original_test_ids.csv --model dne_ann --epochs 500 --seed 42 --save-weights --output outputs/dne_ann
python scripts/train_model.py --data data/cfb_864.csv --train-ids data/split_ids/original_train_ids.csv --test-ids data/split_ids/original_test_ids.csv --model ohe_ann --epochs 500 --seed 42 --save-weights --output outputs/ohe_ann
python scripts/train_model.py --data data/cfb_864.csv --train-ids data/split_ids/original_train_ids.csv --test-ids data/split_ids/original_test_ids.csv --model dne_rf --seed 42 --save-weights --output outputs/dne_rf
python scripts/train_model.py --data data/cfb_864.csv --train-ids data/split_ids/original_train_ids.csv --test-ids data/split_ids/original_test_ids.csv --model ohe_rf --seed 42 --save-weights --output outputs/ohe_rf
```

Random seed `42` is intentionally documented and should be retained when reproducing the verification runs. Model predictions use `model.eval()` for PyTorch ANNs. RF `joblib` objects internally use pickle; never load them from untrusted origins.

## Predict with the provided retrained checkpoints

```bash
python scripts/predict_model.py --model dne_ann --input data/cfb_864.csv --checkpoints trained_models --eval-ids data/split_ids/original_test_ids.csv --output outputs/dne_ann_predictions.csv
python scripts/predict_model.py --model ohe_ann --input data/cfb_864.csv --checkpoints trained_models --eval-ids data/split_ids/original_test_ids.csv --output outputs/ohe_ann_predictions.csv
python scripts/predict_model.py --model dne_rf  --input data/cfb_864.csv --checkpoints trained_models --eval-ids data/split_ids/original_test_ids.csv --output outputs/dne_rf_predictions.csv
python scripts/predict_model.py --model ohe_rf  --input data/cfb_864.csv --checkpoints trained_models --eval-ids data/split_ids/original_test_ids.csv --output outputs/ohe_rf_predictions.csv
```

Prediction commands write all 864 predicted rows while **reporting performance only for the 86 held-out test IDs** supplied through `--eval-ids`. If the evaluation flag is omitted, no R² or error metrics are printed, to prevent confusion between training-fit scores and test-set scores.

## Optuna search (optional; not independently rerun for 200 trials)

```bash
python scripts/optuna_search.py --data data/cfb_864.csv --model rf --encoding dne --outer-fold 1 --trials 200
python scripts/optuna_search.py --data data/cfb_864.csv --model ann --encoding ohe_reduced --outer-fold 1 --trials 200 --epochs 500
```

The requested fixed optimal configurations are **within** the search bounds (`docs/HYPERPARAMETERS.md` and `tests/test_hyperparameter_ranges.py`). These commands perform a **new grouped objective** and are not a replay of historical Optuna trials or historical Table 8.

## Independently verified checkpoint performance

Fixed configurations, original 778/86 pointwise split, seed 42, ANN max 500 epochs:

| Model | Submitted Table 7 R² | Retrained R² | Retrained RMSE | Retrained MAE | Retrained SMAPE (%) |
|---|---:|---:|---:|---:|---:|
| DNE-ANN | 0.917 | 0.957385 | 0.005768 | 0.004228 | 10.811 |
| OHE-ANN | 0.968 | 0.974022 | 0.004504 | 0.003236 | 8.084 |
| DNE-RF | 0.929 | 0.929172 | 0.007436 | 0.005341 | 14.473 |
| OHE-RF | 0.938 | 0.938217 | 0.006945 | 0.004984 | 14.128 |

After reloading, maximum per-observation differences were no greater than 8.1 × 10⁻⁹ for ANNs and 1.0 × 10⁻¹⁶ for RFs (`verification/`). These verified checkpoints **are not the exact historical models from the submitted paper**; in particular, the difference in DNE-ANN R² cannot be explained by rounding alone. The supplied historical Table 8 fold statistics have unresolved provenance and should not be claimed as numerically reproduced.

## Data and code availability

The author has indicated that the compiled dissertation-derived dataset may be publicly shared, and it is now bundled as `data/cfb_864.csv`. The author-supplied dissertation citation is recorded in `docs/DATA_SOURCE.md`. Reuse licenses remain pending as described in `LICENSE_PENDING.md`. The original high-frequency optical-probe measurements and raw calibration records are not included. Source code, retrained checkpoints and dataset are distributed through https://github.com/Ernest-ls/cfb-solids-holdup-ml. See `LICENSE_PENDING.md` and `docs/RELEASE_REVIEW.md`.

## Original data source and reuse terms

Xu, J. (2010). *Hydrodynamics Studies on Macro- and Micro-Flow Structure with Effects of Particle Properties in a Circulating Fluidized Bed*. Ph.D. dissertation, The University of Western Ontario, London, Ontario, Canada.

The author identifies Appendix B1 as the data source. See `docs/DATA_SOURCE.md`. MIT for code and CC BY 4.0 for data and retrained models are proposed, pending rights-holder confirmation. No explicit public reuse license is currently granted. Please acknowledge the original experiments and cite the dissertation and associated publication.

## Download

Download and extract `cfb-solids-holdup-ml-v1.0.0.zip` from this repository. Run the commands above from the extracted directory.
