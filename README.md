# ANN-RF-gas-solid-fluidization-Solids-holdup

## Environments

Python 3.13, PyTorch 2.10.0, NumPy 2.3.5 and scikit-learn 1.8.0 were used to save the supplied models. Install dependencies with `pip install -r requirements.txt`.

## Introduction

This project predicts local solids holdup in a circulating fluidized bed riser using DNE-ANN, OHE-ANN, DNE-RF and OHE-RF. The dataset contains 864 processed, time-averaged observations across five particle systems and 16 operating conditions, compiled from Appendix B1 of Xu, J. (2010), *Hydrodynamics Studies on Macro- and Micro-Flow Structure with Effects of Particle Properties in a Circulating Fluidized Bed*, Ph.D. dissertation, The University of Western Ontario, Canada.

## Code files

- `Fluidization-ANN-RF.py` trains and tests the four model configurations.
- `Fluidization-ANN-RF-optuna.py` optimizes hyperparameters using grouped validation.
- `predict_models.py` predicts with the supplied models.
- `model_core.py`, `model_training.py` and `preprocessing.py` define the models, training and scaling procedures.
- `cfb_864.csv` contains the modeling data. `original_train_ids.csv` and `original_test_ids.csv` define the 778/86 split.
- `*_state_dict.pt`, `*.joblib` and `*_preprocessing.json` contain the retrained models and preprocessing parameters.

## Usage

```bash
python Fluidization-ANN-RF.py --data cfb_864.csv --train-ids original_train_ids.csv --test-ids original_test_ids.csv --model dne_ann --save-weights --output outputs/dne_ann
python predict_models.py --model dne_ann --input cfb_864.csv --eval-ids original_test_ids.csv --output outputs/predictions.csv
python Fluidization-ANN-RF-optuna.py --data cfb_864.csv --model rf --encoding dne --trials 200
```

Use `dne_ann`, `ohe_ann`, `dne_rf` or `ohe_rf` to select a model. Targets are supplied before normalization, and scaling is fitted on training data. Raw probe time series are unavailable in this repository. The checkpoints were independently retrained. The Optuna script performs a new grouped search, with historical study records unavailable.

Reuse licenses are pending rights-holder confirmation. MIT for code and CC BY 4.0 for data and checkpoints are proposed. See `LICENSE_PENDING.md`. Please cite the associated publication and Xu (2010).
