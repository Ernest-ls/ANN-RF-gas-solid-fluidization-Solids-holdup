#!/usr/bin/env python3
"""Train DNE-ANN, OHE-ANN, DNE-RF and OHE-RF on user-provided data.

The processed experimental dataset is included. Retrained checkpoints can be saved locally. Preprocessing is fitted on training
observations only. Inference is deterministic (torch.nn.Module.eval()).

This recreates the **historical pointwise evaluation task**, not independent
condition-grouped generalization. The latter requires a different test protocol.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent

from model_core import (
    read_data, validate_data, original_split, fit_rf_predict,
    ANN_PARAMS, RF_PARAMS, metrics,
)
from model_training import train_ann_predict

NAMES = {
    'dne_ann': ('DNE-ANN', 'dne', 'ann'),
    'ohe_ann': ('OHE-ANN', 'ohe_reduced', 'ann'),
    'dne_rf': ('DNE-RF', 'dne', 'rf'),
    'ohe_rf': ('OHE-RF', 'ohe_reduced', 'rf'),
}

def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group=p.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--data',type=Path,help='Combined CSV/XLSX containing all 864 observations (or equivalent data)')
    input_group.add_argument('--train-data',type=Path,help='Explicit train CSV/XLSX; requires --test-data')
    p.add_argument('--test-data',type=Path,help='Explicit test CSV/XLSX')
    p.add_argument('--train-ids',type=Path,help='CSV with original training order IDs; requires --test-ids')
    p.add_argument('--test-ids',type=Path,help='CSV with original held-out order IDs')
    p.add_argument('--model',choices=list(NAMES),required=True,help='Run one model per Python process to isolate PyTorch training')
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--epochs',type=int,default=500,help='ANN training epochs; original training script max_epochs=500')
    p.add_argument('--patience',type=int,default=100,help='Training-loss early stopping patience')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--save-weights',action='store_true',help='Save retrained model and preprocessing artifacts')
    return p.parse_args()

def ids_from_csv(p:Path):
    d=pd.read_csv(p)
    if 'order' not in d.columns:raise ValueError(f'{p}: required column "order"')
    return d['order'].astype(int).tolist()

def load_split(args):
    if args.train_data is not None:
        if args.test_data is None or args.train_ids or args.test_ids:
            raise ValueError('--train-data requires --test-data, and cannot be combined with --*-ids')
        tr=read_data(args.train_data);te=read_data(args.test_data)
        overlapping=set(tr.order)&set(te.order)
        if overlapping:raise ValueError(f'Train/test order IDs overlap: {len(overlapping)} records')
        return tr,te,'provided_separate_train_and_test_files'
    if args.test_data is not None:raise ValueError('Use --test-data only with --train-data')
    if bool(args.train_ids)!=bool(args.test_ids):raise ValueError('Provide BOTH --train-ids and --test-ids')
    data=read_data(args.data)
    if args.train_ids:
        tr,te=original_split(data,ids_from_csv(args.train_ids),ids_from_csv(args.test_ids))
        return tr,te,'author_supplied_original_pointwise_ids'
    # Fallback is clearly marked; not the original saved 778/86 records.
    from sklearn.model_selection import train_test_split
    tr,te=train_test_split(data,test_size=0.1,random_state=args.seed,
                          stratify=data['Materials'].astype(int))
    return tr.copy(),te.copy(),'NEW_particle_stratified_pointwise_90_10_split_NOT_historical'

def run(args):
    if args.epochs<1 or args.patience<1:raise ValueError('Invalid ANN epochs or patience')
    tr,te,protocol=load_split(args)
    if not len(tr) or not len(te):raise ValueError('Empty train or test set')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    (out/'predictions').mkdir(exist_ok=True)
    if args.save_weights:(out/'local_weights').mkdir(exist_ok=True)
    results=[]
    for name in [args.model]:
        label,enc,kind=NAMES[name]
        if kind=='rf':
            pr,model,sx=fit_rf_predict(tr,te,enc)
            params=RF_PARAMS[enc]
            if args.save_weights:
                # Only load joblib artifacts you trust; pickle-backed formats can execute code.
                import joblib
                joblib.dump({'scaler':sx,'estimator':model},out/'local_weights'/f'{name}.joblib')
            details={'algorithm':'RandomForestRegressor','parameters':params}
        else:
            pr,model,sx,sy,log=train_ann_predict(tr,te,enc,
                    epochs=args.epochs,patience=args.patience,seed=args.seed)
            params=ANN_PARAMS[enc]
            if args.save_weights:
                import torch
                torch.save(model.state_dict(),out/'local_weights'/f'{name}_state_dict.pt')
                from preprocessing import serialize_scaler
                (out/'local_weights'/f'{name}_preprocessing.json').write_text(json.dumps({
                    'model_name':name,
                    'encoding':enc,
                    'feature_columns':list(sx.feature_names_in_),
                    'input_scaler':serialize_scaler(sx),
                    'target_scaler':serialize_scaler(sy),
                    'ann_hyperparameters':params,
                    'seed':args.seed,
                    'notes':'Newly retrained model; not a checkpoint from the originally submitted Table 7.'
                },indent=2,ensure_ascii=False),encoding='utf-8')
            details={'algorithm':'PyTorch dense ANN','parameters':params,'training_summary':log}
        score=metrics(te['εs'],pr)
        row={'Model':label,'n_train':len(tr),'n_test':len(te),**score}
        results.append(row)
        pframe=pd.DataFrame({'order':te['order'].to_numpy(),'y_true':te['εs'].to_numpy(float),
                            'y_pred':np.asarray(pr,float),'residual':np.asarray(pr,float)-te['εs'].to_numpy(float)})
        pframe.to_csv(out/'predictions'/f'{name}_pointwise_test.csv',index=False)
        (out/f'{name}_config.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f"{label:9s} R²={score['R2']:.6f} RMSE={score['RMSE']:.6f} "
              f"MAE={score['MAE']:.6f} SMAPE={score['SMAPE_percent']:.3f}%",flush=True)
    pd.DataFrame(results).to_csv(out/'pointwise_test_metrics.csv',index=False)
    (out/'RUN_METADATA.json').write_text(json.dumps({
        'protocol':protocol,'seed':args.seed,'n_train':len(tr),'n_test':len(te),
        'ann_epochs':args.epochs,'ann_patience':args.patience,
        'warning':'Pointwise evaluation, NOT grouped condition-level validation; independent test records were not used to fit scalers or select model hyperparameters.',
        'models_executed':[args.model],'checkpoints_saved':bool(args.save_weights),
        'weight_provenance':'new_retraining_not_historical_paper_checkpoints' 
    },ensure_ascii=False,indent=2),encoding='utf-8')
    return results

def main():
    run(parse_args())

if __name__=='__main__':
    main()
