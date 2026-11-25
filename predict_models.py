#!/usr/bin/env python3
"""Predict with saved retrained checkpoint. A trained model must be from a trusted source.

Example:
 python predict_models.py --model dne_ann --input /private/cfb_data.csv \\
   --checkpoints trained_models --output outputs/dne_predictions.csv
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

from model_core import FEATURES,build_ann,metrics
from preprocessing import transform_with_saved_scaler,inverse_transform_with_saved_scaler

def predict(name,df,ckpt_dir):
    name=str(name);ckpt_dir=Path(ckpt_dir)
    if name not in {'dne_ann','ohe_ann','dne_rf','ohe_rf'}:raise ValueError('Unknown model name')
    enc='dne' if name.startswith('dne') else 'ohe_reduced'
    columns=FEATURES[enc]
    absent=[c for c in columns if c not in df]
    if absent:raise ValueError(f'Missing features: {absent}')
    x=df[columns].to_numpy(dtype=float)
    if name.endswith('_rf'):
        import joblib
        # joblib is pickle-based: never load models from untrusted sources.
        bundle=joblib.load(ckpt_dir/f'{name}.joblib')
        y=bundle['estimator'].predict(bundle['scaler'].transform(pd.DataFrame(x,columns=columns)))
    else:
        import torch
        data=json.loads((ckpt_dir/f'{name}_preprocessing.json').read_text(encoding='utf-8'))
        if data['feature_columns']!=columns:raise ValueError('Feature order differs from saved training configuration')
        hp=data['ann_hyperparameters']
        model=build_ann(enc,hp['hidden'],hp['dropout'],hp['activation'])
        model.load_state_dict(torch.load(ckpt_dir/f'{name}_state_dict.pt',map_location='cpu',weights_only=True))
        model.eval()
        xx=transform_with_saved_scaler(x,data['input_scaler'])
        with torch.no_grad():y_scaled=model(torch.tensor(xx,dtype=torch.float32)).numpy()
        y=inverse_transform_with_saved_scaler(y_scaled,data['target_scaler']).ravel()
    return np.asarray(y).ravel()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['dne_ann','ohe_ann','dne_rf','ohe_rf'])
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--checkpoints',default=ROOT,type=Path)
    p.add_argument('--output',default=ROOT/'outputs'/'predictions.csv',type=Path)
    p.add_argument('--eval-ids',type=Path,help='Optional CSV with order IDs of the independent evaluation subset; omitting this flag disables printing metrics.')
    args=p.parse_args()
    df=pd.read_excel(args.input) if args.input.suffix.lower() in {'.xlsx','.xls'} else pd.read_csv(args.input)
    y=predict(args.model,df,args.checkpoints)
    out=pd.DataFrame({'prediction_epsilon_s':y})
    if 'order' in df:out.insert(0,'order',df['order'].to_numpy())
    if 'εs' in df:
        out['observed_epsilon_s']=df['εs'].to_numpy(float)
    if args.eval_ids:
        if 'order' not in df or 'εs' not in df:
            raise ValueError('--eval-ids requires order and εs columns in the prediction dataset')
        ids=pd.read_csv(args.eval_ids)
        if 'order' not in ids or ids['order'].duplicated().any():
            raise ValueError('Evaluation IDs must contain a unique order column')
        m=df['order'].isin(ids['order'])
        if int(m.sum())!=len(ids):
            raise ValueError('Some requested evaluation IDs are missing from the prediction dataset')
        print(f'Independent evaluation subset: n={int(m.sum())} (from --eval-ids)')
        print(json.dumps(metrics(df.loc[m,'εs'],y[m.to_numpy()]),indent=2))
    else:
        print('Predictions only: no performance metric computed over mixed training/test rows; pass --eval-ids for valid held-out evaluation.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)
    print(f'Saved {len(out)} predictions to {args.output}')

if __name__=='__main__':main()
