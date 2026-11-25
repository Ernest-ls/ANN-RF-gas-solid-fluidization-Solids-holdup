#!/usr/bin/env python3
"""Optional Optuna tuning script (provided for reviewer transparency; full tuning NOT validated).

Uses group-disjoint train/validation partitions (no outer test leakage). The
specified ranges correspond to the earlier archived code-search settings.
The package includes data and retrained model weights. Historical Optuna study history is unavailable.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
ROOT=Path(__file__).resolve().parent

from model_core import read_data,fixed_condition_folds,fit_rf_predict,metrics
from model_training import train_ann_predict,balanced_inner_folds

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--data',required=True,type=Path)
 p.add_argument('--model',choices=['rf','ann'],required=True)
 p.add_argument('--encoding',choices=['dne','ohe_reduced'],required=True)
 p.add_argument('--outer-fold',type=int,choices=[1,2,3,4,5],default=1)
 p.add_argument('--trials',type=int,default=200)
 p.add_argument('--epochs',type=int,default=500)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--output',type=Path,default=ROOT/'outputs'/'optuna')
 args=p.parse_args()
 df=read_data(args.data)
 outer_train_idx,outer_test_idx=fixed_condition_folds(df)[args.outer_fold-1]
 outer_train=df.iloc[outer_train_idx].reset_index(drop=True)
 inner=balanced_inner_folds(outer_train,n_splits=3)
 def objective(trial):
  if args.model=='rf':
   hp=dict(n_estimators=trial.suggest_int('n_estimators',32,256),
      max_depth=trial.suggest_int('max_depth',2,30),
      min_samples_split=trial.suggest_int('min_samples_split',2,100),
      min_samples_leaf=trial.suggest_int('min_samples_leaf',1,20),
      max_features=trial.suggest_categorical('max_features',['sqrt','log2',None]),
      bootstrap=trial.suggest_categorical('bootstrap',[True,False]),
      random_state=args.seed,n_jobs=1)
  else:
   depth=trial.suggest_int('n_layers',2,8)
   hp=dict(hidden=[trial.suggest_int(f'n_units_{i}',32,256) for i in range(depth)],
      lr=trial.suggest_float('lr',1e-5,1e-1,log=True),
      dropout=trial.suggest_float('dropout',0.0,0.2),
      batch_size=trial.suggest_int('batch_size',16,64),
      activation=trial.suggest_categorical('activation',['relu','gelu']),
      optimizer=trial.suggest_categorical('optimizer',['adamw','adam','sgd']))
  scores=[]
  for k,(tr,va) in enumerate(inner):
   a,b=outer_train.iloc[tr],outer_train.iloc[va]
   if args.model=='rf':pred,*_=fit_rf_predict(a,b,args.encoding,params=hp)
   else:pred,*_=train_ann_predict(a,b,args.encoding,params=hp,epochs=args.epochs,seed=args.seed+trial.number*17+k)
   scores.append(metrics(b['εs'],pred)['RMSE'])
  return float(np.mean(scores))
 study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=args.seed))
 study.optimize(objective,n_trials=args.trials)
 args.output.mkdir(parents=True,exist_ok=True)
 study.trials_dataframe().to_csv(args.output/'optuna_trials.csv',index=False)
 (args.output/'best_parameters.json').write_text(json.dumps({'best_value':study.best_value,
         'best_parameters':study.best_params,'note':'NEW GROUPED SEARCH; not historical Optuna study'},indent=2),encoding='utf-8')
 print('Best inner-group RMSE:',study.best_value)

if __name__=='__main__':main()
