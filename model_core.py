"""Reproducible data preparation, metrics and estimators for POWTEC-D-26-05045.

No data is downloaded, no untrusted artifact is unpickled, and no output is
presented as equivalent to the submitted manuscript unless independently verified.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

COLUMN_SET = ['order', 'Materials', 'A', 'B', 'C', 'D', 'SMD', 'LP', 'H', 'Ug', 'Gs', 'εs']
FEATURES = {
    'dne': ['SMD','LP','H','Ug','Gs'],
    'ohe_reduced': ['A','B','C','D','LP','H','Ug','Gs'],
}
PARTICLE_PROPERTIES = {
    0: ('FCC',67.,1877.,1125.,0.95),
    1: ('Glassbead 1',76.,2453.,1434.,1.00),
    2: ('Glassbead 2',134.,2403.,1421.,1.00),
    3: ('Glassbead 3',288.,2498.,1475.,1.00),
    4: ('Sand',138.,2467.,1453.,0.591),
}
RF_PARAMS = {
    'dne': dict(n_estimators=110,max_depth=17,min_samples_split=5,min_samples_leaf=1,max_features='sqrt',bootstrap=False,random_state=42),
    'ohe_reduced': dict(n_estimators=162,max_depth=25,min_samples_split=5,min_samples_leaf=1,max_features='log2',bootstrap=False,random_state=42),
}
ANN_PARAMS = {
    'dne':dict(lr=.00036,hidden=[207,247,249,36],dropout=.05328,batch_size=58,optimizer='adamw',activation='relu'),
    'ohe_reduced':dict(lr=.00078,hidden=[174,202,146,232,202],dropout=.02668,batch_size=54,optimizer='adamw',activation='relu'),
}

def validate_data(df:pd.DataFrame,expect_864:bool=False)->None:
    missing = [c for c in COLUMN_SET if c not in df.columns]
    if missing: raise ValueError(f'Missing required columns: {missing}')
    if df[COLUMN_SET].isna().any().any(): raise ValueError('Missing required data')
    if df['order'].duplicated().any(): raise ValueError('Duplicated order IDs')
    if expect_864 and len(df)!=864: raise ValueError(f'Expected 864 rows, received {len(df)}')
    if not set(df['Materials'].unique()).issubset(PARTICLE_PROPERTIES):raise ValueError('Unknown particle label')
    if (df['εs']<0).any() or (df['εs']>1).any():raise ValueError('Solids holdup outside [0,1]')


def read_data(path:Path|str)->pd.DataFrame:
    path=Path(path)
    df=pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_excel(path)
    validate_data(df)
    return df


def feature_frame(df:pd.DataFrame,encoding:str)->pd.DataFrame:
    if encoding not in FEATURES:raise ValueError(f'Unknown encoding {encoding}')
    x=df.copy()
    return x.loc[:,FEATURES[encoding]].astype(float)


def metrics(y_true:Iterable[float],y_pred:Iterable[float])->dict[str,float]:
    yt=np.asarray(y_true,float).ravel();yp=np.asarray(y_pred,float).ravel()
    if len(yt)!=len(yp) or len(yt)==0 or not (np.isfinite(yt).all() and np.isfinite(yp).all()):raise ValueError('Invalid paired predictions')
    e=yt-yp;sse=float(np.sum(e*e));ss=float(np.sum((yt-yt.mean())**2))
    smape=float(np.mean(200*np.abs(e)/(np.abs(yt)+np.abs(yp)+1e-8)))
    return {'R2':float(1-sse/ss) if ss>0 else float('nan'),'RMSE':float(np.sqrt(np.mean(e**2))),'MAE':float(np.mean(np.abs(e))),'SMAPE_percent':smape}


def original_split(df:pd.DataFrame,train_ids:Iterable[int],test_ids:Iterable[int]):
    a=set(map(int,train_ids));b=set(map(int,test_ids))
    if a&b or len(a|b)!=len(df) or set(map(int,df['order']))!=a|b:raise ValueError('Split IDs do not partition the input rows')
    return df[df['order'].isin(a)].copy(),df[df['order'].isin(b)].copy()


def group_id(df:pd.DataFrame)->pd.Series:
    return (df['Materials'].astype(int).astype(str)+'|'+df['Ug'].map(lambda x:f'{x:.10g}')+'|'+df['Gs'].map(lambda x:f'{x:.10g}'))


def fixed_condition_folds(df:pd.DataFrame)->list[tuple[np.ndarray,np.ndarray]]:
    # A predefined prospective grouped validation, not the originally submitted CV.
    heldout=[
      [(0,3.5,50),(0,5.5,150),(1,5.5,100),(4,5.5,200)],
      [(0,3.5,100),(0,5.5,200),(2,5.5,100)],
      [(0,3.5,150),(0,5.5,100),(3,5.5,100)],
      [(0,3.5,200),(1,5.5,200),(2,5.5,200)],
      [(0,8.0,200),(3,5.5,200),(4,5.5,150)],
    ]
    existing={(int(a),float(b),float(c)) for a,b,c in zip(df.Materials,df.Ug,df.Gs)}
    intended=set(sum(heldout,[]))
    if existing!=intended:raise ValueError('Operating conditions differ from expected 16 groups; freeze fold design only after verification')
    ret=[];all_index=set(range(len(df)));shown=set()
    for folds in heldout:
        m=np.array([(int(r.Materials),float(r.Ug),float(r.Gs)) in folds for r in df.itertuples()],bool)
        v=np.flatnonzero(m);t=np.flatnonzero(~m)
        if not len(v) or shown.intersection(v):raise ValueError('Invalid held-out fields')
        shown.update(v)
        if set(map(int,df.iloc[t]['Materials']))!=set(map(int,df['Materials'])):raise ValueError('A particle class missing from training fold')
        if set(group_id(df.iloc[v]))&set(group_id(df.iloc[t])):raise ValueError('Field leakage')
        ret.append((t,v))
    if shown!=all_index:raise ValueError('Not all samples evaluated')
    return ret


def fit_rf_predict(train:pd.DataFrame,test:pd.DataFrame,encoding:str,params:dict|None=None):
    if params is None:params=RF_PARAMS[encoding]
    sx=MinMaxScaler().fit(feature_frame(train,encoding))
    m=RandomForestRegressor(**params).fit(sx.transform(feature_frame(train,encoding)),train['εs'].to_numpy(float))
    pred=m.predict(sx.transform(feature_frame(test,encoding)))
    return pred,m,sx


def build_ann(encoding:str,hidden:list[int]|None=None,dropout:float|None=None,activation:str="relu"):
    import torch.nn as nn
    if hidden is None:hidden=ANN_PARAMS[encoding]['hidden']
    if dropout is None:dropout=ANN_PARAMS[encoding]['dropout']
    layers=[];in_dim=len(FEATURES[encoding]);
    for i,n in enumerate(hidden):
        layers.append(nn.Linear(in_dim,n));layers.append(nn.ReLU() if activation.lower()=="relu" else nn.GELU())
        if i<len(hidden)-1:layers.append(nn.Dropout(dropout))
        in_dim=n
    layers.append(nn.Linear(in_dim,1))
    return nn.Sequential(*layers)
