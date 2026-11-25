"""Controlled ANN training for NEW experiments; not historical 2024/2025 checkpoint replay."""
from __future__ import annotations
from copy import deepcopy
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model_core import feature_frame, build_ann, ANN_PARAMS

def train_ann_predict(train,test,encoding:str,params:dict|None=None,epochs:int=500,patience:int=100,seed:int=42):
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    if params is None:params=ANN_PARAMS[encoding]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1)
    sx=MinMaxScaler().fit(feature_frame(train,encoding))
    sy=MinMaxScaler().fit(train[['εs']])
    xx=torch.tensor(sx.transform(feature_frame(train,encoding)),dtype=torch.float32)
    yy=torch.tensor(sy.transform(train[['εs']]).ravel(),dtype=torch.float32)
    model=build_ann(encoding,params['hidden'],params['dropout'],params.get('activation','relu'))
    for m in model.modules():
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim_name=params.get('optimizer','adamw').lower()
    if optim_name=='adamw':opt=torch.optim.AdamW(model.parameters(),lr=params['lr'],weight_decay=.01)
    elif optim_name=='adam':opt=torch.optim.Adam(model.parameters(),lr=params['lr'])
    elif optim_name=='sgd':opt=torch.optim.SGD(model.parameters(),lr=params['lr'],momentum=.9)
    else:raise ValueError(f'Unknown optimizer {optim_name}')
    loader=DataLoader(TensorDataset(xx,yy),batch_size=int(params['batch_size']),shuffle=True)
    best=float('inf');best_state=None;no_improve=0
    for epoch in range(epochs):
        model.train();total=0.;seen=0
        for x,y in loader:
            opt.zero_grad(set_to_none=True)
            pred=model(x).squeeze(-1)
            loss=torch.nn.functional.mse_loss(pred,y)
            loss.backward();opt.step();total+=float(loss.item())*len(x);seen+=len(x)
        av=total/seen
        if av<best-1e-10:
            best=av;best_state=deepcopy(model.state_dict());no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:break
    model.load_state_dict(best_state);model.eval()
    with torch.no_grad():
        pred=model(torch.tensor(sx.transform(feature_frame(test,encoding)),dtype=torch.float32)).numpy()
    return sy.inverse_transform(pred).ravel(),model,sx,sy,{'epochs_completed':epoch+1,'training_scaled_mse':best}

def balanced_inner_folds(df,n_splits=3):
    """Split complete fields, preserving singleton-material fields in training only.
    The inner-validation objective can omit singleton materials; report this clearly.
    """
    from model_core import group_id
    g=group_id(df);unique=pd.DataFrame({'g':g,'mat':df['Materials']}).drop_duplicates().sort_values(['mat','g'])
    assignments={}
    for mat,sub in unique.groupby('mat'):
        if len(sub)<2:continue
        for i,group in enumerate(sub['g']):assignments[group]=i%n_splits
    result=[]
    for i in range(n_splits):
        val=np.flatnonzero(g.map(assignments).eq(i).to_numpy())
        train=np.flatnonzero(~g.map(assignments).eq(i).to_numpy())
        if not len(val):raise ValueError('An inner fold has no validation fields')
        if set(g.iloc[train])&set(g.iloc[val]):raise ValueError('Inner field leakage')
        result.append((train,val))
    return result
