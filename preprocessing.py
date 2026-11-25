"""Small, safe JSON representation for MinMaxScaler inference parameters."""
from __future__ import annotations
import numpy as np

def serialize_scaler(scaler):
    return {
        'scale_':scaler.scale_.tolist(),
        'min_':scaler.min_.tolist(),
        'n_features_in_':int(scaler.n_features_in_),
        'feature_range':list(scaler.feature_range),
        'n_samples_seen_':int(scaler.n_samples_seen_),
    }

def transform_with_saved_scaler(x,record):
    arr=np.asarray(x,dtype=float)
    if arr.ndim!=2 or arr.shape[1]!=record['n_features_in_']:
        raise ValueError('Incompatible input feature dimensions')
    return arr*np.asarray(record['scale_'],dtype=float)+np.asarray(record['min_'],dtype=float)

def inverse_transform_with_saved_scaler(x,record):
    arr=np.asarray(x,dtype=float)
    if arr.ndim!=2 or arr.shape[1]!=record['n_features_in_']:
        raise ValueError('Incompatible output dimensions')
    scale=np.asarray(record['scale_'],dtype=float)
    if not np.all(scale!=0):raise ValueError('Invalid scaler coefficient')
    return (arr-np.asarray(record['min_'],dtype=float))/scale
