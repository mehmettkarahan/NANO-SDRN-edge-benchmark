#!/usr/bin/env python3
"""Recalculate the article's nine-condition TOPSIS (Equal, Entropy, CRITIC).

Run from repository root: python scripts/verify_topsis.py
Requires: numpy, pandas
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'results/topsis'
dm=pd.read_csv(folder/'decision_matrices.csv')
weights=pd.read_csv(folder/'weights.csv')
scores=pd.read_csv(folder/'scores_and_ranks.csv')
assert dm.Condition.nunique()==9 and len(dm)==72
assert len(scores)==216 and len(weights)==27
for condition, df in dm.groupby('Condition',sort=False):
    x=df[['Macro_F1_percent','E2E_ms_per_image','Deployment_size_MiB']].to_numpy(float)
    assert x.shape==(8,3)
    for method in ('Equal','Entropy','CRITIC'):
        if method=='Equal': w=np.ones(3)/3
        elif method=='Entropy':
            p=x/x.sum(axis=0)
            entropy=-(p*np.log(p)).sum(axis=0)/np.log(len(x))
            info=1-entropy
            w=info/info.sum()
        else:
            z=np.column_stack([(x[:,0]-x[:,0].min())/np.ptp(x[:,0]),
                               (x[:,1].max()-x[:,1])/np.ptp(x[:,1]),
                               (x[:,2].max()-x[:,2])/np.ptp(x[:,2])])
            s=z.std(axis=0,ddof=1)
            info=s*(1-np.corrcoef(z,rowvar=False)).sum(axis=1)
            w=info/info.sum()
        expected_w=weights[(weights.Condition==condition)&(weights.Weighting==method)]
        assert len(expected_w)==1
        assert np.allclose(w,expected_w[['F1_weight','Latency_weight','Size_weight']].to_numpy(float)[0],atol=1e-10)
        v=(x/np.linalg.norm(x,axis=0))*w
        plus=np.array([v[:,0].max(),v[:,1].min(),v[:,2].min()])
        minus=np.array([v[:,0].min(),v[:,1].max(),v[:,2].max()])
        dplus=np.linalg.norm(v-plus,axis=1); dminus=np.linalg.norm(v-minus,axis=1)
        ci=dminus/(dplus+dminus)
        expected=scores[(scores.Condition==condition)&(scores.Weighting==method)].set_index('Model')
        assert len(expected)==8
        for model,c in zip(df.Model,ci):
            assert np.isclose(c,expected.loc[model,'Ci'],atol=1e-10), (method,condition,model)
print('PASS: 9 conditions × 3 methods × 8 models; all weights and TOPSIS scores match.')
