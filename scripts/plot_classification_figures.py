#!/usr/bin/env python3
"""Render NANO-SDRN seed 42 confusion matrices and OvR ROC from probabilities."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]
x=pd.read_csv(ROOT/'reproducibility/classification_predictions/nano_sdrn_real_test_predictions_seed_42.csv')
figdir=ROOT/'NANO-SDRN/results/training/figures'; figdir.mkdir(exist_ok=True,parents=True)
names=['Early blight','Late blight','Healthy']
cm=confusion_matrix(x.true_index,x.predicted_index,labels=[0,1,2])
for normal in (False,True):
    a=cm/cm.sum(axis=1,keepdims=True) if normal else cm
    fig,ax=plt.subplots(figsize=(7.1,5.7))
    im=ax.imshow(a,cmap='Blues')
    for (i,j),v in np.ndenumerate(a):
        ax.text(j,i,f'{v:.2%}' if normal else str(int(v)),ha='center',va='center',
                color='white' if v>a.max()/2 else 'black')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=names,yticklabels=names,
           xlabel='Predicted class',ylabel='True class',title='NANO-SDRN — Real-test confusion matrix (seed 42)')
    fig.colorbar(im,ax=ax);fig.tight_layout()
    fig.savefig(figdir/('real_test_confusion_matrix_normalized_seed_42.png' if normal else 'real_test_confusion_matrix_raw_seed_42.png'),dpi=180)
    plt.close(fig)
fig,ax=plt.subplots(figsize=(7,5.6))
for c,name in enumerate(names):
    y=(x.true_index.to_numpy()==c).astype(int)
    p=x[['prob_Tomato__Early_blight','prob_Tomato__Late_blight','prob_Tomato__healthy'][c]].to_numpy(float)
    fpr,tpr,_=roc_curve(y,p)
    auc=roc_auc_score(y,p)
    ax.plot(fpr,tpr,label=f'{name} (AUC={auc:.3f})')
ax.plot([0,1],[0,1],linestyle='--',label='Chance')
ax.set(xlabel='False positive rate',ylabel='True positive rate',title='NANO-SDRN — Real-test OvR ROC (seed 42)')
ax.legend(loc='lower right');fig.tight_layout()
fig.savefig(figdir/'real_test_roc_ovr_seed_42.png',dpi=180)
plt.close(fig)
print('Published seed-42 images rebuilt from probability predictions.')
