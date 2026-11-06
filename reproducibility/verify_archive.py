#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parent

def check(cond,msg):
    if not cond: raise AssertionError(msg)
    print('OK:',msg)

m=pd.read_csv(ROOT/'dataset/all_splits_image_manifest.csv')
check(len(m)==9000,'all-split manifest has 9,000 images')
counts=m.groupby('split').size().to_dict()
check(counts.get('synthetic_train')==6000,'synthetic_train has 6,000 images')
check(counts.get('real_valid')==1500,'real_valid/model-selection source pool has 1,500 images')
check(counts.get('real_test')==1500,'real_test has 1,500 images')

cal=pd.read_csv(ROOT/'deployment/metadata/int8_calibration_manifest.csv')
check(len(cal)==600,'INT8 calibration manifest has 600 images')

env=json.loads((ROOT/'training/training_environment.json').read_text())
check(env.get('cuda_runtime_reported_by_torch')=='13.0','training environment records CUDA 13.0')
penv=json.loads((ROOT/'deployment/environments/pytorch_fp32_environment.json').read_text())
check(penv.get('cuda_reported_by_torch')=='13.2','final PyTorch deployment environment records CUDA 13.2')
ienv=json.loads((ROOT/'deployment/environments/tensorrt_int8_ptq_environment.json').read_text())
check(ienv.get('calibration_images')==600 and ienv.get('calibration_seed')==42,'INT8 runtime metadata records 600 calibration images and seed 42')
check(ienv.get('calibration_uses_real_test') is False,'real_test excluded from INT8 calibration')

# The archived data package does not contain additional paired-comparison inputs.
print('All archive checks passed.')
