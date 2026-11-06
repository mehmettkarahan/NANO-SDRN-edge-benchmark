#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
a=pd.read_csv(ROOT/'results/cross_platform_e2e_batch1_4_8_16.csv')
assert len(a)==288, len(a)
assert a.model.nunique()==8 and a.batch_size.nunique()==4
assert not a.duplicated(['platform','backend','model','precision','batch_size']).any()
assert not a[['accuracy_percent','macro_f1_percent','e2e_ms_per_image','nominal_throughput_images_s']].isna().any().any()
assert all(a.e2e_ms_per_image>0)
for (plat, backend, precision),x in a.groupby(['platform','backend','precision']):
    assert len(x)==32,(plat,backend,precision,len(x))
    assert set(x.batch_size)=={1,4,8,16}
assert set(a[a.backend=='ONNX Runtime CPU'].cpu_threads.astype(int))=={4}
print('PASS: eight models, nine hardware/precision configurations, four batch sizes, 288 E2E rows.')
