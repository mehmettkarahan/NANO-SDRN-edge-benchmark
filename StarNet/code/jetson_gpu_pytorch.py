import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from model import create_model

MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    p.add_argument('--output',type=Path,default=Path('jetson_gpu_pytorch.csv'))
    p.add_argument('--warmup',type=int,default=50)
    a=p.parse_args()
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ds=datasets.ImageFolder(a.data_root/'real_test',transform=tf)
    dl=DataLoader(ds,batch_size=1,shuffle=False,num_workers=2,pin_memory=True)
    device=torch.device('cuda')
    model=create_model(len(ds.classes))
    state=torch.load(a.checkpoint,map_location='cpu')
    if isinstance(state,dict):
        for key in ('state_dict','model_state_dict','model'):
            if key in state and isinstance(state[key],dict): state=state[key]; break
    model.load_state_dict(state,strict=True); model.eval().to(device)
    if a.precision=='fp16': model=model.half()
    sample=next(iter(dl))[0].to(device)
    if a.precision=='fp16': sample=sample.half()
    with torch.inference_mode():
        for _ in range(a.warmup): model(sample)
    torch.cuda.synchronize()
    ys=[]; ps=[]; ms=[]
    with torch.inference_mode():
        for x,y in dl:
            x=x.to(device,non_blocking=True)
            if a.precision=='fp16': x=x.half()
            s=torch.cuda.Event(enable_timing=True); e=torch.cuda.Event(enable_timing=True)
            s.record(); z=model(x); e.record(); torch.cuda.synchronize(); ms.append(s.elapsed_time(e))
            ys.extend(y.tolist()); ps.extend(z.argmax(1).cpu().tolist())
    row={'precision':a.precision,'batch_size':1,'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),'latency_mean_ms':float(np.mean(ms)),'latency_p95_ms':float(np.percentile(ms,95)),'fps':1000.0/float(np.mean(ms))}
    pd.DataFrame([row]).to_csv(a.output,index=False)

if __name__=='__main__': main()
