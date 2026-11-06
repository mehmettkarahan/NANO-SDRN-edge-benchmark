import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import datasets, transforms

MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]
PLATFORM='Jetson Orin Nano CPU'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--onnx',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--precision',choices=['fp32','int8'],default='fp32')
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,4,8,16])
    p.add_argument('--threads',type=int,nargs='+',default=[1,4,6])
    p.add_argument('--output',type=Path,default=Path('cpu_results.csv'))
    a=p.parse_args()
    tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ds=datasets.ImageFolder(a.data_root/'real_test')
    rows=[]
    for nt in a.threads:
        so=ort.SessionOptions(); so.intra_op_num_threads=nt; so.inter_op_num_threads=1
        sess=ort.InferenceSession(str(a.onnx),sess_options=so,providers=['CPUExecutionProvider'])
        inp=sess.get_inputs()[0].name
        for bs in a.batch_sizes:
            ys=[]; ps=[]; times=[]
            for i in range(0,len(ds),bs):
                chunk=ds.samples[i:i+bs]
                x=np.stack([tf(Image.open(path).convert('RGB')).numpy() for path,_ in chunk]).astype(np.float32)
                y=[label for _,label in chunk]
                t=time.perf_counter(); z=sess.run(None,{inp:x})[0]; times.append((time.perf_counter()-t)*1000)
                ys.extend(y); ps.extend(np.argmax(z,axis=1).tolist())
            mean=float(np.mean(times))
            rows.append({'platform':PLATFORM,'precision':a.precision,'threads':nt,'batch_size':bs,'accuracy':accuracy_score(ys,ps),'macro_f1':f1_score(ys,ps,average='macro'),'latency_mean_ms_batch':mean,'latency_mean_ms_image':mean/bs,'throughput_images_s':1000.0*bs/mean})
    pd.DataFrame(rows).to_csv(a.output,index=False)

if __name__=='__main__': main()
