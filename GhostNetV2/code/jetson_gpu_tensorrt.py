import argparse
import re
import subprocess
from pathlib import Path
import pandas as pd

def run(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
    return p.stdout

def value(pattern,text):
    m=re.search(pattern,text,re.I)
    return float(m.group(1)) if m else None

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--onnx',type=Path,required=True)
    p.add_argument('--precision',choices=['fp32','fp16','int8'],required=True)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,4,8,16])
    p.add_argument('--output',type=Path,default=Path('tensorrt_results.csv'))
    p.add_argument('--calibration-cache',type=Path)
    a=p.parse_args(); rows=[]
    for bs in a.batch_sizes:
        engine=a.output.parent/f'{a.onnx.stem}_{a.precision}_b{bs}.engine'
        shape=f'input:{bs}x3x224x224'
        cmd=['trtexec',f'--onnx={a.onnx}',f'--saveEngine={engine}',f'--minShapes={shape}',f'--optShapes={shape}',f'--maxShapes={shape}','--useCudaGraph','--noDataTransfers']
        if a.precision=='fp16': cmd.append('--fp16')
        if a.precision=='int8':
            cmd.append('--int8')
            if a.calibration_cache: cmd.append(f'--calib={a.calibration_cache}')
        build=run(cmd)
        bench=run(['trtexec',f'--loadEngine={engine}',f'--shapes={shape}','--warmUp=1000','--duration=10','--useCudaGraph','--noDataTransfers'])
        mean=value(r'GPU Compute Time:.*?mean = ([0-9.]+) ms',bench)
        p95=value(r'GPU Compute Time:.*?percentile\(95%\) = ([0-9.]+) ms',bench)
        qps=value(r'Throughput: ([0-9.]+) qps',bench)
        rows.append({'precision':a.precision,'batch_size':bs,'gpu_mean_ms':mean,'gpu_p95_ms':p95,'throughput_qps':qps,'engine':engine.name})
    pd.DataFrame(rows).to_csv(a.output,index=False)

if __name__=='__main__': main()
