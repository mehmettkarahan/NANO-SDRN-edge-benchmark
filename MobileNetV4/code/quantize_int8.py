import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]

class Reader(CalibrationDataReader):
    def __init__(self,root,input_name,limit):
        tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
        ds=datasets.ImageFolder(root)
        self.items=[{input_name:tf(Image.open(p).convert('RGB')).unsqueeze(0).numpy().astype(np.float32)} for p,_ in ds.samples[:limit]]
        self.it=iter(self.items)
    def get_next(self):
        return next(self.it,None)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration-dir',type=Path,required=True)
    p.add_argument('--input-name',default='input')
    p.add_argument('--samples',type=int,default=300)
    a=p.parse_args()
    r=Reader(a.calibration_dir,a.input_name,a.samples)
    quantize_static(str(a.input),str(a.output),r,quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8)

if __name__=='__main__': main()
