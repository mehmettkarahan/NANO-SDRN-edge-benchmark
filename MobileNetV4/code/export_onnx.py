import argparse
from pathlib import Path
import torch
from model import create_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('model.onnx'))
    p.add_argument('--num-classes',type=int,default=3)
    a=p.parse_args()
    model=create_model(a.num_classes)
    state=torch.load(a.checkpoint,map_location='cpu')
    if isinstance(state,dict):
        for key in ('state_dict','model_state_dict','model'):
            if key in state and isinstance(state[key],dict): state=state[key]; break
    model.load_state_dict(state,strict=True); model.eval()
    x=torch.randn(1,3,224,224)
    torch.onnx.export(model,x,str(a.output),opset_version=17,input_names=['input'],output_names=['logits'],dynamic_axes={'input':{0:'batch'},'logits':{0:'batch'}})

if __name__=='__main__': main()
