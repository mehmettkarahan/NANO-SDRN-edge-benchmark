import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score, classification_report
from sklearn.model_selection import StratifiedKFold
from model import create_model

MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]
SEEDS=[42,123,2026]

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def tfms():
    train=transforms.Compose([transforms.RandomResizedCrop((224,224),scale=(0.85,1.0),ratio=(0.90,1.10)),transforms.RandomHorizontalFlip(0.5),transforms.RandomRotation(15),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    eva=transforms.Compose([transforms.Resize(256),transforms.CenterCrop((224,224)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    return train,eva

def loader(ds,batch,shuffle,workers,seed):
    g=torch.Generator().manual_seed(seed)
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,generator=g)

def metrics(y,p):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
    return {'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_precision':pr,'macro_recall':rc,'macro_f1':f1,'mcc':matthews_corrcoef(y,p),'cohen_kappa':cohen_kappa_score(y,p)}

def train_epoch(model,dl,criterion,opt,device):
    model.train()
    loss_sum=0.0
    correct=0
    total=0
    for x,y in dl:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        z=model(x); loss=criterion(z,y); loss.backward(); opt.step()
        loss_sum+=loss.item()*y.size(0); correct+=(z.argmax(1)==y).sum().item(); total+=y.size(0)
    return loss_sum/total,correct/total

def evaluate(model,dl,criterion,device):
    model.eval(); ys=[]; ps=[]; loss_sum=0.0; total=0
    with torch.inference_mode():
        for x,y in dl:
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            z=model(x); loss=criterion(z,y)
            loss_sum+=loss.item()*y.size(0); total+=y.size(0)
            ys.extend(y.cpu().tolist()); ps.extend(z.argmax(1).cpu().tolist())
    m=metrics(ys,ps); m['loss']=loss_sum/total
    return m,ys,ps

def cv_reference(train_aug,train_eval,num_classes,device,args,out):
    targets=np.array(train_eval.targets)
    splitter=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    rows=[]
    for fold,(ti,vi) in enumerate(splitter.split(np.zeros(len(targets)),targets),1):
        seed=42+fold
        seed_all(seed)
        tr=loader(Subset(train_aug,ti),args.batch_size,True,args.workers,seed)
        va=loader(Subset(train_eval,vi),args.eval_batch_size,False,args.workers,seed)
        model=create_model(num_classes).to(device)
        criterion=nn.CrossEntropyLoss()
        opt=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20,eta_min=1e-6)
        best_f1=-1.0; best_epoch=0; best_acc=0.0
        for epoch in range(1,21):
            train_epoch(model,tr,criterion,opt,device)
            vm,_,_=evaluate(model,va,criterion,device)
            if vm['macro_f1']>best_f1:
                best_f1=vm['macro_f1']; best_epoch=epoch; best_acc=vm['accuracy']
            sch.step()
        rows.append({'fold':fold,'best_epoch':best_epoch,'best_accuracy':best_acc,'best_macro_f1':best_f1})
    pd.DataFrame(rows).to_csv(out/'cv_fold_summary.csv',index=False)

def run_seed(train_ds,val_ds,test_ds,class_names,num_classes,device,args,out,seed):
    seed_all(seed)
    tr=loader(train_ds,args.batch_size,True,args.workers,seed)
    va=loader(val_ds,args.eval_batch_size,False,args.workers,seed)
    te=loader(test_ds,args.eval_batch_size,False,args.workers,seed)
    model=create_model(num_classes).to(device)
    criterion=nn.CrossEntropyLoss()
    opt=optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.max_epochs,eta_min=1e-6)
    best=-1.0; best_epoch=0; patience=0; hist=[]
    ckpt=out/f'seed_{seed}_best.pth'
    for epoch in range(1,args.max_epochs+1):
        tl,ta=train_epoch(model,tr,criterion,opt,device)
        vm,_,_=evaluate(model,va,criterion,device)
        hist.append({'epoch':epoch,'train_loss':tl,'train_accuracy':ta,'val_loss':vm['loss'],'val_accuracy':vm['accuracy'],'val_macro_f1':vm['macro_f1'],'learning_rate':opt.param_groups[0]['lr']})
        if vm['macro_f1']-best>1e-4:
            best=vm['macro_f1']; best_epoch=epoch; patience=0; torch.save(model.state_dict(),ckpt)
        else:
            patience+=1
        sch.step()
        if patience>=7: break
    pd.DataFrame(hist).to_csv(out/f'training_history_seed_{seed}.csv',index=False)
    model.load_state_dict(torch.load(ckpt,map_location=device))
    vm,vy,vp=evaluate(model,va,criterion,device)
    tm,ty,tp=evaluate(model,te,criterion,device)
    pd.DataFrame(classification_report(vy,vp,target_names=class_names,output_dict=True,zero_division=0)).T.to_csv(out/f'real_validation_classification_report_seed_{seed}.csv')
    pd.DataFrame(classification_report(ty,tp,target_names=class_names,output_dict=True,zero_division=0)).T.to_csv(out/f'real_test_classification_report_seed_{seed}.csv')
    row={'seed':seed,'best_validation_epoch':best_epoch,'best_validation_selection_score':best}
    row.update({f'val_{k}':v for k,v in vm.items()})
    row.update({f'test_{k}':v for k,v in tm.items()})
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('training_output'))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--eval-batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--max-epochs',type=int,default=30)
    p.add_argument('--skip-cv',action='store_true')
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    train_t,eval_t=tfms()
    train_aug=datasets.ImageFolder(a.data_root/'synthetic_train',transform=train_t)
    train_eval=datasets.ImageFolder(a.data_root/'synthetic_train',transform=eval_t)
    val=datasets.ImageFolder(a.data_root/'real_validation',transform=eval_t)
    test=datasets.ImageFolder(a.data_root/'real_test',transform=eval_t)
    if not(train_aug.class_to_idx==val.class_to_idx==test.class_to_idx): raise RuntimeError('Class mappings differ across splits')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not a.skip_cv: cv_reference(train_aug,train_eval,len(train_aug.classes),device,a,a.output_dir)
    rows=[run_seed(train_aug,val,test,train_aug.classes,len(train_aug.classes),device,a,a.output_dir,s) for s in SEEDS]
    pd.DataFrame(rows).to_csv(a.output_dir/'seed_metrics.csv',index=False)
    summary={'test_accuracy_mean':float(np.mean([r['test_accuracy'] for r in rows])),'test_macro_f1_mean':float(np.mean([r['test_macro_f1'] for r in rows])),'validation_accuracy_mean':float(np.mean([r['val_accuracy'] for r in rows])),'validation_macro_f1_mean':float(np.mean([r['val_macro_f1'] for r in rows]))}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
