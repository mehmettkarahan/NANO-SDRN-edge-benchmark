from torch import nn
import torch

class SEBlock(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        hidden=max(1, channels//ratio)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(nn.Linear(channels,hidden),nn.ReLU(inplace=True),nn.Linear(hidden,channels),nn.Sigmoid())
    def forward(self,x):
        b,c,_,_=x.shape
        y=self.fc(self.pool(x).view(b,c)).view(b,c,1,1)
        return x*y

class HybridDomainNorm(nn.Module):
    def __init__(self, channels, instance_ratio):
        super().__init__()
        n=max(1,min(channels-1,int(round(channels*instance_ratio))))
        self.n=n
        self.inorm=nn.InstanceNorm2d(n,affine=True,track_running_stats=False,eps=1e-5)
        self.bnorm=nn.BatchNorm2d(channels-n)
    def forward(self,x):
        a,b=torch.split(x,[self.n,x.shape[1]-self.n],dim=1)
        return torch.cat([self.inorm(a),self.bnorm(b)],dim=1)

class DSConv(nn.Module):
    def __init__(self, cin, cout, kernel_size=3, use_se=True, se_ratio=8):
        super().__init__()
        p=kernel_size//2
        self.dw=nn.Conv2d(cin,cin,kernel_size,padding=p,groups=cin,bias=False)
        self.bn1=nn.BatchNorm2d(cin)
        self.act1=nn.Hardswish(inplace=True)
        self.pw=nn.Conv2d(cin,cout,1,bias=False)
        self.bn2=nn.BatchNorm2d(cout)
        self.act2=nn.Hardswish(inplace=True)
        self.se=SEBlock(cout,se_ratio) if use_se else nn.Identity()
    def forward(self,x):
        x=self.act1(self.bn1(self.dw(x)))
        x=self.act2(self.bn2(self.pw(x)))
        return self.se(x)

class StandardConv(nn.Module):
    def __init__(self, cin, cout, use_se=True, se_ratio=8):
        super().__init__()
        self.conv=nn.Conv2d(cin,cout,3,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(cout)
        self.act=nn.Hardswish(inplace=True)
        self.se=SEBlock(cout,se_ratio) if use_se else nn.Identity()
    def forward(self,x):
        return self.se(self.act(self.bn(self.conv(x))))

class SpatialRefine(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.block=nn.Sequential(nn.Conv2d(channels,channels,5,padding=2,groups=channels,bias=False),nn.BatchNorm2d(channels),nn.Hardswish(inplace=True))
        self.se=SEBlock(channels,8)
    def forward(self,x):
        return self.se(self.block(x))

class NanoSDRN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,32,3,stride=2,padding=1,bias=False),nn.BatchNorm2d(32),nn.Hardswish(inplace=True))
        self.stage1=nn.Sequential(DSConv(32,32,3,False),nn.MaxPool2d(2))
        self.stage2=nn.Sequential(DSConv(32,64,3,True,8),DSConv(64,64,5,True,8),HybridDomainNorm(64,0.50),nn.MaxPool2d(2))
        self.texture=nn.Sequential(StandardConv(64,128,True,8),DSConv(128,128,3,True,8),HybridDomainNorm(128,0.25),SpatialRefine(128))
        self.pool1=nn.MaxPool2d(2)
        self.semantic=nn.Sequential(StandardConv(128,288,True,4),DSConv(288,288,3,True,4))
        self.pool2=nn.MaxPool2d(2)
        self.global_stage=nn.Sequential(DSConv(288,384,3,True,4),DSConv(384,384,3,True,4))
        self.avg=nn.AdaptiveAvgPool2d(1)
        self.max=nn.AdaptiveMaxPool2d(1)
        self.classifier=nn.Sequential(nn.Linear(768,160),nn.Hardswish(inplace=True),nn.Dropout(0.2),nn.Linear(160,num_classes))
    def forward(self,x):
        x=self.stem(x)
        x=self.stage1(x)
        x=self.stage2(x)
        x=self.texture(x)
        x=self.semantic(self.pool1(x))
        x=self.global_stage(self.pool2(x))
        x=torch.cat([self.avg(x).flatten(1),self.max(x).flatten(1)],dim=1)
        return self.classifier(x)

def create_model(num_classes=3):
    return NanoSDRN(num_classes=num_classes)
