import torch.nn as nn
import torch.nn.functional as F
import torch


#基础的残差块，这里直接复制了之前baseline使用的代码
class Baseblock(nn.Module):

    def __init__(self,in_ch,out_ch,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(in_ch,out_ch,3,stride, 1, bias=False)
        self.bn1=nn.BatchNorm2d(out_ch)
        self.conv2=nn.Conv2d(out_ch,out_ch,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(out_ch)
        self.relu=nn.ReLU(inplace=True)


        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self,x):
        out=x
        identity=x
        out=self.conv1(out)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample!=None:
            identity=self.downsample(identity)

        out=self.relu(out+identity)
        return out



class DerainNet(nn.Module):
    def __init__(self,C=64):
        super().__init__()
        self.stem = nn.Sequential(
                    nn.Conv2d(3, C, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                    )
        self.branch1=nn.Sequential(Baseblock(C,C),
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 )
        self.branch2=nn.Sequential(
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 )
        self.branch3=nn.Sequential(
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 Baseblock(C,C),
                                 Baseblock(C,C)
                                 )
        self.pool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.out_conv=nn.Conv2d(3*C,3,1)

    def forward(self,x):
        feat=self.stem(x)
        p1=feat
        p2=self.pool(feat)
        p3=self.pool(p2)


        branch1=self.branch1(p1)+p1
        branch2=self.branch2(p2)+p2
        branch3=self.branch3(p3)+p3

        branch2 = F.interpolate(branch2, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        branch3 = F.interpolate(branch3, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        rain=torch.cat([branch1,branch2,branch3], dim=1)
        rain=self.out_conv(rain)

        return x-rain