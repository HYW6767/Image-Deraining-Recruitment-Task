import torch.nn as nn

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

class BaselineNet(nn.Module):
    def __init__(self):
        super(BaselineNet, self).__init__()
        self.block1=Baseblock(3,64,1)
        self.block2=Baseblock(64,64)
        self.block3=Baseblock(64,64)
        self.block4=Baseblock(64,64)
        self.out_conv=nn.Conv2d(64,3,1)

    def forward(self, x):
        rain=self.block1(x)
        rain=self.block2(rain)
        rain=self.block3(rain)
        rain=self.block4(rain)
        rain=self.out_conv(rain)     
        
        return x-rain
