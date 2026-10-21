import torch.nn as nn
import torch

#加入Resblock，这里依然复制粘贴之前在baseline中写的
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


#实现PReNet里所说的LSTM（torch内自带的LSTM会破坏图像结构，不能在PReNet内使用）
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            hidden_dim * 4,
            kernel_size,
            padding=padding,
            bias=True,
        )
    def forward(self,x,state):
        h_prev, c_prev = state
        #拼接输入和隐藏状态
        combined = torch.cat([x, h_prev], dim=1)
        #算出四个门
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)      # 输入门
        f = torch.sigmoid(f)      # 遗忘门
        o = torch.sigmoid(o)      # 输出门
        g = torch.tanh(g)         # 候选记忆
        #更新细胞状态
        c_next = f * c_prev + i * g

        #输出隐藏状态
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


#PReNet层
class PReNetLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.f_in=nn.Sequential(nn.Conv2d(6,64,3,padding=1),
                                nn.ReLU(inplace=True))#即PReNet论文里提到的输入层
        self.f_res=nn.Sequential(Baseblock(64,64),
                                    Baseblock(64,64),
                                    Baseblock(64,64),
                                    Baseblock(64,64),
                                    Baseblock(64,64))#循环层，内含五个ResBlock
        self.f_recurrent=ConvLSTMCell(64,64)
        self.f_out=nn.Conv2d(64,3,3,padding=1)
    def forward(self,pred_x,y,state):
        feat=self.f_in(torch.cat([pred_x,y],dim=1))#拼接上轮的x和原图y
        h,c=self.f_recurrent(feat,state)#使用LSTM记忆
        out=self.f_out(self.f_res(h))#输出的会进入下一次循环
        return out,(h,c)

#下面的模型为模仿PReNet搭建
class Mymodel(nn.Module):
    def __init__(self,T=6):
        
        super().__init__()
        self.T=T
        self.layer=PReNetLayer()

    def forward(self,x):
        y=x
        #初始化h,c
        h0=torch.zeros(x.shape[0], 64,x.shape[2], x.shape[3], device=x.device)
        c0=torch.zeros(x.shape[0], 64,x.shape[2], x.shape[3], device=x.device)
        #循环层，循环6次，共享参数
        state=(h0,c0)
        for i in range(self.T):
            x,state=self.layer(x,y,state)

        
        return x