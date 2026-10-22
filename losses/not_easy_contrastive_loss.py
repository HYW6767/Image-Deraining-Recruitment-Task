import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from math import log,e



MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class RealContractiveloss(nn.Module):
    def __init__(self):
        super(RealContractiveloss, self).__init__()
        vgg=vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35]
        
        
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg=vgg.eval()
        self.pool=nn.AdaptiveAvgPool2d(1)

    def encode(self,x):
        mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
        x=self.vgg((x-mean)/std)
        return self.pool(x).flatten(1)

    def forward(self,rainy, generated, ground_truth,T=0.2):
        with torch.no_grad():
            v_r=self.encode(rainy)
            v_gt=self.encode(ground_truth)
        v_g=self.encode(generated)
        

        cos=nn.functional.cosine_similarity
        pos=cos(v_g, v_gt).mean()/T
        neg=cos(v_g,v_r).mean()/T
        loss=-torch.log(torch.exp(pos)/(torch.exp(neg)+torch.exp(pos)))
        return loss