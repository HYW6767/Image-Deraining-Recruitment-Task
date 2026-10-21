import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]






class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg=vgg19(pretrained=True).features[:35]


        for p in vgg.parameters():
            p.requires_grad = False
        self.use=vgg.eval()

    def forward(self, generated, ground_truth):
        mean = torch.tensor(MEAN, device=generated.device).view(1, 3, 1, 1)
        std = torch.tensor(STD, device=generated.device).view(1, 3, 1, 1)
        generated=self.use((generated - mean) / std).abs().mean()
        with torch.no_grad():
            ground_truth=self.use((ground_truth - mean) / std).abs().mean()
        loss=F.l1_loss(generated, ground_truth)

        return loss