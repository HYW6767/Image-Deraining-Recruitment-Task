import torch.nn as nn


class DerainNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x