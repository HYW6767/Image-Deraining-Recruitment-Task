from Mymodel import Mymodel
import torch

model = Mymodel()
total = sum(p.numel() for p in model.parameters())
print(total) 