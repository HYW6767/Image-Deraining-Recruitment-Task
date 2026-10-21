import torch
from Mymodel import Mymodel

model=Mymodel().eval()
dummy = torch.randn(1, 3, 128, 128)

torch.onnx.export(model,
                  dummy,
                  "mymodel.onnx",
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=17)

print("保存成功")