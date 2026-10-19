# 任务一：搞半天原来要自己写

出题人提供的代码库中，models 模块下似乎只有 Baseline_net 和 unet 两份代码文件，在 main 文件中的模型索引的选择（choise）等也只有“baseline”和“unet”这两个，选择太少啦！😩

请你根据源代码中这两个部分的设计方式，模仿着加上“DerainNet”和“Mymodel”这两个类似的部分。要求：

   1. 在 get_args 函数中加入“derainnet”和“mymodel”两个 choice

如下

 ```python
parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "unet","derainnet","mymodel"], help="Model to use")
```

在 choices 列表中加入了 mymodel 和 unrainnet 两个新元素

   2. 在 models 中创建 DerainNet 和 Mymodel 两个代码文件，用最简单的方式写一个模型类，并将模型导入到 main 文件中
     如下

```python
#models\derainnet.py
import torch.nn as nn


class DerainNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x     
```

```python
#models\mymodel.py
import torch.nn as nn


class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x
```

以上都是基础的模型类尚未加入各种层

再导入到 `main.py`

```python
from models.derainnet import DerainNet
from models.mymodel import Mymodel
```

   3. 根据“baseline”和“unet”的设计方式补充 main 函数中使用不同 args.model 时的判断条件

在 `main.py` 中对 args.model 进行判断后加入以下几行

```python
elif args.model=="mymodel":
    model=Mymodel().to(device)
elif args.model=="derainnet":
    model=DerainNet().to(device)
```
