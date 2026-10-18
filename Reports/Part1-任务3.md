# 任务三：我有说明书，太好啦😄

1. 环境搭建：小菜一碟
   1. 请创建一个新的虚拟环境，根据 requirements.txt 安装所需的依赖库，附上截图

   ~~我去不早说，我都装好了~~
   
   重新装了一遍
   
   ![](https://picgocloud.com/m/d8c85530-8b27-4628-b9a1-63599504b21a.png)

2. 实现一个简单的模型
在 models 软件包中的 baseline_net 文件中，实现一个 8 层的卷积核大小为 3x3、中间通道层数为 64 的 Residual Block 的 ResNet，并根据上文中的思路决定模型最终的前向传播方法（重要）
   1. 将实现好的模型代码放到 markdown 文件中

```python
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
```

此处包含 8 个 3x3 卷积层 + 一个输出头

   2. 思考下面这个问题，并给出回答：

    根据 CNN 的卷积计算原理，一个 3x3 的卷积核在当前层只能感受到附近 3x3 区域的信息，无法捕捉长距离依赖（例如贯穿图片的雨线）。在分类任务中，模型通过池化层不断减小特征图尺寸来扩大感受野，但在去雨任务中，我们通常希望保持原始分辨率。这意味着需要堆叠非常深的网络才能让像素间进行远距离信息交流，而深层网络会带来梯度消失、训练困难等问题。针对此问题，请你思考并寻找可能的解决方案，将你的想法写入 Markdown 文件中

    进行了一些查询后，我的想法如下：

    方案一：隔一段距离取一个点，变相增大了感受野，缺点是采样稀疏

    方案二：把图分成几路：一路保持原尺寸看细节，一路缩到一半看中等结构，一路缩到四分之一看长程结构，处理完再放大回来拼在一起，在四分之一那一路中 3x3 的卷积核相当于 12x12，缺点是会丢失细节

    方案一改动较小，而方案二更加强大，若条件允许，我应该会选择方案二
    
3. 这个代码是什么意思？😳

   1. 学会在 main 函数中设置命令行参数，成功运行代码，并截图保存使用以下参数时训练的结果
   
   老大，我们真的要用 cpu 跑这个吗😨😨😨

   我重下了一个能使用我显卡的 torch 版本

   进行运行，结果如下
   ![](https://picgocloud.com/m/4878dc00-50a5-46fb-b934-8581a3f458f9.png)

   2. 将你学到的与代码相关的新东西写在这个小问中

   设置 Python 的命令行参数的方法：

   第一步：导入 argparse 库

   ```python
   import argparse
   ```
   
   第二步：建一个解释器：

   ```python
   parser = argparse.ArgumentParser()
   ```

   第三步：声明各种参数

   ```python
   parser.add_argument("--epochs", type=int, default=20)
   #                     ↑参数名    ↑类型     ↑默认值
   parser.add_argument("--data_dir", type=str, required=True, help="数据集路径")
   #                                              ↑是否必填      ↑帮助文字--help时显示
   ```

   第四步：解析

   ```python
   args = parser.parse_args()
   ```

   后续可通过 `args.xxx` 来访问其中的数据

   3. （附加）：根据 PSNR 的公式，编写计算 PSNR 的代码，替换原本的库函数，放在 utils 文件中

   ```python
   import math

   def psnr(image_true,image_test,*,data_range=None):
      MAX_I=1 if data_range is None else data_range

      mse=((image_true-image_test)**2).mean()
      if mse==0:
         return float("inf")
      
      
      result=10*math.log10((MAX_I**2)/mse)
      return result
   ```
