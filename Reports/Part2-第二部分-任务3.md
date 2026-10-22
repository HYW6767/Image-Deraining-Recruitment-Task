# 改造感知损失函数，使用特征做对比😏

1. 在 losses 软件包中创建一名为 easy contrastive loss 的文件，修改 main 文件（包括 get_args、train_one_epoch、main 函数），使得能够通过输入形参的方式启用 easy contrastive loss 进行学习

对 main.py 进行了大改，使其能够适配新加入的 contrastiveloss

2. 在 easy contrastive loss 文件中编写新的损失函数：
   1. 学习 AdaptiveAvgPool2d，明白如何使用其将二维的图片压缩成一条特征向量
   2. 流程：
      1. 使用与感知损失相同的预训练模型，取相同中间层的输出作为特征图
      2. 使用 AdaptiveAvgPool2d 将特征图压缩为一条特征向量
      3. 对输入图像（有雨）、预测图像（去雨结果）和目标图像（无雨）的特征向量计算余弦相似度，并将其作为损失项（即最大化预测图像与目标图像的相似度，同时最小化预测图像与输入图像的相似度）

      已完成

3. 使用新的损失函数训练并测试模型（不用过多关注指标高低）
输入以下内容：

```PowerShell
python main.py --data_dir "data\Rain200L" --model unet --mode train --epochs 100 --use_contrastive_loss
```

训练结束，再输入以下内容进行测试：

```powershell
python main.py --data_dir "data\Rain200L" --model unet --mode test --checkpoint "checkpoints\best_unet_contrastive.pth.tar"
```

结果如下

![](https://picgocloud.com/m/6d0c90e1-8970-4350-b9e4-5cf09011f62a.png)
