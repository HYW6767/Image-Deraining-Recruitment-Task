# 任务二：试试看前人的杰作😘

1. 在 DerainNet 文件中，根据图片编写模型代码（代码需要整洁）

已写完~~虽然不太整洁~~

2. 配置 main 函数命令行参数，使用编写好的代码训练并测试模型性能（epoch 50）

已执行以下参数，进行训练

```powershell
python main.py --data_dir "data\Rain200L" --model derainnet --mode train --epochs 50
```

测试：

```powershell
python main.py --data_dir "./data/Rain200L" --model derainnet --mode test --checkpoint "checkpoints/best_derainnet.pth.tar"
```

结果如下

![](https://picgocloud.com/m/87abea5c-22f5-47f7-b23e-811ef01d6b0d.png)

3. 思考：
   1. **为什么 DerainNet 要先下采样后上采样？**下采样、上采样是什么？什么是 Concat？

**下采样**是一种扩大感受野的方法，通过一些规则（如 maxpooling、averagepooling）将特征图池化缩小，使得同样大小的卷积核能捕捉更多信息

**上采样**则将池化的特征图扩张，还原原本的分辨率，常用双线性插值或转置卷积

**下采样**是为了扩大感受野，使得模型可以捕捉像雨丝这样的长距离依赖，DerainNet 中，将图片分别分至三个分支，分别工作在 1x、1/2x、1/4x 三个尺度，原尺度捕捉细节，1/4 尺度捕捉长雨线，1/2 折中，同时，为了保证结果能够保持原始分辨率，三个图要拼接在一起，于是进行**上采样**，还原成原始分辨率，使得其能与未放大的特征图拼接

concat 可以将多个特征图沿某个维度拼接起来，在这里，concat 将三个分支的特征图沿通道合并，并成一个 192 通道的特征图，后用 1x1 卷积融合降维，输出一个三通道的雨层

   2. 将你的疑惑和学到的新知识写在 markdown 中

**学到的新知识**

1. 下采样的用法

2. 上采样

3. 可以将原特征图一分为三，以此来捕捉不同尺度的信息

4. 拼接

**疑惑**

事实上，雨线还存在空间关系，景深等信息，但单张图像缺乏深度信息，这个信息是否能恢复？🤔🤔
