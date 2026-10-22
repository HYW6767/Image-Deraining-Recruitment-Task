# 来看看标准的对比学习😘

1. 你觉得开头的话有道理吗？对比学习一般具有哪些特征？

有点道理，感知损失变体严格来说只能算是使用了对比学习的思想，理由如下：

- 它训练的是去雨模型而不是编码器，在这里，编码器 VGG 的参数是冻结的
- 它是有监督信号的，而真正的对比学习是无监督学习
- 其只有一个正样本和一个负样本，而对比学习拥有多个负样本

对比学习的特征：

- 让“相似的样本”在特征空间里靠近，让“不相似的样本”远离
- 属于无监督学习
- 拥有正样本和负样本

2. 用对比学习的方式解释一下感知损失变体，对于这个预训练模型来说，什么是正样本？什么是负样本？

正样本是无雨图，负样本是有雨图，锚点是去雨图

3. 三选一：
   1. 如果要让这个感知损失变体满足对比学习中最经典的 InfoNCE 损失 的形式，我们应该怎么修改代码？尝试修改该损失函数并运行模型
   2. 我们使用的预训练模型作为编码器，但他们都是在 ImageNet 这一数据集上训练得到的，这个数据集中并没有雨线存在，因此，预训练模型可能不是很擅长捕捉雨纹这种特定特征。请尝试设计一个训练编码器的算法，微调得到一个更好的编码器权重
   3. 可能你并不是很想继续搞去雨，那么也可以尝试复现一篇简单对比学习的工作，将感受和结果写进 markdown 中

我选第 1 个选做题

InfoNCE 损失的公式长这样  
L = −log [ exp(sim(q, k⁺) / τ) / Σᵢ exp(sim(q, kᵢ) / τ) ]

于是我在 losses 下将原本的感知损失变体复制了一份命名为 `not_easy_contrastive_loss.py`🤓，并在这里尝试满足上述条件

在 main.py 内加入了新的形参

**注意**，这里默认了 contrastiveloss 和新的 loss 不会同时开着

使用以下命令

```powershell
python main.py --data_dir "data\Rain200L" --model unet --mode train --epochs 50 --use_noteasy_contrastive_loss
```

使用以下命令测试

```powershell
python main.py --data_dir "data\Rain200L" --model unet --mode test --checkpoint "checkpoints/best_unet_real_contrastive.pth.tar"
```

结果如下
![](https://picgocloud.com/m/f6970094-680d-40e8-8a1a-d01828ad8af0.png)