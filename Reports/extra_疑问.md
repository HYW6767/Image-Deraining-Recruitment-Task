在训练模型时没事干，靠自己的理解写了一个 `use_model.py` 来输出去雨的图片，使用的原图为 Rain200L 中 test 的 1.png

一开始，我在 load_img 函数里是加入了 transforms.Resize（）的，但是发现图片太糊了根本看不到原图细节，于是突发奇想，将 resize 去掉了，但是发现在原分辨率下，去雨的效果非常差，我推测是由于原模型训练时用的是 resize 过的图片导致的，进行了一番查证，验证了我的推测

现在我产生了一些**疑问**：

1. 如果我想让模型适应更大的分辨率，是否意味着我一定得在训练时就投喂分辨率更大的数据？

2. 如果要使用更大的分辨率进行投喂，我的显存不够用怎么办？

3. 如果我要让模型能适应不同的分辨率，我该怎么做？

4. 如何判断效果差是“尺度不匹配”导致的，而不是模型容量不足或训练不充分？

望得到学长解答（所有结果均存在项目文件夹里）

在尝试多无 resize 的模型训练时，我修改了 dataset 的内容，后面让 ds 检查了一下代码，疑似发现了一个问题：  
在 transform 里这两行随机变化对有雨图和无雨图是独立随机的，会导致有雨图和干净图对不上，我将其注释掉了

```python
transforms.RandomCrop(image_size),
transforms.RandomHorizontalFlip(),
```

不对，在 noresize 的训练中图片尺寸不相同，无法训练，得修改 noresize 的代码

我将 resize 改为裁剪了

对 noresize 进行训练

```powershell
python main.py --data_dir "data\Rain200L" --model derainnet --mode train --epochs 50 --use_noresize
```

使用 `use_model.py` 尝试生成图片

还是不太理想，不过可以发现雨丝没那么明显了

试试用这个将 mymodel 训练 100 轮

```powershell
python main.py --data_dir "data\Rain200L" --model mymodel --mode train --epochs 100 --use_noresize 
```

裁剪至 256 显存撑不住啊，改成 128

开始训练后电脑直接下雨了😨😨

挂一晚上试试看

怎么训一半显存又炸了

加上 `--batch_size 2` 试试

测试一下

生成完成，虽然效果好了，但是仍有部分雨丝

为了防止这张图后续被覆盖，我将其重命名为 `output_mymodel_noresize1.jpg`

换个 loss 试试看

```powershell
python main.py --data_dir "data\Rain200L" --model mymodel --mode train --epochs 100 --use_noresize --use_noteasy_contrastive_loss
```

卧槽这个训的太慢了，训练 10 轮应该差不多了

生成一下图片

![](https://picgocloud.com/m/a7cfe7ce-8147-483f-893d-5c202410dea8.jpg)

说实话，效果不尽人意，或许是底图太白了？

换张图试试

![](https://picgocloud.com/m/15a0b1a6-5555-475c-b650-e98037dccacf.jpg)

也不太行啊

相比原来模糊不少

用之前的模型试试看

好不少，但是距离能完全去雨还是有一段路啊

先建一个 img 目录存一下刚刚生成的图（防覆盖）

卧槽我才发现 dataset 里评估的 resize 我忘记注释了

又更改了一下 dataset 的逻辑，并加入了可选 MSEloss 的选项

现在 checkpoint 的命名太臃肿了，我让 ai 帮助我改一下，先把这版上传 GitHub
