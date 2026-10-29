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

还是不太理想

不过雨丝与原来比少很多，时间不太够了后面再来看看