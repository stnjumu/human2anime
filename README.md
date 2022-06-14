# human2anime

使用mindspore+cyclegan完成人脸到漫画脸的转换

## 算法模型选择

使用[mindspore ModelZoo](https://gitee.com/mindspore/models/tree/master)中的[CycleGAN](https://gitee.com/mindspore/models/tree/master/research/cv/CycleGAN)模型代码

人脸向漫画脸的转换是Domain Adaptation问题，或者说风格迁移问题。可以使用的[Pix2Pix](https://arxiv.org/abs/1611.07004)和[CycleGAN](https://arxiv.org/abs/1703.10593)等，但前者要求训练数据必须是成对的，而现实生活中，要找到两个域(画风)中成对出现的图片是相当困难的；后者的优势是只需要两种域的数据，而不需要他们有严格对应关系，但生成效果不是特别好。基于数据集的获取难度，我选择了CycleGAN来做此任务。

### CycleGAN简介

对于两个数据集A和B，分布为DA和DB，CycleGAN想要训练两个生成器和两个鉴别器。

1. G_A：用B的图片生成A的图片的生成器
2. G_B: 用A的图片生成B的图片的生成器
3. D_A：判断图片是否属于A的鉴别器
4. D_B: 判断图片是否属于B的鉴别器

训练过程使用GAN的对抗训练策略；由于图片是非成对的，所以生成图片的质量难以评价，原论文提出以下两种loss用于训练：

1. GAN loss: 生成图片的质量应尽量骗过鉴别器
2. Cycle Consistency Loss: 对于A中的图片a，G_A(G_B(a))应尽可能接近a，此loss是为了生成图片与原图仅风格不同而内容相同；对于B中图片同样有类似loss.

论文中还有更多细节，我这里只是简单介绍，感兴趣可以查看原论文：[CycleGAN](https://arxiv.org/abs/1703.10593).

## 数据集选择

对于漫画脸，我从网络中搜索到了[动漫人脸数据集](http://www.seeprettyface.com/mydataset_page3.html#anime),
![动漫人脸数据集概览](./imgs_for_README/anime.jpg)
观察发现，动漫人脸数据集中的多为女的，男的不足1成，所以对于人脸数据集，我选择了[网红人脸数据集](http://www.seeprettyface.com/mydataset_page3.html#wanghong)
![网红人脸数据集概览](./imgs_for_README/wanghong.jpg)

### 数据集简单预处理

由于选择的数据集已经是人脸部分，所以我只做了以下预处理工作：

1. 对每个数据集，等间隔采样得到10000张图片作为训练集，1000张图片作为测试集
2. 对每张图片，resize到256*256大小

后来在华为ModelArts平台训练时发现完成一次完整训练需要约70小时，而每小时需要约20元，由于只有500块代金券，所以无奈只能再次采样，**每个数据集的训练集只有1000张图片，测试集有100张图片**

## 实验部分

Mindspore框架可以在linux+GPU平台安装使用，我参考[MindSpore安装指南](https://www.mindspore.cn/install#%E5%AE%89%E8%A3%85cuda)尝试在实验室的服务器上安装该框架，但由于gcc和gmp的安装需要root权限，并未成功；最终只能在华为的ModelArts网上平台跑代码实验。

### 华为ModelArts平台使用

摸索发现，在ModelArts平台完成自定义数据集和自定义python代码的训练主要有以下步骤：

1. 登录OBS对象存储服务，创建OBS桶和文件夹；
2. 上传数据集和上传代码: 由于网页端上传只支持单一文件，在线解压非常复杂，而且还是测试版本，无法上传下载文件夹。无奈我使用了华为提供的工具：[obsutil](https://support.huaweicloud.com/utiltg-obs/obs_11_0001.html)；根据[教程](https://support.huaweicloud.com/utiltg-obs/obs_11_0003.html)下载安装该工具并[正确设置](https://support.huaweicloud.com/utiltg-obs/obs_11_0005.html)后可以传文件夹。
3. 在ModelArts管理控制台的训练管理-训练作业New页面，创建新的训练作业启动训练，作业的关键配置如下图所示：![训练配置](./imgs_for_README/%E8%AE%AD%E7%BB%83%E9%85%8D%E7%BD%AE.png)

