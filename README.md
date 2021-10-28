# Bird Recognition

## 总纲

![image-20211028202831886](https://i.loli.net/2021/10/28/5ZtwzqiNTv6yCFL.png)

![image-20211028205757812](https://i.loli.net/2021/10/28/7ZC8OMJjFxe9mnz.png)

现代深度学习一般使用ampspec和melspec

逻辑可以看https://www.cnblogs.com/BaroC/p/4283380.html

## 时域图

利用python的wave库，很容易得到一段音频的频谱图。

![image-20211028131836040](https://i.loli.net/2021/10/28/1l6Q2MREd4O3yJ9.png)

## 频谱图

使用scipy.fft对时域图进行快速傅里叶变换。

![image-20211028230103750](https://i.loli.net/2021/10/28/fNsw4tq2iv8ylom.png)

采样时一般以25ms或1024个采样点为一个窗口。





## 文献笔记

**MFCC**：梅尔倒谱系数。相比于传统的频谱图，一方面采用了Mel滤波，对频谱进行平滑化，消除谐波的作用。另一方面采用（DFT）去相关滤波器组系数。

https://zhuanlan.zhihu.com/p/88625876

**GMM**：高斯混合模型（不太懂）

**GLCM**：灰度共生矩阵（探究灰度图像中某种形状的像素对，在全图中出现的次数）。使用联合条件概率密度$P(i,j|d,\theta)$表示该矩阵。即在给定空间距离d和方向θ时，灰度以i为起始点（行），出现灰度级j（列）的概率。（很像卷积？？）在纹理分析中有重要意义。

在论文2中，可以根据GLCM计算得到频谱图的特征向量，作为随机森林的输入。

https://cloud.tencent.com/developer/article/1588343

**RF**：随机森林，集成学习（Bagging）的一种方法。利用多个没有关联的决策树进行判断。可以理解成不同的“评委”。

https://easyai.tech/ai-definition/random-forest/

**GTF**：全局纹理特征

**LTF**：局部纹理特征（在论文中，LTF的效果更好）

**LDA**：线性判别分析

**confusion matrix**：混淆矩阵——用于衡量评判的精度

**VQ**：矢量量化（不懂）

**DTW**：动态时间规整。用动态规划的方法判断两个时间序列的相似度。











