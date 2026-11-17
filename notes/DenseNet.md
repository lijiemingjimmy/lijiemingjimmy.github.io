# DenseNet 论文解读：通过密集连接复用特征

**论文：** Densely Connected Convolutional Networks  
**作者：** Gao Huang、Zhuang Liu、Laurens van der Maaten、Kilian Q. Weinberger  
**发表：** CVPR 2017，Best Paper；预印本最初发布于 2016 年  
**资源：** [CVF 论文页面](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html) · [arXiv](https://arxiv.org/abs/1608.06993) · [PDF](https://arxiv.org/pdf/1608.06993) · [作者代码](https://github.com/liuzhuang13/DenseNet)

**一句话概括：DenseNet 让同一个 dense block 内的每一层直接接收此前所有层的特征，以特征复用和更直接的信息通路改善深层卷积网络。**

## 一、核心机制：把此前特征拼接起来

对于同一个 dense block 中的第 $\ell$ 层：

$$
x_{\ell}=H_{\ell}([x_0,x_1,\ldots,x_{\ell-1}]).
$$

其中，$[\cdot]$ 表示沿通道维拼接，$H_{\ell}$ 是该层的特征变换。

后面的层可以直接访问早期层的表示，同时学习新的特征。不同 dense block 之间通过 transition layer 连接，因此“此前所有层”需要在 block 范围内理解。[来源：论文](https://arxiv.org/abs/1608.06993)

## 二、为什么这种连接有用

密集连接使信息和梯度有更直接的传播路径，也减少了后续层重复学习相似特征的需要。每层只新增一定数量的特征通道，既有表示继续被复用。

与残差相加相比，拼接会保留不同层的特征身份，同时也会扩大后续层可见的通道集合。论文的贡献围绕这一结构及其特征复用效果展开。[来源：作者仓库](https://github.com/liuzhuang13/DenseNet)

## 三、它在这组论文中的位置

来稿提及 DenseNet，主要是介绍黄高课题组更早的代表工作。它研究卷积网络结构，与 RLVR、推理 token 熵和 DLM 解码顺序属于不同问题。

因此，在阅读这组论文时，DenseNet 适合作为研究背景单独理解；三篇推理工作之间的直接联系见 [阅读总览](FLORA-RLVR-论文阅读总览-2026-10-22.md)。
