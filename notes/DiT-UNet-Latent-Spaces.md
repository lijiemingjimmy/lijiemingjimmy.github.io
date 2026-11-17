# DiT、U-Net 与 Latent Spaces：把生成目标、网络和表示分开理解

> Flow Matching / Diffusion 规定模型学什么；DiT / U-Net 决定用什么网络学；latent space 决定在哪个空间里学。三者可以组合。

阅读材料：[DiT、U-Net 与 Latent Spaces.pdf](/Users/lijieming/Downloads/DiT、U-Net%20与%20Latent%20Spaces.pdf)，共 14 页。本文按原文第 6.1—6.3 节展开，补充张量形状、VAE 推导、训练与采样流程，以及与视频和机器人模型的联系。

原 PDF 的图 14、15、16 和算法 6 等位置显示的是加载占位符，部分长公式越过右边界。下文结构图和伪代码是根据正文重新整理的解释，不是缺失原图的复原。案例按 SD3、Movie Gen 对应技术论文理解，不把讲义中的“最先进”作为当前排名结论。

## 1. 先把三个问题分开

| 层面 | 回答的问题 | 例子 |
|---|---|---|
| 生成建模方法 | 如何构造训练目标，从噪声得到数据？ | Diffusion、Flow Matching |
| 神经网络架构 | 谁来预测噪声、速度或其他目标？ | U-Net、DiT |
| 数据表示 | 网络操作的是像素，还是压缩表示？ | pixel space、latent space |

因此可以有 pixel-space U-Net，也可以有 latent-space U-Net；可以用 DiT 做扩散，也可以用 DiT 学 Flow Matching 的速度场。

这份材料承接前面的 Flow Matching 推导，讨论如何把抽象的向量场实现为处理图像、视频的大网络。

设网络为：

$$
u_\theta(x_t,t,c).
$$

其中 $x_t$ 是生成路径上时刻 $t$ 的样本，$c$ 是文本、类别等条件。以速度预测为例，网络输出与 $x_t$ 形状相同，告诉求解器当前样本该怎样变化。

**一次网络前向给出当前的变化方向，采样器负责沿着这些方向更新样本。** 这一区分能解释为什么网络结构、训练目标、采样步数是不同的设计选择。

## 2. 时间、类别和文本怎么进入网络？

### 2.1 时间不是视频中的第几帧

生成时间 $t$ 表示噪声到数据路径上的位置。视频帧索引表示一个视频内部的先后顺序，两者独立。

同一张带噪图像，在不同的 $t$ 下应采用不同的预测策略。因此模型需要知道 $t$。常见做法是 Fourier / sinusoidal embedding：

$$
\operatorname{TimeEmb}(t)
=\sqrt{\frac{2}{d}}
[\cos(2\pi w_1t),\ldots,\cos(2\pi w_{d/2}t),
\sin(2\pi w_1t),\ldots,\sin(2\pi w_{d/2}t)].
$$

这一写法假设 $d$ 为偶数。每对正余弦的平方和为 1，因此整个向量的范数为 1。不同频率提供不同尺度的时间变化特征，再经 MLP 转成网络使用的条件。

具体频率、缩放和后续投影可以变化；关键是把一个标量时间变成易于调制网络的表示。

### 2.2 类别条件是可学习的查表向量

如果 $c$ 是类别编号，可以用 embedding table 获取 $e_c$。时间和类别可以一起生成后续网络的归一化参数。

类别嵌入一般跟着生成网络训练，不要求额外的文本编码器。

### 2.3 文本通常保留成一个序列

一句提示可以编码为：

$$
Y=\operatorname{TextEncoder}(c)\in\mathbb R^{M\times d_y}.
$$

$M$ 是文本 token 数。再通过投影调整到网络需要的维度。

一个 pooled 向量适合提供整句信息；token 序列让图像位置可以分别关注具体词语。例如“红杯子在蓝盒子左侧”，颜色、物体与空间关系都需要被保留。

CLIP 也可以提供 token 特征，不能把“CLIP”固定理解为只能输出一个全局向量。实际系统会组合不同编码器的 pooled 表示和序列表示。

## 3. DiT：把空间网格转成 token，再预测每个位置的变化

### 3.1 Patchify 的维度怎么走？

设输入是：

$$
x_t\in\mathbb R^{C\times H\times W}.
$$

使用大小为 $p\times p$ 的 patch，暂且假设整除，则：

$$
N=\frac{H}{p}\frac{W}{p},\qquad
\operatorname{Patchify}(x_t)\in\mathbb R^{N\times Cp^2}.
$$

每个 patch 展平成 $Cp^2$ 个数，线性投影到隐藏维度 $d$：

$$
X=\operatorname{Patchify}(x_t)W_{\rm in}+b,
\qquad W_{\rm in}\in\mathbb R^{Cp^2\times d}.
$$

得到 $N\times d$ 的 token 序列。还需要位置编码，让网络知道各 token 对应的空间位置。

输出端把每个 token 投影回 $Cp^2$ 维，再 unpatchify，恢复为 $C\times H\times W$ 的速度预测。某些扩散实现还会输出额外的方差通道，输出维度随目标而定。

~~~text
带噪图像 / latent
       ↓ patchify + 线性投影 + 位置编码
空间 token 序列 ← 时间条件、文本或类别条件
       ↓ 多层 Transformer
输出投影 + unpatchify
       ↓
与输入同形状的速度 / 噪声预测
~~~

### 3.2 Self-attention 和 cross-attention 分别做什么？

注意力基本操作是：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right)V.
$$

Self-attention 的 $Q,K,V$ 都来自图像 token，用于不同区域之间交换信息。

Cross-attention 可以用图像 token 生成 $Q$，用文本 token 生成 $K,V$。每个图像位置据此读取相关文本信息。

多头注意力在多个投影空间中分别计算，再拼接输出。模型可以学习不同的关系，但每个头并没有预先指定的固定语义。

### 3.3 AdaLN 怎样注入条件？

让时间与其他条件产生逐通道缩放和偏移：

$$
(\gamma,\beta)=\operatorname{MLP}(e_t,e_c),
$$

$$
\operatorname{AdaLN}(X)
=(1+\gamma)\odot\operatorname{LayerNorm}(X)+\beta.
$$

$\gamma,\beta$ 沿 token 维广播。网络在不同噪声水平下，可以使用不同的特征缩放和偏移。条件还可以产生残差分支的门控系数：

$$
X\leftarrow X+g\odot F(\operatorname{AdaLN}(X)).
$$

原始 DiT 比较过不同条件注入方式；adaLN-Zero 是其中的重要设计，通过初始化使残差块初始接近恒等映射。讲义给出的“自注意力 + 文本交叉注意力 + AdaLN”应当读成一类通用结构，不是所有 DiT 的统一定义。[DiT 原论文](https://arxiv.org/abs/2212.09748)

### 3.4 DiT token 与语言 token 有什么区别？

这里的 token 通常是连续向量，一个 token 对应一个空间块。它不必对应离散词表编号，也不要求下一个 token 预测。

一次 DiT 前向通常同时更新整幅图像的所有 token。文本生成中常见的自回归顺序，与扩散采样中反复更新整张带噪图像，是不同的计算流程。

## 4. U-Net：多尺度处理，再把细节传回来

U-Net 同样可以接收 $x_t,t,c$，输出同形状的预测，只是组织计算的方式不同。

~~~text
高分辨率特征 ──────────────────→ 高分辨率特征
    ↓ 下采样                       ↑ 上采样
中分辨率特征 ──────────────────→ 中分辨率特征
    ↓                              ↑
低分辨率特征 ───→ bottleneck ───→ 低分辨率特征
~~~

下采样路径逐渐降低空间分辨率、增加通道；上采样路径逐渐恢复分辨率。横向的 skip connections 让前面保留的空间细节进入后面的重建过程。

这些横向连接常采用拼接，也可以有其他实现。它们与残差块内部的相加连接需要区分，不能把所有 skip 都称为同一种 residual connection。

扩散 U-Net 可以加入时间调制、self-attention 和文本 cross-attention，因此不能简单用“U-Net 只有卷积、DiT 才有注意力”来划分。

| 关注点 | U-Net | DiT |
|---|---|---|
| 典型组织方式 | 下采样、上采样、多尺度跳接 | patch token 与 Transformer 堆叠 |
| 典型结构偏置 | 卷积局部性、多尺度组织 | token 间注意力交互 |
| 条件注入 | 时间调制、交叉注意力等 | AdaLN、交叉或联合注意力等 |
| 工作空间 | 可以是像素或 latent | 可以是像素或 latent |
| 比较性能时 | 需要控制参数、数据、训练预算 | 同样需要控制这些条件 |

最容易混淆的一点是：**U-Net 的下采样是每次预测内部的特征计算；VAE 压缩是在进入生成过程之前，改变整个生成过程的工作空间。**

即使 U-Net 内部最终只剩很小的特征图，如果输入仍是像素，它每个采样步骤仍需要处理高分辨率部分。latent diffusion 则把反复执行的生成网络整体放在压缩空间中。

## 5. Latent space 到底省在哪里？

考虑一个纯粹用于算维度的例子：

$$
x\in\mathbb R^{3\times1024\times1024},\qquad
z=E(x)\in\mathbb R^{4\times128\times128}.
$$

像素数组有 3,145,728 个数，latent 数组有 65,536 个数，相差 48 倍。空间位置数相差 64 倍，但通道数不同，所以标量数并不是 64 倍。这是示意配置，不是所有模型的 VAE 规格。

如果在 latent 上再用 $2\times2$ patch：

$$
N=(128/2)^2=4096.
$$

在原图上使用同样大小的 patch 则有 262,144 个 token。

标准全局注意力的计算包含 $O(N^2d)$ 项，线性投影与 MLP 大致包含 $O(Nd^2)$ 项。减少序列长度可以明显降低成本。但上述维度比例不能直接当作端到端加速比，还要考虑编码解码、网络宽度、内存访问和具体实现。

尤其在采样时，生成网络要执行很多步，VAE 解码通常只在最后执行一次。压缩让最频繁的那部分计算变小。

Patchify 与 VAE 压缩也要区分：单纯把数组切块重排不丢数据；VAE 是学习得到的有损表示变换。更大的 patch 会减少 token 数，但随后投影的维度与信息保留方式也会变化。

## 6. 普通 AE 为什么还不够？VAE 增加了什么？

普通自编码器学习：

$$
z=E_\phi(x),\qquad \hat x=D_\psi(z),
$$

$$
\mathcal L_{\rm AE}=\mathbb E_x\|x-D_\psi(E_\phi(x))\|^2.
$$

这个目标鼓励重建原图，却没有明确约束 latent 的尺度、分布和局部连续性。重建很好，不自动意味着后续生成模型容易学习这个表示。

VAE 引入概率编码器和概率解码器：

$$
q_\phi(z\mid x)=\mathcal N\bigl(\mu_\phi(x),
\operatorname{diag}(\sigma_\phi^2(x))\bigr),
$$

$$
p_\psi(x\mid z)=\mathcal N\bigl(\mu_\psi(z),s^2I\bigr).
$$

这里为了说明采用固定的解码方差 $s^2$，编码方差则可以随输入和维度变化。编码器用一片概率区域表示一张图；解码器需要从区域内采到的 latent 恢复图像。

训练目标写为：

$$
\mathcal L_{\rm VAE}
=\mathbb E_x\mathbb E_{q_\phi(z\mid x)}[-\log p_\psi(x\mid z)]
+\beta\mathbb E_x D_{\rm KL}(q_\phi(z\mid x)\|p(z)).
$$

第一项要求保留可重建的信息，第二项将编码分布向先验约束。常选 $p(z)=\mathcal N(0,I)$。

### 6.1 重建似然为什么变成 MSE？

对于固定方差高斯解码器，忽略常数后：

$$
-\log p_\psi(x\mid z)=\frac{1}{2s^2}\|x-\mu_\psi(z)\|^2+\text{const}.
$$

因此它等价于带权 MSE。如果学习解码方差，还需要保留对数方差项，不能只保留除以方差的误差项。

实际图像自编码器还会使用感知损失和对抗损失，以改善视觉重建。引入这些项后，应按完整训练目标理解模型，不能把它全部解释成最简单的像素高斯似然。

### 6.2 KL 的每一项在惩罚什么？

对角高斯编码器相对标准高斯的 KL 为：

$$
D_{\rm KL}(q_\phi(z\mid x)\|\mathcal N(0,I))
=\frac12\sum_j\left(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\right).
$$

均值偏离 0 会被 $\mu_j^2$ 惩罚；令 $a=\sigma_j^2$，则 $a-1-\log a$ 在 $a=1$ 处达到最小值。

更一般的两个对角高斯应逐维计算：

$$
D_{\rm KL}(q\|p)=\frac12\sum_j
\left[
\log\frac{\sigma_{p,j}^2}{\sigma_{q,j}^2}
+\frac{\sigma_{q,j}^2+(\mu_{q,j}-\mu_{p,j})^2}{\sigma_{p,j}^2}-1
\right].
$$

这个展开形式也避免了讲义公式中向量方差与整体范数相除容易引起的歧义。

### 6.3 这和 ELBO 有什么关系？

标准 VAE 在 $\beta=1$ 时优化负 ELBO：

$$
\log p_\psi(x)\ge
\mathbb E_{q_\phi(z\mid x)}\log p_\psi(x\mid z)
-D_{\rm KL}(q_\phi(z\mid x)\|p(z)).
$$

改变 $\beta$ 就改变了重建与正则的权衡，不能再无条件把加权目标称为原始负 ELBO。[VAE 原论文](https://arxiv.org/abs/1312.6114)

$\beta$ 过大可能让编码器几乎不再保留输入信息，即 posterior collapse。实践中可用 KL warm-up，但具体系数要与重建损失的归一化、latent 维度及其他损失一起理解。

讲义的“所有现代自编码器都用很小的 $\beta$”范围过大。较弱 KL 是一些生成用图像自编码器的选择，其他表示方法也可以采用不同正则，甚至不使用这个高斯 KL 目标。

## 7. 为什么 latent 已经被约束成高斯，还要再训练生成模型？

这是整节最关键的问题。

单张图像的编码分布是 $q_\phi(z\mid x)$，所有图像混合后的 aggregated posterior 是：

$$
q_\phi(z)=\int q_\phi(z\mid x)p_{\rm data}(x)\,dx.
$$

KL 正则只是一个优化项，并不保证训练结束后它们就精确等于标准高斯。特别是在重建优先、KL 较弱的设置里，latent 仍然包含复杂的数据结构。

有一个有助于理解权衡的恒等式：

$$
\mathbb E_xD_{\rm KL}(q_\phi(z\mid x)\|p(z))
=I_q(X;Z)+D_{\rm KL}(q_\phi(z)\|p(z)).
$$

它可通过在对数比值中插入 $q_\phi(z)$ 得到：

$$
\log\frac{q_\phi(z\mid x)}{p(z)}
=\log\frac{q_\phi(z\mid x)}{q_\phi(z)}
+\log\frac{q_\phi(z)}{p(z)}.
$$

对联合分布 $p_{\rm data}(x)q_\phi(z\mid x)$ 求期望，右侧两项分别成为互信息与聚合分布的 KL。

这说明，强行压低每个样本的 KL，还会限制 latent 携带多少关于输入的信息。若每张图都被编码成完全一样的高斯分布，就失去了区分它们的能力。

latent diffusion / flow 因此仍有明确工作：**学习由数据编码得到的复杂 latent 分布，并根据文本等条件生成合适的 latent。** 自编码器主要负责压缩与恢复，两部分各自解决不同问题。

也因此，latent 只要适合后续建模即可，并不要求先被变成精确高斯。普通 AE 的表示也可以作为生成模型的工作空间，VAE 是一种常用的正则化方案。

## 8. 重参数化：怎样对随机编码训练？

直接写 $z\sim q_\phi(z\mid x)$ 会把参数依赖藏在采样分布里。重参数化将它写成：

$$
\epsilon\sim\mathcal N(0,I),\qquad
z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon.
$$

随机数 $\epsilon$ 的分布独立于参数，$z$ 对均值、标准差的依赖则是可微的。重建损失的梯度可以通过解码器和这条确定性表达式返回编码器。

它没有消除随机性，只是重新安排了随机性出现的位置。重参数化也不等于固定一个噪声用于整个训练。

~~~python
# VAE 训练的概念流程；省略感知 / 对抗损失等工程项
mu, logvar = encoder(x)
std = exp(0.5 * logvar)
z = mu + std * randn_like(std)
x_hat = decoder(z)
recon = reconstruction_loss(x_hat, x)
kl = 0.5 * sum_over_latent(mu**2 + exp(logvar) - 1 - logvar)
loss = mean_over_batch(recon + beta * kl)
loss.backward()
~~~

代码里的求和或平均方式必须统一。把 KL 从 latent 求和改成求平均，会改变它相对重建项的权重。

## 9. 把 latent Flow Matching 完整串起来

### 第一阶段：学习压缩表示

训练自编码器，检查 $D(E(x))$ 能否保留任务需要的信息。除总体重建质量外，还要看小文字、细物体、边界以及视频的快速运动。

### 第二阶段：冻结自编码器，训练生成网络

为了统一符号，本文采用 $t=0$ 为噪声、$t=1$ 为数据的约定：

$$
z_1\sim q_\phi(z\mid x),\quad z_0\sim\mathcal N(0,I),
\quad z_t=(1-t)z_0+tz_1.
$$

直线插值的条件速度是：

$$
\frac{dz_t}{dt}=z_1-z_0.
$$

于是训练目标可以写成：

$$
\mathcal L_{\rm FM}
=\mathbb E_{x,c,z_0,z_1,t}
\left\|u_\theta(z_t,t,c)-(z_1-z_0)\right\|^2.
$$

这一步的网络可以是 U-Net，也可以是 DiT。latent 的缩放、时间分布和 loss weighting 要按具体配方设置。

训练时随机抽一个 $t$，就能构造监督，不必为了这一次训练更新先完整运行生成 ODE。

### 第三阶段：采样与解码

先采标准高斯，再数值求解：

$$
\frac{dz_t}{dt}=u_\theta(z_t,t,c),\qquad z_0\sim\mathcal N(0,I).
$$

例如 Euler 更新：

$$
z_{t+\Delta t}=z_t+\Delta t\,u_\theta(z_t,t,c).
$$

最终解码 $\hat x=D(z_1)$。通常使用解码器均值输出，避免额外叠加像素采样噪声。

~~~mermaid
flowchart TD
    A[训练图像 x] --> B[冻结的编码器]
    B --> C[数据 latent z1]
    D[高斯噪声 z0] --> E[插值得到 zt]
    C --> E
    E --> F[DiT 或 U-Net]
    G[时间 t 和文本条件 c] --> F
    F --> H[预测速度并计算训练损失]
    I[推理时的新高斯噪声] --> J[多步速度预测与数值更新]
    G --> J
    J --> K[生成 latent]
    K --> L[解码器]
    L --> M[生成图像]
~~~

还有一个容易忽略的点：每对噪声和数据的监督插值是直线，但条件均值形成的学习向量场，其采样轨迹不一定严格笔直。因此直线训练路径不自动意味着一步就能生成。

## 10. CFG：训练时丢条件，推理时组合预测

Classifier-Free Guidance 训练时以一定概率把文本等条件替换为空条件，让同一个模型同时学习有条件与无条件预测。

以一种常见的系数约定：

$$
u_{\rm cfg}(z,t,c)
=u_\theta(z,t,\varnothing)
+s\bigl[u_\theta(z,t,c)-u_\theta(z,t,\varnothing)\bigr].
$$

$s=1$ 对应普通条件预测；$s>1$ 外推条件方向。不同实现可能采用偏移过的系数定义，比较 guidance scale 时要先看公式。

它没有额外分类器，所以讲义 SD3 段落末尾的“分类器引导”应按上下文改为“无分类器引导”。对于文本到图像任务，丢弃的是文本条件，不宜仅称为“去掉类别标签”。

50 个采样步骤也不必等于只计算 50 份网络预测：CFG 通常需要条件与空条件两路，可合并成一个 batch，但计算量仍需计入两路。

## 11. 案例一：SD3 如何组合这些组件？

SD3 可以按四层来理解：自编码器压缩图像、文本编码器提供条件、MM-DiT 预测 latent 速度、采样器迭代生成后解码。

MM-DiT 的关键是图像和文本采用各自的参数处理，再通过联合注意力交换信息。它与“图像查询固定文本特征”的普通 cross-attention 不完全相同，文本表示也会参与多层更新。SD3 使用 CLIP 系列和 T5 等文本表示，论文研究到约 8B 参数规模。[SD3 原论文](https://arxiv.org/abs/2403.03206)

所以要分别看：表示压缩改善了成本，条件表示影响提示理解，网络结构影响模态交互，Flow Matching 决定监督与采样路径。把性能变化全部归到“换成 Transformer”会遗漏其他变化。

历史上，LDM 在 2021 年提交、2022 年发表，已经系统研究在预训练自编码器空间中做扩散。SD3 应放在这条路线的后续发展中，讲义称它为“最早的大规模 latent 扩散模型之一”容易造成时间线混淆。[LDM 原论文](https://arxiv.org/abs/2112.10752)

## 12. 案例二：视频为什么更依赖表示设计？

视频比图像多一个帧维：

$$
x\in\mathbb R^{T\times C\times H\times W}.
$$

如果只把每帧压缩，然后让 DiT 处理全部时空 token，序列仍然很长。时间压缩可以进一步降低长度，但也要保留运动变化。

Movie Gen Video 使用 temporal autoencoder，在时间和空间上压缩视频，主生成骨干采用 DiT 风格结构和 Flow Matching。技术报告的 TAE 在时间、高、宽各压缩约 8 倍，latent 通道数为 16；边界帧与分块处理应按实现计算。文本条件结合 UL2、Long-prompt MetaCLIP 和面向画面文字的 ByT5，主视频模型规模为 30B。[Movie Gen 技术报告](https://arxiv.org/html/2410.13720v2)

对一般视频模型，忽略边界取整，若自编码器压缩为 $r_t,r_h,r_w$，再采用 patch 大小 $p_t,p_h,p_w$，则：

$$
N\approx
\frac{T}{r_tp_t}
\frac{H}{r_hp_h}
\frac{W}{r_wp_w}.
$$

这比只报参数量更能解释视频训练成本。同一个网络宽度与层数，视频长度和分辨率变化也能显著改变 token 数与注意力开销。

还要区分两个时间轴：$T$ 是输出视频帧数，$t$ 是每个视频样本在生成过程中的噪声时间。增加采样步骤与增加视频长度，会以不同方式增加计算。

分块编码可以降低自编码器的峰值显存，但不能据此认定生成骨干的跨帧注意力成本也一并消失。分块边界如何保持连续，也需要单独处理。

## 13. 对 World Model / WAM 的实际启发

以下是从材料延伸出的研究判断。

**第一，先测表示是否保留任务变量。** 视觉重建清楚，并不保证保留了插孔间隙、微小滑移或接触瞬间。这些信息在普通视频里所占面积很小，在操作任务里却可能决定成败。

如果一种压缩把两个控制后果不同的状态映射成难以区分的表示，后续模型仅凭这个表示就很难恢复差别。可以补充动作、触觉或其他状态条件，但应明确这些信息从哪里来。

**第二，换 tokenizer 后，latent loss 不能直接横向比较。** 通道数、尺度和信息分布改变，会影响 MSE 数值。跨 tokenizer 应比较统一输出空间中的重建与生成质量，并检查下游控制表现。

**第三，生成视频和学习动作后果之间还隔着条件与评测。** Action-conditioned World Model 需要学习给定动作后环境如何变化。仅靠文本条件生成合理视频，并不能证明它能预测真实执行后果。

**第四，时间压缩要与控制频率一起考虑。** 在高时间压缩率下，一次短暂接触可能主要由 decoder 恢复。它是否能准确保留接触发生的时刻，应通过事件级指标验证，而非只看视频整体是否平滑。

可做一个直接的对照：固定生成骨干与总预算，改变压缩率和 latent 通道数，同时测量重建质量、生成质量、接触事件保留率和下游成功率。这样才能知道计算节省是否牺牲了真正重要的信息。

## 14. 复习时应能回答的几个问题

| 问题 | 答案 |
|---|---|
| DiT 是一种新的训练目标吗？ | 它是网络架构，可以配合扩散或 Flow Matching。 |
| Latent diffusion 必须使用 DiT 吗？ | 可以使用 U-Net 等不同骨干。 |
| DiT 是否像 LLM 一样逐 token 生成？ | 典型 DiT 同时预测所有位置，采样器多次更新整体状态。 |
| VAE 压缩与 U-Net 下采样相同吗？ | 前者改变生成工作空间，后者是每次前向内部的特征处理。 |
| KL 能保证所有 latent 精确高斯吗？ | 它提供正则约束，最终还取决于重建与正则的权衡。 |
| 为什么 VAE 后还需要 diffusion / flow？ | 后者学习复杂的 latent 数据分布及其条件分布。 |
| 时间嵌入是视频帧位置吗？ | 这里指生成路径时间；视频帧位置是另一套坐标。 |
| 一步网络输出就是最终图像吗？ | 速度预测模型通常输出当前更新方向，再经求解与解码生成图像。 |
| 图像重建好就适合机器人吗？ | 还要测任务相关的几何、接触、时序信息。 |

读完这一节，可以把完整生成系统拆成四部分检查：编码器保留什么信息，网络怎样融合条件，训练目标要求预测什么，采样器怎样把预测变成最终输出。理解这四部分，再看 SD3、视频生成器或 WAM，结构会清楚得多。
