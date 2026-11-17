# Flow Matching and Diffusion Models（三）：从速度场到 Score，再到 CFG

来源：[第三讲 PDF](</Users/lijieming/Downloads/【开学】Flow Matching and Diffusion Models（三）.pdf>)，共 17 页，正文主要在第 2–16 页，对应第 4 节 Score Functions and Score Matching 和第 5 节 Guidance。承接 [第二讲笔记](<./Flow Matching and Diffusion Models（二）.md>)。

这讲可以沿着一个问题读：**上一讲已经会用速度场把噪声搬成数据了，为什么还需要 score、随机微分方程和 guidance？**

Score 提供另一种描述分布的方式；SDE 允许生成过程中持续加入随机性；guidance 则让生成结果服从文字、类别或其他条件。对于这里的高斯概率路径，这三件事可以用同一套公式连起来。

PDF 有多处未加载的插图，也有横向被裁切的公式。下面按可读正文和公式重新推导，不把缺失的插图当作已观察到的实验结果。对应英文教材可参阅 Peter Holderrieth、Ezra Erives 的 [An Introduction to Flow Matching and Diffusion Models，第 4–5 节](https://arxiv.org/html/2506.02070v3)。本文主要依据所提供的 PDF，具体例子、代数展开和实现注意事项是补充笔记。

## 1. 先统一时间方向和符号

本文始终使用生成方向：

\[
t=0:\text{噪声},\qquad t=1:\text{数据}.
\]

采用高斯条件路径：

\[
x_t=\alpha_t z+\beta_t\epsilon,
\qquad z\sim p_{\mathrm{data}},\quad \epsilon\sim\mathcal N(0,I),
\]

其中数据与噪声独立，端点满足：

\[
\alpha_0=0,\quad \beta_0=1,\qquad
\alpha_1=1,\quad \beta_1=0.
\]

| 符号 | 意义 |
| --- | --- |
| \(z\) | 训练时拿到的干净数据，可以是图像、视频 latent 或动作序列 |
| \(x_t\) | 生成时间 \(t\) 的带噪状态 |
| \(p_t(x\mid z)\) | 围绕某个数据样本构造的条件分布 |
| \(p_t(x)\) | 对所有数据样本混合后得到的边缘分布 |
| \(u_t(x)\) | 输运这条概率路径的 ODE 速度场 |
| \(s_t(x)\) | \(\nabla_x\log p_t(x)\)，对位置求导的 score |
| \(\beta_t\) | 构造训练状态时的噪声标准差 |
| \(\sigma_t\) | SDE 采样过程中持续注入噪声的系数 |
| \(y\) | 文本、类别等生成条件 |

特别要分清 \(\beta_t\) 与 \(\sigma_t\)：前者定义要学习的概率路径，后者定义如何随机地沿这条路径采样。它们承担不同作用。

不少扩散论文用“数据到噪声”作为正向时间，再反向生成。比较公式前要先对齐时间，不能只看 score 前面是正号还是负号。

## 2. Score 到底是什么？

定义为：

\[
s_t(x)=\nabla_x\log p_t(x).
\]

它告诉我们：**从当前位置出发，朝哪个方向移动，局部对数密度会上升得最快。** 它是一个与样本同维度的向量，不是图像质量分数，也不是对模型参数求导的训练梯度。

例如一维高斯 \(p(x)=\mathcal N(x;\mu,\tau^2)\) 满足：

\[
\log p(x)=\text{常数}-\frac{(x-\mu)^2}{2\tau^2},
\qquad s(x)=-\frac{x-\mu}{\tau^2}.
\]

位置在均值右侧时，score 指向左侧；在均值左侧时，score 指向右侧。分布越窄，同样偏移对应的恢复方向越强。

但生成不能简单理解成一直爬到密度最高的位置。若只做梯度上升，很多样本会聚集到局部峰值，难以保持目标分布的多样性。后面的时间变化与随机项，正是理解“采样”所必需的部分。

一个方便的性质是：若 \(p(x)=\widetilde p(x)/Z\)，且 \(Z\) 与 \(x\) 无关，则

\[
\nabla_x\log p(x)=\nabla_x\log\widetilde p(x).
\]

因此 score 不需要知道归一化常数。

## 3. 只能算条件 score，为什么能学到边缘 score？

我们知道：

\[
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
\]

在可以交换积分与求导的条件下：

\[
\begin{aligned}
\nabla_x\log p_t(x)
&=\frac{\int\nabla_xp_t(x\mid z)p_{\mathrm{data}}(z)\,dz}{p_t(x)}\\
&=\int\nabla_x\log p_t(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz\\
&=\mathbb E[s_t(x\mid z)\mid x_t=x].
\end{aligned}
\]

中间的权重就是后验 \(p_t(z\mid x)\)。所以这里是“看到当前带噪状态以后，对可能的干净样本加权”，不能直接用数据先验平均所有条件 score。

直观上，一个模糊轮廓可能来自多种图像。每种图像都提供一个去噪方向，边缘 score 把这些可能性按后验概率合在一起。这个结构与上一讲的“边缘速度是条件速度的后验平均”完全对应。

对于高斯路径：

\[
p_t(x\mid z)=\mathcal N(x;\alpha_tz,\beta_t^2I),
\]

条件 score 可以直接计算：

\[
s_t(x\mid z)=-\frac{x-\alpha_tz}{\beta_t^2}.
\]

将 \(x_t=\alpha_tz+\beta_t\epsilon\) 代入：

\[
s_t(x_t\mid z)=-\frac{\epsilon}{\beta_t}.
\]

**训练时自己加进去的噪声，就提供了 score 的监督信号。** 训练标签知道 \(z,\epsilon\)，网络只看 \(x_t,t\)，因此最优预测是相应的条件期望，并不保证能逐样本找回那一次真实噪声。

## 4. 速度、Score、噪声预测和干净样本预测如何转换？

先对插值路径求导：

\[
\dot x_t=\dot\alpha_tz+\dot\beta_t\epsilon.
\]

消去 \(\epsilon=(x-\alpha_tz)/\beta_t\)：

\[
u_t(x\mid z)=
\left(\dot\alpha_t-\frac{\dot\beta_t\alpha_t}{\beta_t}\right)z
+\frac{\dot\beta_t}{\beta_t}x.
\]

再利用 \(\alpha_tz=x+\beta_t^2s_t(x\mid z)\)，得到：

\[
u_t(x\mid z)=b_tx+a_ts_t(x\mid z),
\]

其中

\[
b_t=\frac{\dot\alpha_t}{\alpha_t},\qquad
a_t=\frac{\beta_t^2\dot\alpha_t}{\alpha_t}-\beta_t\dot\beta_t.
\]

对后验取期望后，边缘量也满足：

\[
\boxed{u_t(x)=b_tx+a_ts_t(x)}.
\]

因此在分母非零的时刻，可以把速度网络转换为 score 网络：

\[
s_\theta(x,t)=\frac{u_\theta(x,t)-b_tx}{a_t}.
\]

再定义干净样本预测器和噪声预测器的理想值：

\[
D_t(x)=\mathbb E[z\mid x_t=x],\qquad
\epsilon_t^*(x)=\mathbb E[\epsilon\mid x_t=x].
\]

它们满足：

\[
s_t(x)=-\frac{\epsilon_t^*(x)}{\beta_t},\qquad
D_t(x)=\frac{x+\beta_t^2s_t(x)}{\alpha_t},
\]

\[
u_t(x)=\dot\alpha_tD_t(x)+\dot\beta_t\epsilon_t^*(x).
\]

也可以直接由速度恢复去噪器：

\[
D_t(x)=\frac{\beta_tu_t(x)-\dot\beta_tx}
{\beta_t\dot\alpha_t-\alpha_t\dot\beta_t}.
\]

这里的等价指理想函数之间的代数关系。不同输出参数化配上不同损失权重，会有不同的优化难度。转换公式还可能在端点数值不稳定，不能据此认为所有训练实现等价。

### 用线性路径算一遍

取 \(\alpha_t=t,\beta_t=1-t\)，则训练速度标签为 \(z-\epsilon\)，而

\[
a_t=\frac{1-t}{t},\qquad b_t=\frac1t.
\]

于是

\[
u_t(x)=\frac{x}{t}+\frac{1-t}{t}s_t(x),
\qquad
s_t(x)=\frac{tu_t(x)-x}{1-t}.
\]

还有两个很实用的关系：

\[
D_t(x)=x+(1-t)u_t(x),\qquad
\epsilon_t^*(x)=x-tu_t(x).
\]

后两式可以直接由 \(x_t=tz+(1-t)\epsilon\)、\(\dot x_t=z-\epsilon\) 联立得到。它们也提示：有些重参数化比先计算含奇异分母的系数更适合实现。

## 5. ODE 怎么变成 SDE，还保持相同分布？

上一讲的 ODE 是：

\[
dX_t=u_t(X_t)\,dt.
\]

本讲的 SDE 扩展为：

\[
\boxed{dX_t=
\left[u_t(X_t)+\frac{\sigma_t^2}{2}s_t(X_t)\right]dt
+\sigma_t\,dW_t.}
\]

在精确向量场、正确初始分布和适当的正则性、可解性条件下，它与原 ODE 具有相同的单时刻边缘分布 \(p_t\)。这里 \(\sigma_t\) 仅依赖时间。

新增两项要一起理解：布朗噪声让分布扩散，score 漂移恰好补偿这部分扩散。只给 ODE 加噪声而省略漂移修正，通常就会改变概率路径。

证明用 Fokker–Planck 方程。对于漂移为 \(v_t\) 的 SDE：

\[
\partial_tp_t=-\nabla\cdot(p_tv_t)+\frac{\sigma_t^2}{2}\Delta p_t.
\]

取 \(v_t=u_t+\sigma_t^2s_t/2\)，并使用

\[
p_ts_t=\nabla p_t,\qquad
\nabla\cdot(p_ts_t)=\Delta p_t,
\]

便得到：

\[
\begin{aligned}
\partial_tp_t
&=-\nabla\cdot(p_tu_t)
-\frac{\sigma_t^2}{2}\Delta p_t
+\frac{\sigma_t^2}{2}\Delta p_t\\
&=-\nabla\cdot(p_tu_t).
\end{aligned}
\]

最后恰好回到 ODE 的连续性方程。

“相同边缘分布”意味着同一时刻看大量样本，两种过程的分布相同。它不意味着单条轨迹相同，也不意味着跨时刻联合分布或转移概率相同。给定初始点，满足唯一性的 ODE 轨迹已经确定；SDE 还需要知道后续随机增量。

若让扩散系数依赖位置 \(\sigma(x,t)\)，Fokker–Planck 的扩散项会改变，需要重新推导补偿项。

## 6. Langevin 动力学解释了什么？

取固定分布 \(p_t=p\)，让 \(u_t=0\)，得到：

\[
dX_t=\frac{\sigma_t^2}{2}\nabla\log p(X_t)\,dt+\sigma_t\,dW_t.
\]

这就是本讲介绍的 Langevin 动力学。若 \(X_0\sim p\)，它会保持这个分布；从别的分布出发能否收敛，以及收敛多快，还需要遍历性等条件，不能只靠平稳性推断。

举一个补充例子：令 \(p=\mathcal N(0,1)\)、\(\sigma=\sqrt2\)，则

\[
dX_t=-X_tdt+\sqrt2\,dW_t.
\]

漂移将样本拉向原点，噪声让样本持续散开，两者平衡时方差为 1。只有前一项会逐渐收缩到原点，只有后一项则会持续扩散。

PDF 的 GLASS Flows 备注没有给出完整算法。这里保留一个必要区别：固定初值和固定确定性向量场的唯一 ODE 解，本身不会额外产生随机性。若用 ODE 实现随机转移，需要在采样流程中另外引入随机变量或随机操作，不能从这段简短备注推出“普通 ODE 自动等价于 SDE”。

## 7. 去噪得分匹配为什么是正确的训练目标？

理想目标是：

\[
\mathcal L_{\mathrm{SM}}
=\mathbb E_{t,x_t}\|s_\theta(x_t,t)-s_t(x_t)\|^2.
\]

边缘 score 不可直接获得，但条件 score 可算，因此训练：

\[
\mathcal L_{\mathrm{DSM}}
=\mathbb E_{t,z,x_t}\|s_\theta(x_t,t)-s_t(x_t\mid z)\|^2.
\]

令 \(Y=s_t(x_t\mid z)\)，则 \(\mathbb E[Y\mid x_t,t]=s_t(x_t)\)。平方误差分解给出：

\[
\mathcal L_{\mathrm{DSM}}
=\mathcal L_{\mathrm{SM}}
+\mathbb E\|Y-\mathbb E[Y\mid x_t,t]\|^2.
\]

最后一项与参数无关，所以两者参数梯度相同。它们的 loss 数值不一定相同；网络容量有限时，能得到的是网络函数族内的最佳近似。

对高斯路径代入监督目标：

\[
\mathcal L_{\mathrm{DSM}}
=\mathbb E\left\|s_\theta(x_t,t)+\frac\epsilon{\beta_t}\right\|^2.
\]

若定义 \(\epsilon_\theta=-\beta_ts_\theta\)，就有：

\[
\mathcal L_{\mathrm{DSM}}
=\mathbb E\left[\frac1{\beta_t^2}
\|\epsilon_\theta(x_t,t)-\epsilon\|^2\right].
\]

常用的简化噪声回归则写成：

\[
\mathcal L_\epsilon
=\mathbb E\|\epsilon_\theta(x_t,t)-\epsilon\|^2.
\]

**这里去掉的是随时间变化的权重，不能把它理解成整个目标乘以常数。** 当每个时间点都能独立拟合、权重为正且风险可积时，两者有相同的逐点最优预测；共享参数的有限网络下，不同时间的训练资源分配已经改变。

当 \(\beta_t\to0\) 时，条件 score 标签可能非常大。直接覆盖端点的未加权风险甚至可能不满足有限二阶矩条件。因此实际实现通常结合截断时间区间、损失加权或噪声／数据参数化处理。PDF 中“加入足够噪声才有效”应理解为对这种数值问题的提醒，而不是去噪得分匹配在低噪声区间数学上失效。

## 8. 训练与采样分别做什么？

训练一次噪声预测的流程非常短：

```text
取干净数据 z
采样时间 t 和标准高斯噪声 epsilon
构造 x_t = alpha(t) * z + beta(t) * epsilon
预测 epsilon_hat = network(x_t, t)
计算 mean((epsilon_hat - epsilon)^2)，反向传播
```

这一步不需要模拟完整生成轨迹。采样时才从新噪声出发，多次查询网络并更新状态。

若使用 SDE，Euler–Maruyama 的一步为：

\[
x_{t+h}=x_t+
\left[u_\theta(x_t,t)+\frac{\sigma_t^2}{2}s_\theta(x_t,t)\right]h
+\sigma_t\sqrt h\,\xi,
\quad\xi\sim\mathcal N(0,I).
\]

随机项是 \(\sqrt h\)，因为布朗增量的方差为 \(h\)。写成 \(h\xi\) 会改变所模拟的随机过程。

取 \(\sigma_t=0\) 就回到 ODE。若通过速度换算 score，可以复用同一个网络；端点处须用稳定公式、适当的噪声日程或单独的末步处理。

理论上的自由选择 \(\sigma_t\) 不代表所有选择在有限步数下效果一样。真实模型有拟合误差，数值求解有离散误差，需要在固定计算预算下比较采样质量。PDF 没有提供这方面的统一最优参数或性能表。

## 9. “条件路径”和“有条件生成”是两回事

前面的条件是已知某个干净样本 \(z\)，为了构造训练标签；现在的条件 \(y\) 是生成时用户提供的文本、类别或任务描述。

例如：\(z\) 是一张训练图片，\(y\) 是它的文字描述。测试时只知道 \(y\)，需要生成新的 \(z\)。

最直接的条件生成方式是在网络输入中加入 \(y\)：

\[
u_\theta(x_t,t,y).
\]

对成对数据 \((z,y)\sim p_{\mathrm{data}}(z,y)\)，训练：

\[
\mathcal L_{\mathrm{CFM}}=
\mathbb E\|u_\theta(\alpha_tz+\beta_t\epsilon,t,y)
-(\dot\alpha_tz+\dot\beta_t\epsilon)\|^2.
\]

噪声构造和速度标签可以保持不变，改变的是网络现在可以利用 \(y\) 缩小对干净样本的推断范围。在精确学习和求解的理想条件下，这就能采样 \(p_{\mathrm{data}}(z\mid y)\)。

CFG 是在此基础上进一步调整生成方向。仅仅给网络输入文本，并不等于已经使用 CFG。

## 10. 从 Bayes 公式推到 CFG

对固定的时间 \(t\)：

\[
p_t(x\mid y)=\frac{p_t(x)p_t(y\mid x)}{p_t(y)}.
\]

对 \(x\) 求对数梯度：

\[
s_t(x\mid y)=s_t(x)+\nabla_x\log p_t(y\mid x).
\]

其中 \(p_t(y\mid x)\) 是“看到当前噪声水平下的样本，它符合条件 \(y\) 的程度”。若训练一个噪声感知分类器，就可以使用其梯度加强条件方向。

引入引导强度 \(w\)：

\[
s_t^{\mathrm{guided}}(x,y)
=s_t(x)+w\nabla_x\log p_t(y\mid x).
\]

将 Bayes 恒等式代回去：

\[
\boxed{s_t^{\mathrm{CFG}}(x,y)
=s_t(x)+w[s_t(x\mid y)-s_t(x)].}
\]

在同一高斯日程下，利用 \(u=bx+as\)，也有：

\[
\boxed{u_t^{\mathrm{CFG}}(x,y)
=u_t(x)+w[u_t(x\mid y)-u_t(x)].}
\]

两项系数之和为 1，因此公共的 \(b_tx\) 正好保留。

| 本文 \(w\) | 含义 |
| --- | --- |
| 0 | 无条件生成 |
| 1 | 普通条件生成 |
| 大于 1 | 向条件方向外推 |

例如无条件速度是 \((1,0)\)，条件速度是 \((1,1)\)。\(w=3\) 得到 \((1,3)\)：放大的是条件引起的差异，而不是把整个速度乘以 3。

另一种代码约定写成 \(u_{\mathrm{cond}}+\gamma(u_{\mathrm{cond}}-u_{\mathrm{uncond}})\)，对应本文 \(w=1+\gamma\)。比较 guidance scale 的数值时，要先核对公式。

## 11. 一个网络怎么学两种预测？

训练时，以固定概率 \(\eta\) 将条件 \(y\) 替换为空条件 \(\varnothing\)：

\[
\widetilde y=
\begin{cases}
\varnothing,&\text{概率 }\eta,\\
y,&\text{概率 }1-\eta.
\end{cases}
\]

然后使用同一个回归目标：

\[
\mathbb E\|u_\theta(x_t,t,\widetilde y)
-(\dot\alpha_tz+\dot\beta_t\epsilon)\|^2.
\]

丢弃事件应独立于数据内容，这样空条件分支看到的仍是原来的边缘数据分布。丢掉的是条件信息，干净目标和速度标签照常保留。

推理时：

```text
u_uncond = network(x_t, t, null_condition)
u_cond   = network(x_t, t, y)
u_cfg    = u_uncond + w * (u_cond - u_uncond)
x_next   = x_t + step_size * u_cfg
```

这是一套参数的两种条件计算，通常需要两份前向计算量，也可以拼成一个 batch。空条件表示必须与训练一致，不能随便用未经训练的全零向量替换。

同样的线性组合可用于噪声预测：

\[
\epsilon_{\mathrm{CFG}}
=\epsilon_{\mathrm{uncond}}+w(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
\]

CFG 省去了单独的分类器，但保留条件与无条件生成分支。原始工作明确将其用于调节样本质量与多样性的取舍。[Ho 与 Salimans，Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

### 为什么不能把 CFG 看成精确后验采样？

在固定 \(t\)、精确 score 且密度可归一化时，可以形式上写出：

\[
s_t^{\mathrm{CFG}}
=\nabla_x\log\big[p_t(x\mid y)^w p_t(x)^{1-w}\big].
\]

但逐时刻写出一个密度的 score，不足以证明这些密度组成的整条路径符合当前采样方程。因而不能据此断言最终样本严格服从该幂次重加权分布。

PDF 也明确强调：\(w>1\) 后，最终分布不保证仍等于原始 \(p_{\mathrm{data}}(\cdot\mid y)\)。CFG 是实用的生成控制方式，增强条件同时可能损失模式覆盖。更大的 \(w\) 还会放大条件与无条件预测误差的差异，需要实际评测。

## 12. 和视频模型、机器人策略放在一起理解

下面是基于本讲公式的研究联系，不是 PDF 的机器人实验结论。

对于视频 latent，\(z\) 可以是整段视频编码，\(y\) 可以包含文本、首帧等信息。学习目标仍是带噪视频上的速度或噪声，CFG 改变条件信息对生成方向的影响。

对于动作生成，\(z\) 可以是一段动作，\(y\) 可以是观测、语言和机器人状态。这里要分清两条时间轴：动作序列有真实执行时间，模型还有从噪声生成动作的去噪时间 \(t\)。SDE 的随机性发生在生成计算中，不等于给真实机器人执行的电机命令持续加噪声。

如果只丢弃语言、保留观测，那么所谓“无条件分支”实际学习的是 \(p(z\mid\mathrm{observation})\)。CFG 加强的是语言相对于已有视觉条件的贡献。若把全部观测也丢掉，则比较的是另一对分布。这必须在训练设计和论文表述中说清楚。

更强的条件遵循也不能替代碰撞、关节限位或接触力约束。动作生成的评估需要记录任务成功率、可执行性、平滑性和推理成本，不能只报告去噪 loss。

我会把这讲最后记成下面这张表，读后续方法时先定位它改了哪一项：

| 层次 | 本讲中的选择 | 直接影响 |
| --- | --- | --- |
| 概率路径 | \(\alpha_t,\beta_t\) | 中间训练状态如何构造 |
| 网络输出 | 速度、score、噪声、干净样本 | 回归目标与数值尺度 |
| 训练加权 | 时间采样、损失权重 | 不同噪声区间受到多少重视 |
| 采样过程 | ODE、SDE、步长、\(\sigma_t\) | 轨迹随机性、误差和成本 |
| 条件控制 | 普通条件输入、CFG、\(w\) | 条件遵循与分布覆盖的取舍 |

真正贯穿这一讲的是同一个结构：训练时利用可计算的条件标签，学到边缘量；采样时再利用这些边缘量组织整体概率分布的运动。高斯路径让速度、score 和去噪之间的关系尤其清楚，但参数化、训练权重、数值求解和条件外推仍然需要分别考虑。
