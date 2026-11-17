# QF3：用 Filtered Q-Gradients 稳定训练 Flow Policy

[项目主页](https://qf3-rl.github.io/)

QF3 针对的是一个很实际的问题：已经有预训练的 Flow Policy，想用离线或 off-policy 强化学习把它调得更好，应该如何利用 Critic 的 Q 值而不把策略推到数据分布之外？论文的答案是过滤 Q 梯度，只允许它在 Replay Buffer 动作附近有限地改变 Flow Matching 的速度场。

## 1. 为什么直接最大化 Q 值容易失败

Flow Policy 适合表达多模态动作，但 Critic 并不会在整个连续动作空间都准确。Replay Buffer 附近有真实数据支撑，远离数据的动作往往只是 Critic 的外推，可能得到虚假的高 Q 值。

如果直接最小化 \(-Q(s,a)\)，策略会迅速追逐这些高估区域。对于预训练策略来说，这种更新尤其危险：几次梯度更新就可能破坏原本具有多模态和安全性的动作分布。

## 2. 在 Flow Matching 速度上做过滤

QF3 先从 Replay Buffer 取一个动作 (a_{\rm buf})，采样噪声 (\epsilon)，构造标准 Flow Matching 路径：

\[
x_\tau=(1-\tau)\epsilon+\tau a_{\rm buf},
\qquad
u_{\rm buf}=a_{\rm buf}-\epsilon。
\]

策略输出条件速度 (v_\theta(x_\tau,s,\tau))。单步动作估计为：

\[
\hat a=x_\tau+(1-\tau)v_\theta。
\]

QF3 不直接裁剪最终动作，而是先计算策略速度相对于数据速度的残差：

\[
\delta_\theta=v_\theta-u_{\rm buf}。
\]

再用逐维 clip 限制残差：

\[
\hat a_{\rm clip}=x_\tau+(1-\tau)
\left[u_{\rm buf}+\operatorname{clip}(\delta_\theta,-\alpha,\alpha)\right]。
\]

最终 actor 目标为：

\[
\mathcal L_{\rm actor}=-Q(s,\hat a_{\rm clip})+
\lambda\|v_\theta-u_{\rm buf}\|^2。
\]

这样做的含义是：Q 梯度可以把策略往更高价值方向推，但超出数据支持范围的部分会被截断；同时，正则项让策略保持在原有 Flow Matching 速度附近。

## 3. 与其他 Flow RL 方法的区别

FPO 或 FPO++ 更接近使用轨迹优势来重新加权 Flow Matching 目标，FlowRL 则可以沿完整采样路径反传 Q 信号。QF3 使用受约束的单步动作估计，不要求每次更新都重新采集 on-policy 轨迹，适合已有 Replay Buffer 和预训练策略的 post-training。

它的关键并不是简单增加一个行为克隆项，而是把“离数据多远”的限制放到了 Flow 速度场上。对生成模型而言，这比只看最终动作距离更贴近模型实际的更新方式。

## 4. 实验与结果

作者在 ABC-VLA 双臂操作、robomimic，以及人形机器人 sim-to-real 任务中测试 QF3。在 ABC-VLA 中，VLM 主干保持冻结，只对 Flow Action Head 做 LoRA 微调；在人形控制中，QF3 actor 更新被接入 FastTD3 体系。

这些实验主要支持两个结论：预训练 Flow Policy 可以在不完全破坏原有行为的情况下获得 Q 值带来的改进；在动作维度较高、Critic 容易外推的任务中，过滤梯度比直接 Q 优化更加稳定。论文还展示了从双臂操作到人形控制的跨任务适用性。

## 5. 适合怎样使用

QF3 更像是一种策略后训练工具，而不是从零开始的算法。比较合适的使用场景是：已有行为克隆或 VLA 策略，Replay Buffer 中积累了成功和失败轨迹，希望通过 RL 提升成功率，同时又担心策略漂移。

实践中可以把 clip 范围、正则项和 Critic 训练质量一起作为实验变量。clip 太小，Q 信号几乎无法改变策略；clip 太大，过滤就失去作用。Critic 如果本身在数据附近都不准确，限制梯度也不能解决根本问题。

## 6. 局限和我的判断

速度残差较小只是“靠近数据”的一个近似，并不是 Critic 误差的严格上界。高维多模态动作中，逐维 clip 也不保证整体分布距离足够小；某些关键变化可能需要多个维度协同完成，逐维限制反而会阻碍策略进入真正更优的模式。

我的判断是，QF3 的价值在于给 Flow Policy 的 RL 微调提供了一个清楚、可实现的安全边界。它不承诺解决离线 RL 的分布外问题，但把最危险的更新方向限制在了可解释的位置。对于机器人策略后训练，这种“先保留预训练行为，再有限地使用 Critic”比直接把生成策略当成普通 actor 更符合实际。

