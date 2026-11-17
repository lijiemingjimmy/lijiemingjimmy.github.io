# Deterministic Policy Gradient（DPG）：确定性策略梯度

## 一句话理解

DPG 是连续动作空间中的 Policy Gradient 方法。它不通过提高某个动作的采样概率来更新策略，而是让 Critic 计算：**当前动作应该朝哪个方向变化，才能提高未来回报。**

核心梯度链是：

\[
Q(s,a)\rightarrow\nabla_a Q(s,a)\rightarrow\nabla_\theta\mu_\theta(s)
\]

这也是 DDPG、TD3 等算法的理论基础，在机器人操作中很常见。

## 1. DPG 和 REINFORCE 的区别

REINFORCE 使用随机策略：

\[
a\sim\pi_\theta(a\mid s)
\]

它通过 score function gradient 更新：

\[
\nabla_\theta J
=\mathbb E\left[
\nabla_\theta\log\pi_\theta(a\mid s)Q^\pi(s,a)
\right]
\]

DPG 使用确定性策略：

\[
a=\mu_\theta(s)
\]

给定状态后，Actor 直接输出动作，而不是输出动作分布。因此不能直接使用 \(\log\pi_\theta(a\mid s)\) 的形式，而是对 Critic 的动作输入求导：

\[
\nabla_\theta J(\theta)
=\mathbb E_{s\sim\rho^\mu}
\left[
\nabla_\theta\mu_\theta(s)
\left.\nabla_a Q^\mu(s,a)\right|_{a=\mu_\theta(s)}
\right]
\]

两种方法的直觉区别是：

- REINFORCE / PPO：提高高 Advantage 动作的概率；
- DPG：直接把当前连续动作推向更高 Q-value 的方向。

## 2. 公式中的两个梯度

### Critic action gradient

\[
\nabla_a Q(s,a)
\]

它回答：在当前状态下，动作的每个分量应该朝哪个方向变化，才能提高累计回报？

例如：

\[
\nabla_a Q(s,a)=(0.3,-0.5,0.1)
\]

表示局部来看，增大第一个和第三个动作分量、减小第二个动作分量可能提高 Q-value。

### Actor parameter gradient

\[
\nabla_\theta\mu_\theta(s)
\]

它回答：应该怎样修改神经网络参数，才能让 Actor 的输出朝刚才的方向变化？

两者通过链式法则连接：

\[
\frac{\partial Q}{\partial\theta}
=\frac{\partial Q}{\partial a}
\frac{\partial a}{\partial\theta}
\]

因此 Critic 只需要对动作可微，不需要环境动力学模型可微，也不需要把梯度反向传播穿过真实机器人。

## 3. DPG 为什么成立

确定性策略的价值函数满足：

\[
V^\mu(s)=Q^\mu(s,\mu_\theta(s))
\]

对策略参数求导：

\[
\nabla_\theta V^\mu(s)
=\nabla_\theta\mu_\theta(s)
\left.\nabla_a Q^\mu(s,a)\right|_{a=\mu_\theta(s)}
+\gamma\mathbb E_{s'}[\nabla_\theta V^\mu(s')]
\]

不断展开未来状态项，就得到每个访问状态上的动作梯度之和：

\[
\nabla_\theta J
=\mathbb E\left[
\sum_{t=0}^{\infty}\gamma^t
\nabla_\theta\mu_\theta(s_t)
\left.\nabla_a Q^\mu(s_t,a)\right|_{a=\mu_\theta(s_t)}
\right]
\]

未来动力学的影响已经被包含在 \(Q^\mu\) 中，所以训练时不需要知道或微分机器人动力学 \(P(s'\mid s,a)\)。

## 4. 一个一维例子

假设 Critic 学到：

\[
Q(s,a)=-(a-2)^2
\]

它在 \(a=2\) 时最大。对动作求导：

\[
\frac{\partial Q}{\partial a}=-2(a-2)
\]

如果当前 Actor 输出 \(a=0\)，则：

\[
\frac{\partial Q}{\partial a}=4
\]

正梯度告诉 Actor 增大动作，逐渐向 \(a=2\) 靠近。真实 DPG 并不是直接把动作变量改成 0.1，而是利用这个方向更新 Actor 网络，使网络在相关状态下输出更好的动作。

## 5. DDPG：DPG 的深度网络实现

DDPG 包含两个网络：

- Actor：\(\mu_\theta(s)\)，输入状态并输出动作；
- Critic：\(Q_\phi(s,a)\)，输入状态和动作并预测未来累计回报。

### Step 1：收集数据

加入探索噪声执行动作：

\[
a_t=\mu_\theta(s_t)+\epsilon_t
\]

并把 \((s_t,a_t,r_t,s_{t+1},d_t)\) 存入 Replay Buffer。确定性策略仍然需要探索，否则同一状态总是输出同一个动作。

### Step 2：更新 Critic

使用 Target Network 构建 TD target：

\[
y_t=r_t+\gamma(1-d_t)
Q_{\phi^-}\left(s_{t+1},\mu_{\theta^-}(s_{t+1})\right)
\]

最小化 Bellman error：

\[
\mathcal L_Q(\phi)=
\mathbb E_{\mathcal D}\left[(Q_\phi(s,a)-y)^2\right]
\]

### Step 3：更新 Actor

最大化 Critic 对 Actor 动作的评价，等价于最小化：

\[
\mathcal L_\pi(\theta)
=-\mathbb E_{s\sim\mathcal D}
\left[Q_\phi(s,\mu_\theta(s))\right]
\]

自动微分得到：

\[
\nabla_\theta\mathcal L_\pi
=-\mathbb E_s\left[
\nabla_\theta\mu_\theta(s)
\left.\nabla_aQ_\phi(s,a)\right|_{a=\mu_\theta(s)}
\right]
\]

DDPG 使用 Replay Buffer 做 off-policy 训练，因此实际采样状态分布是数据集分布的近似，不应无条件地把它视为原始目标 \(J(\theta)\) 的精确梯度。

## 6. TD3 为什么比 DDPG 稳定

TD3 在 DDPG 基础上加入三项机制：

1. **Clipped Double Q-learning**：用两个 Critic 缓解 Q-value 高估；
2. **Delayed Policy Updates**：降低 Actor 更新频率，等待 Critic 更可靠；
3. **Target Policy Smoothing**：对 target action 加平滑噪声，避免 Critic 利用尖锐的错误峰值。

DPG 的关键风险也在这里：如果 Critic 的动作梯度不可靠，Actor 就会被错误方向带偏。

## 7. 回到 SmoothRL：DPG 如何处理 Action Chunk

SmoothRL 的 Actor 输出的不是单个动作，而是一个动作序列：

\[
A_\theta(s)=[a_0,a_1,\ldots,a_{H-1}]
\]

Critic 可以对整个序列求动作梯度：

\[
\nabla_AQ(s,A)
\]

异步执行时，Action Chunk 被分成：

\[
A=[A_{\text{committed}},A_{\text{execution}},A_{\text{discarded}}]
\]

SmoothRL 的核心梯度可以写成：

\[
\nabla_\theta J
=\mathbb E\left[
\nabla_\theta A_{\text{execution}}
\cdot
\nabla_{A_{\text{execution}}}
Q(s,A_{\text{committed}},A_{\text{execution}})
\right]
\]

Committed actions 作为 Critic 的上下文输入，但对当前 Actor 停止反向传播；Discarded actions 不参与这部分 Q-value 优化。

因此 SmoothRL 没有改变 DPG 的基本原理，而是增加了 **execution-aware gradient truncation**：只让 Critic 的动作梯度通过当前策略真正执行的区间。

## 8. 最后记住这条链

\[
\text{Critic }Q(s,a)
\rightarrow\nabla_aQ\;\text{（动作方向）}
\rightarrow\nabla_\theta\mu_\theta\;\text{（参数更新）}
\]

DPG 学习的是“当前连续动作应该朝哪个方向变化”，而 REINFORCE/PPO 学习的是“哪些动作的概率应该提高”。
