# Waypoint-OPD：把 RL 训练过程中的策略变化蒸馏给 Student

论文：[Waypoint-OPD: Transferring RL-Induced Policy Shifts via Gated On-Policy Distillation](https://huggingface.co/papers/2610.21601)

这篇工作的核心问题是：当一个 Teacher 通过强化学习获得了新的推理能力时，Student 应该模仿 Teacher 最终的策略，还是应该学习 Teacher 在 RL 过程中发生的策略变化？

Waypoint-OPD 给出的答案是：策略变化本身更值得蒸馏，但不能把整个 RL 过程中的变化都当成可靠信号。作者使用一个中间 checkpoint，比较 RL 前后两个阶段的变化方向，再决定哪些策略变化值得传给 Student。

## 1. 从 OPD 到 Direct-OPD

On-Policy Distillation（OPD）的特点是，训练轨迹由 Student 自己生成。Teacher 不负责生成完整答案，而是在 Student 实际访问的每个 prefix 上提供 next-token 分布，然后让 Student 调整自己的概率。

传统 OPD 主要让 Student 接近 Teacher 的最终策略。但 Teacher 的最终能力里，既有 RL 之前已经具备的内容，也有 RL 新学到的内容。直接模仿最终策略，可能把 Teacher 原本的偏好和 Student 已经掌握的能力一起重复蒸馏。

Direct-OPD 因此比较 RL 前后的两个 Teacher：

\[
\delta_t(a)=\log \pi_1(a|s_t)-\log \pi_0(a|s_t)
\]

其中 \(\pi_0\) 是 RL 前的 Teacher，\(\pi_1\) 是 RL 后的 Teacher。\(\delta_t(a)>0\) 表示 RL 提高了在当前 prefix 选择 token \(a\) 的概率，负值则表示降低。

这个 log-probability ratio 可以被理解为一种隐式的策略改进信号。它不要求重新调用 reward verifier，而是直接利用 Teacher RL 训练前后的差异。

## 2. 为什么只看起点和终点还不够

Direct-OPD 只观察：

**RL 前的策略 → RL 后的策略。**

但同样的 endpoint shift 可能来自完全不同的训练过程。例如，一个行为的概率可能经历：

- 20% → 40% → 60%，前后两个阶段都在强化；
- 20% → 90% → 60%，前期强化、后期又反向修正。

两种情况最终都从 20% 变成 60%，但第一种变化更稳定，第二种变化说明训练过程中可能出现过冲或方向反复。

Waypoint-OPD 引入一个中间 checkpoint，把 Teacher 的训练过程拆成两段，再用两段策略变化是否一致来判断 endpoint signal 是否可靠。

## 3. 三个 Teacher checkpoint

方法使用三个策略：

- Initial：\(\pi_0\)，RL 开始前；
- Waypoint：\(\pi_m\)，RL 中间阶段；
- Final：\(\pi_1\)，RL 完成后。

定义两段 centered policy shift：

\[
\delta_t^{(1)}(a)=\log\frac{\pi_m(a|s_t)}{\pi_0(a|s_t)},
\qquad
\delta_t^{(2)}(a)=\log\frac{\pi_1(a|s_t)}{\pi_m(a|s_t)}
\]

两段 shift 相加，仍然等于 Initial 到 Final 的 endpoint shift。作者不改变最终要迁移的策略变化，只用两段 shift 的方向一致性给它加权。

## 4. Centering 和 temporal gate

在 Student 当前 prefix 上，方法只考虑 Student 概率最高的 Top-K token。实验中使用 \(K=16\)。对每一段 shift 减去 Student 局部概率分布下的加权均值：

\[
u_t(a)=\delta_t^{(1)}(a)-\mathbb E_q[\delta_t^{(1)}],
\qquad
v_t(a)=\delta_t^{(2)}(a)-\mathbb E_q[\delta_t^{(2)}]
\]

Centering 的作用，是去掉所有 token 共同增加或共同减少的部分，只比较不同 token 之间的相对偏好变化。这和 Advantage 中减去 baseline 的思路类似。

随后计算两个 centered shift 的加权余弦相似度：

\[
c_t=\cos_q(u_t,v_t),
\qquad
g_t=\max(0,c_t)
\]

\(g_t\) 就是 temporal gate。同一个 prefix 下的 Top-K token 共享这个 gate：两段变化方向一致时，保留更多 endpoint signal；方向冲突时，降低该 prefix 的蒸馏权重。

最终的更新系数可以理解为：

**Student 的局部概率权重 × temporal gate × endpoint policy shift。**

作者还使用 batch-level normalization，避免 gate 整体变小后只是简单地把所有更新步长缩小。实际训练使用 PPO 风格的 surrogate objective，并对 gate 系数 stop-gradient。

## 5. 实验结果

实验使用 RL 训练过的 Qwen3-4B 作为 Teacher，Student 包括 Qwen3-8B、14B 和 30B-A3B，测试数学推理与代码生成。

数学任务中，Waypoint-OPD 相比 Direct-OPD 的平均提升约为：

| Student | Direct-OPD | Waypoint-OPD | 提升 |
| --- | ---: | ---: | ---: |
| 8B | 54.0 | 56.1 | +2.18 |
| 14B | 57.6 | 59.2 | +1.59 |
| 30B-A3B | 56.0 | 57.9 | +1.89 |

代码任务中的提升约为：8B +2.66、14B +1.52、30B-A3B +1.19 个百分点。数学实验的 21 个对应比较中有 19 个提升。

Student 甚至可以超过 Final Teacher。原因是这里迁移的是 Teacher 通过 RL 学到的策略变化，而不是把 Student 的绝对分布压到 Teacher 的最终分布上；Student 自己更大的模型容量仍然可以发挥作用。

消融实验也支持 gate 的作用：完整方法在 Qwen3-8B 数学实验中达到 56.14，去掉 gate 后为 53.50，随机打乱不同 prefix 的 gate 后降到 51.20，去掉 batch normalization 后为 55.22。随机 gate 反而更差，说明 gate 与具体 prefix 有关，而不是简单降低整体更新强度。

## 6. 这篇方法真正解决了什么

Waypoint-OPD 的创新点并不是重新设计 RL，而是把 Teacher 的训练历史作为额外监督。

普通蒸馏只看“Teacher 现在是什么样”；Direct-OPD 看“Teacher 相比 RL 之前改变了什么”；Waypoint-OPD 进一步问“这个改变在训练过程中是否稳定地朝同一个方向发生”。

这提供了一个很有用的视角：RL checkpoint 不只是保存最终模型，也保存了策略变化的轨迹。训练轨迹本身可以帮助判断哪些能力变化更值得迁移。

## 7. 局限和进一步思考

第一，temporal consistency 只能说明策略变化稳定，不等于它一定正确。错误偏好也可能在两个阶段持续增强；相反，后期纠正早期错误的过程可能被 gate 降权。

第二，论文主要使用一个 waypoint。更完整的做法可以保存多个 checkpoint，统计相邻 shift 的方向、方差以及后期变化和 endpoint shift 的关系。但 checkpoint 越多，Teacher scoring 的开销也越大。

第三，gate 是手工设计的 cosine similarity。更进一步可以学习一个 transfer utility，直接估计某个 state-token 的策略变化被 Student 吸收后，是否真正改善验证集表现。

第四，实验集中在 Qwen3 模型家族、数学和代码推理任务。对于不同 Teacher-Student 架构、视觉模型、机器人策略或更长时序任务，temporal gate 是否仍然有效，还需要单独验证。

我的判断是，这篇论文最有价值的地方不是 cosine gate 本身，而是把“策略变化的可靠性”引入 OPD。后续可以把三类信息结合起来：Student 是否能访问相关状态、Teacher 与 Student 是否具有兼容的高概率行为、Teacher 的策略变化是否在训练中保持一致。这样，蒸馏目标就不再是简单模仿一个最终模型，而是选择性地迁移真正有用的学习过程。
