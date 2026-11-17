# Rethinking OPD I & II：重新理解 On-Policy Distillation

这两篇文章放在一起看，讨论的不是一个更复杂的蒸馏损失，而是 OPD（On-Policy Distillation）到底靠什么起作用。

Paper I 主要问：**Teacher 给出的什么信号，Student 才能真正学进去？**

Paper II 进一步问：**如果真正重要的是 Student 访问到的状态，训练 prompt 是不是不需要很多？**

两篇文章给出的直觉可以先写成一句话：

> OPD 的效果取决于 Student 是否能访问到有用状态、Teacher 的局部分布是否值得迁移，以及 Student 能否逐步吸收这些信号。

这和把 Teacher 当成一个固定答案生成器有些不同。OPD 更像是让 Student 在自己的轨迹上不断遇到 Teacher，然后一点点调整自己的 next-token 偏好。

## 1. 先把 OPD 看成一个状态访问过程

在普通的 SFT 或 off-policy distillation 中，训练数据通常来自 Teacher 或其他固定数据集。Student 在这些固定的 response 上学习。

OPD 则让 Student 自己生成 response：

\[
y \sim \pi_\theta(\cdot\mid x)
\]

对 response 的每个 prefix：

\[
s_t=(x,y_{<t})
\]

Teacher 在这个状态上给出 next-token 分布，Student 再把自己的分布向 Teacher 靠近。一个常见的写法是：

\[
\mathcal{L}_{\text{OPD}}(\theta)=
\mathbb{E}_{x,y\sim\pi_\theta}
\left[\sum_t D_{KL}\left(\pi_\theta(\cdot\mid s_t)\,\|\,\pi_T(\cdot\mid s_t)\right)\right]
\]

实际训练通常会对 sampled token、Top-K token 或 PPO-style surrogate 做近似，所以工程实现和这个完整目标并不完全一样。但这个框架有两个关键点：

- **状态是 Student 自己访问的**，不是 Teacher 预先写好的固定轨迹；
- **监督是逐 token 的**，不是只有整条 response 的最终分数。

也就是说，prompt 的作用不一定是把一道题的答案教给 Student。它更像一个入口，负责让 Student 走出一批可以被 Teacher 监督的 prefix。

## 2. Paper I：什么样的 OPD 容易成功

Paper I 研究了 OPD 的现象、机制和训练配方。它最重要的提醒是：Teacher 在 benchmark 上更强，并不自动代表它能给当前 Student 提供更好的更新方向。

### 2.1 Teacher 和 Student 的 thinking pattern 要能接上

如果 Teacher 和 Student 在推理方式上差别很大，Student 访问到的 prefix 可能落在 Teacher 不熟悉的区域；即使 Teacher 在同一题上最终能给出好答案，它在这些 prefix 上的 token-level 分布也不一定适合作为监督。

反过来，如果两者有相近的 reasoning pattern，Teacher 的局部概率分布更容易被 Student 吸收，OPD 才能稳定启动。

因此需要区分两件事：

- Teacher 的总体能力或 benchmark 分数；
- Teacher 在 Student 当前状态上的**可迁移局部信号**。

同一个模型家族中，较大的 Teacher 不一定总能带来更大的 OPD 收益。它也可能只是把 Student 拉向一个暂时无法理解的分布。

### 2.2 OPD 主要在重叠的高概率 token 上发生

作者比较了 Student 和 Teacher 在同一个 prefix 上的 Top-K token 集合。令：

\[
S_\theta(s)=\operatorname{TopK}(\pi_\theta(\cdot\mid s)),\qquad
S_T(s)=\operatorname{TopK}(\pi_T(\cdot\mid s))
\]

可以定义重合率：

\[
O(s)=\frac{|S_\theta(s)\cap S_T(s)|}{K}
\]

在成功的 OPD 训练里，常见现象是：

- Student 和 Teacher 的 Top-K overlap 持续上升；
- 重合 token 上的概率差逐渐变小；
- 两者的 entropy gap 缩小；
- 重合 token 已经承载了各自绝大部分概率质量。

作者还做了一个很有说明力的消融：只更新 Student 与 Teacher 共有的 Top-K token，效果已经接近更新完整 Top-K。只优化不重合的 token，收益明显弱很多。

这提示 OPD 的主要机制可能不是“把 Student 带到完全陌生的 token 区域”，而是先在双方已经共享的高概率区域内做 probability reweighting：

**重合 token → 重新分配概率 → overlap 变高 → 对齐继续变容易。**

这也解释了为什么 thinking-pattern compatibility 重要。若初始高概率区域几乎没有交集，OPD 很难形成这个自我强化的循环。

### 2.3 两种让 OPD 更容易启动的方法

**Off-Policy Cold Start。**

先用 Teacher response 做一段 SFT，让 Student 先接近 Teacher 的基础 reasoning pattern，再切换到 OPD：

\[
\pi_{\theta_0}\xrightarrow{\text{SFT}}\pi_{\theta_1}
\xrightarrow{\text{OPD}}\pi_{\theta_2}
\]

这通常能提高初始 token overlap，让后续的 on-policy 更新更容易发生。需要注意，SFT 本身也会带来能力提升，所以最终增益不能全部归因于 overlap 改善。

**Teacher-aligned prompt selection。**

让 OPD 使用的 prompt 模板和 Teacher 后训练数据更接近，可以改善 Student 访问状态与 Teacher 分布的兼容性。但对齐过强也可能把 Student 的 entropy 压得过低，降低开放场景的探索。因此更合适的做法可能是 Teacher-aligned prompts 与分布外 prompts 混合。

### 2.4 长轨迹不一定带来更多可靠监督

Student 生成的 prefix 越长，越可能逐渐偏离 Teacher 原本熟悉的轨迹分布。此时虽然监督 token 数量变多，Teacher 的局部预测却可能不再可靠。

实验中，较短的最大生成长度表现更好；当长度继续拉长，性能出现停滞甚至不稳定。Teacher 接管 Student prefix 后的 continuation advantage 也会随着 prefix 变长而明显减弱。

所以有一个很实际的提醒：

> 更多 supervised tokens，不等于更多可靠 supervision。

在长推理、Agent 或多轮工具调用里，如何检测 prefix 已经离开 Teacher 的有效支持范围，可能比简单增加上下文长度更重要。

## 3. Paper II：One Training Example

Paper II 做了一个看起来很极端的实验：数学 OPD 通常使用上万道题，但如果只保留一道题，反复对它采样和蒸馏，Student 能不能在其他题上获得泛化提升？

实验显示，One-Shot OPD 在早期训练中可以恢复 Full-Data OPD 很大一部分收益。以文中的设置为例，One-Shot 在 300 step 时已经恢复了大约 87% 的 Full-Data 增益；训练到 1000 step 后，恢复比例下降到约 72%，说明它很有竞争力，但不等于长期完全替代多样化数据。

这个实验里每一步仍然会从当前 Student 重新采样多条 response，而不是把同一个答案做 SFT。prompt 只有一个，rollout 和 prefix 却在不断变化。

更反直觉的是，即便选择一道 Student 从未解出的 hard problem，One-Shot OPD 仍然能改善其他数学题。于是，增益不能简单解释成“Student 学会了这道训练题”。

## 4. State Coverage：prompt 更像状态生成器

Paper II 给出的解释是：一个 prompt 的价值，可能主要在于诱导 Student 产生一批可以被 Teacher 监督的状态。

同一道题反复采样时，Student 可能走出很多不同 prefix：

- 直接推导；
- 错误后重新计算；
- 检查中间结果；
- 换一种代数或几何路径；
- 先猜测再验证。

这些 prefix 虽然来自同一个问题，却可能包含许多任务之间共享的局部 reasoning pattern。

为了验证这个解释，作者用 Teacher 的最后一层 hidden state 表示每个 prefix，先做 PCA，再用 K-means 把 Full-Data OPD 的 states 聚成 200 个 cluster。对一个训练设置，定义：

\[
\operatorname{Coverage}(S)=
\frac{|\{c(s):s\in S\}|}{K}
\]

它统计访问到多少个 cluster。这个指标不是严格的真实状态空间覆盖率，而是一个 representation-dependent proxy。

在相同的 300-step 预算下，文中观察到：

| 训练 prompt 数量 | 相对 state coverage（示意） |
| --- | ---: |
| 1 个 query | 约 71.5% |
| 16 个语义不同的 query | 约 98.9% |
| Full-Data held-out states | 100% |

One-Shot 的 coverage 在训练早期就增长得很快，之后增加有限。加入少量语义多样的 query，可以补齐剩余区域。这说明原始 prompt 数量和有效状态数量不是一回事。

但不要把 71.5% 读成“覆盖了真实推理空间的 71.5%”。聚类会把不同状态压进同一个簇，也可能把相近状态分开。更精确的说法是：在这套 hidden-state 表征和聚类尺度下，One-Shot 访问了相当多与 Full-Data 相似的状态区域。

## 5. Algorithm-Starved：真正慢的是吸收信号

如果 One-Shot 很快就能访问很多状态，为什么训练还要持续几百步？Paper II 提出，瓶颈可能不在状态发现，而在 Student 吸收 Teacher 分布的速度。

在 Student 当前访问的 token 上，可以计算平均概率差：

\[
d_t=\frac{1}{|T_t|}\sum_{i\in T_t}
\left|\log\pi_T(y_i\mid s_i)-\log\pi_{\theta_t}(y_i\mid s_i)\right|
\]

然后定义吸收率：

\[
v_t=\frac{d_t-d_{t+1}}{d_t}
\]

直观上，\(v_t\) 表示这一轮更新消除了多少剩余 Teacher-Student gap。

对 1、4、16 以及约 17000 个 query 的设置进行比较时，不同数据规模的 absorption rate 都在逐步下降，下降速度相当接近。到较后训练阶段，它们都消除了训练早期剩余差异的大部分比例，但每一轮能吸收的新信息越来越少。

这个现象可以叫作 **algorithm-starved**：训练变慢不一定是因为 prompt 不够多，而可能是当前优化过程已经难以继续吸收现有监督。

一个很有意思的控制实验是 Fixed-State Distillation。作者把 Student 最开始采样的轨迹固定下来，后面不再产生新 states，只在这些固定轨迹上重复做 off-policy distillation。即使如此，performance 和 token alignment 仍然能继续改善约 200 步，然后才停止。

这说明“不断发现新状态”不是训练持续进步的唯一原因。慢速还可能来自 optimizer dynamics、参数耦合、gradient interference 或目标函数的局部几何结构。论文没有把这些因素完全拆开，这正是后续可以深入的地方。

## 6. 甚至不一定需要数学题

既然 prompt 主要负责生成可学习状态，Paper II 还测试了 content-light template 和通用聊天 prompt。

有些只包含 `<think>` 一类起始模板、或者来自 WildChat 的非数学 prompt，也能得到接近数学训练数据的效果，而且 rollout token 更少。这里的关键不是 prompt 是否明确写了数学任务，而是它能不能让 Student 进入足够长、足够丰富、同时 Teacher 能提供有效分布的推理状态。

但这不代表空 prompt 都有效。有的模板会让模型马上结束，产生很短的 meta response；这种情况下既没有丰富状态，也没有足够 token 供 Teacher 监督。

所以更准确的条件是：

\[
\text{Prompt}\rightarrow\text{Useful Student States}
\rightarrow\text{Exploitable Teacher Signal}
\]

内容少可以，但状态不能少，Teacher 信号也不能离开可迁移范围。

## 7. One-Shot OPD 和 One-Shot RLVR 不是一回事

One-Shot RLVR 依赖整条轨迹的最终结果。如果一道题的所有 rollout 都答对，group advantage 可能接近零，训练就会失去更新信号。

OPD 不依赖 outcome 是否已经正确，它关心的是 Student 和 Teacher 在每个 token 上的概率差：

\[
A_t^{\text{OPD}}=
\log\pi_T(y_t\mid s_t)-\log\pi_\theta(y_t\mid s_t)
\]

因此即便 Student 已经把训练题全部答对，next-token distribution 仍可能与 Teacher 不同，OPD 还可以继续做分布对齐。

这解释了为什么 One-Shot OPD 在实验中能比 One-Shot RLVR 获得更明显的泛化提升。但两者目标并不相同：RLVR 直接优化结果，OPD 受 Teacher 分布质量限制，不能据此推出 OPD 在所有任务上都优于 RLVR。

## 8. 两篇文章放在一起看

Paper I 和 Paper II 正好对应 OPD 的两个层面：

| 问题 | 两篇文章给出的视角 |
| --- | --- |
| Teacher 的什么信号容易被学到？ | 与 Student thinking pattern 兼容、在高概率重合区域内的局部分布变化 |
| 为什么少量 prompt 也能训练？ | 一个 prompt 能诱导许多 Student states，prefix 级监督比题目数量更密集 |
| 为什么训练会变慢？ | 新状态的收益有限，Student 吸收已有 Teacher 信号的速度逐渐下降 |
| 为什么多样化 query 仍然有用？ | 它们能补齐 One-Shot 尚未访问的状态区域 |

可以把两篇文章的启示写成一个研究用的粗略框架：

\[
\text{Effective OPD}
\approx
\text{State Coverage}
\times
\text{Signal Transferability}
\times
\text{Optimization Efficiency}
\]

这不是原论文提出的定理，更像是一个整理思路：

- **State Coverage**：Student 是否访问到了足够多、且与目标任务相关的状态？
- **Signal Transferability**：Teacher 在这些状态上的分布变化，是否兼容、稳定、值得迁移？
- **Optimization Efficiency**：Student 能多快吸收这些 token-level 信号？

## 9. 局限：不要把现象直接扩大成结论

第一，state coverage 依赖 hidden-state 表征、PCA 和聚类尺度。它能帮助比较训练设置，但不能直接证明真实推理状态空间已经被覆盖。

第二，One-Shot OPD 的 Student 已经有相当基础能力，Teacher 也来自相关训练分布。一个 prompt 能获得强泛化，不等于从零开始就能学会整个数学领域。

第三，One-Shot 省掉的是 prompt 数量，不是计算量。它仍然需要反复 rollout、Teacher inference 和多轮优化。数据语义变少，未必意味着系统成本按同样比例下降。

第四，Paper I 观察到高 overlap 和成功 OPD 相关，但 overlap 本身不等于正确性。一个错误的偏好也可能在 Student 和 Teacher 之间高度重合。

第五，长 prefix 上的 Teacher supervision 可能失真。对于 Agent、多轮工具调用和超长推理，需要额外判断哪些状态仍处于 Teacher 的有效支持范围。

第六，absorption rate 的下降说明优化越来越慢，但还不能单独说明具体是梯度冲突、参数耦合还是信赖域限制。这个指标更像一个现象探针，而不是完整解释。

## 10. 可以继续怎么做

### 从 state coverage 走向 gradient coverage

hidden state 相似不一定意味着参数更新相似。更直接的做法，是计算每个 state 的蒸馏梯度：

\[
g(s)=\nabla_\theta D_{KL}
\left(\pi_\theta(\cdot\mid s)\,\|\,\pi_T(\cdot\mid s)\right)
\]

再衡量一组 states 的梯度子空间对 Full-Data 梯度空间的覆盖程度。这样评价的就不只是“访问了多少状态”，而是“提供了多少不同的有效更新方向”。

### 把 absorption rate 用来调度训练

当某一类 token 的 gap 已经很小，继续重复蒸馏的收益可能有限；当 gap 大但更新方向反复冲突，也不应简单增加权重。可以考虑：

- 按剩余 gap 和历史 absorption rate 动态加权；
- 对同一 batch 做多轮更新，同时使用 trust region；
- 检测梯度冲突后做方向去冲突；
- 区分 Student 已经学会、正在吸收和长期吸收不了的 Teacher signals。

### 和 Direct-OPD、Waypoint-OPD 结合

可以把三类问题接到一起：

- Paper I：哪些局部信号是 Student 能吸收的？
- Paper II：哪些 states 能暴露这些信号？
- Waypoint-OPD：哪些 policy shift 在 Teacher 训练过程中更稳定？

一种还可以继续试的 state-token 权重是：

\[
w(s,a)=w_{\text{coverage}}(s)
\cdot w_{\text{compatibility}}(s,a)
\cdot w_{\text{shift}}(s,a)
\]

其中分别表示当前状态提供了多少新增信息、Teacher 和 Student 是否处于可对齐的局部区域，以及策略变化是否在训练过程中保持一致。

这三个权重不一定要直接相乘，也不能默认高 overlap 或高 temporal consistency 就一定代表真实收益。更可靠的方向，是用少量 verifier feedback、短期性能变化或 influence estimate 去学习一个 **value of transfer**，判断某个 state-token pair 是否值得继续蒸馏。

## 11. 作为机器人和 Agent 训练的参考

这两篇文章对机器人策略学习也有直接启发。机器人数据经常很贵，问题不一定是“采集更多完整任务”，而可能是让同一批 rollout 暴露更多可迁移的局部状态，并确认 Teacher 或示范策略在这些状态上给出的动作分布值得学习。

如果把机器人轨迹拆成 prefix，单条示范也可以提供很多局部动作状态；但要注意，状态覆盖、动作分布兼容和长期控制稳定性是三件事。一个视觉状态在表征空间里相似，不代表它们的接触动作、时序约束和失败后果也相似。

因此，OPD 的研究思路可以借到机器人上，但最好保留真实环境的成功判定、动作安全和 reset 约束。少量示范有机会启动学习，真正决定能否泛化的，仍然是可迁移信号和吸收过程。

## 12. 总体判断

Paper I 的价值在于把“Teacher 更强所以一定更适合蒸馏”的直觉拆开了：OPD 更依赖局部 thinking-pattern compatibility，以及双方高概率区域里的可吸收变化。

Paper II 则把“训练数据量越大越好”的直觉往前推了一步：一个 prompt 也能生成很多有效 states，训练变慢时，瓶颈可能已经从 state discovery 转向 signal absorption。

两篇文章合起来，给 OPD 留下了一个比较清楚的研究地图：先看 Student 访问了哪些状态，再看 Teacher 的信号能否迁移，最后看优化过程能不能把它们吸收进去。这样理解 OPD，比只盯着训练 prompt 数量、Teacher benchmark 或最终 KL 更接近它真正的学习动力学。
