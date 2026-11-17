# Beyond Motion Tracking：可操控的物理交互能力

> **原文**：Beyond Motion Tracking: Steerable Physical Interaction Competence  
> **作者**：Haoyang Weng  
> **时间**：2026 年 10 月  
> **主题**：Humanoid Robotics、VLA、Whole-Body Control、Physical Interaction  
> **链接**：[原文](https://haoyangweng.pages.dev/blog/2026-10-steerable-interaction-competence/en/)

## 一句话总结

当前人形机器人的瓶颈不只是上层 VLM/VLA 能否理解任务，也在于底层控制器能否理解物理交互，并让上层策略精确指定这种交互。

文章主张，下一代通用底层控制器 **General S0** 应同时具备：

\[
\text{General S0}
=\text{Dynamics Grounding}+\text{Expressive Interface}
\]

这不是严格的数学等式，而是两个设计要求：

- **Physical Interaction Competence**：理解物体动力学、接触、摩擦和负载，并根据物理反馈调整行为；
- **Steerability**：让上层策略不仅指定身体怎么运动，也能指定物体怎么运动、在哪里接触、施加多大力以及采用什么控制模式。

## 1. 为什么 Motion Tracking 不够

一类典型的人形机器人 VLA 系统可以抽象为：

- **S1：高层策略**：理解场景、规划任务并生成身体运动参考；
- **S0：Whole-Body Motion Tracker**：根据本体感知跟踪关节和身体姿态。

这种 S0 并不是完全开环。它可以利用关节状态、IMU 和其他 proprioception 稳定地跟踪身体轨迹。问题在于，它对外部物体和接触状态缺乏足够的感知与控制能力。

例如，让机器人把一个很重的箱子扔到平台上：

1. S1 生成一个看起来合理的投掷动作；
2. S0 也准确跟踪了身体轨迹；
3. 但箱子的质量、转动惯量、抓握接触和释放时机会改变完整系统的动力学。

只会空手动作跟踪的 S0，不一定能在负载和接触变化时维持平衡，更无法保证箱子按照预期飞出。

因此需要区分：

\[
\text{Spatial Grounding}\neq\text{Dynamics Grounding}
\]

Spatial Grounding 关注杯子在哪里、应该移动到哪里；Dynamics Grounding 关注推箱子时物体是否滑动、摩擦力是否足够、施力方向是否需要调整，以及动作会造成什么物理后果。

把上层模型的推理频率提高到 50 Hz，也不会自动获得这种能力。控制频率和通过闭环交互学到的动力学能力是两个不同维度。

## 2. General S0 的三个阶段

文章将控制器的发展抽象为三个阶段。

### A. Motion-Centric Tracker

输入身体运动参考和本体状态，目标是跟踪动作。它通常缺乏直接的场景感知，主要通过自由空间动作跟踪训练获得能力。

### B. Object-Aware Task-Specific Controller

控制器能够感知物体和环境，并通过 RL 等方式学习物体动力学，但通常针对特定任务训练，接口也高度任务化，例如搬箱子或跑酷。

### C. Interaction-Competent General S0

使用统一的 Scene Representation 和 Interaction Intent Interface，学习跨物体、跨任务、跨接触模式的物理交互控制能力。这才是文章真正希望推进的方向。

可以将目标写成条件控制策略：

\[
a_t=\pi^{S0}_\theta(o_t^{robot},o_t^{scene},c_t)
\]

其中：

- \(o_t^{robot}\)：关节状态、IMU 和其他本体感知；
- \(o_t^{scene}\)：RGB-D、点云、物体运动和接触反馈；
- \(c_t\)：S1 提供的 Interaction Intent；
- \(a_t\)：S0 最终执行的关节命令、力矩或底层控制量。

相比传统 Motion Tracker 的：

\[
a_t=\pi_\theta(o_t^{robot},q_t^{ref})
\]

这个策略额外接收两类信息：外部世界正在发生什么，以及希望外部世界发生什么。

## 3. Scene Awareness：从物体位姿到通用场景表示

一种直接的物体表示是：

\[
s^{obj}=(p^{obj},R^{obj})
\]

它使用物体的位置和姿态描述场景。但这种表示需要预先定义物体坐标系，也不容易扩展到多个物体、非刚体和复杂接触场景。

因此文章倾向于使用 RGB、Depth、Point Cloud 等更通用的表示。需要注意：感知几何形状不等于理解物体动力学。

- 点云可以告诉控制器物体在哪里、表面是什么形状；
- 质量、摩擦系数、惯性和接触稳定性仍需要通过观测与交互估计，或者由策略隐式学习。

也就是说，Scene Awareness 是 Dynamics Grounding 的必要条件，但不是充分条件。

## 4. Interaction Intent：上层不只指定身体动作

文章认为，S1 不应只输出机器人身体轨迹，也可以直接描述期望的物体运动。

例如使用 General-Flow 一类的表示：

\[
F=\{(x_i,\Delta x_i)\}_{i=1}^{N}
\]

其中 \(x_i\) 是物体表面的三维点，\(\Delta x_i\) 是这些点期望发生的位移，也可以扩展为未来轨迹。

如果机器人要把箱子旋转 90°，S1 可以指定箱子表面不同点未来应到达的位置，而不必精确规定每个关节该怎样运动。S0 决定如何通过抓取、施力和全身协调实现这一世界状态变化。

这建立了一个不同于 Body Motion Tracking 的接口：

\[
\text{Desired World Change}\rightarrow\text{Robot Control}
\]

高层策略表达“希望外部世界发生什么”，底层控制器负责把它转化为可执行的身体动作。

## 5. 为什么还需要 Contact-Centric Interface

Object Flow 可以描述物体如何运动，但不能完整描述接触行为。

例如机器人擦黑板：黑板本身可能几乎不移动，

\[
\Delta x_{board}=0
\]

但机器人必须保持接触并施加合适的法向力，才能真正擦掉字迹。只指定身体轨迹或物体轨迹，无法区分轻轻划过和用力擦拭。

因此 Interaction Intent 可以拆成三个互补部分：

\[
c_t=(c_t^{body},c_t^{object},c_t^{contact})
\]

Contact Intent 可以用下式帮助理解：

\[
c_t^{contact}=(m_t,f_t^{des},K_t^{des},D_t^{des})
\]

其中：

- \(m_t\)：期望接触模式；
- \(f_t^{des}\)：目标接触力；
- \(K_t^{des}\)、\(D_t^{des}\)：期望刚度和阻尼。

这只是便于理解的形式化表达，不代表文章给出了固定的具体实现。

对于灵巧手操作，这种区分尤其重要。捏住积木并旋转时，物体轨迹可能相同，但抓取力不同会导致积木滑落、被挤压或稳定完成旋转。

因此 General S0 不仅需要 Kinematic Tracking，还需要 Contact-Aware Control，甚至需要学习隐式的阻抗控制或力控制。

## 6. Steerability：能力强不等于容易控制

一个有能力的 S0，未必是一个容易被 S1 控制的 S0。

可以把底层策略写成：

\[
\pi_\theta(a\mid s,c)
\]

如果条件 \(c\) 只包含末端执行器的位置轨迹，那么同样的轨迹可能对应完全不同的接触力和行为结果。对于行为克隆，这会表现为条件信息不足造成的动作歧义：

\[
H(A\mid S,C)>0
\]

增加与行为差异相关的 Contact Intent，有机会减少这种条件歧义，使模型更容易学习多种执行模式。

文章将这一思想和 \(\pi_{0.7}\) 中的多种条件控制联系起来。语言、速度、质量、控制模态和 Visual Subgoal 等条件，可以让模型不仅理解做什么，也理解怎么做。区别在于，\(\pi_{0.7}\) 更强调 VLA 层面的多模态任务引导，而本文聚焦底层 S0 的物理交互接口。

## 7. Latent Action 不一定自动增加控制能力

文章在 Q&A 中讨论了 Latent Action 的边界。假设物理接口为 \(q\)，先经编码器得到：

\[
z=E(q)
\]

再由底层控制器执行：

\[
a=D(s,z)
\]

如果 S1 直接输出 \(z\)，而不是先输出 \(q\)，那么在编码器可达的 latent 范围内，理论上不会自动产生新的交互能力，因为：

\[
D(s,E(q))
\]

已经涵盖原有物理接口能够表达的行为。

这不意味着 Latent Interface 没有价值。它仍然可能改善平滑性、可优化性、技能插值和组合。文章反对的是“仅仅更换 latent representation，就能凭空产生缺失的物理交互能力”这一推断。

这个区分同样适用于 Skill Latent、Action Tokenizer 和 Hierarchical RL：表示空间可以改善学习和控制，但不能替代新的可观测信息、物理反馈或行为能力。

## 8. 研究路线

文章给出的研究路线可以概括为：

```text
Human-Object Interaction Data
            ↓
   Retargeting / Simulation
            ↓
Diverse Object Dynamics Training
            ↓
General Interaction-Competent S0
            ↑
Scene + Body/Object/Contact Intent
```

作者展示了基于 HDMI 代码和 Weave 人物交互数据重定向训练的模拟策略，涉及箱子、桌椅等家具交互。但这不能直接等同于已经实现跨任务、跨接触模式的通用 S0。

## 9. 仍未解决的研究问题

### 9.1 General S0 的泛化能力从何而来

把大量 Task-Specific RL Policy 的数据合并训练，不代表模型自动获得跨任务的物理交互泛化。需要系统研究：

- 训练任务的覆盖度；
- 物体动力学分布；
- 接触模式的多样性；
- 测试时的 OOD 物体与 OOD 接触条件。

### 9.2 接触意图如何标注和学习

视觉视频能提供丰富的几何与轨迹信息，却未必能准确识别接触力、摩擦系数和柔顺性。如果训练数据主要来自 Human Video，Contact-Centric Interface 会面临明显的监督困难。

可能需要融合力传感器、触觉、系统辨识、仿真标注和交互式数据采集。

### 9.3 S0 应该自主恢复到什么程度

例如灵巧手发现积木即将滑落，S0 应当立即增大抓取力；但如果积木已经掉下去，可能需要 S1 重新规划任务。

这里存在一个需要明确的分工：

- 什么属于 S0 的 Reactive Control；
- 什么需要 S1 的 Recovery Planning；
- 失败恢复的边界是否应该由 Intent Interface 显式定义。

## 10. 对训练范式的一个限定

文章将 Dynamics Grounding 重点放在通过闭环交互获得的能力，但这并不意味着必须使用 RL。

大规模真实示教同样可以训练出响应性控制器；显式动力学建模、系统辨识和 MPC 也可以成为可行路线。因此不能简单得出“BC 不行、只有 RL 才行”的结论。

更准确的说法是：训练方法必须让 S0 接触到与目标交互相关的状态、反馈和行为结果，并能够在闭环中学会利用这些信息。

## 11. 总结

这篇文章把人形机器人底层控制器的问题从“能否准确跟踪身体动作”推进到三个更具体的问题：

1. S0 是否理解物体和接触的动力学后果；
2. S1 是否能表达期望的世界变化，而不只是身体轨迹；
3. 接触力、接触模式和柔顺性是否能通过统一接口被上层策略操控。

因此，General S0 的目标不是一个更快的 Motion Tracker，而是一个同时具备 Dynamics Grounding 和 Expressive Interface 的 Interaction-Competent Controller。

