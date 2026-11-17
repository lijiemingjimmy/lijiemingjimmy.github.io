# HDMI：从人类视频学习人形机器人全身交互控制

> **论文**：HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos  
> **团队**：CMU LeCAR Lab；Haoyang Weng、Yitang Li 等  
> **时间**：2025 年 9 月  
> **项目**：[HDMI](https://hdmi-humanoid.github.io/#/)  
> **论文**：[arXiv:2509.16757](https://arxiv.org/abs/2509.16757)  
> **代码**：[LeCAR-Lab/HDMI](https://github.com/LeCAR-Lab/HDMI)

## 1. 核心问题：做出动作，不等于完成交互

HDMI 将普通人类视频转换成机器人与物体的参考运动轨迹，再通过强化学习同时追踪两者，让人形机器人学习开门、搬运和下跪抓取等接触丰富的全身技能。

传统人形机器人学习通常分别关注 Locomotion 和 Manipulation，但真实任务往往需要同时完成两者。例如，蹲下抓住箱子、站起并搬运，涉及复杂的机器人—物体耦合动力学，不能简单拼接走路策略和抓取策略。

**只模仿人体运动还不够，还必须让物体产生预期的运动。**机器人做出了推门姿势，不代表门真的被推开。这是 Robot–Object Co-Tracking 的出发点。

## 2. 整体流程：Video → Reference Motion → RL → Real Robot

### 2.1 从视频提取参考轨迹

输入是一段人类与物体交互的单目 RGB 视频。使用 GVHMR 恢复人体三维运动，再通过 LocoMujoco 进行动作重定向，将人体运动映射到 Unitree G1 的关节空间，同时处理物体轨迹和接触信息。

每帧参考状态可以概念性地表示为：

\[
s_t^{\mathrm{ref}}=
\left(s_t^{\mathrm{robot}},s_t^{\mathrm{object}},c_t\right)
\]

- \(s_t^{\mathrm{robot}}\)：机器人根节点位置、姿态和关节角；
- \(s_t^{\mathrm{object}}\)：物体位置、姿态和关节状态；
- \(c_t\)：期望接触标记。

此外还需要物体局部坐标系中的目标接触位置。

视频主要用于离线构建技能参考轨迹。HDMI 并不是把 RGB 视频直接输入 VLA，再输出机器人动作；实际控制策略通过仿真 RL 训练获得。

### 2.2 Robot–Object Co-Tracking

传统 Motion Tracking 重点奖励机器人身体接近参考姿态，HDMI 同时加入物体跟踪和交互奖励。概念上可写成：

\[
r_t=r_{\mathrm{robot}}+r_{\mathrm{object}}
+r_{\mathrm{interaction}}+r_{\mathrm{regularization}}
\]

实际各项具有不同权重。

| 奖励类别 | 关注的问题 |
| --- | --- |
| Robot Tracking | 是否完成预期的全身动作？ |
| Object Tracking | 门是否打开、箱子是否移动？ |
| Interaction | 是否在目标位置建立有效接触？ |
| Regularization | 动作是否平滑，是否满足关节和力矩等约束？ |

训练采用 PPO，并使用类似 DeepMimic 的参考状态初始化、相位条件和跟踪误差终止机制。

### 2.3 Sim-to-Real

技能在 Isaac Sim 中训练，再部署到真实 Unitree G1。展示包括连续开门穿越、手提箱操作、搬运泡沫垫及复杂全身动作。

真实实验依赖 Motion Capture 标记获取机器人和物体的位姿，尚不是完全依靠机载视觉完成感知的系统。

## 3. 三个关键技术设计

### 3.1 Unified Object Representation

HDMI 将物体状态和目标接触点转换到机器人 Root Frame，使策略主要学习相对几何关系，减少对世界坐标的依赖。

这种表示可以统一描述门、箱子和椅子等物体，但统一表示不等于已经获得对任意新物体的零样本泛化能力。

### 3.2 Residual Action Space

普通关节位置策略可以围绕默认姿态输出动作：

\[
\theta_t^{\mathrm{target}}=\theta_{\mathrm{default}}+a_t
\]

HDMI 则围绕当前参考姿态输出残差：

\[
\theta_t^{\mathrm{target}}=\theta_t^{\mathrm{ref}}+a_t
\]

例如训练下跪技能时，若动作始终围绕站立姿态探索，机器人容易突然站起或失去平衡。围绕参考下跪姿态学习修正量，可以让探索更接近目标动作，改善训练稳定性与收敛速度。

### 3.3 Unified Interaction Reward

提供的解读将接触奖励写为：

\[
R_{\mathrm{contact},i}
=\exp\left(-\frac{\lVert p_i-p_i^*\rVert^2}{\sigma_{\mathrm{pos}}}\right)
\min\left(
\exp\left(\frac{\lVert F_i\rVert-F_{\mathrm{thres}}}{\sigma_{\mathrm{frc}}}\right),1
\right)
\]

这里 \(p_i\) 和 \(p_i^*\) 分别表示实际与目标接触位置，\(F_i\) 表示接触力。奖励同时鼓励靠近目标接触点和形成足够的接触力。

接触力项在超过阈值后饱和，不会继续惩罚过大的力。因此，不能单凭这一项推断接触力受到了上界约束。

消融结果表明，Interaction Reward 对多数任务最终成功率的影响有限，但在参考轨迹不精确或需要准确建立接触的任务中更重要。

## 4. 实验结果与证据边界

| 项目 | 提供材料中的结果 |
| --- | --- |
| 仿真技能 | 14 个 |
| 真机技能 | 6 个 |
| 连续门穿越 | 67 次，约 34 分钟 |
| 仿真并行环境 | 4096 个 |

连续门穿越实验中，机器人用手推门、穿过后转身，再用脚踢门返回。实验还考察了起始位置扰动和地面条件变化。

结果支持 Residual Action 的有效性，同时说明接触奖励的收益依赖任务性质和参考轨迹质量。

需要区分三种不同的泛化结论：

1. **任务覆盖面**：同一方法可以训练多种技能，有实验支持。
2. **单技能鲁棒性**：某个策略能适应一定的位置或环境扰动，有部分实验证据。
3. **统一策略的任务泛化**：尚未证明。HDMI 仍采用 One Policy per Skill，并非统一的 Generalist Humanoid Policy。

## 5. 主要贡献与局限

HDMI 将 Human Motion Imitation 扩展到 Contact-Rich Humanoid–Object Interaction。其价值在于使用相对可复用的训练机制，从视频参考中获得可部署的全身交互技能，而不是提出全新的 RL 算法。

主要局限包括：

- 依赖参考轨迹提取和接触标注；
- 真机实验依赖外部位姿观测；
- 每个技能单独训练策略；
- 尚未解决高层任务规划、未知环境探索和复杂视觉反馈下的在线技能选择。

## 6. 与 VLA、General S0 和技能库的关系

HDMI 更接近 Low-Level Whole-Body Skill Learning。它提供的路线是：

```text
Human Video → Reference Motion → RL Skill → Skill Library
```

高层 VLA 或 Planning Agent 可以进一步决定何时调用哪个技能，但这属于系统扩展方向，不应当被当作 HDMI 已完成的能力。

与 General S0 的设想联系起来，HDMI 展示了物体感知和机器人—物体联合跟踪的重要性；从多个任务专用控制器走向统一、可操控的通用交互控制器，仍需要解决策略统一、接口设计和跨任务泛化。

一个自然的后续研究方向是：自动发现交互技能、生成参考轨迹、通过 RL 训练，再利用真实执行反馈判断可靠性，将通过验证的技能加入可复用技能库。

