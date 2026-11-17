# SmoothRL：异步执行下的机器人在线强化学习

> **论文**：SmoothRL: Online Reinforcement Learning During Asynchronous Execution  
> **机构**：Astribot（星尘智能）  
> **时间**：2026 年 8 月  
> **arXiv**：2608.29768

## 一句话总结

SmoothRL 解决的是 VLA 在异步推理（asynchronous inference）过程中如何进行在线强化学习的问题。它没有发明一种全新的 RL 算法，而是把基于 value gradient 的 Actor-Critic 更新和真实机器人的异步 Action Chunk 执行机制对齐。

核心原则是：**Critic 评估真正影响回报的动作，Actor 的 Q-value 梯度只更新当前策略实际控制的动作。**

## 1. 异步执行为什么会改变 RL 问题

VLA 根据观测一次生成长度为 \(H\) 的 Action Chunk：

\[
\pi_\theta(o_t)\rightarrow(a_t,a_{t+1},\ldots,a_{t+H-1})
\]

假设 VLA 每次推理需要 200 ms，而机器人控制频率为 50 Hz。同步执行时，机器人执行完一段动作后必须等待下一次推理，快速抓取、抛掷等任务就可能因为停顿而失败。

异步执行让机器人执行当前 Chunk 的同时并行推理下一段 Chunk。但新 Chunk 返回时，时间窗口已经向前移动，因此它不一定会完整执行：

- **Committed region**：推理期间仍由上一段 Chunk 提供的动作；
- **Execution region**：当前 Chunk 真正接管机器人的动作；
- **Discarded region**：尚未执行就被下一段 Chunk 覆盖的动作。

如果把整段 Chunk 都当成当前策略的有效决策，RL 就会优化一部分从未执行的动作，造成训练与部署不一致。

## 2. 固定 latency budget 和三段动作区间

SmoothRL 为异步推理设置固定的 latency budget \(n\)。论文使用 \(H=32\) 的 Action Chunk，并划分为：

\[
[0,n)\quad [n,2n)\quad [2n,H)
\]

分别对应 Committed、Execution 和 Discarded 区间。

固定 \(n\) 的作用，是把波动的推理延迟转换成确定的执行窗口。较早返回的推理结果会等到预定交接时刻再生效，从而让每个 transition 的动作归属明确。

这是一种用调度约束换取 RL 可定义性的设计：系统牺牲了一部分调度自由度，但得到了稳定的 Bellman backup 和 policy gradient。

## 3. Execution-aware Value Gradient

普通连续动作 Actor-Critic 最大化：

\[
\max_\theta\;\mathbb E_s[Q_\phi(s,\pi_\theta(s))]
\]

在 SmoothRL 中，Critic 需要同时看到已经提交的动作和当前执行的动作：

\[
Q_\phi\left(s,\tilde a_{[0,n)},a_{[n,2n)}\right)
\]

其中 \(\tilde a\) 是已经发给机器人的动作，当前策略不能修改。Actor 的目标写成：

\[
\max_\theta Q_\phi\left(
 s,\operatorname{sg}(\tilde a_{[0,n)}),
 \pi_\theta(s)_{[n,2n)}
\right)
\]

\(\operatorname{sg}\) 表示 stop-gradient。

这意味着：

1. Critic 观察完整的有效动作区间，以准确估计真实 return；
2. Actor 的 Q-gradient 只通过 Execution region，避免优化未执行动作；
3. Discarded region 不参与这部分 value-gradient 更新。

TD target 可以从执行窗口结束后的状态开始计算，并结合后续动作与折扣回报。实现中还使用 REDQ 风格的 Critic ensemble 计算 target，以改善 off-policy value estimation 的稳定性。

## 4. Residual RL：冻结 VLA，只学习动作修正

实验没有端到端更新整个 \(\pi_{0.5}\)，而是采用 RLT 风格的残差策略：

\[
\bar a=\pi_{\text{base}}(o),\qquad
\Delta a=f_\theta(z,s,\bar a),\qquad
a=\bar a+\Delta a
\]

其中：

- \(\pi_{\text{base}}\)：冻结的预训练 VLA；
- \(z\)：从 VLA 提取的 RL token representation；
- \(f_\theta\)：可训练的轻量 residual actor；
- \(\Delta a\)：对基础动作的修正。

因此整体形态是：

**Frozen VLA + Residual RL + Execution-aware Asynchronous Training**。

这种设计适合基础 VLA 已经接近正确、但动态性、接触精度或动作连续性仍有缺陷的任务。如果模型发生严重感知错误、选错 skill 或生成完全错误的动作模式，有限幅度的 residual correction 很难从根本上修复。

## 5. Actor loss

论文中的 Actor loss 可以概括为：

\[
\mathcal L_{\text{actor}}
=-Q_\phi(s,\operatorname{sg}(\tilde a),a_{\text{exec}})
+\lambda_{\text{BC}}\lVert a-a_{\text{target}}\rVert^2
+\lambda_{\text{smooth}}\sum_{k=1}^{3}w_k\lVert\Delta^k a\rVert^2
\]

三项分别是：

- **Q maximization**：利用 value gradient 提高任务回报；
- **BC regularization**：避免策略过度偏离预训练 VLA，人工接管的 Chunk 可以作为监督目标；
- **Smoothness regularization**：约束动作的一阶、二阶和三阶差分，减少速度、加速度和 jerk 的剧烈变化。

Gradient truncation 只限制 Q-value 的梯度路径。BC 和 smoothness regularization 仍可按照各自的动作范围计算，因此既能避免未执行动作干扰策略更新，也能保持整段动作的连续性。

## 6. 真机结果

SmoothRL 在 Astribot S1 上测试了动态抛掷、笔帽操作和开盒任务。经过约 250 个 episode 的在线 RL 后，成功率变化如下：

| 任务 | Frozen Base | SmoothRL |
| --- | ---: | ---: |
| Dynamic Tossing | 39% | 94% |
| Pen Capping | 8% | 83% |
| Box Opening | 30% | 90% |

这三个任务分别考察动态连续性、双臂相对定位和毫米级接触操作。论文还报告了一次真实抛掷 rollout：末端加速度 RMS 降低约 52%，jerk RMS 降低约 47%。

评估配置和样本量较小，训练中也存在人工干预，因此这些结果主要说明 SmoothRL 在特定真机任务上有效，尚不足以单独证明方法具有普遍的统计稳定性和跨任务泛化能力。

## 7. 论文真正的贡献

SmoothRL 的关键贡献，是把 asynchronous execution 正式写进 RL 的状态表示、Critic 输入、TD target 和 Actor gradient 中。

同步执行时，模型输出的动作块通常可以看成一个完整决策；异步执行时，Chunk 中每个位置拥有不同的决策归属。固定 latency window 与 execution-aware gradient truncation 让这种归属变得明确。

它和普通 chunk-level RL 的主要区别，不是换了一个更复杂的 Actor-Critic，而是重新定义了：**哪些动作应该被当前 RL 更新优化。**

## 8. 局限与后续方向

1. 论文没有完全隔离 Gradient Truncation、Residual RL、Smoothness Regularization 和人工干预各自的贡献，尤其缺少相同异步 rollout 下“不做 execution-aware truncation”的直接对比。
2. 固定 latency budget 是较强假设。推理延迟存在长尾时，可以根据调度日志使用动态 action mask、可变长度 execution region 和对应的 TD transition。
3. Residual RL 主要适合基础策略已接近正确的任务，不能替代对严重 perception failure 或 skill selection failure 的修复。
4. 每个任务的训练运行次数和评估样本量有限，跨设备、跨任务的稳定性仍需更多实验。

## 9. 总结

SmoothRL 可以看作一套面向异步机器人执行的 Online RL 框架。生成出来的动作不一定都会执行，真正执行的动作也不一定都由当前策略决定。它把这两个事实写进 Critic 和 Actor 的更新路径，使 VLA 能在保持异步流畅性的同时进行在线策略改进。
