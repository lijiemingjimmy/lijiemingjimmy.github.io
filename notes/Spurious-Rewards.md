# Spurious Rewards：随机奖励为什么也能提高部分模型的数学分数

> **论文**：Spurious Rewards: Rethinking Training Signals in RLVR  
> **作者**：Rulin Shao、Shuyue Stella Li、Rui Xin、Scott Geng 等  
> **机构**：University of Washington、Ai2、UC Berkeley 等  
> **版本**：arXiv:2506.10947v2，2026-02-25；arXiv 首版发布于 2025-06-12  
> **来源**：[论文与版本记录](https://arxiv.org/abs/2506.10947)、[v2 全文](https://arxiv.org/html/2506.10947v2)  
> **整理与核对**：2026-10-20；基于提供的解读，核对 Figure 1、关键表格和附录。本文没有重新训练模型。

## 1. 论文想区分的两种解释

RLVR 指利用可验证奖励进行强化学习，例如数学最终答案正确得 1 分，错误得 0 分。通常，人们会把训练后的准确率提升解释为：正确性奖励教会了模型更好的推理。

这篇论文问：如果奖励与答案正确性无关，甚至刻意奖励某些错误答案，模型还能提高多少？

作者在部分 Qwen 模型上观察到明显提升，并分析了两个可能相互作用的因素：

- **优化过程**：GRPO 的 clipping 会改变随机奖励下的有效更新行为。
- **模型先验**：预训练中已有的某些行为恰好与正确性相关，进一步放大这些行为就可能提高分数。

这使得“准确率提高”不足以单独证明“正确性奖励创造了新的求解能力”。但它没有证明真实奖励毫无价值，也没有证明随机奖励对所有模型有效。

## 2. 最关键的实验数字

Figure 1 对 Qwen2.5-Math-7B 报告了以下 MATH-500 结果，训练前准确率为 49.4%。表中 Post-RL 按“49.4 加上图示增益”计算。

| 训练信号 | 训练后准确率 | 相对初始模型增益 |
| --- | ---: | ---: |
| Ground Truth | 78.5% | +29.1 pp |
| Majority Vote | 76.5% | +27.1 pp |
| One-Shot RL | 75.4% | +26.0 pp |
| Format | 63.2% | +13.8 pp |
| Incorrect Label | 73.5% | +24.1 pp |
| Random | 70.8% | +21.4 pp |

Figure 1 展示 300 steps 后的结果；Appendix A.6 说明图中的终值采用跨十个训练步平滑后的指标。因此，带一位小数的终值不应都被理解为“单个 checkpoint 在 500 道题上一次评测的整数答对数”。[来源：Figure 1 与 Appendix A.6](https://arxiv.org/html/2506.10947v2)

随机奖励获得的增益约为真实奖励增益的：

\[
\frac{21.4}{29.1}\approx73.5\%
\]

这是这个实验中两种**增益幅度之比**，不能解释为“随机奖励掌握了真实奖励 73.5% 的推理能力”。

还要区分模型：Qwen2.5-7B 的初始值是 41.6%，Qwen2.5-Math-7B 的初始值是 49.4%。两者不是同一个 checkpoint。2025 年的 Baseline 批评文章使用了前者的部分早期数字，不能直接与本表拼接。

## 3. 奖励究竟怎样定义

| 奖励 | 给 1 分的条件 | 包含的监督信息 |
| --- | --- | --- |
| Ground Truth | 最终答案匹配真实标签 | 正确性监督 |
| Majority Vote | 匹配训练前模型对该题 64 次采样得到的多数答案 | 模型生成的伪标签，可能错误 |
| Format | 包含至少一个非空的 `\boxed{}` | 格式要求 |
| Incorrect Label | 匹配多数投票选出的某个特定错误答案 | 刻意错误的标签，仍与模型先验有关 |
| Random | 独立抽取 Bernoulli 随机变量 | 无答案正确性信息 |

Incorrect Label 不是“任何错答案都奖励”。作者先筛出多数投票标签错误的训练题，再奖励匹配该错误标签的输出。因此，这一条件还涉及训练子集选择，不能完全等同于在同一数据集上只替换一个 reward 函数。

主要随机奖励条件为：

\[
r(x,y)\sim\operatorname{Bernoulli}(0.5)
\]

回答正确可能得 0，回答错误也可能得 1。其他非零奖励概率的实验得到类似趋势；常数零奖励没有产生改进。[来源：§2.2 与 Appendix B](https://arxiv.org/html/2506.10947v2#S2)

训练使用 DeepScaleR 数据和 GRPO。提供解读对应的配置为学习率 \(5\times10^{-7}\)，每批 64 个 prompt，每题采样 16 条 rollout，共 1024 条回答，mini-batch 为 128 条；不使用 KL 或 entropy loss。

因此，无信息奖励不等于完全没有任务信息：训练依然使用数学问题、预训练权重以及确定的优化过程。

## 4. 为什么随机奖励不应直接产生正确性梯度

理想的 on-policy policy gradient 为：

\[
\nabla_\theta J
=\mathbb E_{y\sim\pi_\theta}
[r(x,y)\nabla_\theta\log\pi_\theta(y\mid x)]
\]

若奖励独立于回答，则在通常的正则条件下：

\[
\mathbb E_{r,y}[r\nabla_\theta\log\pi_\theta(y\mid x)]
=\mathbb E[r]\,
\mathbb E_y[\nabla_\theta\log\pi_\theta(y\mid x)]
=0
\]

这利用了 score function identity。期望为零，并不表示有限样本 SGD 每次更新都为零；它也不能直接说明训练动态中的模型表现一定不变。

GRPO 实际使用有限样本、组内 advantage、重复 mini-batch 更新和 clipped surrogate，不能直接用这个理想结果替代对具体算法的分析。

## 5. GRPO clipping 怎样改变随机更新

### 5.1 优化目标

省略 token、样本求和和正则项后，常见 GRPO 目标可写为：

\[
J(\theta)=\mathbb E\left[
\min\left(\rho\widehat A,
\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)\widehat A\right)
\right]
\]

其中：

\[
\rho_t=\frac{\pi_\theta(y_t\mid x,y_{<t})}
{\pi_{\mathrm{old}}(y_t\mid x,y_{<t})},
\qquad
\widehat A_i=\frac{r_i-\bar r}{\sigma_r}
\]

实际实现还需要处理组内奖励标准差为零等情况。这里用于说明 clipping 的作用，不作为完整实现规范。

### 5.2 一个有限定条件的偏置推导

固定当前策略与行为策略，假设随机 advantage 独立于 rollout，且满足 \(\mathbb E[\widehat A]=0\)。记：

\[
c=\mathbb E[\widehat A\mathbf1_{\widehat A>0}]>0
\]

对单个 token 的 surrogate，在不讨论边界点的情况下：

\[
\mathbb E_{\widehat A}[\nabla_\theta J]
=c\begin{cases}
\nabla_\theta\rho,&\rho<1-\epsilon,\\
0,&1-\epsilon<\rho<1+\epsilon,\\
-\nabla_\theta\rho,&\rho>1+\epsilon.
\end{cases}
\]

直观上，clipping 会对超出允许区间的概率比产生回拉偏置。这里保留 \(\nabla\rho\)；若改写成 \(\nabla\log\pi_\theta\)，应保留 \(\rho\) 因子，或明确它被合并到正的权重中，不能无条件当作全局常数删除。

**当策略尚未变化，\(\pi_\theta=\pi_{\mathrm{old}}\)、\(\rho=1\) 时，上述期望偏置仍为零。**有限样本更新先使策略发生移动，后续使用同一批 rollout 更新时，部分 token 才可能触及 clip 边界。

这不是“随机奖励含有隐藏的正确答案信息”，而是优化目标对随机扰动具有非对称的处理方式。完整训练中，参数本身依赖先前随机奖励，因此固定策略下的推导也不能直接充当整个训练过程的收敛证明。[来源：§4 与 Appendix B.1](https://arxiv.org/html/2506.10947v2#S4)

### 5.3 高概率 token 的直观例子

设 \(\epsilon=0.2\)。若某 token 原始概率是 0.85，上侧概率边界为：

\[
0.85(1+0.2)=1.02
\]

概率不能超过 1，因此它不会触及这一上边界。若原始概率是 0.02，上侧边界只有 0.024，更容易触发上侧 clipping。

这个例子说明了更新空间的不对称，但 token 通过 softmax 和共享参数耦合，不能据此证明任意 GRPO 更新都会单调提高所有高概率 token 的概率。

### 5.4 消融证据与局限

作者比较了三种使 clipping 不起作用的设置：关闭 clipping、将 mini-batch 增大到整个 rollout batch，以及减小 rollout batch，使每批数据只更新一次。

有 clipping 的随机奖励训练更稳定地改善分数；关闭 clipping 后，不同种子的行为不稳定。但部分无 clipping 运行仍会偶然上升，而且 batch 改动也改变了更新过程。因此，实验支持 clipping 是重要机制，还未完全排除其他机制。

## 6. 为什么模型先验如此关键

### 6.1 Code reasoning 是可观察的行为代理

Qwen2.5-Math-7B 在训练前已有 65.0% 的回答包含 Python 相关内容。这里的 code reasoning 是模型自回归生成代码及其声称的输出，**实验没有给模型提供实际 Python 解释器**。

论文 Table 1 报告，Qwen2.5-Math-7B 含代码回答准确率为 60.9%，自然语言回答为 35.0%；随机奖励训练后的代码比例最高达到 95.6%。这些观察支持“增加既有有效行为的出现频率”这一解释。[来源：§5 与 Table 1](https://arxiv.org/html/2506.10947v2#S5)

不过，原文存在一个需要保留的数字冲突：v2 引言将无代码回答准确率写为 **28.0%**，Table 1 则写为 **35.0%**。若这些统计来自同一批输出，按代码占比加权应重构总体准确率：

\[
0.65\times60.9\%+0.35\times35.0\%=51.835\%
\]

它与 49.4% 初始值不吻合；使用 28.0% 则得到：

\[
0.65\times60.9\%+0.35\times28.0\%=49.385\%\approx49.4\%
\]

这说明相关数字需要结合原始输出核查，可能涉及笔误或未说明的统计口径。本文不替作者擅自修正，也不把这一不一致上升为对研究诚信的判断。

### 6.2 策略重加权足以改变整体准确率

把 \(z\) 看作潜在推理策略，可写成：

\[
\pi_\theta(y\mid x)
=\sum_z\pi_\theta(z\mid x)\pi_\theta(y\mid x,z)
\]

在固定题目分布上：

\[
\operatorname{Acc}(\pi_\theta)
=\mathbb E_x\sum_z\pi_\theta(z\mid x)
\Pr_\theta(\mathrm{correct}\mid x,z)
\]

即使策略内部正确率不变，只改变策略选择概率也可能提高整体分数。

例如，假设代码策略准确率为 70%，语言策略为 30%，代码选择率从 50% 提高到 90%，整体准确率就从 50% 变为 66%。这是解释机制的假设例子，不是原文实验。

现实中 RL 也会改变策略内部的条件输出分布，所以这个分解不能证明所有收益都来自策略选择。

### 6.3 直接干预比相关性更有说服力

作者用提示词要求模型以 Python 解题，得到以下 MATH-500 结果：

| 模型 | 原始准确率 | 代码提示后 | 差值 |
| --- | ---: | ---: | ---: |
| Qwen2.5-Math-1.5B | 36.2% | 60.4% | +24.2 pp |
| Qwen2.5-Math-7B | 49.4% | 64.4% | +15.0 pp |
| Llama3.2-3B-Instruct | 36.8% | 8.2% | −28.6 pp |

增加代码行为对不同模型可能产生相反效果，因此 code reasoning 的价值取决于模型已有的能力与习惯，不能当成通用增强方法。[来源：Table 2](https://arxiv.org/html/2506.10947v2#S5)

提供解读还引用 Appendix H：对 Qwen2.5-Math-7B，平均增益中约 58.3% 来自被归为 Language→Code 的题目子集。这是子集贡献统计，不是已经识别出的“代码策略的因果贡献比例”。

## 7. Prompt 敏感性对解释结果有什么影响

Appendix I 对同一个 Qwen2.5-Math-7B 给出：

| Prompt | MATH-500 准确率 | 可解析回答比例 |
| --- | ---: | ---: |
| Qwen Default | 49.4% | 78.9% |
| Math Problem | 55.8% | 72.1% |
| SimpleRL-Zoo | 63.2% | 85.4% |
| Sober | 61.6% | 93.1% |
| Lorem Ipsum 占位文本 | 68.8% | 84.1% |

无关文本相对默认提示词提高了 19.4 个百分点。作者同时说明，随机挑选的其他 prompt 并不总有这种收益。[来源：Appendix I，Table 5](https://arxiv.org/html/2506.10947v2#A9)

这个结果对双方都有约束：研究 RL 时需要加入合理的无训练调优基线；批评 RL 时也不能把所有改善都归结为格式，因为最高解析率与最高正确率不对应。

此外，本表中的 Sober prompt 在作者当前实现里得到 61.6%，不能直接等同于另一篇论文在其完整软件和解码配置下测出的 64.3%。仅复制提示词并不等于复现完整协议。

## 8. 这篇论文支持哪些结论

支持较强的结论是：在某些 Qwen 模型及作者设置中，没有正确性信息的奖励也能产生明显 benchmark 改善；因此，分数改善不充分识别奖励中的正确性监督贡献。解释这一现象时，需要同时考虑优化偏置和模型先验。

它仍有以下边界：

- 真实奖励取得更高成绩，并出现超出单纯增加代码频率的改善迹象。
- Llama、OLMo 等家族对同类信号的反应不同；跨家族失败也未经过全面超参数优化，不能写成永远不可能有效。
- 用 `python` 字符串识别策略较粗糙，不能直接等同于内部推理算法。
- Prompt、答案提取和训练动态相互作用，不能由单一终值完成因果归因。

## 9. 对后续 RL 与具身研究的启发

奖励设计研究可以加入 Random、Format-only、无需更新权重的 prompt/decoding 调优，以及可比预算下的跨 backbone 对照。不同对照回答不同问题，不宜认为只超过随机奖励就已经证明全部机制主张。

对 Agent/VLA，同样需要区分“更常选择已有成功行为”与“新增规划或恢复能力”。可以通过固定初始状态、分析动作策略分布、检验新布局和主动扰动来区分解释。

论文没有证明随机奖励对机器人控制有效。具身方向值得迁移的是实验设计和归因要求，而非直接迁移这一训练现象。

相关笔记：[A Sober Look](A-Sober-Look-Reasoning-Evaluation.md)、[Baseline 争议](LLM-RL-Baseline-Evaluation-Critique.md)、[综合核查：打假](打假.md)。
