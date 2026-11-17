# LLM-RL 的 Baseline 评测争议：分数提升能否证明能力提升？

> **原文**：Incorrect Baseline Evaluations Call into Question Recent LLM-RL Claims  
> **作者**：Nikhil Chandak、Shashwat Goel、Ameya Prabhu  
> **时间**：2025 年 5 月 29 日  
> **链接**：[原文](https://safe-lip-9a8.notion.site/Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37)

文中的早期汇总表保留评论当时的口径，并单独说明模型与版本差异；本文没有重新运行全部训练与评测。

## 1. 核心问题：训练收益是否被低估的 Baseline 放大

这是一篇评测方法学评论，而不是提出新 RL 算法的论文。它考察了七篇 LLM 强化学习相关工作，指出多篇论文报告的 Pre-RL baseline 明显低于官方或其他标准化评测结果。

由此产生一个基本问题：**RL 后的 benchmark score 更高，究竟说明模型获得了新的推理能力，还是更适应了当前的评测协议？**

如果基础模型没有被充分、合理地评测，论文展示的提升可能混合了推理能力、格式遵循、采样设置和测试时计算量的变化。

这类异常足以要求重新检查实验，但不能单独证明训练无效。

## 2. 关键证据：MATH-500 上的 Baseline 差异

以下数字对应评论原文的早期汇总，单位为准确率百分比；不是各论文最新版本的统一重测结果。

| 工作 | 对应起点模型 | 论文 Pre-RL | 论文 Post-RL | 外部参考 Pre-RL |
| --- | --- | ---: | ---: | ---: |
| Spurious Rewards | Qwen2.5-7B | 41.6 | 70.1 | 64.6 |
| RL with 1 Example | DeepSeek-R1-Distill-1.5B | 71.9 | 78.0 | 83.9 |
| Entropy Minimization | Qwen2.5-7B | 43.8 | 70.8 | 64.6 |

来源：[评论原文的 MATH-500 汇总表](https://safe-lip-9a8.notion.site/Incorrect-Baseline-Evaluations-Call-into-Question-Recent-LLM-RL-Claims-2012f1fbf0ee8094ab8ded1953c15a37)。这里保留三组有完整数值的案例；不能据此推断其余工作的结果。

**外部参考与原论文可能使用不同协议，因此这不是严格的配对实验。**评论引用 Qwen 官方数据或 Sober Reasoning 等外部评测，它们也不是脱离协议的唯一“真实准确率”。

还需要明确：表中 41.6% 对应 **Qwen2.5-7B**，不能与 Spurious Rewards v2 中重点分析的 **Qwen2.5-Math-7B、49.4%** 混为一谈。后者在 v2 Figure 1 中的真实奖励训练后成绩为 78.5%。

### Spurious Rewards

原论文所报告的提升是：

\[
70.1-41.6=28.5\text{ 个百分点}
\]

若仅将 Post-RL 数字与外部 Pre-RL 参考相减：

\[
70.1-64.6=5.5\text{ 个百分点}
\]

这个差异说明 baseline 的选择会大幅影响读者对收益的判断，但 **5.5 个百分点并不是经过统一协议验证的真实训练效应**。

### RL with 1 Example

论文 Post-RL 为 78.0%，低于外部 Pre-RL 的 83.9%。这不能直接说明 RL 造成退化，因为 prompt、decoding 和 evaluation harness 可能不同；它说明现有数字不足以排除“基础模型本来就能达到类似水平”的解释。

## 3. 四类影响训练收益归因的评测因素

### 3.1 Format：学会答题格式不等于学会解题

假设模型回答 `The answer is 42.`，而评测器只能识别 `\boxed{42}`。即使数学答案正确，也可能因提取失败被判错。

在将未成功提取的正确答案计为错误的简化评测中：

\[
A_{\mathrm{obs}}=P(C\cap F)=P(C)P(F\mid C)
\]

其中 \(C\) 表示答案正确，\(F\) 表示答案被成功提取。

一个说明机制的假设例子：

- 训练前真实解题正确率为 80%，正确答案提取率为 60%，观测准确率为 48%；
- 训练后解题正确率仍为 80%，但提取率变为 95%，观测准确率达到 76%。

分数提高了 28 个百分点，但这个例子中并没有新增数学解题能力。这不是原文实验数据，而是对混杂机制的演示。

Format reward，甚至某些看似与正确性无关的训练信号，可能通过改变输出行为提高分数；是否实际如此，仍需实验验证。

仅增加 format-reward RL 对照还不充分，因为该对照也更新了权重。还应加入**不训练、只改进 prompt、输出格式要求和 parser 的 baseline**。

### 3.2 Temperature：输出分布不只取决于权重

采样温度影响 token 分布：

\[
P_T(x_i)=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}
\]

当温度趋近于零时，采样趋近 greedy decoding。低温可能在某些长推理任务中导致重复或固定的错误路径，但不能据此推出低温对所有模型和任务都更差。

评论质疑了一些工作使用 0 或 0.1 等低温设置，而这些设置可能偏离模型发布方建议。

严谨做法是对 Pre-RL 和 Post-RL 使用相同的温度扫描，例如：

\[
T\in\{0,0.1,0.3,0.6,1.0\}
\]

比较完整性能曲线，而不是只选择一个有利于某个模型的设置。

### 3.3 Output Length：Token Budget 是测试时计算预算

模型正确率可以概念性地写成：

\[
P(\mathrm{correct})=f(\theta,\mathrm{prompt},T,L,\ldots)
\]

这里 \(L\) 是最大生成长度。更长的预算可能帮助解题，也可能增加重复和无效推理，不能简单假设越长越好。

如果训练前后长度上限不同，或训练改变了模型实际使用的 token 数，仅看最终准确率就难以区分参数更新、计算预算和答案格式的作用。

应同时报告准确率与实际 token 消耗。相同的最大长度限制，也不保证相同的实际计算成本。

### 3.4 Small Benchmarks：小幅差异需要统计分析

MATH-500 有 500 道题。在独立二项近似下，若准确率为 70%，单模型准确率的标准误约为：

\[
\mathrm{SE}=\sqrt{\frac{0.7(1-0.7)}{500}}\approx 2.05\%
\]

对应的近似 95% 置信区间半宽约为 4 个百分点。

这只是单模型准确率不确定性的近似，不能直接作为两个模型差值的显著性检验。对于相同题目上的模型比较，应优先使用 McNemar test 或 paired bootstrap 等配对分析；生成随机性还需要多次采样或多个随机种子来评估。

## 4. 七篇工作的主要争议

下表整理的是提供材料中对评论观点的概括，不代表已经对各论文结论作出最终裁定。

| 工作 | 研究主张或训练信号 | 评论关注的问题 |
| --- | --- | --- |
| Spurious Rewards: Rethinking Training Signals in RLVR | 随机、格式或错误标签奖励也能改善表现 | Qwen baseline 偏低，难以确定实际推理收益 |
| Maximizing Confidence Alone Improves Reasoning | 用 confidence 或 entropy 训练 | 答案格式和 baseline 评测可能解释部分收益 |
| Reinforcement Learning for Reasoning in Large Language Models with One Training Example | 单个样本即可产生 RL 收益 | 低温 decoding 和评测设置影响收益解释 |
| Learning to Reason without External Rewards | 使用 self-certainty 训练信号 | 基础模型经过合理 few-shot 等设置后是否已能达到类似表现 |
| VeriFree: Reinforcing General Reasoners without Verifiers | 不依赖显式答案 verifier | Qwen3 的低温评测可能偏离适宜推理设置 |
| The Unreasonable Effectiveness of Entropy Minimization | 熵最小化改善推理分数 | 报告的 Pre-RL 与外部参考存在较大差距 |
| Can Large Reasoning Models Self-Train? | 使用 self-consistency 等自身信号 | Pre-RL 低于其他标准化评测，训练增益归因不明确 |

这些问题要求统一协议复测，不能直接推出七种方法的改善全部消失。

## 5. 对批评本身的关键限制

LessWrong 原始讨论中，Neel Nanda 提出关键质疑：**更好的评测协议可能同时提高 Pre-RL 和 Post-RL 的成绩。**[来源：原始讨论](https://www.lesswrong.com/posts/p8rcMDRwEGeFAzCQS/incorrect-baseline-evalutions-call-into-question-recent-llm)

例如原协议下：

\[
A_{\mathrm{pre}}=40\%,\qquad A_{\mathrm{post}}=70\%
\]

外部协议测得 Pre-RL 为 65%，不能据此认定训练收益只剩 5 个百分点；同一外部协议下 Post-RL 也可能达到 85%。

正确比较应为：

\[
\Delta_{\mathrm{RL}}(e)
=A(\theta_{\mathrm{post}},e)-A(\theta_{\mathrm{pre}},e)
\]

其中 \(e\) 包含 prompt、temperature、parser、token limit 等完整评测配置。

评论作者认可这一限制，并解释当时缺少相关 Post-RL 开放权重，因此无法统一复测。这描述的是当时的复核条件，不代表这些项目今天仍未提供权重。

因此，这篇评论最有力的结论是：**现有证据不足以支持部分强主张，需要进一步验证。**它给出了评测不一致的证据，但没有为每一种方法提供完整的配对训练效应估计。

## 6. 两个不同的目标估计量

### 6.1 固定评测协议下的训练收益

\[
\Delta_{\mathrm{train}}(e)
=A(\theta_{\mathrm{RL}},e)-A(\theta_0,e)
\]

它回答：在相同评测条件下，参数更新是否改善表现？

### 6.2 允许评测配置调优后的净收益

\[
\Delta_{\mathrm{best}}
=\max_{e\in E}A(\theta_{\mathrm{RL}},e)
-\max_{e\in E}A(\theta_0,e)
\]

它回答：给基础模型充分的 prompt 和 decoding 调优机会后，RL 是否仍有额外价值？

两者不是同一个问题。实践中，配置选择应在独立验证集上完成，并给予两类模型可比的搜索和计算预算，避免直接在测试集上选最高分。

即使 \(\Delta_{\mathrm{best}}\approx 0\)，RL 仍可能提高效率、减少对 prompt 的敏感度，或将昂贵的测试时计算转移到模型参数中。

反过来，\(\Delta_{\mathrm{train}}>0\) 也不自动证明出现了新的 reasoning capability。

## 7. 能力主张需要哪些额外证据

若论文主张新增推理能力，而非仅提高某个 benchmark 分数，还应进一步检验：

- 在未见过的任务分布上是否改善；
- 控制 token budget 和实际计算成本后是否仍有收益；
- 增益是否主要来自答案提取或格式遵循；
- 是否出现新的成功解题路径，而非主要提高已有路径的采样概率；
- 结论是否对 prompt、随机种子和温度保持稳健。

提高正确答案的采样概率（elicitation），与扩展模型可解任务的能力边界，是不同层面的结论。后者需要更强的证据，不能仅由单一准确率差值推出。

## 8. 对 Embodied Agent Evaluation 的启发

这篇评论的论证方式可以借鉴到具身智能评测中：先发现跨论文的异常，再检查是否存在更简单的解释，最后提出能够区分这些解释的受控实验。

具身 Agent 的观测成功率同样依赖多个因素：

\[
S_{\mathrm{obs}}=f(
\mathrm{policy},\mathrm{task\ distribution},\mathrm{environment},
\mathrm{harness},\mathrm{execution\ budget})
\]

当任务分布、环境、工具接口或执行预算不同，排行榜上的高分未必足以支撑关于长程规划、失败恢复或泛化能力的强主张。

对 Beyond the Score: Rethinking the Evaluation of Embodied Agents 这类研究，值得借鉴的是以下论证顺序：

1. 明确社区正在接受的能力结论。
2. 找出分数与结论之间的解释歧义。
3. 用具体案例和跨论文数据证明歧义值得关注。
4. 统一协议，对目标能力进行受控比较。
5. 用过程指标与规模化实验检验结论是否改变。

这样可以从指出 baseline 问题，推进到建立更完整的评测框架。这里是研究延伸，并非评论本身已完成的具身智能实验。

## 9. 研究规范与最终判断

可靠的复核需要公开 checkpoint、完整评测配置和逐样本预测结果，使第三方能够区分格式失败、推理错误、截断及采样波动。

这篇评论的价值在于指出：观测到的 benchmark improvement 不能未经分析就归因于能力提升。它的局限在于缺少所有模型在统一协议下的配对复测。

一个可信的研究贡献，需要证明方法带来的改善，并检验这种改善能否被更简单的 baseline、测试时计算或评测协议差异解释。

## 10. 与另外两篇材料一起阅读时，需要防止哪些混用

### 10.1 评论时间与论文版本不是一回事

LessWrong 链接帖发表于 **2025-05-29**。Spurious Rewards 的 arXiv 首版发布于 **2025-06-12**，v2 更新于 **2026-02-25**。评论的 Spurious Rewards 链接当时指向 GitHub 上的公开材料。

因此，不能因 arXiv 首版日期较晚就断言评论不存在；也不能把 2025 年的评论说成已经逐条反驳了 2026 年 v2 新增的全部机制实验。[来源：LessWrong 发帖记录](https://www.lesswrong.com/posts/p8rcMDRwEGeFAzCQS/incorrect-baseline-evalutions-call-into-question-recent-llm)、[arXiv 版本记录](https://arxiv.org/abs/2506.10947)

### 10.2 两类批评提供不同的证据

| 材料 | 主要证据 | 能支持的结论 |
| --- | --- | --- |
| 本文讨论的 Baseline 评论 | 跨论文、跨协议的数字不一致 | 原有能力归因不充分，需要复核 |
| A Sober Look | 统一流程下的模型复测与逐题统计 | 部分原报告收益缩水，部分方法仍保留收益 |
| Spurious Rewards | 改变奖励、跨模型实验与行为干预 | 正确性奖励并非某些模型分数提升的唯一解释 |

这些证据不能相互替代。尤其不能用“外部基线更高”直接计算每一种 RL 方法的真实净收益，也不能把某些模型上的随机奖励现象推广成所有 RL 的性质。

### 10.3 批评也需要给出可检验的命题

“当前证据不足以支持新增推理能力”与“该方法已被证明完全无效”对应不同强度的结论；后者需要更完整的受控实验。

同样，格式遵循改善可以有实际使用价值。需要重新界定的是论文声称的贡献：格式、效率、现有能力的调用，还是新问题上的求解能力。

相关笔记：[A Sober Look](A-Sober-Look-Reasoning-Evaluation.md)、[Spurious Rewards](Spurious-Rewards.md)、[综合核查：打假](打假.md)。
