# ProRL 论文解读：训练时长与探索控制怎样改变能力覆盖

**论文：** ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models  
**作者：** Mingjie Liu、Shizhe Diao、Ximing Lu、Jian Hu、Xin Dong、Yejin Choi、Jan Kautz、Yi Dong  
**发表：** NeurIPS 2025  
**资源：** [论文](https://arxiv.org/abs/2505.24864) · [PDF](https://arxiv.org/pdf/2505.24864) · [会议收录](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html) · [发布模型](https://huggingface.co/nvidia/Nemotron-Research-Reasoning-Qwen-1.5B)

**一句话概括：ProRL 报告，延长 RL 训练并结合 KL 控制、参考策略重置和多样任务，可以扩大有限采样预算下的可解题覆盖，为 RLVR 的边界讨论提供了不同训练条件下的证据。**

## 一、它在回应什么问题

[Limit of RLVR](Limit-of-RLVR.md) 观察到，部分训练设置提高 pass@1 的同时，大 k 的 pass@k 没有同步改善。ProRL 则追问：这一现象会不会取决于训练持续时间、探索约束和任务分布？

因此，ProRL 的重点是研究 RL 运行条件怎样影响最终可达的解题范围。

## 二、方法与主要证据

作者将长时间训练与以下设计结合：

- KL divergence control：控制策略更新与参考分布之间的偏离。
- Reference policy resetting：在训练过程中调整参考策略。
- 多样任务：让模型在不同类型的可验证任务上持续学习。

论文报告，在多组 pass@k 评价中，训练模型能够覆盖更多问题，包括 Base Model 在所测采样预算中未能解决的问题；覆盖收益还与基础能力和训练时长相关。[来源：会议论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html)

## 三、怎样与 Limit of RLVR 对照

两项工作应放在各自训练与评测设置中理解。值得逐项对齐的变量包括基础模型、训练任务、训练步数、rollout 数量、参考策略、解码参数以及最大生成长度。

“Base Model 在大量采样中没有解出”是有限预算下的经验观察。它能够支持实际可达性改善的结论，但不能单独证明某条轨迹在数学意义上的概率从零变为非零。

ProRL 将讨论推进到更具体的问题：**什么训练条件能让 RL 持续改善较大采样预算下的覆盖？**

## 四、对机器人自我改进实验的启发

以下属于应用推论。

若 Agent 在短期改进后只提高已有任务的稳定性，值得继续检查更长训练、不同任务组合或探索控制能否产生新增可解布局。

不过，延长训练本身会消耗更多环境交互。比较时应同时给出收益随累计执行次数和时间的变化，并用独立测试布局检验新增能力是否迁移。

这篇论文适合与 [Curriculum RL](Curriculum-RL.md) 配合阅读：一个强调持续训练及探索条件，另一个强调围绕能力边界组织外部指导。
