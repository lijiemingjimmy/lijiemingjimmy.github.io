# AI 性能分析与插装：工具原理、Attention 优化与集群诊断

AI 性能分析要回答一个具体问题：**程序为什么慢，哪些观测支持这个判断，修改以后完整任务到底快了多少？**

这份笔记以一个两层 TinyDecoder 的 Attention 优化为主例。报告中的四步 decode 从 18.206ms 降到 3.099ms，约加速 5.9 倍。理解这个结果，需要把算子调用次数、张量形状、CPU/GPU 时间线、kernel 硬件指标和独立基准测量对应起来。

全文沿着 **性能指标 → 插装与工具原理 → Attention 案例 → 大规模 Trace → 实际调用与研究方向** 展开。基础概念服务于后面的诊断，工具命令则帮助把同样的方法用到自己的程序上。

**资料范围与日期**：本次于 2026-10-30 根据以下材料改写。

- [《交流汇报-性能分析工具-final.pptx》](/Users/lijieming/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_cuzqyadwb3jn12_2c1d/temp/drag/交流汇报-性能分析工具-final.pptx)：沈哲浩于 2026-10-29 的工具介绍与 Attention 案例。文件实际有 21 页，本文按实际页序引用；中间插入了 4 页截图，后续页脚编号与实际页序不同。
- [《02-调用说明书.md》](/Users/lijieming/Library/Containers/com.tencent.xinWeChat/Data/Documents/xwechat_files/wxid_cuzqyadwb3jn12_2c1d/temp/drag/02-调用说明书.md)：环境检查、采集命令、文件阅读及环境相关问题。
- 前次提供的会议纪要与部分逐字稿：补充性能指标、插装、1024 卡排障和 Trace 压缩讨论。

PPT 已给出 Attention 实验配置、计数与基准结果。以下将其作为**报告中的实验数据**解释；附件未包含 `model.py`、`benchmark.json`、原始 Trace 等文件，本次未重跑实验。说明书中的 Windows CPU 接口测试、WSL GPU 实验和通用命令模板分别使用各自的统计口径。

## 1. 首先明确：我们希望什么变快

同一个程序可能有多个性能目标，优化一个目标时，另一个目标未必同时改善。

| 指标 | 含义 | 在 AI 工作负载中的例子 |
| --- | --- | --- |
| 吞吐量（Throughput） | 单位时间完成的有效工作量 | samples/s、tokens/s、训练 steps/s |
| 延迟（Latency） | 指定任务从开始到完成的时间 | 一次推理、一轮控制、一个训练 step 的耗时 |
| P99 延迟 | 约 99% 的观测延迟不超过的分位点 | 一批请求中较慢部分的响应时间 |
| 端到端时间 | 完整流程的完成时间 | 数据读取、推理、环境交互和更新共同决定的 RL 迭代耗时 |

例如，把更多请求合成一个 batch，可能提高 tokens/s，也可能让请求等待更久。报告“性能提升”时，应同时写清测量对象、输入规模和完成条件。

### 1.1 为什么少量慢任务能影响整个系统

会议提到的尾部延迟放大，适用于需要等待多个子任务全部完成的场景。若第 \(i\) 个子任务耗时为 \(T_i\)，忽略其他开销时：

$
T_{\mathrm{job}}=\max(T_1,T_2,\ldots,T_n).
$

假设各子任务独立，每个任务超过某个延迟阈值的概率都是 1%，那么至少一个子任务超过该阈值的概率为：

$
P(\text{至少一个超时})=1-0.99^n.
$

当 \(n=100\) 时，这个概率约为 63.4%；当 \(n=1000\) 时，约为 99.996%。这里的“超时”指超过所选阈值，不一定是程序触发 timeout。

这解释了为什么单卡看起来很少出现的慢事件，在需要同步的大规模任务里可能频繁拖慢整步。真实集群中的负载和故障常有相关性，上式只是独立假设下的示意，不能直接预测实际集群的 P99。

## 2. FLOPs、GPU 利用率和 MFU 分别说明什么

这几个指标很容易混用，但它们回答的问题不同。

### 2.1 FLOPs 是运算量，FLOP/s 是运算速率

本文统一使用：

- **FLOPs**：浮点运算次数，例如一次模型前向需要多少浮点运算；
- **FLOP/s**：每秒浮点运算次数，例如芯片的峰值算力；
- **TFLOP/s**：\(10^{12}\) FLOP/s。

实际资料常把 FLOPS、FLOPs 混写，阅读时要结合单位判断。一次乘加通常按两次浮点运算计数，比较不同结果前也要统一这个约定。

### 2.2 GPU 利用率高，只能说明某种活动持续存在

以 `nvidia-smi` 的 GPU utilization 为例，官方定义是采样窗口内至少有一个 kernel 执行的时间比例。它没有直接报告所有计算单元完成有效数学运算的比例。[NVIDIA-SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)

因此，一个持续运行、但主要在等待数据或执行低效指令的 kernel，也可能对应很高的利用率。会议中的“假忙”，可以理解为**设备持续活跃，但完成目标工作的效率较低**。

另外，显存占用、访存带宽、SM occupancy 和 GPU utilization 是不同指标。显存装得满，不能推出带宽已经用满；occupancy 高，也不能单独推出程序高效。

### 2.3 MFU 要根据模型的有效工作量计算

MFU 的完整名称是 **Model FLOPs Utilization**。一个常用写法是：

$
\mathrm{MFU}
=\frac{F_{\mathrm{model,step}}/T_{\mathrm{step}}}
{N_{\mathrm{device}}P_{\mathrm{peak,device}}}.
$

其中 \(F_{\mathrm{model,step}}\) 是一个全局 step 所需的模型运算量，\(T_{\mathrm{step}}\) 是对应实测时间，分母是全部设备在指定精度和运算模式下的理论峰值之和。不能把局部设备的工作量与整个集群的峰值混用。

PaLM 对稠密 decoder-only Transformer 给出的简化估算，在忽略 Attention 的额外开销时，每个训练 token 约需 \(6N_{\mathrm{param}}\) 次运算。因此也可用全局 tokens/s 估算 MFU。完整计算还应考虑结构相关开销；MoE、长序列和不同任务需要相应的运算量模型。[PaLM，第 4.1 节与附录 B](https://arxiv.org/html/2204.02311v5)

**MFU 的分子侧重模型需要完成的工作。** PaLM 的口径不把 activation recomputation 产生的重复运算计入模型必需 FLOPs；把这些运算算进去，会得到另一种硬件运算利用率口径。

因此，会议中的“实测算力除以峰值”还需要补充：实测的是哪种工作量，峰值对应哪种精度、稀疏性和设备规模。MFU 适合检查训练系统效率，推理和机器人控制仍应结合吞吐、延迟与任务结果判断。

## 3. 瓶颈：计算、访存、通信和调度

GPU 上执行的 kernel，是一段设备端程序。一个 PyTorch 算子可能启动多个 kernel，多个算子也可能经编译融合成一个 kernel。因此，算子和 kernel 不能总是一一对应。

会议把主要开销概括为 Kernel、Memory、Communication。为了定位问题，可以进一步区分：

| 类型 | 主要成本 | 值得检查的现象 |
| --- | --- | --- |
| 计算 | 矩阵乘法、归约和其他指令 | kernel 较长，是否接近对应计算能力 |
| 设备内访存 | HBM、缓存、shared memory 与寄存器之间的数据访问 | 是否反复读写中间张量、访问是否高效 |
| 主机与设备传输 | CPU 内存与 GPU 显存之间的数据移动 | 小块传输、同步拷贝、数据准备延迟 |
| 设备间通信 | AllReduce、AllGather、Send/Recv 等 | 消息大小、到达时间差、链路和计算重叠 |
| 主机调度与同步 | Python、运行时调用、任务提交和等待 | GPU 空档、频繁同步、kernel 启动间隔 |

原材料把 Memory 描述成“数据从外部搬到 GPU 内部”，范围偏窄。分析一个 GPU kernel 时，设备内部不同存储层级之间的访问同样关键。

通信在大规模并行中可能成为主要瓶颈，但这个结论需要由具体工作负载验证。小模型可能受启动开销限制，数据管线可能供给不足，大模型推理又可能受显存带宽限制。

### 3.1 为什么各部分耗时不能直接相加

计算、传输和通信可能重叠。如果某次通信耗时 20ms，其中 15ms 与独立计算重叠，直接把 20ms 全算作可消除的整步开销，就会夸大优化空间。

更有用的问题是：**这次事件是否推迟了后续必需工作，最终使任务更晚完成？**

可以把程序抽象成带依赖的事件图。若事件 \(v\) 的时长为 \(d_v\)，只考虑已建模的依赖和资源约束，则完成时间可递推为：

$
C(v)=d_v+\max_{u\in\operatorname{pred}(v)}C(u).
$

这说明性能分析既要看事件持续多久，也要看依赖它的工作何时才能开始。实际系统还需要把 stream 顺序、资源竞争和同步约束纳入图中。

## 4. 插装原理：在正确的位置记录事件

插装（Instrumentation）是在程序或运行时的特定位置加入观测逻辑，记录事件名称、时间、线程、设备和关联关系。

最简单的形式是“开始时记录一次，结束时记录一次”。但 AI 系统同时存在 CPU、GPU、多 stream 和多个进程，必须先确定记录的是哪一层的开始与结束。

### 4.1 CPU 提交完成，不等于 GPU 执行完成

典型 CUDA 调用是异步的。CPU 把工作提交给 GPU 后，可以继续执行后面的代码。[PyTorch CUDA 执行语义](https://docs.pytorch.org/docs/2.9/notes/cuda.html)

```text
时间向右 →

CPU 线程：   提交 A ─ 提交 B ─ 继续运行 ───────── 等待结果
GPU stream：       [ 执行 A ] [ 执行 B ]
```

所以，用 CPU 计时器直接包住 `y = model(x)`，通常测到的是主机侧调用和提交过程；若调用内发生同步，结果又会包含相应等待。要测 GPU 工作完成，必须明确完成边界。

两种计时回答不同问题：

- **同步后用 CPU 时钟测量**：观察指定代码段从主机视角到相关设备工作完成所需的时间；
- **使用 CUDA Event**：观察设备 stream 中两个事件之间的经过时间。

微基准中可在测量边界同步。若给生产代码的每个算子都加同步，会改变原有重叠关系，测到另一个执行过程。

### 4.2 框架层：知道程序在做什么

PyTorch 的算子记录，以及用户添加的 `record_function` 区间，可以赋予执行事件语义。例如把一段代码标为 `rollout`、`forward` 或 `optimizer_step`，方便对应业务阶段。[record_function 文档](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.profiler.record_function.html)

其思路是在区间进入和退出时触发记录逻辑。框架层方便解释“这是什么操作”，但一个 CPU 区间本身不能替代 GPU 时间线。

这里还要区分 Python 中用于标记区域的 `record_function` 接口，以及框架内部更广泛的算子观测机制。不能据此假定所有设备后端都经由同一条实现路径采集。

### 4.3 CUDA 层：知道工作何时真正执行

CUPTI 提供两类相关机制：

| 机制 | 主要观测对象 | 适合回答的问题 |
| --- | --- | --- |
| Callback API | CUDA API 进入、退出及指定运行时事件 | CPU 在何时发起某次 CUDA 调用 |
| Activity API | CPU/GPU 活动记录，经过缓冲区交付 | kernel、拷贝等活动在何时执行 |

关联标识（correlation ID）帮助工具把一次主机 API 调用与它发起的设备活动对应起来。Activity API 的异步交付是指记录经缓冲区返回，不代表用交付时刻作为事件发生时刻。[CUPTI 使用指南](https://docs.nvidia.com/cupti/main/main.html)

GPU 活动的时间戳还需要映射到可比较的时间域。CUPTI 文档明确描述了 GPU 时间戳向 CPU 时间域的转换；问题涉及时钟校准，而不只是两个时钟“精度不一致”。[CUPTI Activity API 时间戳说明](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)

### 4.4 CUDA Event：适合测量明确的设备区间

CUDA Event 在指定 stream 中记录事件，可计算两个 Event 之间的毫秒数。它也能用于同步；PyTorch 的 `torch.cuda.Event` 是对相关 CUDA 能力的封装。[CUDA Event 文档](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html)

原纪要把 CUDA Event、C10 和 Kineto 串成一条底层依赖，容易误解。更清楚的分法是：

- CUDA Event 是 CUDA 提供的事件机制；
- C10 是 PyTorch 基础设施的一部分；
- Kineto 是 CPU/GPU profiling 库，可为 PyTorch Profiler 提供相关能力。[Kineto 官方仓库](https://github.com/pytorch/kineto)

理解设备计时不需要把三者当成同一个组件。会议提到的“热路径/冷路径”具体对应哪些函数和版本，现有材料不足以确认，记为**待复核**。

### 4.5 通信层：补充一次集体操作的语义

GPU 上出现一个很长的通信 kernel 时，我们还想知道：它属于哪个 communicator、哪次 collective、涉及哪些 rank、传输多少数据。

NCCL 的 Profiler Plugin 用于扩展通信观测。官方支持通过 `NCCL_PROFILER_PLUGIN` 指定插件库；插件必须与实际 NCCL 版本和接口匹配。设置环境变量负责选择和加载插件，具体事件记录能力由插件实现决定。[NCCL 环境变量文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-profiler-plugin)

会议另一个案例使用国产加速卡，其采集后端和通信库未给出。可以借鉴相同的分析思路，但不能直接假定其底层也是 CUDA/CUPTI/NCCL。

## 5. 动态采集与观察者效应

### 5.1 “按需抓取”需要区分已有能力与运行时附加

会议提出：一个已经训练多日的任务出现异常时，应能按需抓取短时间 Trace。

这里至少有两种机制：

1. 程序已经接入采集组件，平时关闭或少量记录，需要时打开详细采集。
2. 工具支持向运行中的进程附加相应组件，再启动采集。

PyTorch Profiler 提供动态开关部分活动采集的接口；这以进程已经运行相关 profiler 上下文为前提。CUPTI 也提供动态 attach/detach 机制，但实际工具仍需解决进程内接入、版本支持和安全的生命周期管理。[PyTorch Profiler 动态采集](https://docs.pytorch.org/docs/2.9/profiler.html#torch.profiler._KinetoProfile.toggle_collection_dynamic)、[CUPTI 动态附加与分离](https://docs.nvidia.com/cupti/main/main.html#dynamic-attach-and-detach)

因此，会议里的“遥控器”可以理解为采集控制入口。它是否支持完全未预埋的任务，现有材料没有说明。

### 5.2 不改变计算语义，仍然可能改变运行时序

插装希望保持程序的功能语义，但记录事件需要执行代码、写缓冲区并消耗资源。即使输出仍在允许的数值误差内，调度和重叠也可能发生变化。

可以先用下面的量检查扰动：

$
\mathrm{Overhead}
=\frac{T_{\mathrm{profiled}}-T_{\mathrm{baseline}}}
{T_{\mathrm{baseline}}}.
$

该比值应在相同配置和相近运行阶段下重复测量，并报告波动。较小的总开销也不能保证每个短 kernel、每次通信的局部时序都未受影响。

| 测量陷阱 | 影响 | 应对方法 |
| --- | --- | --- |
| 高频探针 | 记录成本与被测事件耗时接近 | 缩小采集范围，比较开启和关闭采集的运行 |
| 输入 shape 与调用栈记录 | 增加记录成本，可能改变张量引用生命周期 | 先用低成本配置，定位后按需补充 |
| 频繁同步 | 打断原有计算和传输重叠 | 区分微基准计时与完整系统分析 |
| 缓冲区不足或事件丢失 | 时间线出现缺口 | 检查丢失统计和采集完整性 |
| 根据文件顺序判断先后 | 异步缓冲交付不等于执行顺序 | 使用事件时间、标识和依赖重建关系 |
| 跨时钟域直接相减 | 制造虚假的间隔或顺序 | 核对时钟来源、单位和校准方法 |

PyTorch 文档特别指出，`record_shapes=True` 可能暂时保留张量引用，影响依赖引用计数的优化。[PyTorch Profiler 开销说明](https://docs.pytorch.org/docs/2.9/profiler.html)

会议给出过类似 0.01 的采样比例。由于未说明具体接口，这里只将其理解为降低观测频率的例子，不当作所有工具通用的配置项。随机稀疏采样也可能漏掉罕见故障；诊断分布式依赖时，应考虑保留连续、对齐的事件窗口。

## 6. 四种工具怎样配合，以及结果怎样读

| 工具 | 主要视角 | 优先回答的问题 | 典型产物 |
| --- | --- | --- | --- |
| Nsight Systems | CPU、CUDA API、GPU 与 NVTX 的系统时间线 | 谁在提交？谁在等待？计算和传输是否重叠？ | `.nsys-rep`、stats、可导出的 SQLite |
| PyTorch Profiler | 框架算子、用户区间与设备活动 | 哪个阶段慢？调用多少次？输入多大？ | 算子表、Chrome JSON Trace |
| Nsight Compute | 指定 kernel 的硬件指标 | 任务是否太小？计算、访存和并行资源怎样使用？ | kernel 报告、指标 CSV |
| Perfetto | 已采集事件的时间线与查询 | 阶段、算子和设备活动怎样对应？ | 可交互时间线、SQL 查询结果 |

PPT 第 3–9 页把这四个观察层次连在一起：先确定阶段，再检查算子和启动关系，最后深入选定的 kernel。Perfetto 读取采集结果；打开一个文件不会重新测量模型。[Nsight Systems 指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)、[Perfetto 外部格式](https://perfetto.dev/docs/getting-started/other-formats)

### 6.1 算子表：Self、Total、次数和形状一起读

| 字段 | 应如何理解 | 常见误读 |
| --- | --- | --- |
| Self CPU | 事件自身的 CPU 时间，按工具规则排除子调用 | 当作对应 GPU 的执行时间 |
| CPU total | 包含子调用的 CPU 时间 | 再与所有子行相加，重复计数 |
| Self CUDA / CUDA total | 按关联与层级规则归属的设备时间 | 当作某阶段的墙钟时间或全卡独占时间 |
| `# of Calls` | 采集窗口内该组事件的调用次数 | 忽略实际记录了几步 |
| Input Shapes | 该组算子的输入维度 | 只看算子名称，忽略输入规模变化 |

阅读时先确认 active 窗口，随后找到热点，再用次数和形状解释它。一次只需几微秒的操作，若由 Python 循环反复提交，也可能形成明显的主机开销。相同的 `aten::mm` 则可能分别处理 prefill 的完整序列和 decode 的单 token。

`key_averages(group_by_input_shape=True)` 可以按输入形状拆分统计，前提是采集时开启 `record_shapes=True`。父阶段和子算子的 total 可能重叠，多个 stream 的 GPU 活动也可能并行，因此这些数字不能直接相加成端到端耗时。（PPT 第 6–7 页；说明书第 3 节）

还要区分 **Trace 中的持续时间 `dur` 与 Self 时间**：一个父区间的 `dur` 包含嵌套区间。按名字累计 `dur`，得到的是该类事件的累计区间时长；计算 self time 需要处理子区间覆盖。同样，标准火焰图的横向位置不表示执行先后，父框通常汇总自身及后代栈的权重。[Perfetto slice 定义](https://perfetto.dev/docs/analysis/sql-tables#slice)、[Flame Graphs 官方说明](https://www.brendangregg.com/flamegraphs.html)

### 6.2 时间线：先找阶段，再看实际设备活动

![PPT 中的四步 decode Perfetto 时间线](assets/AI性能分析与插装/perfetto-decode.png)

图来自 PPT 第 8 页。上方可以沿 `decode`、`attention` 和框架算子展开，设备轨道展示对应的 GPU 活动。图中颜色用于区分事件；一条很长的阶段范围并不表示某个 kernel 在整个区间持续计算。

阅读顺序可以是：选定一步 decode，找到 attention，检查 CPU 的算子与启动调用，然后查看 GPU kernel 之间的间隔。如果导出文件提供关联 ID 或 flow，可以回查具体启动来源。关联标识需要结合进程和设备作用域解释，跨文件合并后不能只凭同一个数字匹配。

Nsight Systems 的 stats 则提供另一种汇总视图：`Num Calls` 反映次数，`Total Time` 反映该类记录的累计时长，`Median / Min / Max` 帮助识别通常情况和少数慢调用。表中百分比只描述该表的分布，不能直接当作完整程序的时间占比。（PPT 第 5、8 页）

### 6.3 硬件指标：先看任务规模，再解释吞吐

PPT 第 9 页强调了三个方面：

- **Grid / Waves Per SM**：一次启动提供多少线程块，相对于设备的并发执行容量是否足够。
- **SM / DRAM Throughput**：对应计算或访存子系统的吞吐比例。
- **Occupancy 与资源限制**：活跃 warp 比例，以及寄存器、shared memory 等资源对并行度的约束。

如果 grid 很小，SM 和 DRAM 的吞吐都可能很低。这个组合首先提示任务不足以充分使用硬件，不能只凭“SM 吞吐低”就判定为访存瓶颈。即使任务规模足够，还需检查具体资源、访问模式与停顿原因。

Nsight Compute 为收集指标可能重放 kernel 或应用，影响缓存、并发和主机运行时间。它适合解释 kernel 内部行为，最终性能收益需要关闭这些采集后另测。[Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## 7. Attention 案例：怎样从记录推导出 5.9 倍加速

这个案例的价值在于，几种工具分别提供了能够互相印证的证据。PPT 第 16–21 页报告了一段有意保留低效写法的 Naive 实现，以及 SDPA 与 KV Cache 的组合改写。

### 7.1 实验对象与测量范围

| 项目 | PPT 记录的配置 |
| --- | --- |
| 硬件与环境 | RTX 5060 Laptop GPU，WSL / CUDA |
| 模型 | 两层 TinyDecoder，每层包含 Attention、残差与 MLP |
| hidden size / MLP 宽度 | 256 / 512 |
| Attention heads | 8，每个 head 维度为 32 |
| batch / dtype | 1 / BF16 |
| 输入流程 | 随机张量模拟 128 token prefill，之后进行 4 步增量 decode |
| decode 历史长度 | 129、130、131、132 |
| 对比区间 | prefill 之后，整个两层 TinyDecoder 的四步 decode |

第 16 页备注记录的版本为 PyTorch `2.12.0.dev20260407+cu128`、Nsight Systems `2026.3.2`、Nsight Compute `2026.3.1`。这些是报告所用环境，不代表本文运行环境或通用版本要求。

这里的 prefill 一次处理完整前缀，decode 随后逐次处理新增位置。单个 head 的 Attention 可写为：

$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
\qquad d_h=32,
$

其中 \(M\) 控制哪些位置可被访问。Naive 与缓存实现的区别，既涉及执行这些运算的方式，也涉及每步需要重新计算哪些位置。

该任务用小型解码块解释性能机制，未报告完整语言模型的文本生成质量，也不等于集群训练基准。

### 7.2 Profiler：调用次数暴露逐 head 循环

报告中，四步 decode 的 attention 阶段出现 8 次：

$
4\ \text{步}\times2\ \text{层}=8.
$

`aten::bmm` 却出现了 128 次。每个 head 分别计算一次 \(QK^\top\) 和一次 \(PV\)，刚好有：

$
4\ \text{步}\times2\ \text{层}\times8\ \text{head}\times2\ \text{矩阵乘}=128.
$

这把算子表中的次数对应到了上层代码结构：同一次 Attention 被拆成多个 head，再逐个提交小矩阵乘。（PPT 第 17 页）

### 7.3 输入形状暴露完整历史重算

首步 Naive 的 \(QK^\top\) 输入是：

$
[1,129,32]\times[1,32,129],
$

每个 head 形成 \(129\times129\) 的分数矩阵。这个形状表明，增加一个 token 后，模型仍在计算所有历史位置的 query 和输出。

对于只需要最后一个位置结果、且历史状态可合法复用的因果解码，可以保存各层历史 K/V，只对新 token 做增量计算。首步的 Attention 分数规模就从每个 head 的 \(129\times129\) 变成 \(1\times129\)。

这一比较解释的是工作量变化，不等于整个程序会按 129 倍加速。QKV 投影、MLP、缓存更新、kernel 启动和其他开销仍然存在。

### 7.4 Systems：逐 head 同步打断任务提交

PPT 第 18 页报告的四步采集窗口包含：

| 记录 | Naive 次数 |
| --- | ---: |
| GPU kernel | 468 |
| `cudaLaunchKernel` / `cuLaunchKernel` | 308 / 160 |
| `cudaDeviceSynchronize` | 64 |

其中 64 次同步对应：

$
4\ \text{步}\times2\ \text{层}\times8\ \text{head}=64.
$

在这个实现里，每处理一个 head，CPU 都显式等待设备完成，再继续提交后续工作。大量小 kernel 与频繁等待结合起来，使主机提交和同步成为优先检查的开销。

这一判断需要同时看代码顺序与时间线。launch 次数、kernel 次数和同步次数各自描述一类事件，不能在其他程序里默认相等，或只用次数推算精确时间占比。

### 7.5 Compute：一次 kernel 的任务量很小

PPT 第 19 页给出两个被选中的 Naive kernel：

| 指标 | bmm | softmax |
| --- | ---: | ---: |
| Compute (SM) Throughput | 4.24% | 5.97% |
| DRAM Throughput | 2.76% | 2.93% |
| Waves Per SM | 0.04 | 0.11 |

这两个 kernel 的计算与显存吞吐都较低，任务网格也不足以形成一个完整 wave。结合前面的 128 次 bmm、64 次同步和历史重算，合理的优化方向是减少重复工作、合并小任务并去掉逐 head 等待。

这些数据说明选中调用的执行情况，尚不足以量化三种问题各自占用了多少总时间。Compute 采集本身包含 replay，也不能拿其中的 Duration 替代独立的四步 decode 基准。

### 7.6 改写：多 head 一起计算，历史 K/V 按层缓存

PPT 第 20 页的改写同时做了三件事：

1. 用 `F.scaled_dot_product_attention` 处理全部 head，去掉逐 head 的 Python 调用循环。
2. 移除循环内显式同步，让 CPU 可以继续提交工作。
3. prefill 后保存每层 K/V，decode 只输入当前新 token，并将新 K/V 追加到对应层缓存。

首步 decode 的主要张量为：

| 张量 | 形状 |
| --- | --- |
| 当前 token 输入 | `[1, 1, 256]` |
| Q | `[1, 8, 1, 32]` |
| 追加后的 K / V | `[1, 8, 129, 32]` |

SDPA 是调用接口，具体后端由输入和运行配置决定。PPT 报告本轮实际执行了 Flash Attention kernel，因此可以说明这次进入了对应路径，不能把所有 SDPA 调用都默认等同于 Flash Attention。

这里的 mask 也值得注意：第 20 页备注说明，单 token query 已位于序列末尾，缓存仅包含合法的历史位置与当前位置，因此该 decode 步使用 `is_causal=False`。在这一前提下，当前 query 应能访问缓存中全部 K/V。

如果 query 一次包含多个新 token，或 cache 中含 padding、未写入位置，就需要相应的 mask。PyTorch 文档中的非方形 `is_causal=True` 使用左上对齐的因果约束，不能无条件套到带历史缓存的单 token 解码上。推理时还需把 SDPA 的 `dropout_p` 设为 0。[SDPA API 与 mask 语义](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)

### 7.7 独立基准与数值检查

PPT 第 21 页报告：关闭所有 Profiler，预热 3 轮，交替顺序重复测量 12 次，以四步 decode 的墙钟中位数比较。

| 指标 | Naive | SDPA + KV Cache 组合改写 |
| --- | ---: | ---: |
| 四步模型 decode 墙钟中位数 | 18.206ms | 3.099ms |
| Systems 窗口中的 GPU kernel 数 | 468 | 96 |
| Systems 窗口中的设备同步数 | 64 | 0 |
| Flash Attention kernel 数 | 0 | 8 |

据报告数值计算：

$
S=\frac{18.206}{3.099}\approx5.875.
$

所以“约 5.9 倍”成立，对应的耗时下降约为 83.0%。这里的墙钟测量与 Systems 计数来自不同采集运行；表格用于对应实现变化，不能把它们当成同一次测量。

8 个 Flash Attention kernel 对应四步、每步两层。设备同步数降为 0，指报告采集窗口中的该类显式同步记录，不代表程序没有依赖等待，也不代表基准计时不需要确认设备完成。

数值检查使用同权重、同输入，比较每一步最后一个 token 的输出，报告最大绝对差为 **0.015625**。这支持对这些已比较输出描述数值接近程度；是否满足应用要求，还要看预设容差、输出尺度和任务指标。材料没有提供逐位一致或完整模型质量验证。

因此，这一结果的准确表述是：**在该小模型、batch 1、BF16 和四步 decode 设置下，SDPA、去同步与 KV Cache 的组合改写，相对刻意低效的 Naive 基线获得约 5.9 倍加速。** 单项贡献需要消融实验。

### 7.8 如何把案例的方法迁移到自己的程序

可先逐项增加改动：去掉无必要的逐 head 同步，合并多 head 计算，再加入合法的缓存复用；每一步都保留独立计时和数值检查。这个顺序是建议的后续实验，不是 PPT 已报告的消融结果。

对更大的系统，还应考虑被优化部分的占比。若它占总时间 \(p\)，加速 \(s\) 倍，其余开销不变，则：

$
S_{\mathrm{total}}=\frac{1}{(1-p)+p/s}.
$

例如某部分只占完整流程的 20%，该部分快 5.9 倍，完整流程约快 1.20 倍。PPT 的 5.9 倍已经覆盖这个 TinyDecoder 的四步 decode，但不能继续外推成整个服务、训练或机器人闭环也快 5.9 倍。

## 8. 多进程 Trace：从“某张卡在等”找到“谁让它等”

单个进程里的一段 GPU 空闲，只能直接说明这个区间没有观察到相应执行活动。原因可能是本地供给不足，也可能是等待远端数据。

例如流水线并行中：

```text
Rank 0： [计算 microbatch 0] ─── Send ─── [计算 microbatch 1]
Rank 1：       等待输入 ─────── [计算 microbatch 0] ─── Send
Rank 2：                等待输入 ─────────────────── [计算]
```

若只看 Rank 2，它可能空闲很久；根因可能在 Rank 0 的计算、Rank 0 到 Rank 1 的通信，或者更早的数据准备过程。

会议描述了一个 **1024 张国产加速卡**的训练案例：把多个进程的 Trace 后处理后导入 Perfetto，从全局视角分析流水线等待，并定位到光模块相关故障。这里保留为**会议报告的排障案例**；未提供原始 Trace、链路计数器或维修验证记录。

### 8.1 合并前要解决四件事

1. **身份唯一**：保留 host、rank、device、process、thread、stream 等层级。不同机器的同一 PID 不能当作同一进程。
2. **时间可比**：确认时间单位、原点与跨主机校准。不能把各文件独立减去各自首事件时间后，就认为完成了对齐。
3. **依赖可追踪**：保留 step、microbatch、collective 序号、communicator、peer 和可用关联标识；具体字段取决于采集后端。
4. **采集范围明确**：记录每个 rank 的采集窗口、是否缺失以及是否丢事件。未采集的区间应保留为未知。

跨主机对齐可抽象为：

$
t_{\mathrm{global}}\approx a_h t_{\mathrm{local}}+b_h,
$

其中 \(b_h\) 表示时钟偏移，\(a_h\) 表示漂移校正。实际应记录校准依据和残余误差；当误差大于事件间隔时，不应强行判断两者的精确先后。

### 8.2 长通信事件不一定意味着链路带宽低

某个 collective 持续很久，可能是传输较慢，也可能是其他 rank 很晚才到达。后一种情况中，本地通信事件包含等待远端参与者的时间。

因此，诊断应对比各 rank 的到达时刻、通信前的计算、传输规模和链路状态。要确认光模块故障，还应结合重传、错误计数、硬件状态或更换后的复测。

全局 Trace 的价值，是帮助缩小根因范围并追踪等待传播；最终根因仍需要相应层面的证据支持。

## 9. Trace 压缩课题：从 500GB 到 5GB 是什么要求

会议提出在现有工具上增加流处理和压缩，将千卡场景约 500GB 的 Trace 缩到约 5GB。

按同一容量单位计算，这相当于 **100:1 压缩比，体积减少约 99%**。这是会议提出的目标，现有材料没有给出已完成实现或达到该比例的实验。

### 9.1 先说明“压缩”保留什么

| 方案 | 保留能力 | 必须说明的代价 |
| --- | --- | --- |
| 通用无损压缩 | 可恢复原始字节 | 解压成本，随机访问是否方便 |
| 结构化编码 | 可设计为恢复选定 schema 内全部事件语义 | JSON 格式细节或未知字段是否仍保留 |
| 重复模式编码 | 用模板与实例参数表示重复结构 | 是否保存每次执行的真实时间和异常偏差 |
| 聚合与采样 | 保留统计或部分事件 | 个别事件与完整依赖可能无法恢复 |
| 多级摘要 | 先看全局，再读取局部详细数据 | 摘要不代表底层信息一定可以丢弃 |

**文件更小和诊断能力不变，是需要分别验证的两个结果。** 训练 step 结构虽然重复，真正需要诊断的往往就是其中一次异常。用平均 step 替代全部 step，可能把问题一并删掉。

### 9.2 一个可验证的工具方案

以下是基于会议问题提出的设计建议，并非会议已实现的系统：

```text
各 rank 的事件流
        ↓
统一字段、身份与时间域
        ↓
字典编码 / 时间增量 / 分块无损压缩
        ↓
按 step、rank、时间建立索引
        ↓
全局摘要 + 保留原始异常窗口
        ↓
查询候选等待关系，按需载入局部完整事件
```

可先从低风险的数据冗余入手：把重复名称与调用栈改为引用，对有序时间使用增量表示，按连续时间窗口分块。未知字段和原始身份映射需要保留，避免转换后无法回查。

进一步可以研究重复 step 的结构模板，但每次执行的持续时间、缺失事件、通信序号和异常残差应单独表达。存储结构重复，与抹平运行差异，是两种不同处理。

“流处理”也需要说明位置：若先写出 500GB，再离线压缩为 5GB，只改善后续存储和传输；若目标包含减少首次落盘 IO，就需要在采集或写入阶段处理，并测量额外 CPU 和训练扰动。

### 9.3 现有能力与研究增量

Perfetto 已支持包括动态字符串 interning、flow 关联在内的机制，原生格式也有压缩能力。对巨大 Trace，可使用本地原生 Trace Processor 缓解浏览器运行时的内存限制。[Perfetto 能力说明](https://perfetto.dev/docs/)、[Trace 配置与压缩](https://perfetto.dev/docs/concepts/config)、[大 Trace 分析](https://perfetto.dev/docs/visualization/large-traces)

因此，“换一种格式加压缩算法”首先是工程方案。若希望形成研究工作，需要证明额外解决了什么问题，例如：

- 在大量重复执行结构中，保留异常时间与跨 rank 依赖，同时降低存储和查询成本；
- 在有限观测预算下，选择哪些 rank 和连续窗口能更好地定位异常；
- 数据不完整、时钟有误差时，仍能给出有依据的等待关系与不确定性。

这些方向的创新性需要进一步检索相关工作，不能仅凭本次会议判断。

### 9.4 怎样验证这个方向成立

| 验证维度 | 建议测量 |
| --- | --- |
| 数据口径 | 设备数、rank 数、采集时长、事件数、字段和原始格式 |
| 存储与开销 | 压缩比、处理速度、峰值内存、首次落盘字节数、训练扰动 |
| 恢复正确性 | 声明保留的事件、时间、身份和关联关系是否能恢复 |
| 查询性能 | 首次打开、step 检索、跨 rank 对比、异常窗口读取耗时 |
| 诊断效果 | 能否定位相同异常，是否漏掉慢 rank 或错误链路 |
| 对照方案 | 原始 JSON、通用压缩、已有 Trace 表示、相同存储预算下的采样 |

先在单机多卡和小规模跨机数据上比较完整事件与转换结果，再扩大规模。若使用有损摘要，需要另外报告遗漏率和诊断质量；不能沿用无损恢复的结论。

## 10. 实际调用：从环境确认到读回结果

本节根据调用说明书与 PPT 第 14–15 页整理。通用代码展示接入方式；`run_paired_*.py` 属于报告作者的项目，目前附件没有这些脚本。取得对应工程后才能直接运行案例命令。

### 10.1 先确认采集环境

在实际运行模型的 GPU 环境中检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.profiler.supported_activities())"
nsys --version
ncu --version
nvidia-smi
```

`torch.cuda.is_available()`、Profiler 支持的 activities 和 Nsight 是否可用，回答不同问题。如果只有 CPU activity，应继续检查 PyTorch 构建、驱动、CUPTI 与采集支持，不能只凭一项检查认定根因。

说明书记录了两组环境：

| 环境 | 材料中的用途与结果 | 可以得出的结论 |
| --- | --- | --- |
| Windows、PyTorch `2.14.0+cpu` | 小模型 CPU 示例生成算子表、89,676 字节 Trace 和火焰图；未发现 `nsys`、`ncu` | CPU 接口与文件链路在该环境跑通 |
| WSL / CUDA、RTX 5060 Laptop GPU | PPT 中的 Naive/SDPA 四步 decode 实验 | 按报告配置解释 GPU 指标与基准 |

第一组示例是 batch 2、序列 32、宽度 64；它与第 7 节的 batch 1、hidden size 256 案例不同。未发现 Nsight 表示该环境没有相应 GPU 采集结果，不是 GPU 利用率为 0。

### 10.2 阶段标记与采集窗口分别控制

在 PyTorch Profiler 中，可用自定义区间标记模型阶段：

```python
from torch.profiler import record_function

with record_function("decode"):
    output = model(tokens)
```

在独立的 Nsight 采集运行中，可以显式添加同名 NVTX 区间：

```python
import torch

with torch.cuda.nvtx.range("decode"):
    output = model(tokens)
```

`record_function` 与 NVTX range 负责命名区间。`profile(...)`、`nsys profile` 等工具负责采集。说明书中的 `emit_nvtx()` 还能为 autograd 操作生成 NVTX 标记，但自定义的同名阶段最好显式标注，避免假定不同接口自动产生相同结构。[NVTX range API](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.nvtx.range.html)

`model.eval()` 设置模块的评估行为，`torch.inference_mode()` 或 `no_grad()` 控制梯度记录。两者的作用不同，推理测试应按模型需要共同设置。

### 10.3 PyTorch Profiler：只记录需要分析的几步

下面是接入已有 `model` 和 `tokens` 的 CUDA 示例，不是完整模型实现：

```python
from pathlib import Path
import os
import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

out_dir = Path("traces")
out_dir.mkdir(exist_ok=True)
rank = os.environ.get("RANK", "0")
model.eval()

def save_trace(prof):
    print(prof.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=20
    ))
    prof.export_chrome_trace(
        str(out_dir / f"rank{rank}_step{prof.step_num}.json")
    )

with torch.inference_mode():
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(skip_first=10, wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=save_trace,
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        for _ in range(14):
            with record_function("decode"):
                output = model(tokens)
            prof.step()
```

这里先跳过 10 步，再等待 1 步、Profiler 预热 1 步、正式记录 2 步，共需 14 次 `prof.step()`。跳过 10 步只是例子；编译或模型预热是否结束，必须看真实程序。实际自回归解码还应在循环中更新输入和缓存。

`prof.step()` 推进采集 schedule。多个周期的 Trace 应在 `on_trace_ready` 中分别导出，文件名加入 rank 和周期信息；若需要跨多次运行保留记录，再增加运行 ID。只在退出上下文后导出，可能只拿到最后一个周期。[PyTorch Profiler API](https://docs.pytorch.org/docs/2.9/profiler.html)

需要查源码时，再单独开启 `with_stack=True` 并按调用栈分组。调用栈和 shape 会增加开销，最终对外报告的吞吐要回到关闭详细采集的运行中测量。

PPT 所用工程还提供了现成入口：

```bash
# 在作者的 scenarios/naive_attention 目录中执行。
python run_paired_torch_profiler.py --variant naive
python run_paired_torch_profiler.py --variant sdpa
```

PPT 记录的输出目录为 `outputs/paired_2026-10-08/`。其中 `*_torch_table.txt` 是算子表，`*_torch_trace.json` 是可视化数据。这是报告工程中的相对路径，不是本文工作区已生成的结果。

### 10.4 Nsight Systems：把窗口限定到稳态 decode

PPT 的 `run_paired_gpu.py` 已用 `cudaProfilerStart/Stop` 包围预热和 prefill 之后的四步 decode，因此可使用：

```bash
nsys profile --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --sample=none --cpuctxsw=none -o naive_nsys \
  python run_paired_gpu.py --variant naive
```

如果换成自己的脚本，必须同时实现对应的 start/stop；单独复制 `--capture-range=cudaProfilerApi` 不会自动找到模型的稳态区间。只想对短程序做初步检查时，可使用普通采集并限制时长。[Nsight Systems 采集区间](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#focused-profiling)

报告生成后，可在不重跑模型的情况下读取汇总或导出数据库：

```bash
nsys stats --report cuda_gpu_kern_sum naive_nsys.nsys-rep
nsys export --type=sqlite --output=naive_nsys.sqlite naive_nsys.nsys-rep
sqlite3 naive_nsys.sqlite '.schema'
```

在集群上优先把报告和 SQLite 写到节点本地目录，再复制回共享存储。说明书记录过共享文件系统上的 SQLite 锁问题；具体表名、权限和可用指标以实际版本为准。每个 rank 使用独立输出文件，避免覆盖。

### 10.5 Nsight Compute：筛选一个明确的 kernel

先从 Systems 找到实际 kernel 名，再用 Compute 筛选。下面沿用 PPT 的 SDPA 场景入口：

```bash
ncu --profile-from-start off --kernel-name-base demangled \
  --kernel-name 'regex:.*flash_fwd_kernel.*' --launch-count 1 \
  --section SpeedOfLight --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --export sdpa_flash \
  python run_paired_gpu.py --variant sdpa
```

这条命令依赖脚本中的 profiler start/stop。正则需加引号，防止 shell 先解释 `*`；`flash_fwd_kernel` 也只适用于实际匹配的后端，换版本或输入后应重新确认名称。[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)

PPT 与说明书记录该环境导出 `.ncu-repz`，其他版本可能使用 `.ncu-rep` 等扩展名。以后缀和报告日志为准，用兼容版本 GUI 打开。需要 Roofline 时，先通过 `ncu --list-sections` 确认并采集相应 section。

同一实验优先分开运行 PyTorch Profiler、Nsight Systems 和 Nsight Compute，再单独跑无 Profiler 的基准。这样既减少采集器之间的相互影响，也容易区分每组数字的来源。CUPTI 对多订阅者的支持有版本条件；不应把说明书的环境经验写成所有版本永远无法共存。

### 10.6 Perfetto 与 SQL：让汇总保持正确含义

打开 [Perfetto UI](https://ui.perfetto.dev)，导入 Chrome Trace，按阶段名、算子名或 kernel 名定位。也可以用已安装的 Trace Processor 汇总：

```bash
trace_processor query trace.json \
  "SELECT name, COUNT(*) AS n, SUM(dur)/1e6 AS summed_duration_ms
   FROM slice WHERE dur >= 0
   GROUP BY name ORDER BY summed_duration_ms DESC LIMIT 20"
```

这里使用 Trace Processor 的纳秒单位，所以除以 \(10^6\) 得到毫秒；Chrome JSON 原始 `dur` 使用微秒。命令语法以安装版本为准。

这条查询用于初步查看事件名称与累计区间时长。父子区间会重叠，不同线程也可能并行，因此结果既不是 self time，也不是完整任务的墙钟占比。比较某一步时，应先按时间、track、rank 或阶段限制查询范围。原始 JSON 的简单统计还需限定事件类型，避免把内存计数器等非耗时事件纳入热点排行。[Perfetto Trace 格式](https://perfetto.dev/docs/getting-started/other-formats)

### 10.7 CUDA Event 与最终基准

同一 stream 上的明确设备区间可用 Event 计时。下面是独立示例：

```python
import torch

x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
for _ in range(10):
    z = x @ y

a = torch.cuda.Event(enable_timing=True)
b = torch.cuda.Event(enable_timing=True)
a.record()
b.record()
b.synchronize()

a.record()
for _ in range(100):
    z = x @ y
b.record()
b.synchronize()
print(f"平均设备区间: {a.elapsed_time(b) / 100:.3f} ms")
```

这得到 stream 中经过的平均时间，可能包含提交空档与竞争。完整模型的墙钟基准则需要在明确边界确认设备完成，并固定模型状态、输入和缓存起点。对于 decode，先建立同样的 prefill cache，再测同样数量的新 token；每次试验应重新准备等价状态。

本笔记的代码做了静态语法检查，未在本机完成 CUDA 实测。第 7 节的实验结果仍来自 PPT。

### 10.8 文件与版本问题怎样处理

说明书记录了中文路径导出失败、JSON 中反斜杠转义错误，以及火焰图导出链路的兼容性问题。这些经验适合保留为排查线索：

- 导出失败时，可先尝试临时 ASCII 路径，核对文件是否存在且可解析，再复制到目标目录。
- `use_python_export`、私有 `_ExperimentalConfig` 和显存历史接口依赖版本，使用前核对当前安装的函数签名。
- JSON 损坏时保留原文件，优先修复或更换导出路径；无差别替换反斜杠可能改变合法内容。
- 火焰图无数据时，分别检查是否采集到调用栈、导出是否成功和输入格式。`export_stacks` 的维护状态与 FlameGraph 项目的维护状态应分别说明。

例如，PyTorch 2.14 文档已经列出 `use_python_export` 参数，并将 `export_memory_timeline` 标为 deprecated；2.9 文档的对应接口不同。CUDA 显存分配分析可按所用版本选择 memory history/snapshot，不能把 CUDA 专用接口直接当作全部 CPU 内存分析的替代品。[PyTorch 2.14 Profiler](https://docs.pytorch.org/docs/2.14/profiler.html)、[PyTorch 2.9 Profiler](https://docs.pytorch.org/docs/2.9/profiler.html)

这些问题不会改变分析原则：先证明产物确实包含需要的事件，再解释数据；缺失的硬件指标写为“未采集”，不要补成零。

## 11. 与 RL、VLA 和仿真研究的联系

下面是结合已有研究主题的分析，不属于会议原始结论。

### 11.1 RL：先找完整迭代中谁占时间

一个 RL 流程可能包括 rollout、reward、logprob、训练更新、权重同步和环境交互。局部 kernel 优化的收益，取决于该部分在整个迭代中的占比与依赖位置。

这与 [RL-Kernel 笔记](RL-Kernel.md) 中区分算子窗口和完整 RL step 的思路一致。系统观测还应保留 policy version、批次和阶段标识，避免把不同时期的活动错误对应起来。

### 11.2 VLA：关注动作何时真正可执行

机器人控制链路可以包括相机采集、预处理、VLA 推理、动作排队和控制执行。只优化前向时间，可能无法改善被旧动作队列或执行端等待限制的闭环响应。

因此，可以分别记录观察时间、动作生成时间和真实执行时间。这与 [SmoothRL 笔记](SmoothRL.md) 中区分生成动作和实际执行区间有关。推理耗时、动作新鲜度和最终成功率需要分别测量。

### 11.3 仿真：先说明要模拟哪一层

会议讨论了仿真器对访存与通信行为研究的帮助，也表达了对大规模仿真实验成本的顾虑。

这一问题应结合保真度和研究目标判断。体系结构仿真、集群调度模拟和机器人物理仿真，分别描述不同对象。已有 Trace 可以支持负载重放或性能建模，但模拟器要用于预测优化收益，仍需验证依赖、争用和通信模型是否足够准确。

比较稳妥的研究流程是：用小规模实测校准模型，明确模型覆盖范围，再用模拟探索配置，并回到实机验证关键结论。

## 12. 下一步可以做什么

### 12.1 把 Attention 案例复现成可比较的实验

PPT 已补齐硬件、模型、输入、计数、基准方法和末 token 误差。继续复现时，重点取得报告中引用的 `model.py`、`run_paired_gpu.py`、`run_paired_torch_profiler.py`、`benchmark.json`、`equivalence.json` 和对应的原始报告。

优先补三组分析：逐项去同步、合并 head、增加 KV Cache 的消融；不同 batch 和上下文长度的扩展；数值误差相对预设容差与下游任务的检查。这些实验可以判断 5.9 倍收益来自哪些改动，以及适用范围怎样变化。

### 12.2 保留会议中的集群与工具课题

| 工作 | 材料中的安排 | 目前资料支持到哪里 |
| --- | --- | --- |
| 性能工具调用说明 | 沈哲浩提供说明 | 本次已获得说明书，可整理调用流程 |
| 在各自科研课题中实践并分享 | 团队成员 | 可按第 10 节建立基准、采集和复测记录 |
| 大规模 Trace 流处理与压缩 | 金煜阳探索 | 约 500GB → 5GB 仍是目标，未提供达成实验 |

1024 卡光模块排障仍缺少原始 Trace、链路指标和故障修复后的复测。动态采集控制器、C10/Kineto 的具体热路径与冷路径，以及会议中 0.01 采样率的代码位置，也没有在这两份新附件中补齐。

若从这些材料发展一个研究问题，值得聚焦的是：**在有限观测预算下，怎样同时保留重复程序结构、真实时间变化和跨 rank 依赖，使大规模异常仍然可定位。** Attention 案例说明了细粒度证据怎样支持优化；集群课题则要求这套分析在更多进程、更长运行时间和不完整观测下依然成立。

## 参考资料

正文开头列出本次两份附件；PPT 的案例引用按实际页序，说明书引用按章节。下列官方资料用于核对通用机制。基础资料于 2026-10-29 核对，本次于 2026-10-30 补查 SDPA、NVTX、采集参数及版本相关接口。

1. 用户提供的会议文字材料、21 页 PPT 和《02-调用说明书.md》：分别提供背景讨论、案例数据与调用流程。
2. [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/html/2204.02311v5)：MFU 的定义与计算。
3. [NVIDIA-SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：GPU utilization 的口径。
4. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.9/notes/cuda.html)：异步执行与同步。
5. [PyTorch Profiler API](https://docs.pytorch.org/docs/2.9/profiler.html) 与 [官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)：采集、调度、导出与开销。
6. [record_function](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.profiler.record_function.html)、[torch.cuda.Event](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html)、[Kineto](https://github.com/pytorch/kineto)：区域标记、设备事件与采集组件。
7. [CUPTI 使用指南](https://docs.nvidia.com/cupti/main/main.html) 与 [Activity API](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html)：回调、活动、关联与时间戳。
8. [NCCL Profiler Plugin 配置](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-profiler-plugin)：插件选择与加载。
9. [Nsight Systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) 与 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)：系统时间线、kernel 指标和 replay。
10. [PyTorch SDPA 教程](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html) 与 [SDPA API](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)：Attention 实现、mask 与后端选择。
11. [Perfetto 外部格式](https://perfetto.dev/docs/getting-started/other-formats)、[大 Trace 分析](https://perfetto.dev/docs/visualization/large-traces)、[Trace 配置](https://perfetto.dev/docs/concepts/config)：导入、分析与压缩能力。
12. [NVTX range](https://docs.pytorch.org/docs/2.9/generated/torch.cuda.nvtx.range.html) 与 [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html)：阶段标记与 kernel 筛选。
13. [PyTorch 2.14 Profiler](https://docs.pytorch.org/docs/2.14/profiler.html)、[Perfetto slice 定义](https://perfetto.dev/docs/analysis/sql-tables#slice)、[Flame Graphs](https://www.brendangregg.com/flamegraphs.html)：接口版本、区间统计与火焰图语义。
