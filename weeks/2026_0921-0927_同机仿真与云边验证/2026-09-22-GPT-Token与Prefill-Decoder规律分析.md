# GPT Token 与 Prefill／Decoder 规律分析

版本：v1.0 · 日期：2026-09-22

## 1. 分析对象与统计口径

【E2／实验范围】本文分析 `decision_v2_20260922` 中 LIBERO-10 task 0、Ori、init 0 的 GPT Direct 和 π0.5 + GPT Hybrid 两条路线。Direct 共 97 轮 GPT 决策，Hybrid 共 51 轮；模型配置为 `gpt-6-astra / medium`。

本文关注四个量：输入 Token、可见输出 Token、首个可见输出前的等待，以及首个至最后一个可见输出的生成时间。

【E1／Token 来源】Token 数量直接取模型接口的 `thread/tokenUsage/updated`，不是本地根据文字长度估算。程序用“当前累计用量减去上一轮累计用量”得到本轮用量，再把各轮增量求和；总和必须等于最后一轮的累计用量，否则报告生成失败。

【E1／时间边界】现有接口没有提供纯服务端 `prefill_time` 和纯 `decode_time`。因此本文使用以下可观测代理：

```text
TTFT
= 网络发送 + 服务端排队 + prefill + 内部处理／reasoning + 首个输出返回

可见输出生成时间
= 最后一个可见输出分片时间 − 第一个可见输出分片时间
≈ decoder 阶段的客户端代理
```

后文不能把 TTFT 全部解释为 prefill，也不能把可见输出生成时间写成服务端精确 decoder kernel 时间。

## 2. 核心结论

1. **两条路线的输入长度都随决策轮次近似线性增长。** 原因是同一 thread 保留历史输入和输出；Hybrid 的历史增长速度约为 Direct 的 2.4 倍。
2. **输入变长没有带来同步增长的 TTFT。** 两条路线约 96% 的输入被标记为缓存，输入长度与 TTFT 的相关性都较弱；本次数据不支持“上下文越长，TTFT 必然越长”的简单判断。
3. **输出已经高度规范化。** 所有响应都是符合固定 Schema 的单行 JSON，字段顺序稳定，能够逐轮解析并与结构化记录完全匹配。
4. **Hybrid 的主要瓶颈是固定而冗长的结构化输出。** 它平均生成约 332 个可见 Token，首个输出后的生成阶段平均需要 9.62 秒；输出长度与生成时间的相关系数达到 0.82。
5. **Direct 的主要瓶颈在首个输出之前。** Direct 输出较短，但平均 TTFT 为 7.34 秒。其 reasoning Token 与 TTFT 有一定关系，但排队、网络和输入处理仍混在其中。
6. **精简 Hybrid Schema 会同时改善三项指标。** 它可以缩短当前轮 decoder 时间、抑制后续输入增长，并降低 Token 消耗。

## 3. Token 是怎样统计的

### 3.1 每轮增量

模型接口返回的是 thread 累计用量。第 `n` 轮的独立用量按下式计算：

```text
本轮输入 Token = 第 n 轮累计输入 − 第 n−1 轮累计输入
本轮输出 Token = 第 n 轮累计输出 − 第 n−1 轮累计输出
```

第一轮以前的累计值视为 0。每轮至少保存以下字段：

| 字段 | 含义 |
| --- | --- |
| `inputTokens` | 本轮完整模型输入，包括协议、Schema、图像、当前状态和历史上下文 |
| `cachedInputTokens` | `inputTokens` 中由 provider 标记为缓存命中的部分 |
| `outputTokens` | 本轮全部输出 Token，包含 reasoning Token |
| `reasoningOutputTokens` | 接口单独报告的 reasoning Token |

其中：

```text
可见输出 Token = outputTokens − reasoningOutputTokens
```

`cachedInputTokens` 是 `inputTokens` 的子集，`reasoningOutputTokens` 是 `outputTokens` 的子集，不能再次与总数相加。

### 3.2 输入 Token 包含什么

接口报告的完整输入可能包括：固定控制协议、开发者指令、输出 JSON Schema、任务要求、当前两张图像、机械臂状态、Hybrid 的 π 候选，以及同一 thread 中此前的输入与输出。provider 没有把这些内容的 Token 分项返回，因此现有数据不能精确拆出“图像 Token”“Schema Token”和“历史文本 Token”各自占多少。

### 3.3 输出内容如何核验

【E1／完整性】每轮可见输出的传输分片按接收顺序拼接，拼接结果必须等于保存的原始 JSON；原始 JSON 解析结果必须等于结构化 `response`。同时，所有轮次的 Token 增量之和必须等于接口最终累计用量。

因此，本文的 Token 数不是根据字符数反推，也没有把传输分片数量误当作 Token 数。

## 4. 输入长度的规律

### 4.1 输入随轮次增长

| 指标 | GPT Direct | Hybrid |
| --- | ---: | ---: |
| 第一轮输入 Token | 13,734 | 14,984 |
| 最后一轮输入 Token | 65,941 | 74,329 |
| 每轮输入增长中位数 | 493 | 1,183 |
| 全回合缓存输入占比 | 96.5% | 96.4% |
| 首轮之后非缓存输入中位数 | 676 | 1,323 |

两条路线的输入长度与轮次的 Pearson 相关系数都接近 1，说明它们几乎线性增长。这与持久 thread 保留历史的协议一致。

Hybrid 每轮输入增量明显更大：

```text
Direct：约 493 Token／轮
Hybrid：约 1,183 Token／轮
```

本轮较长的 Hybrid 输出会进入下一轮历史，从而形成累积效应：

```text
本轮输出更长
    ↓
下一轮历史上下文更长
    ↓
后续输入 Token 与缓存上下文继续增长
```

所以，输出协议不仅影响当前轮的生成时间，也影响后续所有轮次的输入长度。

### 4.2 输入长度没有明显推高 TTFT

| 关系 | Pearson 相关系数 |
| --- | ---: |
| Direct 输入 Token 与 TTFT | -0.22 |
| Hybrid 输入 Token 与 TTFT | +0.18 |
| Direct 非缓存输入与 TTFT | +0.04 |
| Hybrid 非缓存输入与 TTFT | +0.15 |

这些相关性都较弱。本次记录中，即使输入总量持续增长，TTFT 也没有同步增长。最后一轮 Direct 的输入为 65,941 Token，但 TTFT 为 4.79 秒；第一轮只有 13,734 Token，TTFT 却为 12.64 秒。

最合理的解释是：大量历史已命中缓存，同时网络、排队、模型内部处理和 reasoning 波动共同影响 TTFT。该结果不能证明输入长度没有成本，只能说明它不是本回合 TTFT 波动的主要解释变量。

## 5. 输出格式与内容规律

### 5.1 Direct 输出

Direct 的 97 轮全部采用 `eef` 模式，顶层字段和顺序完全相同：

```json
{
  "request_id": "...",
  "mode": "eef",
  "steps": 5,
  "reason": "...",
  "target": {
    "position": [0, 0, 0],
    "quaternion_wxyz": [1, 0, 0, 0],
    "gripper_closed": false
  }
}
```

Direct 的变化主要集中在 `reason`、目标位姿、步数和夹爪状态。其可见输出为 109–135 Token，平均约 124.4 Token，长度变异系数只有 4.0%。

### 5.2 Hybrid 输出

Hybrid 的 51 轮全部采用 `student` 模式，即全部放行 π0.5；但每轮仍输出完整固定结构：

```json
{
  "request_id": "...",
  "mode": "student",
  "steps": 5,
  "reason": "...",
  "edit": {...},
  "target": {...},
  "assessment": {...}
}
```

Hybrid 可见输出为 315–362 Token，平均约 331.8 Token，长度变异系数仅为 2.4%。这说明它几乎每轮都在填写同一份固定审核表。

按字符数拆分，Hybrid 平均每轮约 1,397 个字符：

| 内容 | 平均字符占比 | 本回合是否用于执行 |
| --- | ---: | --- |
| `assessment` | 64.3% | 用于记录和解释，不直接转换为动作 |
| `edit` 与 `target` | 约 16.5% | `student` 模式下不执行 |
| `request_id`、`mode`、`steps`、`reason` | 约 18.3% | 决策核心字段 |

规范化输出提高了解析、校验和审计的可靠性，但当前 Hybrid Schema 把“控制所需字段”和“详细分析字段”捆绑在一起，形成了稳定而昂贵的固定输出成本。

## 6. Prefill 代理时间的规律

| 指标 | GPT Direct | Hybrid |
| --- | ---: | ---: |
| 平均 TTFT | 7.34 s | 4.45 s |
| 最小 TTFT | 3.63 s | 2.67 s |
| 中位 TTFT | 6.88 s | 4.28 s |
| 最大 TTFT | 22.00 s | 9.11 s |

Direct 的 TTFT 明显高于 Hybrid，但这不能仅由输入长度解释：Hybrid 平均输入反而更长。Direct 的 reasoning Token 与 TTFT 的 Pearson 相关系数约为 0.42，表明 reasoning 较多的轮次通常更晚出现首个可见输出，但相关性只达到中等水平。

Hybrid 的 reasoning Token 在 51 轮中均由接口报告为 0，TTFT 仍在 2.67–9.11 秒之间变化，说明排队、网络、输入处理等因素同样重要。

基于现有打点，能够得出的结论是：

- Direct 的主要等待发生在首个可见输出以前；
- 总输入 Token 与 TTFT 没有明显正相关；
- reasoning 可能解释 Direct 的部分 TTFT 波动；
- 还不能把 TTFT 拆成准确的网络、排队和 prefill 时间。

## 7. Decoder 代理时间的规律

| 指标 | GPT Direct | Hybrid |
| --- | ---: | ---: |
| 平均可见输出 Token | 124.4 | 331.8 |
| 平均可见输出生成时间 | 3.16 s | 9.62 s |
| 平均 TPOT 估算 | 25.55 ms | 29.06 ms |
| TPOT 中位数 | 26.19 ms | 28.86 ms |
| 输出长度与生成时间相关系数 | 0.43 | 0.82 |

Hybrid 的 TPOT 很稳定：25%–75% 区间仅为 28.79–29.01 ms。按平均值估算：

```text
约 331 个 Token × 29 ms／Token ≈ 9.6 秒
```

这与实测 9.62 秒基本一致。Hybrid 慢的主要原因不是每个 Token 突然变慢，而是固定 Schema 要求它持续生成三百多个 Token。

Direct 的输出长度范围很窄，输出长度与生成时间只呈中等相关。其平均 3.16 秒生成阶段已经明显短于 TTFT，因此 Direct 更应优先调查首输出前的等待，而不是继续压缩已经较短的输出。

## 8. 面向性能的 Schema 优化

### 8.1 按决策模式返回不同字段

当前 Hybrid 无论选择哪种模式都返回 `edit`、`target` 和完整 `assessment`。建议改为模式相关的最小结构。

放行 π：

```json
{
  "request_id": "...",
  "mode": "student",
  "steps": 5,
  "reason": "Candidate remains aligned with the current subgoal."
}
```

修正 π 时增加 `edit`；GPT 接管时增加 `target`；停止时只保留停止理由。即：

```text
student → 核心字段
edit    → 核心字段 + edit
eef     → 核心字段 + target
stop    → request_id + mode + reason
```

本回合中，`request_id + mode + steps + reason` 的平均字符数约为完整 Hybrid JSON 的 18.3%。字符数与 Token 数并非严格同比例，但该结果表明输出仍有很大的压缩空间。

### 8.2 将控制字段和分析字段分层

控制闭环只保留执行所需信息；在线审核如果需要状态记录，可以保留三个短字段：

```json
{
  "current_subgoal": "...",
  "execution_status": "progressing",
  "evidence": "..."
}
```

详细的任务进度、预期意图、预测意图和对齐解释可以在离线报告中生成，或只在检测到异常时要求输出。离线内容必须明确标记，不能伪装成模型执行当时的原始回答。

### 8.3 同时控制历史上下文

精简输出后，还应避免把每轮完整审核文字永久保留在活动上下文中。可以考虑：

- 保留固定协议作为可缓存前缀；
- 将历史压缩为简短的结构化任务状态；
- 只保留最近若干轮动作和异常事件；
- 始终在独立日志中保存原始输入输出，避免上下文裁剪损失实验证据。

## 9. 建议的验证实验

建议先做一个单变量对照，只改变 Hybrid 输出 Schema：

| 项目 | 基线组 | 精简组 |
| --- | --- | --- |
| 任务与初态 | 相同 | 相同 |
| 模型与 reasoning effort | 相同 | 相同 |
| π 候选动作 | 相同协议 | 相同协议 |
| 审核频率 | 每 5 步 | 每 5 步 |
| 输出 | 当前完整 Schema | 模式相关最小 Schema |

至少比较：每轮输入 Token、非缓存输入 Token、可见输出 Token、TTFT、可见输出生成时间、整轮墙钟、接口拒绝数、GPT 介入数和原生任务结果。

如果精简 Schema 能稳定降低输出和整轮时间，再测试事件触发审核。这样可以区分“输出协议优化”和“减少 GPT 调用”各自贡献了多少收益。

## 10. 证据入口

- [GPT 决策表现与推理性能分析](2026-09-22-GPT决策表现与推理性能分析.md)：GPT 能力、耗时瓶颈和总体优化顺序。
- [逐轮决策重新采集记录](2026-09-22-LIBERO10-task0-Ori-init0-逐轮决策实录.md)：实验协议、Token 与计时定义。
- [逐轮汇总数据](evidence/2026-09-22-decision-v2-summary.json)：路线级 Token 和平均耗时。
- [完整性核验结果](evidence/2026-09-22-decision-v2-verification.json)：逐轮数据、轨迹、视频与页面逻辑核验。
- 私有自包含 HTML：`GPT-as-Policy-repro/data/decision_v2_20260922/index.html`，用于查看每轮输入、原始输出、Token、输出分片和时间轴。
