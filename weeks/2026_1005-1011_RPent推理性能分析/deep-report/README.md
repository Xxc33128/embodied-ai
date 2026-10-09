# 两场景独立深度计时报告

这是两场景独立深度测试的报告入口。最终入选固定 8 个身份，计时验收通过：6 集采用 `attempt-001`，Astra / Sol 的 t9、seed1 两集采用独立审查后的 `attempt-002`；环境结果为 3 成功、5 失败。旧 120 集只用于选题与历史参考，不进入本报告的时间、Token 或成功率分母。〔E2〕

两次 t9、seed1 的原 `attempt-001` 因 SAM NPU OOM 排除，原轨迹、录像、结果与 hash 留存。其 3,205.2 秒 episode 累计工作量与 326.209433478 秒冷加载工作量另记复现成本；前者不是并行墙钟，后者已包含于全部 attempt 加载工作量，不能重复加总。参见 [排除与资源 epoch 账本](evidence/excluded-attempt-cost-ledger.json) 和 [排除 attempt 的冷加载证据](evidence/excluded-attempt-model-init-evidence.json)。非正式探针单列，不加入正式 8 集。〔E2〕

所有未知量保留 `null`，页面显示 `—`；即使批次完成，没有服务端遥测的 prefill / 完整 thinking 也不填数值。

生成：

```sh
python3 build_deep_report.py
```

默认读取同目录 `report-data.json`，生成 `RPent-两场景深度计时报告.html`。可用 `--data` / `--output` 指定路径。输出自包含，支持离线筛选、CSV 导出、手机阅读和打印；只更新 JSON 即可重建。

## 单一 JSON 数据契约

- `schema_version: 1`；`title`、`batch`、`updated_at`（时间字符串或 null）。
- `data_status`：`awaiting_data` / `partial` / `complete`。最终发布为 `complete`，六集等中间预览保持 `partial`；这是数据发布状态，不能用来推断实际进程是否在运行。
- `note`：当前批次数据范围与完成度说明。
- `conclusion`：首屏结论；`optimization`：按优先级排列的字符串列表，主张附 E2/E3 等级。
- `questions.q1/q2/q3`：各含 `conclusion`、`evidence`（字符串列表）、`boundary`。
- `scenes`：t0、t9 两项。`conclusion`、`trajectory_findings`（字符串列表）、`ideal_nodes`、`ideal_assumptions`（列表）、`baseline_s`、`baseline_kind`（必须明确实测或条件估算）；`representative_episode`留作图像/轨迹扩展。未验证的节点与基线不预填历史数值。
- `io`：列表，每项为 `module`、`input`、`output`、`context`、`evidence`，均为脱敏说明字符串；空列表时页面仅展示核验项目。
- `evidence`：整个新批次的脱敏来源/哈希/计时边界说明列表。

`episodes`始终包含固定 8 个身份：`libero_10_swap` 的 t0/t9 × seed0/1 × Astra/Sol，均为 medium。身份字段 `id/model/reasoning_effort/suite/task/seed` 不改变。

| 字段 | 含义 |
|---|---|
| `attempt` | 独立 attempt 名称，尚未接入为 null |
| `status` | `unreported` / `planned` / `running` / `completed` / `infra_failed` / `needs_review` |
| `environment_success` | 真实环境判定，true/false/null；未知不能写 false |
| `measurement_invalid` | 计时独立验收结果，true/false/null；不覆盖任务结果 |
| `elapsed_s`、`load_s` | episode与另计模型加载秒数，未知为 null |
| `request_count`、`vla_chunks` | 计量响应数量与 π 调用块数，未知为 null |
| `manual_primitive_s`、`vla_primitive_s` | 非 π / π 动作原语累计秒数 |
| `tool_other_s`、`sdk_non_mcp_s`、`outside_sdk_s` | 工具附加残差、SDK 内非工具残差、episode 内 SDK 外开销残差 |
| `stages` | `{name, seconds, scope, status, evidence}`；scope注明inclusive/exclusive，不把阶段再加到五桶 |
| `requests` | `{stream_lifetime_s, ttft_s, ttft_request_linked, input_tokens, cached_input_tokens, reasoning_tokens}`；缺项为 null。请求流墙钟包含客户端可见全过程，不是纯服务端推理；仅 `ttft_request_linked=true` 才展示本请求 TTFT，禁止按行号强配。旧 `elapsed_s` 不回退作流寿命 |
| `trajectory` | `{step, action, seconds, progress, assessment}`；只写本批可复核进展与判定 |
| `evidence` | 该集脱敏引用/哈希/缺测说明字符串列表 |

计时验收与任务结果独立。首屏顶层公式为 `T_episode报告计时 = T_SDK报告时长 + T_episode−SDK残差`；可选每集 `top_timing={verified,sdk_interval_s,outside_sdk_s,evidence}`。仅显式 `verified=true` 且已结束、计时有效的样本进入顶层累计数值。verified 必须来自源码计时口径与字段对账核验，不代表残差为独立实测；两个非负项之和还须与 elapsed_s 在 0.01 秒内闭合。

五桶是否展开取决于逐集 `five_bucket_evidence.classification` 和数值闭合检查（见下文）。`source_backed_step_association` 由源码与逐步关联证据支持；仅有 `conditional_serial_primitive` 的样本仍按条件模式展示。代数闭合不能独立证明互斥，builder 不据此提升证据等级或设置 verified。

## 发布边界

只接受脱敏汇总，不输入认证、IP、原始会话或完整请求正文。原始文件保留在独立实验现场。客户端 TTFT 与可见 reasoning 不等于真实云端 prefill / 完整 thinking；缺失阶段不补零。未做策略重放时，压缩轮数和理想路径只作为条件推导。

## 云端与批次时长扩展（可选，仍为 schema v1）

每集 `cloud` 可包含 `logical_rounds`、`sampling_attempts`、`completed_responses`、`http_attempts`、`websocket_attempts`（不同计数，不能互换），以及 `transport_union_s`、`prefill_s`、`hidden_thinking_s`。无服务端遥测时后两项为 null，不从 TTFT 或 SDK 差值推算。`transport_union_s` 为可关联传输区间并集，不把重叠请求时长重复相加。字段未提供也展示未知。

可另给 `cloud.response_ttft_distribution={count,p50_s,p95_s}`，展示独立响应 TTFT 分布；不绑定逐请求，也不解释成精确 prefill。

顶层可选 `batch_timing={wall_s,model_load_work_s,pair_wait_union_s,review_pause_union_s}`：`wall_s` 来自批次真实起止；`model_load_work_s` 是两 lane 模型加载累计工作量；两个 union 值均按批次时钟轴取区间并集。它们仅作解释量，可能互相/与运行重叠，不能直接相加得墙钟。未结束批次的最终 `wall_s` 保持 null。T_episode 的五桶仍单集闭合，首屏累计五桶表示工作量，不表示 T_batch。

## Q3 Token 用量语义

每集可选 `cloud.token_usage={response_count,input_min,input_median,input_max,input_sum,cached_input_sum,output_sum,reasoning_sum}`。范围与中位数描述单个有用量证据响应的 input；input_sum 是跨响应累计（含重复历史），绝不是一次请求的 context 长度。cached_input_sum 是 input_sum 的子集，reasoning_sum 是 output_sum 的子集，均不得相加重复计量。模板不预填实验数值、不从字段间差异反推真实 prefill/thinking 时长。Q3 主线只放每集一行摘要，逐请求保留折叠。

π prompt 字段在 `io.context` 明确写实际 `nonpadding_count` 与 padding 张量 `shape`，不要只给“上下文长度”一个模糊数值；Astra input 计量可能包含多模态、工具 Schema 和历史，不要求等于可见字符串的 tokenizer 数。

计时边界补充：episode 在 init_runtime 后起算，toolkit.close / daemons.stop 后取差，随后仍有 transcript 写盘和 finalize_run；SDK 在 MCP 启动后起算、MCP stop 后取差。outside_sdk_s 是各自 round(…,1) 时长的残差，不解释为独立实测准备/收尾，也不称全进程墙钟。源码证据见 cloud-first-pair-contract.json。

## 前置公式的证据切换

每集可选 `five_bucket_evidence={classification,evidence}`，其中 `classification="source_backed_step_association"` 表示已由 step_id 唯一关联原语与 MCP、验证 MCP 串行/时长包含，并有同步 handler 源码依据；`evidence` 留脱敏出处。不可仅凭相加闭合赋予该分类。

所有纳入样本（completed、计时有效且 elapsed_s 存在）都具备该分类、五个非负有限数齐全且与 episode 闭合，首屏才展开五项公式与累计秒数。任一纳入样本缺证据或缺字段时整体退回顶层 / 条件模式，不筛掉弱证据样本来改变分母。数值不闭合仍由 validate 拒绝生成。工具附加、SDK 非工具、SDK 外开销均保留“残差”含义；SDK 非工具项位于 SDK 区间内部，SDK 外开销位于 episode 计时内部、SDK 区间外部。语义包含验证不使它们变成纯物理 / 纯 thinking。顶层具体计时边界收在公式后折叠。

前置结构模型只解释优化变量：轮数 × 每轮模型等待 / 工具成本；π 的前处理、prefix 合并、denoise、后处理与工具内部控制 / 物理 / 渲染 / RPC 属嵌套项，不是新增记账桶。

末尾证据折叠含两篇 GPT-as-Policy 方法报告的固定相对链接；只引用分析方法，不混用历史数值。三问的补充 evidence 默认折叠，结论与边界保持可见。

## Q1 实测重点与批次主表

`questions.q1.measured_summary` 是可选列表，最多 6 项，每项 `{label,seconds,scope,evidence}`。seconds 为非负有限数或 null；其他字段为字符串。它紧随 Q1 结论显示实际秒数，不需要展开证据。scope 必须明确原语窗/进程和 inclusive / exclusive；render、physics、π handler、prefix、denoise 等可能嵌套，禁止直接相加。空列表不展示该表。

批次主表展示 `batch_timing.gate_envelope_wall_s`（首个 before_pair 到最终 after_pair，未结束为 null）、`model_load_work_s`（累计工作量）和 `observed_inter_pair_gap_s`（可观测配对间隔）。纯审查和关键等待未知时只在脚注说明，不用大块未知单元格代替已知量。`wall_s` 若有独立全进程起止证据则脚注另列；闸门范围不冒充全进程墙钟。三列可能包含或重叠，不相加。

Q3 可选顶层 `io_details=[{module,field,type_shape,meaning,evidence}]`，各值为脱敏说明字符串，显示在默认关闭的“逐字段接口证据”内；主表 `io` 继续仅放模块级收发与长度概要。空列表/未提供时不渲染折叠。字段内容均 HTML 转义，不支持任意 HTML。

可选 `evidence_files=[{href,label,description}]` 在末尾提供脱敏 JSON 证据链接。href 仅接受 `report-data.json` 或 `evidence/` 下 JSON，拒绝路径上跳、协议和反斜线；最终交付前核验目标文件存在。

Q1 实测表主视图只保留组件、累计秒数和可选 episode 占比；逐行 scope/evidence 收入“组件计时边界与证据”折叠。若 `q1.measured_summary_basis` 提供 `episode_count/episode_denominator_s`，显示统一分母与占比，不改变秒数。嵌套不可相加提示常显。


## 排除 attempt 与资源 epoch（可选，schema v1）

`episodes` 只保留最终入选的 8 个身份及其选定 attempt；不要把被排除 attempt 追加成第 9 / 10 集。`excluded_attempts` 与 `resource_epochs` 可直接接入上述成本账本的同名数组。未提供时不展示成本附加区域。

- `excluded_attempts[]`：`model/task/seed/attempt` 精确标识；`episode_elapsed_s` 为原 episode 工作秒数，`load_s` 为另计冷加载工作秒数；`exclusion_category` 和 `review_id` 给出原因与审查出处。账本还保留原 state、artifact、measurement 等 hash。页面兼容同义字段 `elapsed_s/reason/evidence`，但同时提供两种 elapsed 字段时数值必须一致。
- `resource_epochs[]`：账本的 `epoch`、`sam_physical_device`、`device_model`、`role` 或 `change`、`review_sha256` 或 `sam_launch_script_sha256`。页面也接受展示形式 `{id,description,evidence}`；可选 `attempt_count/model_load_work_s` 仅在有覆盖证据时填写，不从设备变更推算性能收益。
- `evidence_files` 添加成本账本与冷加载证据的安全相对链接；逐文件 hash 明细保留在证据 JSON，主文不铺开。

主文只区分“验收通过的分析分母”与“排除 attempt 的复现消耗”；原因和资源变更放默认关闭的成本明细。汇总只加明确存在的 `episode_elapsed_s`，缺项标为已观测下限，不补零。排除项不得与同一已入选 attempt 重复；重复身份、非有限或负耗时均拒绝生成。

`batch_timing.model_load_work_s` 覆盖全部实际 attempt 的已闭合模型初始化 span，包括排除项；它可能跨 lane 并行，不是墙钟。最终入选 8 集的冷加载子集与全部 attempt 工作量必须分别保留；不能把排除项的 `load_s` 再加到全部加载总数。资源 epoch 仅解释运行条件变化，不能将 OOM 前后两组直接当成相同条件下的优化收益。〔E2/E3〕

## 2026-10-09 逐集时间轴增补

公式后新增 8 集事件时间轴，可点选阶段并展开完整顺序。`extract_timeline.py` 只读提取本批已验收 attempt 的工具事件，输出 `evidence/timeline-eight.json`，与既有 stream SHA256 核对后接入 `report-data.json`。原点为首个有时间戳的 item 事件；不是 episode 起点。模型加载另计，事件窗外差额不虚构首尾位置，工具间隔不标作 thinking。

此次增量审查及新版产物 hash 见 `evidence/timeline-review.json`；原三份 `final-*-review.json` 保留对应时间轴增补前版本的审查，不代表新增页面已由三人重审。
