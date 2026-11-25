# DeepSeek 代理计时源码审计

审计日期：2026-10-12。范围：六份当前代理源码与已脱敏 `proxy-inventory.json` 数值。未启动请求或实验，未读取凭据或公开原始会话。

**证据边界：E1 表示当前源码行号；E2 表示现存日志白名单数值。当前源码的 SHA256 不能证明历史运行时加载的源码版本。** 尚未取得各历史代理进程与启动时源码哈希的完整绑定，以下字段解释不可直接升级为每个历史批次均采用相同实现的保证。日志算式相符也不补足这一版本链缺口。

## 1. 当前源码指纹

路径均相对于远程工作根 `${RPENT_DATA_ROOT}/`；不含连接地址。

| 文件 | SHA256 |
|---|---|
| files_proxy.py | `818c09e657f86134785811427de62502c97ecdb6764bc43d85a8633f3183a84b` |
| files_proxy_v2.py | `2dab638341d51df2b6a0fd35d45d5004f2ac518b726d18768f8a87530296d0ac` |
| phase_proxy.py | `f8e42c04b59bff64cac40d439c5675f7bee7ca1ace28ae36ac8741268d3a21a6` |
| llm_timing_proxy.py | `70c392f06754a925a16138e824e2de21e31b3c8a8d6b8095fb8d4299286d47d4` |
| capture_proxy.py | `f7d2fa12bba8c3cd99aedb5fd49f434b4998ab5f1118144ed3e6e6ba6502236f` |
| complete_proxy.py | `0310f5ef67f731d7bc34340d3bc5d222e35355a5fbc024fe60ca0c9bbb7e52ec` |

## 2. 各实现的实际边界（E1）

| 源码与行号 | 可确认的定义与限制 |
|---|---|
| files_proxy.py:178–180、197–200、228–254 | t0 在代理读取本地请求体前；图像 Files API 上传/替换发生于 chat 请求计时前。`proxy_prep_s`（230）覆盖这些前处理；`upload_s` 包围 `urlopen()`，终点是上游响应头返回。 |
| files_proxy.py:212–219、282–303、330–335 | 非流式客户端请求可被转换为上游流式。首个非空 reasoning/content/tool_calls delta 设置首输出时间；`ttft_s=t_first-t0`、`api_wait_s=t_first-t_post_send`、`gen_s=t_end-t_first`、`duration_s=t_end-t0`。无首 delta 时 303 使用结束时间回退，不能认作真实首 token。 |
| files_proxy.py:390–401、415–418 | 另一流式透传路径也记录首 delta/首 content，但该分支不输出 `gen_s`。不能将一个分支的字段假定为所有请求共有。 |
| files_proxy_v2.py:159–161、178–181、200–225 | t0 在本地读取前，图像替换在 chat 请求前；`upload_s` 同样止于上游响应头返回。 |
| files_proxy_v2.py:227–244 | 非流式分支先完整 `resp.read()`，随后回复及处理，再记录 `ttft_s`。该字段是接近全响应完成的耗时，**不是 TTFT**。 |
| files_proxy_v2.py:299–310、324–327 | 流式分支以首个非空 delta 计时；没有 delta 时回退到全程耗时。 |
| phase_proxy.py:29–33、50–71、75–77、100 | `upload` 从 ready 到 `urlopen()` 返回；非流式路径完整读取后才设 `t_first`，即使注释称 first byte 也不是首字节计时。 |
| phase_proxy.py:141–143、177–183、203–208 | 流式路径单字节读取记录首次到达；`api_wait` 为响应头返回→首字节，`download` 为首字节→首个非空 delta。它们都是代理端网络/解析时间段，不是服务端算子阶段。 |
| llm_timing_proxy.py:100–114、124–134 | `times` 对每个非空 reasoning/content 字段追加到达时间；TPOT p50/p90/max 是相邻 delta 字段的到达间隔，**不是每 token 时间**。同一 SSE 可含两个字段，产生极短间隔；tool delta 不进入该列表。`n_chunks` 实际为记录的字段事件数。 |
| llm_timing_proxy.py:115–122 | `download_first_tok` 使用结束时的 `now - t_first_resp`，实际覆盖首响应字节→结束；字段名称与实现不符，不能作为首 token 下载耗时。 |
| llm_timing_proxy.py:185–200、210–230、246、269–281 | 非流式客户端可转换为流式；若上游仍返回非流式，完整读完后才赋 TTFT。重建流式时 reasoning/content 记入 `times`，工具输出可设置 TTFT/首 content，却不追加 `times`。 |
| llm_timing_proxy.py:325、355–364、387–390 | 普通流式分支的 request-to-headers、首输出字段与首字节边界；不能由这些时间恢复服务端 thinking 起止。 |
| capture_proxy.py:49–51、98–119 | t0 在读取请求前；`upload_s` 是 `urlopen()` 前后；非流式完整读取后才记录 `ttft_s`，不是首 token。 |
| capture_proxy.py:163–180 | 流式取首非空 delta，缺失则回退全程。 |
| complete_proxy.py:67–69、102–122、125–143 | `upload_s` 止于响应头返回，源码注释亦如此；非流式 `ttft_s` 在完整读取后记录，不能解释为 TTFT。 |
| complete_proxy.py:198–228 | 流式首 reasoning/content/tool delta 设置 TTFT，首 content/tool 设置 first_content；依次输出 upload、api_wait、ttft、first_content、duration。无 delta 的回退值不是真实首 token。 |

### 跨实现应采用的名称

- `upload_s`：**上游请求开始至响应头返回**。可能同时含 DNS/TCP/TLS、请求发送、服务端排队/处理及响应头等待，不能标为纯上传、纯网络或 prefill。
- 流式 `ttft_s`：**代理处理开始至首个非空输出 delta**，可能是 reasoning、正文或工具字段；工具字段不一定是自然语言 token。
- 非流式分支的同名 `ttft_s`：按上述实现为**完整响应到达附近的耗时**，不能和流式首输出时间直接混合统计。
- `first_content - ttft`：最多是首输出 delta 到首正文/工具 delta 的代理观测间隔；不能命名为真实 thinking 时长。六份源码均未提供服务端 thinking 的真实起止观测。
- `gen_s`：首输出 delta 至流读取结束，包含可能的 reasoning、正文、工具参数、网络等待及尾部协议；不能标为纯 decode。
- `llm_timing_proxy.py` 的 TPOT：应改称 **delta 字段到达间隔**；未按 token 数归一，不能据此比较 token 解码速度。

## 3. 已保存数值的恒等式核验（E2）

仅对所有相关字段均为数值的行核验，未从原始请求正文补值。残差采用绝对最大值；日志以毫秒舍入，表中 0.001 秒量级兼容独立字段舍入。

| 聚合来源文件名 | 行数 | TTFT 分解可核验行 | 最大残差（秒） | duration 分解可核验行 | 最大残差（秒） |
|---|---:|---:|---:|---:|---:|
| llm_10t0.jsonl | 34 | 34 | 0.001 | 34 | 0.001 |
| llm_10t2.jsonl | 38 | 38 | 0.001 | 38 | 0.001 |
| llm_t2par.jsonl | 15 | 15 | 0.001 | 15 | 0.001 |
| llm_t8.jsonl | 17 | 17 | 0.001 | 17 | 0.001 |
| llm_v5.jsonl | 15 | 15 | 0.001 | 15 | 0.001 |
| llm_stream_conv.jsonl | 14 | 0（字段不齐） | — | 14 | 0.001 |

核验式：

```text
ttft_s ≈ proxy_prep_s + upload_s + api_wait_s        （119 行）
duration_s ≈ ttft_s + gen_s                         （133 行）
```

这证明这些记录的字段在算术上闭合；不证明 TTFT 一定观察到真实 delta、不证明历史代码版本、不证明任何一段是纯模型计算时间。

## 4. 报告可用的分解口径

对于字段完整且确认有首 delta 的上游流式记录，可使用：

```text
代理请求耗时 ≈ 前处理 + 请求至响应头 + 响应头至首输出 + 首输出至流结束
            ≈ proxy_prep_s + upload_s + api_wait_s + gen_s
```

前处理可以包含图像文件接口请求；整式从代理 handler 入口起算，不含 planner 调用代理之前的时间。四段应称为**代理观测分段**，不能画成“上传 / prefill / thinking / decode”的四种服务端计算成本。缺字段、无首输出回退、非流式、失败/中断记录应单独标记；不得为使堆叠图闭合而补猜。不同批次还须先按代理实现与流模式分组，再比较同义字段。

聚合证据：`proxy-inventory.json`，SHA256 `7f4460629ac394157cdcc6652e5bb8336cd76874d02ad7cb4c490ce9f9952294`。
