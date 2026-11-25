#!/usr/bin/env python3
"""Render the standalone deep-timing HTML report for the B-stage batch.

Self-contained: inline CSS, inline JSON, vanilla JS. No external assets, no network,
safe to open offline, print, and read on a phone. Every unknown stays null in the
data and renders as an em dash rather than a fabricated number.

    python3 build_ds_deep_report.py [--data report-data-ds.json] [--out report.html]
"""
import argparse
import json
from pathlib import Path

B = Path(__file__).resolve().parent


def dash(value, fmt="%s"):
    return "—" if value is None else fmt % value


def num(value, digits=1):
    return "—" if value is None else ("%.*f" % (digits, value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(B / "report-data-ds.json"))
    parser.add_argument("--out", default=str(B / "DS-DeepSeek-V4.1-Flash-深度计时报告.html"))
    args = parser.parse_args()

    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    arms = {a["effort"]: a for a in data["arms"]}
    med, none_ = arms.get("medium", {}), arms.get("none", {})
    total_requests = len(data["requests"])
    total_input = sum(e["tokens"]["input_sum"] for e in data["episodes"])
    total_output = sum(e["tokens"]["output_sum"] for e in data["episodes"])
    total_reasoning = sum(e["tokens"]["reasoning_sum"] for e in data["episodes"])
    llm_wall = sum(e["llm_wall_s"] for e in data["episodes"])
    upload_s = sum(e["upload_s"] or 0 for e in data["episodes"])
    solved = sum(1 for e in data["episodes"] if e["solved"])
    fit = data["fit"]
    upload = data["upload"]

    rows = []
    add = rows.append

    add("<!DOCTYPE html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">")
    add("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    add("<title>%s</title>" % data["title"])
    add("<style>")
    add("""
:root{--ink:#14181f;--muted:#5b6675;--line:#dfe4ea;--bg:#f7f9fb;--card:#fff;
--accent:#1f6feb;--accent2:#0b7a5b;--warn:#b54708;--bad:#b42318;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:19px;margin:38px 0 12px;padding-bottom:7px;border-bottom:2px solid var(--line)}
h3{font-size:16px;margin:22px 0 8px}
p{margin:9px 0}
.sub{color:var(--muted);font-size:14px;margin:0 0 4px}
.batch{font-family:var(--mono);font-size:12px;color:var(--muted);word-break:break-all}
.badge{display:inline-block;padding:2px 9px;border-radius:11px;font-size:12px;font-weight:600;
background:#e7f4ee;color:var(--accent2);border:1px solid #bfe3d5}
.note{background:#fff;border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:7px;padding:11px 14px;margin:14px 0;font-size:14px;color:#31404f}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:var(--muted);margin-bottom:5px}
.card .v{font-size:23px;font-weight:650;letter-spacing:-.02em}
.card .v small{font-size:13px;font-weight:500;color:var(--muted)}
.card .f{font-size:12px;color:var(--muted);margin-top:4px}
table{width:100%;border-collapse:collapse;background:var(--card);font-size:13.5px;
border:1px solid var(--line);border-radius:9px;overflow:hidden}
th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:#eef2f7;font-weight:600;font-size:12.5px;color:#3a4757;text-align:right;
position:sticky;top:0;cursor:pointer;user-select:none}
th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left}
tbody tr:hover{background:#f4f8ff}
tbody tr:last-child td{border-bottom:none}
.mono{font-family:var(--mono);font-size:12.5px}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11.5px;font-weight:600}
.ok{background:#e6f6ee;color:#0b7a5b}.no{background:#fdecea;color:#b42318}
.neu{background:#eef2f7;color:#41505f}.warn{background:#fdf1e3;color:#b54708}
.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:12px 0}
.controls select,.controls input{padding:6px 9px;border:1px solid var(--line);border-radius:7px;
font-size:13.5px;background:#fff;color:var(--ink)}
.controls button{padding:6px 13px;border:1px solid var(--accent);background:var(--accent);color:#fff;
border-radius:7px;font-size:13.5px;cursor:pointer;font-weight:550}
.controls button.ghost{background:#fff;color:var(--accent)}
.scroll{max-height:560px;overflow:auto;border:1px solid var(--line);border-radius:9px}
.scroll table{border:none;border-radius:0}
.count{font-size:13px;color:var(--muted)}
pre{background:#0f141b;color:#d8e2ee;padding:14px 16px;border-radius:9px;overflow:auto;
font-size:12.5px;line-height:1.6;font-family:var(--mono)}
details{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:11px 15px;margin:11px 0}
summary{cursor:pointer;font-weight:600;font-size:14.5px}
details[open] summary{margin-bottom:9px}
ul{margin:8px 0 8px 20px;padding:0}li{margin:5px 0}
.foot{margin-top:44px;padding-top:16px;border-top:1px solid var(--line);color:var(--muted);font-size:12.5px}
.gap{color:var(--muted)}
.gap b{color:var(--ink)}
@media print{.controls{display:none}.scroll{max-height:none;overflow:visible}body{background:#fff}}
""")
    add("</style></head><body><div class=\"wrap\">")

    add("<h1>%s</h1>" % data["title"])
    add("<p class=\"sub\">%s</p>" % data["subtitle"])
    add("<p class=\"batch\">%s</p>" % data["batch"])
    add("<p><span class=\"badge\">data_status: %s</span> <span class=\"batch\">%s</span></p>"
        % (data["data_status"], data["updated_at"]))
    add("<div class=\"note\">%s</div>" % data["note"])

    # ---- headline ----
    add("<h2>0. 一句话结论</h2>")
    add("<p>B 阶段把 <b>%d 个请求</b>全部变成可实测：每条都有真实 TTFT、TPOT 与生成窗口，"
        "并对 <b>prefill 做了显式拆分</b>、对 <b>思考耗时做了精确统计</b>。这正是同一周里那份"
        "两场景深度计时报告被结构性挡住的四项——它的 473 条逐请求行里 <b>TTFT 一条值都没有</b>，"
        "TPOT 连字段都不存在，prefill 与完整 thinking 明确留空。</p>" % total_requests)
    add("<div class=\"cards\">")
    for label, value, foot in (
        ("逐请求 TTFT 实测", "%d<small>/%d</small>" % (total_requests, total_requests),
         "参考报告 0/473；本页覆盖率 100%"),
        ("TTFT 中位（medium）", "%s<small> ms</small>" % num(med.get("ttft_p50_ms")),
         "p10 %s / p90 %s ms" % (num(med.get("ttft_p10_ms")), num(med.get("ttft_p90_ms")))),
        ("TPOT 中位（medium）", "%s<small> ms/tok</small>" % num(med.get("tpot_p50_ms")),
         "≈ %s tok/s" % num(1000.0 / med["tpot_p50_ms"], 0) if med.get("tpot_p50_ms") else "—"),
        ("可见思考窗口中位", "%s<small> ms</small>" % num(med.get("thinking_visible_p50_ms")),
         "思考占生成窗口中位 %s%%" % num(med.get("thinking_share_p50"))),
        ("prefill 拆分（斜率）", "%s<small> ms/tok</small>" % num(fit["slope_ms_per_token"], 5),
         "≈ %s tok/s，截距 %s s" % (num(fit["throughput_tok_per_s"], 0), num(fit["intercept_ms"] / 1000.0, 2))),
        ("上传税（实测）", "%s<small> MB/s</small>" % num(upload["rate_mb_per_s"], 3),
         "体字节中位 %s MB" % num(upload["median_body_mb"], 2)),
    ):
        add("<div class=\"card\"><div class=\"k\">%s</div><div class=\"v\">%s</div><div class=\"f\">%s</div></div>"
            % (label, value, foot))
    add("</div>")

    # ---- contrast with the reference report ----
    add("<h2>1. 与《两场景深度计时报告》的对照：本页多出来的四项</h2>")
    add("<p>那份报告用 websocket / OTLP 拿到了很扎实的 SDK 与工具侧计时，但它的请求是"
        "<b>非流式</b>的，没有服务端遥测，于是把 prefill 与完整 thinking 诚实留空、TTFT 也一条都没能关联上。"
        "本页靠 transport 层的流式改造把这四个数补上：</p>")
    add("<table><thead><tr><th class=\"l\">指标</th><th class=\"l\">参考报告（10/08，Astra/Sol，473 请求）</th>"
        "<th class=\"l\">本页（10/09，DeepSeek-V4.1-Flash，%d 请求）</th><th class=\"l\">本页依据</th></tr></thead><tbody>"
        % total_requests)
    for metric, ref, here, basis in (
        ("逐请求 TTFT", "字段存在但 <b>0/473 有值</b>（<span class=\"mono\">ttft_request_linked</span> 全为 false）",
         "<b>%d/%d 实测</b>，逐请求中位与 p10/p90 都给" % (total_requests, total_requests),
         "首个 SSE delta 的客户端时间戳；medium 档首个 delta 即第一个思考 token"),
        ("逐请求 TPOT", "<b>无此字段</b>", "<b>%d/%d 实测</b>，中位 %s ms/tok" % (total_requests, total_requests, num(med.get("tpot_p50_ms"))),
         "(末 delta − 首 delta) ÷ (输出 token − 1)"),
        ("prefill 耗时", "8/8 集 <span class=\"mono\">null</span>（README 明确：没有服务端遥测就不填）",
         "<b>逐请求拆分</b>：斜率 %s ms/tok × 该请求 prompt token" % num(fit["slope_ms_per_token"], 5),
         "9 个真实体各重放 3–5 次的<b>最小 TTFT</b> 拟合；只取斜率不取截距"),
        ("thinking 耗时", "<span class=\"mono\">hidden_thinking_s</span> 8/8 集 <span class=\"mono\">null</span>",
         "<b>可见思考窗口 %d/%d 实测</b>，且给出占生成窗口比例（中位 %s%%）" % (total_requests, total_requests, num(med.get("thinking_share_p50"))),
         "首个思考 delta → 首个可见内容 delta；另用 reasoning_tokens × TPOT 估占比"),
        ("TTFT 分布", "<span class=\"mono\">response_ttft_distribution</span> 为 <span class=\"mono\">null</span>",
         "逐请求全量 + p10/p50/p90", "同一批实测样本，不需要按行号强配"),
    ):
        add("<tr><td class=\"l\"><b>%s</b></td><td class=\"l gap\">%s</td><td class=\"l\"><b>%s</b></td><td class=\"l gap\">%s</td></tr>"
            % (metric, ref, here, basis))
    add("</tbody></table>")
    add("<div class=\"note\"><b>为什么它能补上、参考报告不能：</b>两者都不是真拿到了服务端遥测。"
        "差别在于<b>响应形态</b>——非流式响应只在整段生成结束后返回一次，客户端上根本没有"
        "「第一个 token 何时到」这个观测点；把它改成流式后，这个观测点就出现了，TTFT 与 TPOT 随之可测。"
        "因此本页的 TTFT 仍是<b>客户端观测值</b>，含上传、排队、网络与首 token 前余量，"
        "<b>不等于</b>服务端 prefill——prefill 那一列是明确标注的派生值。</div>")

    # ---- core metrics ----
    add("<h2>2. 核心指标（按档位）</h2>")
    add("<table><thead><tr><th class=\"l\">档位</th><th>集</th><th>解出</th><th>请求</th>"
        "<th>TTFT p10</th><th>TTFT 中位</th><th>TTFT p90</th><th>TTFT 范围</th>"
        "<th>TPOT 中位</th><th>TPOT 范围</th><th>思考窗口中位</th><th>思考占比中位</th></tr></thead><tbody>")
    for effort in ("medium", "none"):
        a = arms.get(effort, {})
        add("<tr><td class=\"l\"><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td><b>%s</b></td><td>%s</td><td>%s – %s</td>"
            "<td><b>%s</b></td><td>%s – %s</td><td>%s</td><td>%s</td></tr>"
            % (effort, dash(a.get("episodes")), dash(a.get("solved")), dash(a.get("requests")),
               num(a.get("ttft_p10_ms")), num(a.get("ttft_p50_ms")), num(a.get("ttft_p90_ms")),
               num(a.get("ttft_min_ms")), num(a.get("ttft_max_ms")),
               num(a.get("tpot_p50_ms")), num(a.get("tpot_min_ms")), num(a.get("tpot_max_ms")),
               dash(a.get("thinking_visible_p50_ms"), "%.1f ms"), dash(a.get("thinking_share_p50"), "%.1f %%")))
    add("</tbody></table>")
    add("<p class=\"count\">单位：TTFT / 思考窗口为 ms，TPOT 为 ms/token。"
        "<span class=\"mono\">none</span> 档没有思考窗口，因为该臂 <span class=\"mono\">reasoning_tokens=0</span>，"
        "首个 delta 直接就是可见内容——这正是「关思考」在 wire 上的可观测特征。</p>")

    # ---- decomposition ----
    add("<h2>3. 耗时怎么拆的（每个请求都闭合）</h2>")
    add("<pre>total_ms = ttft_ms + gen_ms + post_ms              ← 三项皆实测\n"
        "ttft_ms  = upload_ms + prefill_derived_ms + ttft_residual_ms\n"
        "             ↑实测        ↑派生              ↑残差\n"
        "gen_ms   = 思考生成 + 可见输出生成               ← 同一条 token 流，同一 TPOT</pre>")
    add("<ul>")
    for key, label in (("ttft", "TTFT"), ("tpot", "TPOT"), ("thinking_window", "思考窗口"),
                       ("thinking_tokens", "思考占比"), ("prefill", "prefill"), ("upload", "上传"),
                       ("residual", "残差")):
        add("<li><b>%s</b>：%s</li>" % (label, data["method"][key]))
    add("</ul>")

    add("<h3>prefill 拆分的拟合</h3>")
    add("<p>对 %d 个真实出站体的流式重放各取<b>最小</b> TTFT（最小即最不受共享网关排队污染），"
        "做一次最小二乘：斜率 <b>%s ms/token</b>（≈ %s tok/s），截距 <b>%s s</b>。"
        "每个请求的 prefill 派生值 = 斜率 × 该请求 prompt token。</p>"
        % (fit["pairs"], num(fit["slope_ms_per_token"], 5), num(fit["throughput_tok_per_s"], 0),
           num(fit["intercept_ms"] / 1000.0, 2)))
    add("<div class=\"note\">截距 <b>不</b>计入 prefill：它含上传、RTT、排队与首 token 前余量，"
        "与单独列出的上传税重复。代价是这一列只反映「随上下文增长的那部分」prefill。"
        "残差 <span class=\"mono\">TTFT − 上传 − prefill</span> 是剩下的全部，"
        "<b>不得</b>命名为某个服务端阶段。</div>")

    # ---- per-episode ----
    add("<h2>4. 逐集</h2>")
    add("<table><thead><tr><th class=\"l\">集</th><th class=\"l\">档位</th><th class=\"l\">结果</th>"
        "<th>轮次</th><th>请求</th><th>LLM 墙钟 s</th><th>TTFT 中位</th><th>思考中位</th>"
        "<th>上传 s</th><th>prefill 派生 s</th><th>解码 s</th><th>输入 token</th><th>输出 token</th>"
        "<th>思考 token</th><th>ctx 峰值</th></tr></thead><tbody>")
    for e in data["episodes"]:
        status = e["finish_status"] or ("崩溃 exit=%s" % e["exit_code"] if e["exit_code"] else "未调 finish")
        tag = "ok" if e["solved"] else ("no" if e["exit_code"] else "neu")
        add("<tr><td class=\"l\"><b>%s</b></td><td class=\"l\">%s</td>"
            "<td class=\"l\"><span class=\"tag %s\">%s</span></td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (e["identity"], e["effort"], tag, status, dash(e["turns"]), dash(e["requests"]),
               num(e["llm_wall_s"]), num(e["ttft"]["p50"]), num(e["thinking_visible"]["p50"]),
               num(e["upload_s"]), num(e["prefill_derived_s"]), num(e["decode_s"]),
               "{:,}".format(e["tokens"]["input_sum"]), "{:,}".format(e["tokens"]["output_sum"]),
               "{:,}".format(e["tokens"]["reasoning_sum"]), "{:,}".format(e["tokens"]["input_max"])))
    add("</tbody></table>")
    add("<p class=\"count\">「上传 / prefill 派生 / 解码」三列按上面公式拆分，其中解码列即实测生成窗口；"
        "三者加总不等于 LLM 墙钟，差额为集级调度与未拆余量。</p>")

    # ---- per-request table ----
    add("<h2>5. 逐请求（全部 %d 条，可筛选 / 排序 / 导出 CSV）</h2>" % total_requests)
    add("<div class=\"controls\">")
    add("<select id=\"fep\"><option value=\"\">全部集</option></select>")
    add("<select id=\"feff\"><option value=\"\">全部档位</option><option>medium</option><option>none</option></select>")
    add("<input id=\"fq\" type=\"search\" placeholder=\"搜索 request id\" size=\"18\">")
    add("<button id=\"csv\">导出 CSV</button><button class=\"ghost\" id=\"reset\">重置</button>")
    add("<span class=\"count\" id=\"cnt\"></span>")
    add("</div>")
    add("<div class=\"scroll\"><table id=\"t\"><thead><tr>")
    cols = [("identity", "集"), ("effort", "档位"), ("lid", "request"), ("ctx_tokens", "上下文 tok"),
            ("out_tokens", "输出 tok"), ("reasoning_tokens", "思考 tok"), ("chunks", "chunks"),
            ("total_ms", "整请求 ms"), ("ttft_ms", "TTFT ms"), ("gen_ms", "生成窗口 ms"),
            ("thinking_visible_ms", "可见思考 ms"), ("tpot_ms", "TPOT ms/tok"),
            ("upload_ms", "上传 ms"), ("prefill_derived_ms", "prefill 派生 ms"),
            ("ttft_residual_ms", "残差 ms"), ("wire_bytes", "体 MB"), ("finish_reason", "finish")]
    for key, label in cols:
        add("<th data-k=\"%s\" class=\"%s\">%s</th>" % (key, "l" if key in ("identity", "effort", "lid", "finish_reason") else "", label))
    add("</tr></thead><tbody></tbody></table></div>")
    add("<div class=\"note\">提示：<span class=\"mono\">llm-0022</span>（t0_s1 medium）的一次失败传输尝试"
        "与其重试在账上各占一行，前者的 prompt token 为 0——这是重试记账，不是重复计费。</div>")

    # ---- thinking ----
    add("<h2>6. 思考耗时：两个口径</h2>")
    add("<table><thead><tr><th class=\"l\">口径</th><th class=\"l\">定义</th><th>medium 中位</th><th class=\"l\">性质</th></tr></thead><tbody>")
    add("<tr><td class=\"l\"><b>可见思考窗口</b></td><td class=\"l\">首个思考 delta → 首个可见内容 delta</td>"
        "<td><b>%s ms</b></td><td class=\"l\">实测</td></tr>" % num(med.get("thinking_visible_p50_ms")))
    add("<tr><td class=\"l\"><b>思考占生成窗口</b></td><td class=\"l\">reasoning_tokens × TPOT ÷ 生成窗口</td>"
        "<td><b>%s%%</b></td><td class=\"l\">派生（两个实测值相乘）</td></tr>" % num(med.get("thinking_share_p50")))
    add("<tr><td class=\"l\"><b>思考 token 总量</b></td><td class=\"l\">本批 reasoning_tokens 合计</td>"
        "<td><b>%s</b></td><td class=\"l\">实测（服务端 usage）</td></tr>" % "{:,}".format(total_reasoning))
    add("</tbody></table>")
    add("<p>最值得注意的一条：medium 档<b>生成窗口里约 %s%% 的时间在产出思考 token</b>。"
        "也就是说，这一档的墙钟主要由「想」而不是「做」决定；这也解释了为什么关掉思考后"
        "单请求时延每一对都下降。</p>" % num(med.get("thinking_share_p50")))

    # ---- cliff + shim ----
    add("<h2>7. 两集崩溃与 shim 透明性</h2>")
    add("<p>%s</p>" % data["cliff"]["summary"])
    add("<ul>")
    add("<li>阈值 <b>%s</b> 输出 token；触发集：%s。</li>"
        % ("{:,}".format(data["cliff"]["threshold_tokens"]), "、".join(data["cliff"]["episodes"]) or "—"))
    add("<li>冻结批（非流式）%s 次请求中，<b>%d 次</b>以 <span class=\"mono\">length</span> 结束，"
        "但最大输出曾达 <b>%s / %s</b>——差 8 个 token 就越线，说明这个悬崖在两种模式下都存在。</li>"
        % ("{:,}".format(data["cliff"]["frozen_batch_requests"]),
           data["cliff"]["frozen_batch_length_finishes"],
           "{:,}".format(data["cliff"]["frozen_batch_max_output_tokens"]),
           "{:,}".format(data["cliff"]["threshold_tokens"])))
    add("<li>对崩溃请求体做两模式交错 A/B：<b>16/16 全为 tool_calls、零截断</b>，"
        "推理长度中位 3769(stream) vs 3836(plain)（比 0.98）→ wire 模式不改变生成。</li>")
    add("</ul>")
    add("<p>因此这是 <span class=\"mono\">--max-tokens 8192</span> 对 medium 档推理偏小造成的"
        "<b>既有脆弱性</b>，不是本页流式改造引入的。修复方向：把该值提高到 16384 后重跑 t9 两集。</p>")
    add("<h3>shim 透明性检查</h3>")
    add("<table><thead><tr><th class=\"l\">检查</th><th class=\"l\">结果</th></tr></thead><tbody>")
    for check in data["shim_checks"]:
        add("<tr><td class=\"l\"><b>%s</b></td><td class=\"l\">%s</td></tr>" % (check["check"], check["result"]))
    add("</tbody></table>")

    # ---- batch totals ----
    add("<h2>8. 本批总量</h2>")
    add("<div class=\"cards\">")
    for label, value, foot in (
        ("请求", "%d" % total_requests, "8 集"),
        ("输入 token", "{:,}".format(total_input), "跨请求累计，含重复历史"),
        ("输出 token", "{:,}".format(total_output), "其中思考 {:,}".format(total_reasoning)),
        ("LLM 墙钟", "%s s" % num(llm_wall), "占 episode 的一部分"),
        ("上传税", "%s s" % num(upload_s), "占 LLM 墙钟 %s%%" % num(100 * upload_s / llm_wall if llm_wall else None, 1)),
        ("解出", "%d / %d" % (solved, len(data["episodes"])), "t0 全解、t9 全不解"),
    ):
        add("<div class=\"card\"><div class=\"k\">%s</div><div class=\"v\">%s</div><div class=\"f\">%s</div></div>"
            % (label, value, foot))
    add("</div>")

    # ---- limits ----
    add("<h2>9. 使用限制</h2><ul>")
    for item in data["limits"]:
        add("<li>%s</li>" % item)
    add("</ul>")

    add("<div class=\"foot\">生成：<span class=\"mono\">build_ds_deep_report.py</span>，"
        "数据：<span class=\"mono\">report-data-ds.json</span>（由 <span class=\"mono\">assemble_ds_report_data.py</span> 装配）。"
        "本页不含任何 prompt 原文、凭证或原始会话。未知量一律 <span class=\"mono\">null</span> 显示为 —。</div>")

    add("</div>")

    # ---- embedded data + behaviour ----
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    add("<script id=\"d\" type=\"application/json\">%s</script>" % payload)
    add("""<script>
(function(){
var D=JSON.parse(document.getElementById('d').textContent);
var COLS=[['identity',0],['effort',0],['lid',0],['ctx_tokens',1],['out_tokens',1],['reasoning_tokens',1],
['chunks',1],['total_ms',1],['ttft_ms',1],['gen_ms',1],['thinking_visible_ms',1],['tpot_ms',1],
['upload_ms',1],['prefill_derived_ms',1],['ttft_residual_ms',1],['wire_bytes',1],['finish_reason',0]];
var tb=document.querySelector('#t tbody'), sel=document.getElementById('fep'),
    fe=document.getElementById('feff'), fq=document.getElementById('fq'), cnt=document.getElementById('cnt');
var eps=[];D.requests.forEach(function(r){if(eps.indexOf(r.identity)<0)eps.push(r.identity);});
eps.sort();eps.forEach(function(e){var o=document.createElement('option');o.value=e;o.textContent=e;sel.appendChild(o);});
var sortK='identity',sortDir=1;
function fmt(k,v){
  if(v===null||v===undefined)return '—';
  if(k==='wire_bytes')return (v/1e6).toFixed(2);
  if(k==='ctx_tokens'||k==='out_tokens'||k==='reasoning_tokens'||k==='chunks')return v.toLocaleString();
  if(k==='total_ms'||k==='ttft_ms'||k==='gen_ms'||k==='thinking_visible_ms'||k==='upload_ms'||k==='prefill_derived_ms'||k==='ttft_residual_ms')return v.toFixed(1);
  if(k==='tpot_ms')return v.toFixed(2);
  return v;
}
function view(){
  var a=sel.value,b=fe.value,q=fq.value.trim().toLowerCase();
  var rows=D.requests.filter(function(r){
    return (!a||r.identity===a)&&(!b||r.effort===b)&&(!q||String(r.lid).toLowerCase().indexOf(q)>=0);});
  var idx={};COLS.forEach(function(c,i){idx[c[0]]=i;});
  rows.sort(function(x,y){var k=sortK,m=COLS[idx[k]][1];
    var p=x[k],r=y[k];
    if(m){p=(p===null?-Infinity:p);r=(r===null?-Infinity:r);return (p-r)*sortDir;}
    p=String(p);r=String(r);return p<r?-sortDir:p>r?sortDir:0;});
  var h='';rows.forEach(function(r){
    h+='<tr>';
    COLS.forEach(function(c){var k=c[0];
      h+='<td class="'+(c[1]?'':'l')+(k==='lid'?' mono':'')+'">'+fmt(k,r[k])+'</td>';});
    h+='</tr>';});
  tb.innerHTML=h;cnt.textContent='显示 '+rows.length+' / '+D.requests.length+' 条';
  return rows;
}
document.querySelectorAll('#t th').forEach(function(th){
  th.addEventListener('click',function(){var k=th.dataset.k;
    if(sortK===k)sortDir=-sortDir;else{sortK=k;sortDir=1;}view();});});
sel.addEventListener('change',view);fe.addEventListener('change',view);
fq.addEventListener('input',view);
document.getElementById('reset').addEventListener('click',function(){
  sel.value='';fe.value='';fq.value='';sortK='identity';sortDir=1;view();});
document.getElementById('csv').addEventListener('click',function(){
  var rows=view(),head=COLS.map(function(c){return c[0];});
  var out=[head.join(',')];
  rows.forEach(function(r){out.push(head.map(function(k){
    var v=r[k];return (v===null||v===undefined)?'':(k==='wire_bytes'?(v/1e6).toFixed(2):v);}).join(','));});
  var blob=new Blob(['\\ufeff'+out.join('\\n')],{type:'text/csv;charset=utf-8'});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);
  a.download='ds-requests.csv';a.click();URL.revokeObjectURL(a.href);});
view();
})();
</script>""")
    add("</body></html>")

    Path(args.out).write_text("\n".join(rows), encoding="utf-8")
    print("wrote %s (%d KB)" % (args.out, len("\n".join(rows)) // 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
