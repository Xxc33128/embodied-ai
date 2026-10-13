#!/usr/bin/env python3
"""Assemble the deep-timing report data for the B-stage (stream-shim) batch.

Reads the same evidence the report cites and emits one desensitised JSON: no prompt
text, no headers, no key material -- only numbers, ids and body sha prefixes. Every
quantity carries its basis so the page can distinguish what was measured in-batch
from what was derived.

Per request the timing closes as:

    total_ms  = ttft_ms + gen_ms + post_ms                 (all measured)
    ttft_ms   = upload_ms + prefill_derived_ms + ttft_residual_ms
                  ^measured   ^derived            ^residual (queue/network/jitter)

so prefill is split out explicitly rather than left unknown, and the leftover is
named as a residual instead of being attributed to a server stage.
"""
import gzip
import json
import statistics
from pathlib import Path

B = Path("/data/rpent2/ds-adapter-20261009")
BODY_DIR = B / "private/llm-bodies"
INCLUDE = "shim-deepseek"          # the B2 batch only; the frozen batch stays separate


def med(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def pct(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return round(values[idx], 1)


def fit_slope(probe):
    """Least-squares fit of the per-body minimum TTFT against prompt size."""
    xs, ys = [], []
    for entry in probe["probes"]:
        rows = [r["first_delta_ms"] for r in entry["rows"]
                if r.get("stream") and r.get("first_delta_ms") is not None]
        if rows:
            xs.append(entry["prompt_tokens"])
            ys.append(min(rows))
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    return slope, (sy - slope * sx) / n, n


def body_bytes_for(key, attempt):
    """logical request -> bytes actually put on the wire, via the body snapshot."""
    full = {}
    for path in sorted((B / "measurements" / key / attempt).glob("spans-*.jsonl")):
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if row.get("name") == "llm.transport":
                full[(row.get("logical_request_id"), row.get("attempt"))] = row.get("body_sha256")
    return full


def main():
    metrics = json.loads((B / "metrics-shim.json").read_text())
    probe = json.loads((B / "stream-probe.json").read_text())
    ladder = json.loads((B / "upload-ladder.json").read_text())
    cliff = json.loads((B / "cliff-ab.json").read_text())
    equiv = json.loads((B / "stream-equivalence.json").read_text())

    rate = ladder["inferred_mb_per_s"]
    slope, intercept, npairs = fit_slope(probe)

    episodes, requests, missing_bytes = [], [], 0
    for episode in metrics["episodes"]:
        if INCLUDE not in episode["episode_key"]:
            continue
        key, attempt = episode["episode_key"], episode["attempt"]
        full = body_bytes_for(key, attempt)
        rows = []
        for request in episode["llm"]["per_request"]:
            sha = full.get((request["logical_request_id"], request["attempt"]))
            path = BODY_DIR / (sha + ".json.gz") if sha else None
            wire = None
            if path and path.is_file():
                wire = len(gzip.open(path).read())
            else:
                missing_bytes += 1
            total = request["duration_ms"]
            ttft = request["ttft_ms"]
            last = request["last_chunk_ms"]
            gen = round(last - ttft, 3) if (last is not None and ttft is not None) else None
            post = round(total - last, 3) if (last is not None and total is not None) else None
            upload = round(wire / (rate * 1e6) * 1000.0, 1) if wire else None
            prefill = round(slope * (request["prompt_tokens"] or 0), 1) if request["prompt_tokens"] else None
            residual = (round(ttft - upload - prefill, 1)
                        if (ttft is not None and upload is not None and prefill is not None) else None)
            think_tok_ms = (round(request["reasoning_tokens"] * request["tpot_ms"], 1)
                            if request["reasoning_tokens"] and request["tpot_ms"] else None)
            rows.append({
                "lid": request["logical_request_id"], "attempt": request["attempt"],
                "ctx_tokens": request["prompt_tokens"], "cached_tokens": request["cached_tokens"],
                "out_tokens": request["completion_tokens"], "reasoning_tokens": request["reasoning_tokens"],
                "chunks": request["chunk_count"], "streamed": request["streamed"],
                "wire_bytes": wire,
                "total_ms": total, "ttft_ms": ttft, "gen_ms": gen, "post_ms": post,
                "thinking_visible_ms": request["thinking_visible_ms"],
                "thinking_tok_ms": think_tok_ms,
                "first_reasoning_ms": request["first_reasoning_ms"],
                "first_content_ms": request["first_content_ms"],
                "tpot_ms": request["tpot_ms"], "decode_ms": gen,
                "prefill_derived_ms": prefill, "upload_ms": upload, "ttft_residual_ms": residual,
                "finish_reason": request["finish_reason"],
            })
        requests.extend(dict(row, identity=episode["identity"], effort=episode["effort"]) for row in rows)

        live = [r for r in rows if r["ttft_ms"] is not None]
        episodes.append({
            "identity": episode["identity"], "effort": episode["effort"], "attempt": attempt,
            "episode_key": key,
            "finish_status": episode["result"]["finish_status"],
            "exit_code": episode["result"].get("process_exit_code"),
            "solved": episode["result"]["solved"],
            "turns": episode["result"]["turns"],
            "elapsed_s": episode["result"]["rpent_elapsed_s"],
            "tool_calls": episode["decisions"]["tool_calls_total"],
            "requests": len(rows),
            "llm_wall_s": round(episode["llm"]["wall_sum_ms"] / 1000.0, 1),
            "tokens": {
                "input_sum": episode["tokens"]["prompt_tokens"],
                "cached_sum": episode["tokens"]["cached_tokens"],
                "output_sum": episode["tokens"]["completion_tokens"],
                "reasoning_sum": episode["tokens"]["reasoning_tokens"],
                "input_max": max((r["ctx_tokens"] or 0) for r in rows),
            },
            "ttft": {"p50": med([r["ttft_ms"] for r in rows]), "min": min((r["ttft_ms"] for r in live), default=None),
                     "max": max((r["ttft_ms"] for r in live), default=None)},
            "tpot": {"p50": med([r["tpot_ms"] for r in rows])},
            "thinking_visible": {"p50": med([r["thinking_visible_ms"] for r in rows]),
                                 "sum": round(sum(r["thinking_visible_ms"] for r in live
                                                  if r["thinking_visible_ms"] is not None), 1)
                                 if any(r["thinking_visible_ms"] is not None for r in live) else None},
            "upload_s": round(sum(r["wire_bytes"] for r in rows if r["wire_bytes"]) / (rate * 1e6), 1)
            if any(r["wire_bytes"] for r in rows) else None,
            "prefill_derived_s": round(sum(r["prefill_derived_ms"] for r in rows if r["prefill_derived_ms"]) / 1000.0, 1),
            "decode_s": round(sum(r["gen_ms"] for r in rows if r["gen_ms"]) / 1000.0, 1),
        })

    arms = []
    for effort in ("medium", "none"):
        sub = [e for e in episodes if e["effort"] == effort]
        rows = [r for r in requests if r["effort"] == effort]
        live = [r for r in rows if r["ttft_ms"] is not None]
        arms.append({
            "effort": effort, "episodes": len(sub), "requests": len(rows),
            "solved": sum(1 for e in sub if e["solved"]),
            "ttft_p50_ms": med([r["ttft_ms"] for r in rows]),
            "ttft_p10_ms": pct([r["ttft_ms"] for r in rows], 0.10),
            "ttft_p90_ms": pct([r["ttft_ms"] for r in rows], 0.90),
            "ttft_min_ms": min((r["ttft_ms"] for r in live), default=None),
            "ttft_max_ms": max((r["ttft_ms"] for r in live), default=None),
            "tpot_p50_ms": med([r["tpot_ms"] for r in rows]),
            "tpot_min_ms": min((r["tpot_ms"] for r in live if r["tpot_ms"] is not None), default=None),
            "tpot_max_ms": max((r["tpot_ms"] for r in live if r["tpot_ms"] is not None), default=None),
            "thinking_visible_p50_ms": med([r["thinking_visible_ms"] for r in rows]),
            "thinking_share_p50": med([round(100.0 * r["thinking_tok_ms"] / r["gen_ms"], 1)
                                       for r in rows
                                       if r["thinking_tok_ms"] and r["gen_ms"]]),
            "reasoning_tokens_sum": sum(r["reasoning_tokens"] or 0 for r in rows),
            "upload_s": round(sum(r["wire_bytes"] for r in rows if r["wire_bytes"]) / (rate * 1e6), 1),
        })

    # One A/B record per (episode, wire mode), each covering that mode's own sends.
    ab_rows = []
    for entry in cliff["models"]:
        for mode, name in ((True, "stream"), (False, "plain")):
            rows = [r for r in entry["rows"] if r["stream"] == mode and not r["error"]]
            ab_rows.append({
                "label": entry["label"], "mode": name, "n": len(rows),
                "reasoning_p50": med([r["reasoning_tokens"] for r in rows]),
                "completion_p50": med([r["completion_tokens"] for r in rows]),
                "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
            })

    all_live = [r for r in requests if r["ttft_ms"] is not None]
    data = {
        "schema_version": 1,
        "title": "DeepSeek-V4.1-Flash · RPent Long Swap 深度计时报告",
        "subtitle": "B 阶段：stream-shim 全批重跑，逐请求实测 TTFT / TPOT / 思考窗口 / prefill 拆分",
        "batch": "ds-adapter-20261009 · shim-deepseek-v4.1-flash-{medium,none}/libero_10_swap/t{0,9}_s{0,1}",
        "updated_at": "2026-10-10",
        "data_status": "complete",
        "note": "本页数据来自 B 阶段 8 集重跑（transport 层把非流式请求翻成流式后再重组回填 SDK）。与 10/09 冻结批并存，不混算。",
        "method": {
            "ttft": "首个 SSE delta 的客户端时间戳（first_delta）。实测。medium 档首个 delta 即第一个思考 token。",
            "tpot": "(末 delta − 首 delta) ÷ (输出 token − 1)。实测。",
            "thinking_window": "首个思考 delta → 首个可见内容 delta 的窗口（可见思考）。实测。",
            "thinking_tokens": "reasoning_tokens × TPOT，用于估计思考在生成窗口中的占比。派生。",
            "prefill": "TTFT 对 prompt 大小做最小二乘拟合的斜率 × 该请求 prompt token。派生，只取斜率不取截距以免与上传税重复计数。",
            "upload": "该请求出站体解压后字节 ÷ 实测链路速率。链路速率为字节阶梯实测。",
            "residual": "TTFT − 上传 − prefill。装排队、网络、网关抖动与拟合截距，不得当作某个服务端阶段。",
        },
        "fit": {"slope_ms_per_token": round(slope, 5), "throughput_tok_per_s": round(1000.0 / slope, 1),
                "intercept_ms": round(intercept, 1), "pairs": npairs,
                "source": "stream-probe.json：9 个真实体 × 各 3–5 次流式重放，取每个体的最小 TTFT"},
        "upload": {"rate_mb_per_s": rate,
                   "rate_source": "upload-ladder.json：向同网关发无效模型的字节阶梯 0.05/0.5/1.5/4.3 MB × 3",
                   "median_body_mb": round(med([r["wire_bytes"] for r in requests if r["wire_bytes"]]) / 1e6, 2)
                   if any(r["wire_bytes"] for r in requests) else None,
                   "missing": missing_bytes},
        "arms": arms,
        "episodes": sorted(episodes, key=lambda e: (e["identity"], e["effort"])),
        "requests": sorted(requests, key=lambda r: (r["identity"], r["effort"], r["lid"] or "")),
        "cliff": {
            "summary": "两集 t9/medium 以 exit=1 终止：pydantic-ai 在「响应只有思考、无可见输出」且 finish_reason=length 时硬抛，不重试。",
            "threshold_tokens": 8192,
            "episodes": [e["identity"] + "/" + e["effort"] for e in episodes if e["exit_code"] == 1],
            "frozen_batch_max_output_tokens": 8184,
            "frozen_batch_length_finishes": 0,
            "frozen_batch_requests": 306,
            "ab": ab_rows,
            "ab_note": "对每集崩溃请求体，两模式交错各 4 次；见 evidence/cliff-ab.json",
        },
        "shim_checks": [
            {"check": "保真度（in/out vs RPent 自报）", "result": "5/8 集三项全 MATCH；1 集多 1 行（失败传输尝试与其重试各一行，逻辑请求数不变）；2 集 n/a（崩溃集 RPent 只能报 ?）"},
            {"check": "wire 形态", "result": "8/8 集 100% 标流式，且 100% 记录原始请求为非流式"},
            {"check": "相位标记真实性", "result": "全部真流式：正常响应 12–105 chunks；非流式会退化为 1"},
            {"check": "tool_call id 完整性", "result": "0 孤儿引用：两集 66 / 60 个调用，id 全唯一、全匹配 tool 调用结果"},
            {"check": "崩溃请求 A/B", "result": "16/16 全为 tool_calls、零截断；推理长度中位 3769(stream) vs 3836(plain)"},
            {"check": "结构等价预检", "result": "stream/plain 交错各 5 次，结构等价 20/20 通过、0 错误、usage 20/20"},
        ],
        "equiv": {"checks": equiv.get("summary") or {}, "note": equiv.get("notes")},
        "limits": [
            "TTFT 含上传、排队、网络与首 token 前余量，不是纯服务端 prefill；prefill 列是派生值。",
            "residual（TTFT − 上传 − prefill）只作残差，不得命名为某个服务端阶段。",
            "cached_tokens 只记录，不用于折算 prefill。",
            "B2 与 10/09 冻结批整集口径不可比（2 集崩溃 + 逐集工作量随机波动 43–80 次工具调用），逐请求时延口径可比。",
            "同一身份重发的 TTFT max/min 可达 16.6，网关长尾极重，故一律用中位与分位而非均值。",
        ],
    }
    (B / "report-data-ds.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("episodes=%d requests=%d missing_bytes=%d" % (len(data["episodes"]), len(data["requests"]), missing_bytes))
    print("slope=%.5f ms/tok intercept=%.0f ms rate=%.3f MB/s" % (slope, intercept, rate))
    print("wrote report-data-ds.json (%d KB)" % (len(json.dumps(data, ensure_ascii=False)) // 1024))
    for a in arms:
        print("  %-7s req=%3d ttft p50=%s p10=%s p90=%s  tpot p50=%s  思考中位=%s 思考占比中位=%s%%" % (
            a["effort"], a["requests"], a["ttft_p50_ms"], a["ttft_p10_ms"], a["ttft_p90_ms"],
            a["tpot_p50_ms"], a["thinking_visible_p50_ms"], a["thinking_share_p50"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
