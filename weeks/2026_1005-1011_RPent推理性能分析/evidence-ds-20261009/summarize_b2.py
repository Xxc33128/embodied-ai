#!/usr/bin/env python3
"""B2 summary: fidelity vs RPent's own accounting, plus the per-request timing that
only a streamed wire can produce (TTFT / visible-thinking span / TPOT)."""
import json
import statistics
from pathlib import Path

B = Path("/data/rpent2/ds-adapter-20261009")


def med(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 1) if values else None


def main():
    data = json.loads((B / "metrics-shim.json").read_text())
    episodes = data["episodes"]
    print("B2 集数 = %d" % len(episodes))

    print("\n=== 1) 保真度（hook 捕获 vs RPent 自报）===")
    print("%-9s %-7s %5s %-10s %6s %10s %10s %8s %8s" % (
        "identity", "effort", "reqs", "对账", "解出", "in_match", "out_match", "req_match", "流式"))
    for e in sorted(episodes, key=lambda e: (e["identity"], e["effort"])):
        f = e["fidelity_check_vs_rpent"]
        checks = [f.get("requests_match"), f.get("input_tokens_match"), f.get("output_tokens_match")]
        # None means RPent itself could not report (the run died before it could),
        # so "match" is not defined rather than failing.
        verdict = "n/a" if None in checks else ("OK" if all(checks) else "DIFF")
        shown = lambda v: "n/a" if v is None else bool(v)  # noqa: E731
        print("%-9s %-7s %5d %-10s %6s %10s %10s %8s %8d/%d" % (
            e["identity"], e["effort"], e["decisions"]["llm_requests"], verdict,
            "是" if e["result"]["solved"] else "否",
            shown(f.get("input_tokens_match")), shown(f.get("output_tokens_match")), shown(f.get("requests_match")),
            e["llm"]["streamed_requests"], e["decisions"]["llm_requests"]))

    print("\n=== 2) 逐集结果与耗时 ===")
    print("%-9s %-7s %-24s %5s %9s %10s %12s %9s" % (
        "identity", "effort", "结束状态", "轮次", "LLM墙钟s", "TTFT中位", "思考可见中位", "TPOT中位"))
    for e in sorted(episodes, key=lambda e: (e["identity"], e["effort"])):
        llm = e["llm"]
        print("%-9s %-7s %-24s %5s %9.1f %10s %12s %9s" % (
            e["identity"], e["effort"],
            (e["result"]["finish_status"] or "run died (exit=1)")[:24],
            e["result"]["turns"], llm["wall_sum_ms"] / 1000.0,
            med([r["ttft_ms"] for r in llm["per_request"]]),
            med([r["thinking_visible_ms"] for r in llm["per_request"]]),
            med([r["tpot_ms"] for r in llm["per_request"]])))

    print("\n=== 3) 按档位汇总（逐请求中位）===")
    for effort in ("medium", "none"):
        subset = [e for e in episodes if e["effort"] == effort]
        ttft = [r["ttft_ms"] for e in subset for r in e["llm"]["per_request"]]
        think = [r["thinking_visible_ms"] for e in subset for r in e["llm"]["per_request"]]
        tpot = [r["tpot_ms"] for e in subset for r in e["llm"]["per_request"]]
        reqs = sum(e["decisions"]["llm_requests"] for e in subset)
        live = [v for v in ttft if v is not None]
        print("  %-7s 集=%d 请求=%3d  TTFT 中位=%s (min %s / max %s)  思考可见中位=%s  TPOT 中位=%s" % (
            effort, len(subset), reqs, med(ttft),
            round(min(live), 1) if live else None, round(max(live), 1) if live else None,
            med(think), med(tpot)))

    print("\n=== 4) 逐请求样本（每集首/末各一条）===")
    for e in sorted(episodes, key=lambda e: (e["identity"], e["effort"])):
        rows = e["llm"]["per_request"]
        for label, row in (("首", rows[0]), ("末", rows[-1])):
            print("  %-9s %-7s %s req=%s ctx=%7s ttft=%8s 思考=%8s tpot=%6s out=%5s chunks=%s" % (
                e["identity"], e["effort"], label, row["logical_request_id"], row["prompt_tokens"],
                row["ttft_ms"], row["thinking_visible_ms"], row["tpot_ms"],
                row["completion_tokens"], row["chunk_count"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
