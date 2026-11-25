#!/usr/bin/env python3
"""Does the wire mode shift reasoning length at an identical history?

The two B2 t9-medium episodes died on the 8192-token planner guard. The cliff A/B
replayed their exact crashing bodies in both modes. If streaming inflated reasoning
length, the shim would be the cause; if the two modes draw from the same
distribution, the crash is sampling variance near a cap the frozen batch also
nearly hit (its closest draw was 8184 of 8192).
"""
import json
import statistics
from pathlib import Path

B = Path("/data/rpent2/ds-adapter-20261009")


def main():
    data = json.loads((B / "cliff-ab.json").read_text())
    print("A/B：同一份 crashing 请求体，两种 wire 模式交错各 %d 次" % data["repeats_per_mode"])
    print("%-7s %4s %10s %10s %8s %12s %10s %5s" % (
        "mode", "n", "推理中位", "推理均值", "推理max", "输出中位", "输出max", "截断"))
    for mode, name in ((True, "stream"), (False, "plain")):
        rows = [r for m in data["models"] for r in m["rows"] if r["stream"] == mode and not r["error"]]
        reason = [r["reasoning_tokens"] or 0 for r in rows]
        comp = [r["completion_tokens"] or 0 for r in rows]
        trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
        print("%-7s %4d %10d %10.0f %8d %12d %10d %5d" % (
            name, len(rows), statistics.median(reason), statistics.mean(reason), max(reason),
            statistics.median(comp), max(comp), trunc))
    print()
    print("所有 16 次发送的 finish_reason：", sorted({r.get("finish_reason") for m in data["models"] for r in m["rows"]}))
    print("input_tokens = %d" % data["input_tokens_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
