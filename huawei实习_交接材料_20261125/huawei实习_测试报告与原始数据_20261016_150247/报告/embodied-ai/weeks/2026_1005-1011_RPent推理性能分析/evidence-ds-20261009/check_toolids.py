#!/usr/bin/env python3
"""Tool-call id integrity across the shim boundary.

A streamed response delivers tool calls as deltas that must be re-joined by index.
If the shim dropped or corrupted the call id, then the `tool_call_id` carried by
each subsequent `role: "tool"` message would no longer match any assistant
tool_call -- the model would receive self-contradictory history, which would show
up as the agent flailing (many more tool calls). This checks that invariant on the
last request body of an episode, where the entire history is present, and compares
shimmed runs against the frozen non-streaming batch.
"""
import gzip
import json
from pathlib import Path

B = Path("/data/rpent2/ds-adapter-20261009")

EPISODES = [
    ("B2 shim  t9_s1 medium(失败集)", "shim-deepseek-v4.1-flash-medium/libero_10_swap/t9_s1/attempt-001"),
    ("B2 shim  t0_s0 medium", "shim-deepseek-v4.1-flash-medium/libero_10_swap/t0_s0/attempt-001"),
    ("B1 shim  t0_s0 medium", "shim-validation-deepseek-v4.1-flash-medium/libero_10_swap/t0_s0/attempt-001"),
    ("原批      t9_s1 medium", "deepseek-v4.1-flash-medium/libero_10_swap/t9_s1/attempt-001"),
    ("原批      t0_s0 medium", "deepseek-v4.1-flash-medium/libero_10_swap/t0_s0/attempt-001"),
]


def last_body(key):
    shas = []
    for path in sorted((B / "measurements" / key).glob("spans-*.jsonl")):
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if row.get("name") == "llm.transport" and row.get("body_sha256"):
                shas.append(row["body_sha256"])
    for sha in reversed(sorted(set(shas))):
        path = B / "private/llm-bodies" / (sha + ".json.gz")
        if path.is_file():
            try:
                return json.loads(gzip.open(path).read()), sha
            except Exception:  # noqa: BLE001
                continue
    return None, None


def main():
    for label, key in EPISODES:
        body, sha = last_body(key)
        if body is None:
            print("%-24s 未找到快照" % label)
            continue
        messages = body.get("messages") or []
        calls = 0
        null_id = 0
        ids = set()
        tool_ids = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                for call in message["tool_calls"]:
                    calls += 1
                    if not call.get("id"):
                        null_id += 1
                    else:
                        ids.add(call["id"])
            if message.get("role") == "tool":
                tool_ids.append(message.get("tool_call_id"))
        orphan = sum(1 for value in tool_ids if value not in ids)
        print("%-24s 消息=%3d assistant工具调用=%3d id为空=%d 唯一id=%3d tool结果=%3d 孤儿引用=%d" % (
            label, len(messages), calls, null_id, len(ids), len(tool_ids), orphan))
        sample = next((m["tool_calls"][0] for m in messages
                       if m.get("role") == "assistant" and m.get("tool_calls")), None)
        if sample:
            print("     样例 id=%r name=%r args=%s" % (
                sample.get("id"), (sample.get("function") or {}).get("name"),
                str((sample.get("function") or {}).get("arguments"))[:50]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
