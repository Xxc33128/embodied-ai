#!/usr/bin/env python3
"""Matched A/B on the exact request that killed the two B2 t9-medium episodes.

The failure pydantic-ai raised was:
    UnexpectedModelBehavior: Model token limit (8192) exceeded before any response
    was generated.
Its trigger (pydantic_ai/_agent_graph.py) is: the response has ONLY thinking parts
(no content, no tool call) AND finish_reason == 'length'.

The frozen non-streaming batch never hit it in 306 requests -- but it came within 8
tokens once (t9_s1 medium, completion_tokens=8184, reasoning_tokens=7933), and it
never once returned finish_reason='length'. So the cliff is real in non-streaming
too; nobody has observed either mode at the edge.

This replays the crashing request's own body in both wire modes, interleaved in a
shuffled order, and compares the outcome distribution. If non-streaming essentially
never truncates while streaming often does, the shim changed behaviour and B2's
episode-level results are not comparable. If both truncate at a similar rate, the
shim is innocent and this is a planner fragility near the 8192 cap.
"""
import copy
import gzip
import json
import os
import random
import time
from pathlib import Path

import httpx

B = Path("/data/rpent2/ds-adapter-20261009")
BODY_DIR = B / "private/llm-bodies"
BASE_URL = "https://api.modelarts-maas.com/openai/v1/chat/completions"
KEY = os.environ["OPENAI_API_KEY"]
REPEATS = 4
TARGETS = [
    ("t9_s0 medium", "shim-deepseek-v4.1-flash-medium/libero_10_swap/t9_s0/attempt-001"),
    ("t9_s1 medium", "shim-deepseek-v4.1-flash-medium/libero_10_swap/t9_s1/attempt-001"),
]


def last_body_sha(key):
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
    return shas[-1] if shas else None


def reassemble(blob):
    content, reasoning, finish, usage = [], [], None, None
    calls = {}
    tool_deltas = 0
    for line in blob.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload == b"[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:  # noqa: BLE001
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                tool_deltas += 1
                slot = calls.setdefault(call.get("index", 0), {"args": ""})
                slot["args"] += ((call.get("function") or {}).get("arguments") or "")
    return {"finish_reason": finish, "usage": usage,
            "content_chars": len("".join(content)), "reasoning_chars": len("".join(reasoning)),
            "tool_calls": len(calls), "tool_deltas": tool_deltas}


def call(body, stream):
    payload = copy.deepcopy(body)
    payload["max_completion_tokens"] = 8192          # exactly what the batch set
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    else:
        payload["stream"] = False
        payload.pop("stream_options", None)
    started = time.monotonic()
    out = {"stream": stream, "error": None}
    try:
        with httpx.Client(timeout=900.0) as client:
            if not stream:
                response = client.post(BASE_URL, json=payload, headers={"Authorization": "Bearer " + KEY})
                response.raise_for_status()
                native = response.json()
                choice = (native.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                out.update({"finish_reason": choice.get("finish_reason"), "usage": native.get("usage"),
                            "content_chars": len(message.get("content") or ""),
                            "reasoning_chars": len(message.get("reasoning_content") or ""),
                            "tool_calls": len(message.get("tool_calls") or []), "tool_deltas": 0})
            else:
                with client.stream("POST", BASE_URL, json=payload, headers={"Authorization": "Bearer " + KEY}) as response:
                    response.raise_for_status()
                    blob = bytearray()
                    for chunk in response.iter_bytes():
                        blob += chunk
                out.update(reassemble(bytes(blob)))
    except Exception as exc:  # noqa: BLE001
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:120])
    out["wall_s"] = round(time.monotonic() - started, 1)
    usage = out.get("usage") or {}
    out["completion_tokens"] = usage.get("completion_tokens")
    out["reasoning_tokens"] = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return out


def main():
    random.seed(20261010)
    report = {"schema_version": 1, "repeats_per_mode": REPEATS, "models": [], "input_tokens_total": 0}
    for label, key in TARGETS:
        sha = last_body_sha(key)
        if not sha:
            print("%s: 未找到请求体" % label)
            continue
        body = json.loads(gzip.open(BODY_DIR / (sha + ".json.gz")).read())
        entry = {"label": label, "sha": sha, "rows": []}
        modes = []
        for _ in range(REPEATS):
            pair = [True, False]
            random.shuffle(pair)
            modes.extend(pair)
        print("\n=== %s （上下文 %s token，交互顺序）===" % (label, len(json.dumps(body))), flush=True)
        for stream in modes:
            row = call(body, stream)
            entry["rows"].append(row)
            report["input_tokens_total"] += (row.get("usage") or {}).get("prompt_tokens") or 0
            print("  %-6s finish=%-10s out=%-6s reason=%-6s content=%-6d tools=%d wall=%5.1fs %s" % (
                "stream" if stream else "plain", row.get("finish_reason"), row.get("completion_tokens"),
                row.get("reasoning_tokens"), row.get("content_chars") or 0, row.get("tool_calls") or 0,
                row["wall_s"], row["error"] or ""), flush=True)
        report["models"].append(entry)
        Path(B / "cliff-ab.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print("\n=== 汇总：截断率（finish_reason=length 且无可见输出）===")
    for entry in report["models"]:
        for mode, name in ((True, "stream"), (False, "plain")):
            rows = [r for r in entry["rows"] if r["stream"] == mode and not r["error"]]
            trunc = sum(1 for r in rows if r.get("finish_reason") == "length")
            empty = sum(1 for r in rows if r.get("finish_reason") == "length"
                        and not (r.get("content_chars") or 0) and not (r.get("tool_calls") or 0))
            print("  %-14s %-6s n=%d  截断=%d  截断且无可见输出=%d" % (entry["label"], name, len(rows), trunc, empty))
    Path(B / "cliff-ab.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\ninput_tokens=%d" % report["input_tokens_total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
