#!/usr/bin/env python3
"""LP-A2 辅助：从参考 episode JSON 提取 fixture 位姿生成钉扎布局文件。
用法: lp_a2_make_pin.py <ep_json> <out_pin_json>
"""
import json
import sys

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
poses = d["start_identity"]["fixture_poses"]
json.dump({"fixture_poses": poses, "source_episode": src},
          open(out, "w"), indent=1)
print(f"PIN_WRITTEN {out} fixtures={sorted(poses)}", flush=True)
