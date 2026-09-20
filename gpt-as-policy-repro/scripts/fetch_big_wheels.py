#!/usr/bin/env python3
"""LP1：大体积 wheel 的可续传预取（服务器执行）。

本网络对大 HTTP 单流会截断/损坏，pip 单发下载不可靠；而 curl -C - + 哈希校验
已被 12GB 权重下载验证可行。本脚本解析 PyPI JSON API 拿 aarch64 cp38 wheel 的
URL+sha256，逐个断点续传到 data/wheels/，供 setup_libero_env.sh 本地安装。
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

# 位置必须经环境变量或挂载路径传入：容器内是 /workspace/data/wheels，
# 宿主机是 /data_nv0/gpt-as-policy-repro/data/wheels。写死绝对路径曾导致
# wheel 下载进容器 rootfs（宿主不可见、容器重建即丢）。
WHEELS_DIR = os.environ.get("WHEELS_DIR", "/workspace/data/wheels")
PKGS = [
    ("torch", "1.11.0"),
    ("torchvision", "0.12.0"),
    ("torchaudio", "0.11.0"),
    ("opencv-python", "4.6.0.66"),
    ("matplotlib", "3.5.3"),
    ("pillow", "9.3.0"),
]
PATTERN = "aarch64"


def pick(urls):
    """选 cp38 + aarch64 的 manylinux wheel；优先 manylinux2014 命名。"""
    cands = [u for u in urls if "cp38" in u["filename"] and PATTERN in u["filename"]
             and u["filename"].endswith(".whl") and "manylinux" in u["filename"]]
    if not cands:
        return None
    cands.sort(key=lambda u: ("manylinux_2_" in u["filename"], len(u["filename"])))
    return cands[0]


def fetch(url, sha, out):
    want = f"{sha}  {out}"
    for attempt in range(12):
        r = subprocess.run(["curl", "--http1.1", "-L", "-C", "-", "--fail",
                            "--connect-timeout", "30",
                            "--speed-limit", "65536", "--speed-time", "60",
                            "-o", out, url])
        if r.returncode == 0:
            c = subprocess.run(["sha256sum", "-c"], input=want.encode(),
                               capture_output=True)
            if c.returncode == 0:
                return True
            # 哈希不符：本地残留坏块，续传修不了，删掉重来
            os.remove(out)
        time.sleep(10)
    return False


def main():
    os.makedirs(WHEELS_DIR, exist_ok=True)
    failed = []
    for pkg, ver in PKGS:
        api = f"https://pypi.org/pypi/{pkg}/{ver}/json"
        try:
            data = json.load(urllib.request.urlopen(api, timeout=60))
        except Exception as e:
            print(f"META_FAIL {pkg} {ver}: {e}", flush=True)
            failed.append(pkg)
            continue
        pick_ = pick(data["urls"])
        if not pick_:
            print(f"NO_WHEEL {pkg} {ver}", flush=True)
            failed.append(pkg)
            continue
        out = os.path.join(WHEELS_DIR, pick_["filename"])
        # 预检：文件已存在且大小/哈希正确则跳过（curl -C - 对完整文件会 416）
        if os.path.isfile(out) and os.path.getsize(out) == pick_["size"]:
            h = hashlib.sha256()
            with open(out, "rb") as f:
                while c := f.read(1 << 22):
                    h.update(c)
            if h.hexdigest() == pick_["digests"]["sha256"]:
                print(f"SKIP {pick_['filename']}", flush=True)
                continue
        print(f"FETCH {pick_['filename']} {pick_['size']} bytes", flush=True)
        if fetch(pick_["url"], pick_["digests"]["sha256"], out):
            print(f"OK {pick_['filename']}", flush=True)
        else:
            print(f"FAIL {pick_['filename']}", flush=True)
            failed.append(pkg)
    print(f"DONE failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
