#!/usr/bin/env python3
"""pi05_libero 权重 crc32c 校验（服务器执行；对 GCS 元数据逐对象核对）。

背景：6 个大对象（≈12.4GB）在 GCS 为复合对象无 md5，下载脚本仅尺寸校验；
本脚本以 crc32c（Castagnoli）对照 GCS 元数据闭合完整性。在 gap-repro 容器执行：
  python3 scripts/verify_pi05_libero_crc32c.py
"""
import base64
import json
import struct
import urllib.request

SRC = "/workspace/data/checkpoints"          # 容器内挂载路径
OUT = "/workspace/data/pi05_libero_crc32c_verify.json"
PREFIX_LEN = len("checkpoints/")

# CRC32C (Castagnoli) 表驱动实现；测试向量 crc32c(b"123456789") == 0xE3069283
_POLY = 0x82F63B78
_TBL = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ (_POLY if _c & 1 else 0)
    _TBL.append(_c)


def crc32c(data: bytes, crc: int = 0) -> int:
    crc ^= 0xFFFFFFFF
    for b in data:
        crc = _TBL[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def main():
    url = ("https://storage.googleapis.com/storage/v1/b/openpi-assets/o"
           "?prefix=checkpoints/pi05_libero/&maxResults=1000")
    items = json.load(urllib.request.urlopen(url, timeout=120)).get("items", [])
    out, bad = [], []
    for it in items:
        rel = it["name"][PREFIX_LEN:]
        path = f"{SRC}/{rel}"
        want = it.get("crc32c", "")
        size = int(it["size"])
        entry = {"path": rel, "size": size}
        try:
            h = 0
            with open(path, "rb") as f:
                while chunk := f.read(1 << 22):
                    h = crc32c(chunk, h)
            ok = base64.b64encode(struct.pack(">I", h)).decode() == want
            entry["crc32c_ok"] = ok
            if not ok:
                bad.append(rel)
        except FileNotFoundError:
            entry["crc32c_ok"] = None
            bad.append(rel + " (missing)")
        out.append(entry)
    result = {"files": out, "bad": bad, "n": len(out),
              "algorithm": "CRC32C Castagnoli, table-driven",
              "test_vector_ok": crc32c(b"123456789") == 0xE3069283}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
    print(f"DONE n={len(out)} bad={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
