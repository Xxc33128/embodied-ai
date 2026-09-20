#!/usr/bin/env bash
# 原 π₀.₅ checkpoint 推理身份 17 文件的内容级下载与逐字节验证。
#
# 运行位置（2026-09-17 分工）：NPU 服务器数据盘。Mac 不下载完整权重。
#   用法：bash scripts/download_checkpoint.sh /path/to/server/data_dir
#
# 行为：
#   - 断点续传（curl --http1.1 -C -），规避间歇网络的 HTTP/2 流错误与卡死。
#   - 局部文件超过期望大小 → 判损坏，隔离到 _quarantine/ 后从零重下
#     （续传只能补尾部，修不了已下载字节中的错误）。
#   - 下载完成但哈希不符 → 同样隔离后从零重下，最多 5 轮。
#   - 结束写出验证记录 _verification_record.json（逐文件状态/哈希/字节/尝试次数），
#     作为 W1 内容级验收证据；失败文件单列，不算通过。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:?用法: download_checkpoint.sh <服务器数据目录>}"
LOCK="$ROOT/configs/upstream.lock.json"
DEST="$DATA/ckpt/RoboDojo-sim-arx_x5-joint-0/59999"
QUARANTINE="$DATA/ckpt/_quarantine"
mkdir -p "$DEST" "$QUARANTINE"

python3 - "$LOCK" "$DEST" "$QUARANTINE" <<'EOF'
import datetime, hashlib, json, shutil, subprocess, sys, time
from pathlib import Path

lock_path, dest, quarantine = (Path(p) for p in sys.argv[1:4])
lock = json.load(open(lock_path))
rev = lock['pinned_sources']['hf_dataset']['revision']
sub = lock['checkpoint']['hf_subdir']
files = lock['checkpoint']['per_file']

def sha256_of(p: Path) -> str:
    hh = hashlib.sha256()
    with open(p, 'rb') as fh:
        while chunk := fh.read(1 << 22):
            hh.update(chunk)
    return hh.hexdigest()

def quarantine(p: Path, why: str) -> None:
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    target = quarantine / f"{p.name}.{stamp}.bad"
    shutil.move(str(p), target)
    print(f"QUARANTINE {p.name} -> {target.name} ({why})", flush=True)

record = {"finished_at": None, "dest": str(dest), "files": []}
failed = []
for f in files:
    out = dest / f['path']
    out.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    status = "failed"
    # 已有完整且正确的文件直接跳过
    if out.is_file():
        if out.stat().st_size > f['size']:
            quarantine(out, f"size {out.stat().st_size} > expected {f['size']}")
        elif sha256_of(out) == f['lfs_sha256']:
            print(f"SKIP  {f['path']} (already verified)", flush=True)
            record['files'].append({"path": f['path'], "status": "already_verified",
                                    "sha256": f['lfs_sha256'], "bytes": f['size'],
                                    "attempts": 0})
            continue
        else:
            quarantine(out, "existing file hash mismatch")
    for round_ in range(5):
        url = (f"https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo"
               f"/resolve/{rev}/{sub}/{f['path']}")
        while True:
            attempts += 1
            # --speed-time 60s 低于 64KB/s 视为卡死；返回非零视为网络错误，续传重试
            r = subprocess.run(["curl", "--http1.1", "-L", "-C", "-", "--fail",
                                "--connect-timeout", "30",
                                "--speed-limit", "65536", "--speed-time", "60",
                                "-o", str(out), url])
            if r.returncode == 0:
                break
            if out.is_file() and out.stat().st_size > f['size']:
                quarantine(out, "oversized after resume")
            time.sleep(15)
        if out.stat().st_size != f['size']:
            quarantine(out, f"final size {out.stat().st_size} != expected {f['size']}")
            continue
        got = sha256_of(out)
        if got == f['lfs_sha256']:
            status = "verified"
            print(f"OK    {f['path']}  {f['size']} bytes", flush=True)
            break
        quarantine(out, f"hash mismatch got {got[:16]}")
    if status != "verified":
        failed.append(f['path'])
    record['files'].append({"path": f['path'], "status": status,
                            "sha256": f['lfs_sha256'] if status == "verified" else None,
                            "bytes": f['size'], "attempts": attempts})

record['finished_at'] = datetime.datetime.now().isoformat(timespec='seconds')
record['failed'] = failed
out_rec = dest / '_verification_record.json'
out_rec.write_text(json.dumps(record, ensure_ascii=False, indent=1))
print(f"DONE verified={sum(1 for x in record['files'] if x['status'] in ('verified','already_verified'))}"
      f"/{len(files)} failed={len(failed)}", flush=True)
print(f"RECORD {out_rec}", flush=True)
if failed:
    print("FAILED_FILES:")
    for p in failed:
        print(" ", p, flush=True)
    sys.exit(1)
EOF
