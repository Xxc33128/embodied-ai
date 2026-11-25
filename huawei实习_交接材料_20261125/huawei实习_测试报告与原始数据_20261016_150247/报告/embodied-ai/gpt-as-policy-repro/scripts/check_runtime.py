#!/usr/bin/env python3
"""T0：运行时体检与部署身份（审计 F12 / 执行计划 T0）。

用法：
  check_runtime.py --profile mac|libero|robodojo
  check_runtime.py --deploy-manifest     # 输出源码文件 SHA256（本地↔目标机对账）

只输出：组件版本、必需资产存在性、源码 SHA256、blocked 原因。
不输出环境变量值、路径细节以外的配置内容或任何凭据。
"""
import argparse
import hashlib
import importlib
import json
import os
import sys

# 部署对账的源码清单（相对工程根；与目标机 repo 内同路径比较）
DEPLOY_FILES = [
    "src/gap_repro/results.py",
    "src/gap_repro/transport.py",
    "src/gap_repro/sim/environment.py",
    "src/gap_repro/sim/libero_session.py",
    "src/gap_repro/agent/contract.py",
    "src/gap_repro/agent/client.py",
    "src/gap_repro/agent/gate.py",
    "scripts/lp_a2_closed_loop.py",
    "scripts/lp_a2_policy_server.py",
]

ASSETS_MAC = [
    "data/hf_cache/Assets/Robots/x5/X5A.urdf",
]
ASSETS_LIBERO = [
    "upstream/LIBERO-PRO/libero/libero",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def try_version(module_name):
    try:
        m = importlib.import_module(module_name)
    except Exception as e:  # ImportError 及后端初始化失败都算 blocked
        return None, f"import failed: {type(e).__name__}"
    v = getattr(m, "__version__", "unknown")
    return v, None


def check(profile, root):
    components, assets, blocked = {}, {}, []

    py = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    components["python"] = py

    wants = {
        "mac": ["mujoco", "numpy", "pytest"],
        "libero": ["libero", "robosuite", "mujoco", "torch"],
        "robodojo": ["torch", "numpy"],
    }[profile]
    for mod in wants:
        v, err = try_version(mod)
        components[mod] = v if v else None
        if err:
            blocked.append(f"component {mod}: {err}")

    asset_list = {"mac": ASSETS_MAC, "libero": ASSETS_LIBERO,
                  "robodojo": []}[profile]
    data_root = os.environ.get("GAP_REPRO_DATA")
    for rel in asset_list:
        p = os.path.join(root, rel)
        assets[rel] = os.path.exists(p)
        if not assets[rel]:
            blocked.append(f"asset missing: {rel}")
    if profile in ("mac", "robodojo") and not data_root:
        blocked.append("GAP_REPRO_DATA not set (asset root unavailable)")

    if profile == "robodojo":
        # NPU 侧只报告组件事实；设备/服务健康由既有命令检查
        try:
            importlib.import_module("torch_npu")
            components["torch_npu"] = True
        except Exception as e:
            components["torch_npu"] = None
            blocked.append(f"torch_npu: {type(e).__name__}（目标机之外属预期）")

    hashes = {}
    for rel in DEPLOY_FILES:
        p = os.path.join(root, rel)
        hashes[rel] = sha256(p) if os.path.isfile(p) else None
        if hashes[rel] is None:
            blocked.append(f"deploy file missing: {rel}")

    return {
        "profile": profile,
        "components": components,
        "assets_present": assets,
        "deploy_file_sha256": hashes,
        "blocked": blocked,
        "ok": not blocked,
    }


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--profile", choices=["mac", "libero", "robodojo"])
    g.add_argument("--deploy-manifest", action="store_true")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.deploy_manifest:
        out = {rel: (sha256(os.path.join(root, rel))
                     if os.path.isfile(os.path.join(root, rel)) else None)
               for rel in DEPLOY_FILES}
        print(json.dumps(out, indent=1, sort_keys=True))
        return 0
    report = check(args.profile, root)
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
