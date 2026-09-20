"""W1 输入身份核查：源码、评测布局、支持轨迹与 checkpoint 的内容级哈希。

设计约束（对应计划 §6.3 W1）：
- 只依据冻结的 panel/scope 与 HuggingFace 固定 revision 取数，不猜文件名。
- 下载文件一律拒绝 Git-LFS 指针文本冒充资产（version https://git-lfs...）。
- checkpoint 聚合身份沿用上游原配方（{relpath: sha256} 映射、sort_keys JSON 的 sha256），
  已对照作者 previous_load_sha256=d15fb8bd… 精确验证；不得更换配方。
- 本模块不缓存网络失败：每次调用都重新取数，证据以落盘记录为准。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

HF_DATASET = "RoboDojo-Benchmark/RoboDojo"
HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"

# 原 π₀.₅ checkpoint 中属于推理身份的 17 个文件：params/**（16 个）+ assets/**（1 个）。
# train_state/**、_CHECKPOINT_METADATA 是训练态产物，不计入推理身份。
CHECKPOINT_IDENTITY_PREFIXES = ("params/", "assets/")


class LfsPointerError(RuntimeError):
    """下载到的内容是 LFS 指针而非资产本体。"""


class RevisionMismatchError(RuntimeError):
    """锁定 revision 与期望不符。"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> tuple[str, int]:
    """返回 (sha256hex, 字节数)，流式读取。"""
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def is_lfs_pointer(data: bytes) -> bool:
    return data[: len(LFS_POINTER_MAGIC)] == LFS_POINTER_MAGIC


def read_asset_bytes(path: str | Path) -> bytes:
    """读取已落盘资产；内容若为 LFS 指针文本则拒绝（不得当作资产哈希）。"""
    data = Path(path).read_bytes()
    if is_lfs_pointer(data):
        raise LfsPointerError(f"{path} is an LFS pointer, not downloaded content")
    return data


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fetch_dataset_file(
    revision: str,
    repo_path: str,
    dest: Path,
    *,
    repo: str = HF_DATASET,
    timeout: int = 300,
    retries: int = 4,
) -> dict:
    """从固定 revision 下载单个文件并返回内容哈希；LFS 指针一律拒绝。

    网络类失败（URLError/超时）按有限次数退避重试；重试耗尽后抛出最后一次异常。
    """
    url = HF_RESOLVE.format(repo=repo, rev=revision, path=repo_path)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers={"User-Agent": "gap-repro/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                head = resp.read(4096)
                if is_lfs_pointer(head):
                    raise LfsPointerError(
                        f"LFS pointer returned for {repo_path} (revision={revision})")
                out.write(head)
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
            sha, size = sha256_file(tmp)
            tmp.replace(dest)
            return {"path": repo_path, "sha256": sha, "bytes": size}
        except LfsPointerError:
            if tmp.exists():
                tmp.unlink()
            raise
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_exc = e
            if tmp.exists():
                tmp.unlink()
            if attempt < retries:
                time.sleep(10 * (attempt + 1))
    raise RuntimeError(
        f"fetch failed after {retries + 1} attempts: {repo_path} @ {revision}") from last_exc


def verify_native_sources(native_root: str | Path, entries: list[dict]) -> dict:
    """对 panel 的 native_source_files 逐文件内容哈希。

    entries: [{"path": str, "sha256": str}, ...]
    返回 {"results": [...], "passed": int, "failed": int}；缺文件记 status="missing"。
    """
    root = Path(native_root)
    results = []
    for e in entries:
        p = root / e["path"]
        if not p.is_file():
            results.append({"path": e["path"], "expected": e["sha256"], "actual": None,
                            "match": False, "bytes": None, "status": "missing"})
            continue
        actual, size = sha256_file(p)
        results.append({"path": e["path"], "expected": e["sha256"], "actual": actual,
                        "match": actual == e["sha256"], "bytes": size, "status": "ok"})
    failed = [r for r in results if not r["match"]]
    return {"results": results, "passed": len(results) - len(failed), "failed": len(failed)}


def verify_revision(actual: str, expected: str) -> None:
    if actual != expected:
        raise RevisionMismatchError(f"revision mismatch: actual={actual} expected={expected}")


def selected_cases(panel: dict, scope: dict) -> list[dict]:
    """按 scope_v2 的 50 个 case_id 从 panel60 取完整 case 记录（含支持轨迹）。"""
    by_id = {c["case_id"]: c for c in panel["cases"]}
    rows = []
    for s in scope["cases"]:
        cid = s["case_id"]
        if cid not in by_id:
            raise KeyError(f"scope case {cid} not in source panel")
        c = dict(by_id[cid])
        c["scope_layout_sha256"] = s["layout_sha256"]
        rows.append(c)
    return rows


def _fetch_cached(revision: str, repo_path: str, expected_sha: str, dest: Path) -> dict:
    """已有内容且哈希匹配的缓存直接复用（重跑可离线）；否则重新下载。"""
    if dest.is_file():
        sha, size = sha256_file(dest)
        if sha == expected_sha and not is_lfs_pointer(dest.open("rb").read(64)):
            return {"path": repo_path, "sha256": sha, "bytes": size, "cached": True}
        dest.unlink()
    got = fetch_dataset_file(revision, repo_path, dest)
    got["cached"] = False
    return got


def verify_case_inputs(
    panel: dict,
    scope: dict,
    revision: str,
    cache_dir: str | Path,
    *,
    prior: dict | None = None,
) -> dict:
    """下载所选 case 的布局 JSON 与支持轨迹并逐字节验证。

    prior: 上轮输入一致性核查 JSON（可选）。提供时额外回比上轮 actual_sha256。
    布局同时比对 panel.layout.sha256 与 scope.layout_sha256，二者不一致即失败。
    """
    cache = Path(cache_dir)
    results = {"layouts": [], "trajectories": []}
    rows = selected_cases(panel, scope)
    for c in rows:
        lay = c["layout"]
        dest = cache / lay["path"]
        got = _fetch_cached(revision, lay["path"], lay["sha256"], dest)
        match = got["sha256"] == lay["sha256"] == c["scope_layout_sha256"]
        rec = {"case_id": c["case_id"], "path": lay["path"], "expected": lay["sha256"],
               "actual": got["sha256"], "match": match, "bytes": got["bytes"]}
        if prior is not None:
            pr = prior["layout_index"].get(lay["path"])
            rec["prior_match"] = (pr == got["sha256"]) if pr else None
        results["layouts"].append(rec)
        for t in c.get("native_support_trajectories", []):
            dest = cache / t["path"]
            got = _fetch_cached(revision, t["path"], t["sha256"], dest)
            rec = {"case_id": c["case_id"], "path": t["path"], "expected": t["sha256"],
                   "actual": got["sha256"], "match": got["sha256"] == t["sha256"],
                   "bytes": got["bytes"]}
            if prior is not None:
                pr = prior["trajectory_index"].get(t["path"])
                rec["prior_match"] = (pr == got["sha256"]) if pr else None
            results["trajectories"].append(rec)
    return results


def aggregate_identity(pairs: list[tuple[str, str]]) -> str:
    """上游身份聚合配方，已对照作者 previous_load_sha256=d15fb8bd… 精确验证：
    pairs 转为 {relpath: sha256} 映射，取
    sha256(json.dumps(mapping, sort_keys=True).encode("utf-8"))。
    该规则属于上游原实现身份，不得更换；内容级校验在服务器下载完成后执行，
    届时同一配方作用于实际文件哈希。
    """
    mapping = dict(pairs)
    return sha256_bytes(json.dumps(mapping, sort_keys=True).encode("utf-8"))


def checkpoint_identity_files(tree: list[dict], checkpoint_subdir: str) -> list[dict]:
    """从 HF tree 列表筛出推理身份 17 文件（params/** + assets/**）。"""
    out = []
    for f in tree:
        if f.get("type") != "file":
            continue
        rel = f["path"]
        if not rel.startswith(checkpoint_subdir.rstrip("/") + "/"):
            continue
        inner = rel[len(checkpoint_subdir.rstrip("/")) + 1:]
        if inner.startswith(CHECKPOINT_IDENTITY_PREFIXES):
            lfs = f.get("lfs") or {}
            out.append({"path": inner, "lfs_sha256": lfs.get("oid"),
                        "size": lfs.get("size", f.get("size"))})
    return sorted(out, key=lambda r: r["path"])
