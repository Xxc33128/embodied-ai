"""L2：参数化学生链路（pi05 LIBERO，单臂 7D）。

口径钉定：
- SUITE_MAX_STEPS 逐字取自 openpi examples/libero/main.py @ pin
  （spatial 220 / object 280 / goal 300 / 10 520；libero_90=400 不在本面板）。
  变体 benchmark（libero_goal_lan 等）与基线共享 suite 前缀 → 同 max_steps。
- 预处理：180° 旋转（[::-1, ::-1]，训练预处理对齐）+ resize_with_pad 224
  uint8；224 捕获下 resize 恒等（LP-A2 已验证，保留调用以保客户端一致性）。
- state 8 维 = eef_pos(3) + axisangle(eef_quat)(3) + 双指 qpos(2)（LP-A2 契约）。
- prompt = task.language（benchmark 属性；evaluate.py 同源，L1 裁决 #3）。
- 正式噪声契约（与 LP-A2 验证机制分离）：正式路径请求不带 noise 字段 →
  服务器 infer(noise=None) → openpi 内部采样（官方客户端行为逐字一致）；
  LP-A2 的显式 (10,32) NumPy noise 是设备对照专用的验证机制，不进正式协议。
  候选身份 = actions sha256 + state sha256 + prompt sha256 + request_id。
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gap_repro.agent.contract import ContractError
from gap_repro.libero.actions import (
    LIBERO_ACTION_DIM,
    pose_to_policy_state,
    validate_student_chunk,
)

RESOLUTION = 224
REPLAN_STEPS = 5
WAIT_STEPS = 10
ACTION_HORIZON = 10  # 服务器响应 (10, 7)

# openpi examples/libero/main.py（E1 逐字；验证见 tests/test_libero_policy.py）
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

CONDITION_SUFFIXES = ("_object", "_swap", "_lan", "_task", "_env")


def suite_of_benchmark(benchmark_name: str) -> str:
    """变体 benchmark 名 → 套件名（剥离条件后缀；Ori 无后缀恒等）。

    先精确匹配（libero_object 本身是基线套件，不得误剥 _object 后缀），
    未命中再按条件后缀剥离（libero_object_object → libero_object）。
    """
    if benchmark_name in SUITE_MAX_STEPS:
        return benchmark_name
    for suf in CONDITION_SUFFIXES:
        if benchmark_name.endswith(suf):
            return benchmark_name[: -len(suf)]
    return benchmark_name


def max_steps_for(benchmark_name: str) -> int:
    suite = suite_of_benchmark(benchmark_name)
    if suite not in SUITE_MAX_STEPS:
        raise ValueError(f"unknown benchmark/suite: {benchmark_name}")
    return SUITE_MAX_STEPS[suite]


@dataclass(frozen=True)
class CaseSpec:
    """L1 清单 case → 正式 episode 的完整身份（冻结 manifest 的单元）。"""
    campaign_id: str
    episode_id: str
    benchmark: str          # 变体名（libero_goal_lan 等）
    condition: str          # Ori/Obj/Pos/Sem/Task/Env
    task_index: int
    task_name: str
    instruction: str        # = task.language（benchmark 属性）
    init_index: int
    max_steps: int
    method: str = "student_only"

    @classmethod
    def from_inventory(cls, case: dict, *, campaign_id: str, episode_id: str,
                       init_index: int, method: str = "student_only"):
        """l1_case_inventory.json 的单条 case → CaseSpec（max_steps 按套件解析）。"""
        return cls(
            campaign_id=campaign_id, episode_id=episode_id,
            benchmark=case["benchmark"], condition=case["condition"],
            task_index=case["task_index"], task_name=case["task_name"],
            instruction=case["benchmark_language"],
            init_index=init_index, max_steps=max_steps_for(case["benchmark"]),
            method=method)


def sha256_of(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def preprocess_images(obs) -> tuple:
    """官方客户端预处理：180° 旋转 + uint8。返回 (agentview, wrist)。"""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if wrist.dtype != np.uint8:
        wrist = wrist.astype(np.uint8)
    return img, wrist


def build_state(obs) -> np.ndarray:
    """8 维 state 向量（openpi concat 顺序；四元数保持 robosuite
    (x,y,z,w) 原样，由 axisangle_from_xyzw 逐字编码）。"""
    return pose_to_policy_state(
        obs["robot0_eef_pos"],
        np.asarray(obs["robot0_eef_quat"], dtype=np.float64),
        obs["robot0_gripper_qpos"]).astype(np.float32)


def build_element(obs, prompt: str) -> dict:
    """policy 输入元素（字段名与 LP-A2 服务器/openpi 客户端契约一致）。"""
    img, wrist = preprocess_images(obs)
    state = build_state(obs)
    if state.shape != (8,):
        raise ContractError(f"state 契约 8 维，实得 {state.shape}")
    # 图像尺寸不设硬门：openpi resize_with_pad 接受任意尺寸（P3-1 修复
    # 时曾误加 224 硬门——过度校验会拒收合法观测，撤销）
    return {"agentview": img, "wrist": wrist, "state": state,
            "prompt": str(prompt)}


class LiberoStudentPolicy:
    """学生策略客户端：观测元素构造 → 传输 → chunk 校验 + 身份记录。"""

    def __init__(self, transport):
        self._transport = transport

    def infer_chunk(self, obs, prompt: str, request_id: str):
        """返回 (执行 chunk (replan, 7) float32, identity dict)。

        正式路径不传 noise（模块 docstring 噪声契约）；服务器全程 chunk
        过 validate_student_chunk（结构门）后切片 replan 段执行；候选
        身份哈希 = 全程 chunk（服务器产物原样）。
        """
        element = build_element(obs, prompt)
        actions, meta = self._transport.infer(element, request_id=request_id)
        full = validate_student_chunk(np.asarray(actions, dtype=np.float32))
        if full.shape[0] < REPLAN_STEPS:
            raise ValueError(
                f"policy chunk {full.shape[0]} < replan budget {REPLAN_STEPS}")
        chunk = full[:REPLAN_STEPS]
        identity = {
            "request_id": request_id,
            "actions_sha256": sha256_of(full),
            "full_steps": int(full.shape[0]),
            "state_sha256": sha256_of(element["state"]),
            "prompt_sha256": hashlib.sha256(
                element["prompt"].encode()).hexdigest(),
            "agentview_shape": list(element["agentview"].shape),
            "wrist_shape": list(element["wrist"].shape),
            "replan_steps": REPLAN_STEPS,
            "meta": meta,
            "noise": None,  # 正式契约：无显式 noise（见 docstring）
        }
        return chunk, identity


class FileNpzTransport:
    """LP-A2 npz 文件交换传输（复用 lp_a2_policy_server；原子写纪律）。

    请求 <dir>/<request_id>.npz：agentview/wrist/state/prompt(,noise 可选,
    meta 可选)；响应同名 .npz：actions + meta；错误路径 .err 文本。
    原子写：文件句柄 savez + os.rename（np.savez 路径自动加缀 .npz 的
    两次踩坑修复，LP-A2 验证过）。
    """

    def __init__(self, req_dir, resp_dir, *, poll_s=0.1, timeout_s=120.0,
                 noise=None, meta_extra=None):
        self.req_dir = Path(req_dir)
        self.resp_dir = Path(resp_dir)
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.noise = noise          # None = 正式协议；数组 = LP-A2 验证机制
        self.meta_extra = meta_extra or {}

    def infer(self, element: dict, request_id: str):
        import os
        self.req_dir.mkdir(parents=True, exist_ok=True)
        self.resp_dir.mkdir(parents=True, exist_ok=True)
        # P2-3：确定性 request_id 重跑时必须清理上一 attempt 的残留响应，
        # 否则会静默消费旧候选（重发=换候选禁令的镜像面）
        stale_resp = self.resp_dir / f"{request_id}.npz"
        stale_err = self.resp_dir / f"{request_id}.err"
        if stale_resp.exists():
            stale_resp.unlink()
        if stale_err.exists():
            stale_err.unlink()
        payload = {"agentview": element["agentview"],
                   "wrist": element["wrist"], "state": element["state"],
                   "prompt": np.array(str(element["prompt"]))}
        if self.noise is not None:
            payload["noise"] = self.noise
        if self.meta_extra:
            payload["meta"] = np.array(json.dumps(self.meta_extra))
        tmp = self.req_dir / (request_id + ".npz.tmp")
        with open(tmp, "wb") as f:
            np.savez(f, **payload)
        os.rename(tmp, self.req_dir / f"{request_id}.npz")
        deadline = time.time() + self.timeout_s
        resp = self.resp_dir / f"{request_id}.npz"
        err = self.resp_dir / f"{request_id}.err"
        while time.time() < deadline:
            if resp.exists():
                d = np.load(resp)
                actions = np.asarray(d["actions"], dtype=np.float32)
                meta = json.loads(str(d["meta"])) if "meta" in d else {}
                resp.unlink()
                return actions, meta
            if err.exists():
                text = err.read_text()
                err.unlink()
                raise RuntimeError(f"policy server error: {text}")
            time.sleep(self.poll_s)
        raise TimeoutError(f"policy server timeout: {request_id}")
