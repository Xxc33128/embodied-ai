"""W8：transport/RPC 会话层——请求身份、幂等 ACK 账本、陈旧拒绝、中毒标记。

原文语义（计划 §6.3 W8 + session.py）：
- 请求身份 {campaign_id, episode_id, request_id, step_id, control_epoch}；
- 已执行请求重复到达只返回原 ACK；执行次数不明 → poisoned，禁止盲重放；
- 图像按内容哈希传输并校验。

本模块不承担任务规划；网络等待期间物理暂停由 runner 保证。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import hashlib


@dataclass(frozen=True)
class RequestIdentity:
    campaign_id: str
    episode_id: str
    request_id: int
    step_id: int
    control_epoch: int


@dataclass
class Ack:
    request: RequestIdentity
    executed_steps: int
    final_obs_hash: str | None = None
    status: str = "ok"  # ok | poisoned


class ActionLedger:
    """幂等动作账本（T2，audit F02）。

    判定顺序（修正：重取检查先于陈旧检查）：
    1. 已有记录（同 campaign+episode+request_id+epoch）：step 与 payload 摘要
       完全一致才返回原 ACK（不再次执行）；step 或 payload 不一致 → 身份冲突
       明确拒绝（不冒充缓存命中，也不静默新记录）；poisoned 记录重取仍返回
       poisoned（禁重放）。
    2. 新请求：陈旧/超前置拒 → 记录。
    3. 同 (campaign, episode, epoch) 内存在 poisoned 记录时拒绝一切新动作
       （执行状态未知不得被后续动作掩盖；恢复/新 epoch 才解除）。

    payload_sha 为动作内容摘要（由调用方提供；缺省 None 表示无法校验，
    不参与冲突判定）。executed_steps=None 表示执行发起但结果不明 → poisoned。
    账本为进程内状态，进程崩溃后的恢复依赖 §8.4 日志落盘（runner 职责），
    此处如实不声称可恢复。"""

    def __init__(self):
        self._acks: dict[tuple, Ack] = {}
        self._payloads: dict[tuple, str | None] = {}
        self._poisoned_scopes: set[tuple] = set()
        self.current_step = 0

    @staticmethod
    def _key(request: RequestIdentity) -> tuple:
        # R2：跨 episode/epoch 复用 request_id 不得互相命中；
        # step 不入键（重取可能带不同 step——那是身份冲突，须显式拒绝）
        return (request.campaign_id, request.episode_id,
                request.request_id, request.control_epoch)

    @staticmethod
    def _scope(request: RequestIdentity) -> tuple:
        return (request.campaign_id, request.episode_id, request.control_epoch)

    def get_ack(self, request: RequestIdentity) -> Ack | None:
        return self._acks.get(self._key(request))

    def validate_stale(self, request: RequestIdentity) -> str | None:
        """陈旧/超前检查（session 要求精确步号）：落后即陈旧，超前亦拒。"""
        if request.step_id < self.current_step:
            return f"stale step_id {request.step_id} < current {self.current_step}"
        if request.step_id > self.current_step + 1:
            return (f"future step_id {request.step_id} > "
                    f"current+1 {self.current_step + 1}")
        return None

    def submit(self, request: RequestIdentity,
               executed_steps: int | None = None,
               final_obs_hash: str | None = None,
               payload_sha: str | None = None) -> tuple[Ack | None, str | None]:
        """组合式提交门（T2 顺序）：
        1. 已有记录：身份+payload 全一致 → 返回原 ACK（幂等重取，不执行）；
           step/payload 不一致 → 身份冲突拒绝；poisoned 记录 → 仍返回 poisoned。
        2. poisoned scope 封锁：同 episode+epoch 存在中毒记录 → 拒绝新动作。
        3. 新请求：陈旧/超前置拒 → 记录；executed_steps=None → poisoned。"""
        key = self._key(request)
        prev = self._acks.get(key)
        if prev is not None:
            if prev.request.step_id != request.step_id:
                return None, (f"identity conflict: request_id {request.request_id} "
                              f"recorded at step {prev.request.step_id}, "
                              f"re-sent at step {request.step_id}")
            stored = self._payloads.get(key)
            if stored is not None and payload_sha is not None and stored != payload_sha:
                return None, (f"identity conflict: request_id {request.request_id} "
                              f"payload mismatch (recorded {stored[:12]}, "
                              f"got {payload_sha[:12]})")
            if prev.status == "poisoned":
                return prev, "poisoned"  # 中毒禁重放：重取仍返回中毒态
            return prev, None
        if self._scope(request) in self._poisoned_scopes:
            return None, ("poisoned execution state in this episode/epoch; "
                          "recovery or new control_epoch required before new actions")
        stale = self.validate_stale(request)
        if stale:
            return None, stale
        if executed_steps is None:
            ack = self.mark_poisoned(request)
            return ack, "poisoned"
        ack = self.record(request, executed_steps, final_obs_hash,
                          payload_sha=payload_sha)
        return ack, None

    def record(self, request: RequestIdentity, executed_steps: int,
               final_obs_hash: str | None = None,
               payload_sha: str | None = None) -> Ack:
        key = self._key(request)
        if key in self._acks:  # record 层面同样幂等（保守不覆盖）
            return self._acks[key]
        ack = Ack(request=request, executed_steps=executed_steps,
                  final_obs_hash=final_obs_hash)
        self._acks[key] = ack
        self._payloads[key] = payload_sha
        # R1：陈旧判定用执行后步号（chunk 执行 1..15 步后窗口随之推进）
        self.current_step = max(self.current_step, request.step_id + executed_steps)
        return ack

    def mark_poisoned(self, request: RequestIdentity):
        ack = self._acks.get(self._key(request))
        if ack is None:
            ack = Ack(request=request, executed_steps=-1, status="poisoned")
            self._acks[self._key(request)] = ack
            self._payloads[self._key(request)] = None
        else:
            ack.status = "poisoned"
        self._poisoned_scopes.add(self._scope(request))
        return ack


def image_content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_image(payload: bytes, expected_sha256: str) -> bool:
    """图像按内容哈希校验（传输后解码一致性）。"""
    return image_content_hash(payload) == expected_sha256
