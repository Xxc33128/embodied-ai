"""W8 transport 故障注入测试：幂等 ACK、陈旧拒绝、中毒、图像哈希。"""

from __future__ import annotations

import pytest

from gap_repro.transport import (
    ActionLedger,
    RequestIdentity,
    image_content_hash,
    verify_image,
)


def _req(rid=1, step=1, epoch=0):
    return RequestIdentity(campaign_id="c", episode_id="e", request_id=rid,
                           step_id=step, control_epoch=epoch)


class TestIdempotentAck:
    def test_duplicate_request_returns_original_ack(self):
        led = ActionLedger()
        a1 = led.record(_req(1, 1), executed_steps=10, final_obs_hash="h1")
        a2 = led.get_ack(_req(1, 1))
        assert a2 is a1 and a2.executed_steps == 10

    def test_re_record_does_not_overwrite(self):
        led = ActionLedger()
        led.record(_req(1, 1), executed_steps=10)
        again = led.record(_req(1, 1), executed_steps=99)
        assert again.executed_steps == 10  # record 层面也幂等（保守）


class TestStaleRejection:
    def test_stale_step_rejected_after_chunk_execution(self):
        # 审查 R1：chunk 执行 1..15 步后窗口推进 → 窗口内旧步号全部陈旧
        led = ActionLedger()
        led.record(_req(1, 1), executed_steps=10)
        reason = led.validate_stale(_req(2, 5))
        assert reason is not None and "stale" in reason

    def test_next_step_allowed_after_chunk(self):
        # R1 语义：chunk 执行后窗口推进到 step+executed，下一步 = 4
        led = ActionLedger()
        led.record(_req(1, 3), executed_steps=1)
        assert led.validate_stale(_req(2, 4)) is None

    def test_future_step_rejected(self):
        # 审查 Y3：session 要求精确步号，超前亦拒
        led = ActionLedger()
        reason = led.validate_stale(_req(9, 5))
        assert reason is not None and "future" in reason

    def test_cross_episode_request_id_not_deduped(self):
        # 审查 R2：跨 episode 复用 request_id 不得命中旧 ACK。
        # 顺序流：episode e 执行完 chunk 后 current=11 → e2 首步 step 11 起。
        led = ActionLedger()
        led.record(_req(1, 1), executed_steps=10)
        ack, why = led.submit(RequestIdentity("c", "e2", 1, 11, 0), executed_steps=2)
        assert ack is not None and ack.executed_steps == 2  # 新 ACK，非旧值
        assert ack.request.episode_id == "e2"

    def test_submit_poisoned_when_unknown(self):
        # 审查 Y1/Y2：组合式提交门 + dispatch-unknown 中间态（顺序执行流）
        led = ActionLedger()
        led.record(_req(1, 1), executed_steps=1)
        ack, why = led.submit(_req(2, 2), executed_steps=None)
        assert ack is not None and ack.status == "poisoned" and why == "poisoned"
        again, _ = led.submit(_req(2, 2), executed_steps=5)
        assert again.status == "poisoned"  # 中毒后禁重放

    def test_submit_rejects_stale(self):
        led = ActionLedger()
        led.record(_req(1, 3), executed_steps=10)
        ack, why = led.submit(_req(2, 2), executed_steps=1)
        assert ack is None and why is not None and "stale" in why


class TestPoisoned:
    def test_unknown_execution_marked_poisoned(self):
        led = ActionLedger()
        ack = led.mark_poisoned(_req(7, 5))
        assert ack.status == "poisoned" and ack.executed_steps == -1
        assert led.get_ack(_req(7, 5)).status == "poisoned"

    def test_poisoned_existing_keeps_step_count(self):
        led = ActionLedger()
        led.record(_req(7, 5), executed_steps=4)
        ack = led.mark_poisoned(_req(7, 5))
        assert ack.status == "poisoned" and ack.executed_steps == 4


class TestImageHash:
    def test_verify_roundtrip(self):
        payload = b"\x89PNG fake image bytes"
        assert verify_image(payload, image_content_hash(payload))

    def test_corrupted_rejected(self):
        assert not verify_image(b"corrupted", image_content_hash(b"original"))


class TestIdempotentReplay:
    """T2（audit F02）：执行后重取同一请求必须返回原 ACK，而非判陈旧。"""

    def test_submit_replay_returns_original_ack(self):
        ledger = ActionLedger()
        request = RequestIdentity("c", "e", 1, 0, 0)
        first, error = ledger.submit(request, executed_steps=5)
        again, error = ledger.submit(request, executed_steps=5)
        assert error is None
        assert again == first
        assert ledger.current_step == 5

    def test_replay_with_executed_steps_none_returns_cached(self):
        # 重取方可能不知道执行步数——仍应返回原 ACK，不得判 poisoned
        ledger = ActionLedger()
        req = RequestIdentity("c", "e", 1, 0, 0)
        first, _ = ledger.submit(req, executed_steps=5)
        again, why = ledger.submit(req, executed_steps=None)
        assert why is None
        assert again is not None and again.executed_steps == 5
        assert again.status == "ok"

    def test_replay_same_id_different_payload_rejected(self):
        # 复用 request_id 但改 payload：冲突拒绝，不得返回缓存冒充
        ledger = ActionLedger()
        req = RequestIdentity("c", "e", 1, 0, 0)
        ledger.submit(req, executed_steps=5, payload_sha="sha-A")
        again, why = ledger.submit(req, executed_steps=5, payload_sha="sha-B")
        assert again is None and why is not None and "payload" in why

    def test_replay_same_id_different_step_rejected(self):
        # 同 ID 改 step：身份冲突，明确拒绝（不静默缓存命中也不静默新记录）
        ledger = ActionLedger()
        ledger.submit(RequestIdentity("c", "e", 1, 0, 0), executed_steps=1,
                      payload_sha="s")
        again, why = ledger.submit(RequestIdentity("c", "e", 1, 7, 0),
                                   executed_steps=1, payload_sha="s")
        assert again is None and why is not None

    def test_future_step_rejected_before_replay_check(self):
        # 未执行过的未来 step 仍拒绝（重取检查不能放行超前请求）
        ledger = ActionLedger()
        reason = ledger.validate_stale(RequestIdentity("c", "e", 9, 5, 0))
        assert reason is not None and "future" in reason


class TestFaultSemantics:
    """T2：故障 fixture——状态与执行计数证明，不止异常文本。"""

    def test_ack_lost_then_redispatch_unknown_stays_poisoned(self):
        # ACK 丢失 → dispatch unknown → poisoned；后续不能继续执行新动作
        ledger = ActionLedger()
        led1, why = ledger.submit(RequestIdentity("c", "e", 1, 0, 0),
                                  executed_steps=None)
        assert led1.status == "poisoned"
        nxt, why2 = ledger.submit(RequestIdentity("c", "e", 2, 1, 0),
                                  executed_steps=1)
        assert nxt is None and why2 is not None  # poisoned 状态禁止新动作

    def test_duplicate_response_does_not_double_count(self):
        ledger = ActionLedger()
        r = RequestIdentity("c", "e", 1, 0, 0)
        a1, _ = ledger.submit(r, executed_steps=5, payload_sha="s")
        a2, _ = ledger.submit(r, executed_steps=5, payload_sha="s")
        assert ledger.current_step == 5  # 物理步数不重复增长
        assert a1 is a2

    def test_old_epoch_is_separate_identity(self):
        # 旧 epoch 的已执行请求重取仍返回其原 ACK（不串到新 epoch）
        ledger = ActionLedger()
        old = RequestIdentity("c", "e", 1, 0, 0)
        ledger.submit(old, executed_steps=5, payload_sha="s")
        ledger.submit(RequestIdentity("c", "e", 2, 5, 1), executed_steps=3,
                      payload_sha="s")
        again, why = ledger.submit(old, executed_steps=5, payload_sha="s")
        assert why is None and again.executed_steps == 5

    def test_step_never_double_executes_across_faults(self):
        # 执行后断连→重取→新步：全程物理步数只增一次/chunk
        ledger = ActionLedger()
        r1 = RequestIdentity("c", "e", 1, 0, 0)
        ledger.submit(r1, executed_steps=5, payload_sha="p1")
        ledger.submit(r1, executed_steps=5, payload_sha="p1")   # 断连后重取
        r2 = RequestIdentity("c", "e", 2, 5, 0)
        ack2, why = ledger.submit(r2, executed_steps=3, payload_sha="p2")
        assert why is None and ack2.executed_steps == 3
        assert ledger.current_step == 8  # 5+3，无重复执行
