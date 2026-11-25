"""L2 学生链路参数化测试（本地；传输用 fake，无需服务器/环境）。"""

from __future__ import annotations

import numpy as np
import pytest

from gap_repro.libero.actions import LIBERO_ACTION_DIM
from gap_repro.libero.policy import (
    REPLAN_STEPS,
    CaseSpec,
    FileNpzTransport,
    LiberoStudentPolicy,
    build_element,
    max_steps_for,
    suite_of_benchmark,
)


def _obs(resolution=224):
    rng = np.random.default_rng(0)
    return {
        "agentview_image": rng.integers(0, 255, (resolution, resolution, 3),
                                        dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 255,
                                                 (resolution, resolution, 3),
                                                 dtype=np.uint8),
        "robot0_eef_pos": np.array([0.4, 0.0, 0.2], dtype=np.float64),
        "robot0_eef_quat": np.array([0.0, 1.0, 0.0, 0.0]),  # (x,y,z,w) robosuite
        "robot0_gripper_qpos": np.array([0.0208, -0.0208]),
    }


class TestSuiteResolution:
    def test_pinned_max_steps(self):
        # openpi examples/libero/main.py 逐字（E1 钉定）
        assert max_steps_for("libero_spatial") == 220
        assert max_steps_for("libero_object") == 280
        assert max_steps_for("libero_goal") == 300
        assert max_steps_for("libero_10") == 520

    def test_variants_share_suite_budget(self):
        for v in ("libero_goal_lan", "libero_goal_swap", "libero_goal_object",
                  "libero_goal_task", "libero_goal_env"):
            assert max_steps_for(v) == 300
        assert max_steps_for("libero_10_lan") == 520

    def test_unknown_rejected(self):
        with pytest.raises(ValueError, match="unknown benchmark"):
            max_steps_for("libero_90_lan")  # 90 不在本面板
        with pytest.raises(ValueError, match="unknown benchmark"):
            max_steps_for("libero_foo")

    def test_suite_of_benchmark(self):
        assert suite_of_benchmark("libero_goal") == "libero_goal"
        assert suite_of_benchmark("libero_spatial_env") == "libero_spatial"


class TestCaseSpec:
    def test_from_inventory(self):
        case = {"benchmark": "libero_goal_lan", "condition": "Sem",
                "task_index": 3, "task_name": "open_the_middle_drawer",
                "benchmark_language": "pull out the middle drawer"}
        spec = CaseSpec.from_inventory(
            case, campaign_id="c1", episode_id="e1", init_index=7)
        assert spec.max_steps == 300
        assert spec.instruction == "pull out the middle drawer"
        assert spec.method == "student_only"
        assert spec.init_index == 7


class TestElement:
    def test_state_is_8d_and_images_rotated(self):
        obs = _obs()
        el = build_element(obs, "open the drawer")
        assert el["state"].shape == (8,)
        assert el["state"].dtype == np.float32
        assert el["agentview"].dtype == np.uint8
        # 180° 旋转 = 双轴翻转
        assert np.array_equal(el["agentview"], obs["agentview_image"][::-1, ::-1])
        assert el["prompt"] == "open the drawer"

    def test_quat_xyzw_converted(self):
        # robosuite (x,y,z,w)=(0,1,0,0)：y=1 → 绕 y 轴 180° → rotvec (0,π,0)
        el = build_element(_obs(), "p")
        assert np.isclose(abs(el["state"][4]), np.pi, atol=1e-5)


class FakeTransport:
    def __init__(self, chunk_steps=10):
        self.chunk_steps = chunk_steps
        self.calls = []

    def infer(self, element, request_id):
        self.calls.append((request_id, element))
        actions = np.zeros((self.chunk_steps, LIBERO_ACTION_DIM),
                           dtype=np.float32)
        actions[:, 6] = -1.0
        return actions, {"server": "fake"}


class TestStudentPolicy:
    def test_infer_slices_replan_and_hashes_full(self):
        t = FakeTransport()  # 10 步全程
        p = LiberoStudentPolicy(t)
        chunk, ident = p.infer_chunk(_obs(), "open the drawer", "req-1")
        assert chunk.shape == (REPLAN_STEPS, LIBERO_ACTION_DIM)  # 执行段 5
        assert ident["full_steps"] == 10                          # 哈希全程
        assert ident["request_id"] == "req-1"
        assert ident["noise"] is None  # 正式噪声契约
        chunk2, ident2 = p.infer_chunk(_obs(), "open the drawer", "req-1")
        assert ident["actions_sha256"] == ident2["actions_sha256"]
        assert ident["state_sha256"] == ident2["state_sha256"]

    def test_short_chunk_rejected(self):
        p = LiberoStudentPolicy(FakeTransport(chunk_steps=4))
        with pytest.raises(ValueError, match="replan"):
            p.infer_chunk(_obs(), "p", "req-2")

    def test_nonfinite_chunk_rejected(self):
        t = FakeTransport()
        orig = t.infer

        def bad(element, request_id):
            a, m = orig(element, request_id)
            a[0, 0] = np.nan
            return a, m

        t.infer = bad
        p = LiberoStudentPolicy(t)
        with pytest.raises(Exception, match="non-finite"):
            p.infer_chunk(_obs(), "p", "req-3")

    def test_transport_receives_element(self):
        t = FakeTransport()
        p = LiberoStudentPolicy(t)
        p.infer_chunk(_obs(), "open", "req-4")
        rid, el = t.calls[0]
        assert rid == "req-4"
        assert el["state"].shape == (8,)


class TestFileNpzTransport:
    def test_atomic_write_and_response(self, tmp_path):
        req, resp = tmp_path / "req", tmp_path / "resp"
        req.mkdir()
        resp.mkdir()
        import numpy as _np
        import threading
        payload = {"actions": np.zeros((10, 7), dtype=np.float32),
                   "meta": np.array('{"server": "x"}')}

        def serve_later():  # 模拟服务器：infer 启动后才应答
            import time
            time.sleep(0.2)
            with open(resp / "r1.npz", "wb") as f:
                _np.savez(f, **payload)

        threading.Thread(target=serve_later, daemon=True).start()
        t = FileNpzTransport(req, resp, timeout_s=5.0)
        el = build_element(_obs(), "p")
        actions, meta = t.infer(el, "r1")
        assert actions.shape == (10, 7)
        assert meta["server"] == "x"
        # 请求侧：最终名存在、无 .tmp 残留
        assert (req / "r1.npz").exists()
        assert not list(req.glob("*.tmp"))

    def test_stale_response_cleaned(self, tmp_path):
        # P2-3：同 request_id 的旧响应在 infer 开始时必须被清除
        req, resp = tmp_path / "req", tmp_path / "resp"
        resp.mkdir(parents=True)
        import numpy as _np
        stale = {"actions": np.full((10, 7), 9.0, dtype=np.float32),
                 "meta": np.array('{"server": "stale"}')}
        with open(resp / "r1.npz", "wb") as f:
            _np.savez(f, **stale)
        t = FileNpzTransport(req, resp, poll_s=0.01, timeout_s=0.15)
        el = build_element(_obs(), "p")
        with pytest.raises(TimeoutError):
            t.infer(el, "r1")  # 旧响应被清掉 → 超时（而非消费旧值）
        assert not (resp / "r1.npz").exists()

    def test_timeout(self, tmp_path):
        t = FileNpzTransport(tmp_path / "a", tmp_path / "b",
                             poll_s=0.01, timeout_s=0.05)
        el = build_element(_obs(), "p")
        with pytest.raises(TimeoutError):
            t.infer(el, "nope")

    def test_error_file_surfaced(self, tmp_path):
        req, resp = tmp_path / "a", tmp_path / "b"
        resp.mkdir(parents=True)
        import threading
        import time

        def err_later():
            time.sleep(0.2)
            (resp / "r9.err").write_text("boom")

        threading.Thread(target=err_later, daemon=True).start()
        t = FileNpzTransport(req, resp, timeout_s=5.0)
        el = build_element(_obs(), "p")
        with pytest.raises(RuntimeError, match="boom"):
            t.infer(el, "r9")
        assert not (resp / "r9.err").exists()  # 消费后清除
