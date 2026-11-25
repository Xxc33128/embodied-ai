"""LP3：GPT 路线契约测试（纯逻辑，Mac/gap-sim 均可跑；上游 clone 缺失时相关用例 skip）。"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from gap_repro.agent import (
    contract, gate, client,
    UPSTREAM_COMMIT, GATE_PROMPT, response_schema, direct_response_schema,
    tool_specs, validate_eef_target, validate_mode_steps, apply_edit,
    dls_clamp_joint_step, EDIT_GRIPPER_TO_STUDENT, quat_angle_distance,
    validate_assessment, build_observation_packet, OBSERVATION_KEYS,
    AppServerTransport, ApiGatewayTransport, ConfigMissingError,
    dispatch_tool_call, DLS_MAX_PER_JOINT,
)

UPSTREAM_SCHEMA = pathlib.Path(
    "upstream/GPT-as-Policy/hybrid_rollout/robodojo/skill/schema.py")
UPSTREAM_RUN = pathlib.Path(
    "upstream/GPT-as-Policy/hybrid_rollout/robodojo/skill/run.py")
UPSTREAM_GATE_PROMPT = pathlib.Path(
    "upstream/GPT-as-Policy/hybrid_rollout/robodojo/skill/gate_prompt.md")


def load_upstream(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_upstream", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def upstream_schema():
    if not UPSTREAM_SCHEMA.exists():
        pytest.skip("upstream GPT-as-Policy clone missing")
    return load_upstream(UPSTREAM_SCHEMA)


def test_schema_bitwise_equivalent_to_upstream(upstream_schema):
    """移植正确性的黄金锚：与钉定 commit 的上游实现逐字节对等。"""
    for rid in (None, "req-1"):
        assert json.dumps(response_schema(rid), sort_keys=True) == \
            json.dumps(upstream_schema.response_schema(rid), sort_keys=True)
    assert json.dumps(direct_response_schema(), sort_keys=True) == \
        json.dumps(upstream_schema.direct_response_schema(), sort_keys=True)


def test_tool_specs_equivalent_to_upstream_source():
    """工具表：描述文本与参数键逐字对照上游 run.py::tool_specs 源码（skip 若无 clone）。

    上游描述是跨行相邻字符串字面量的编译期拼接：用 ast 抽取该函数内的字符串
    常量序列，验证每个描述可由一段连续常量拼接而成（比去空白子串匹配更严格，
    后者会跨不过引号边界）。
    """
    if not UPSTREAM_RUN.exists():
        pytest.skip("upstream GPT-as-Policy clone missing")
    import ast
    tree = ast.parse(UPSTREAM_RUN.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "tool_specs")
    consts = [n.value for n in ast.walk(fn)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)]

    def formed_by_consecutive_literals(desc):
        for i in range(len(consts)):
            acc = ""
            for j in range(i, len(consts)):
                acc += consts[j]
                if acc == desc:
                    return True
                if len(acc) > len(desc):
                    break
        return False

    src_text = UPSTREAM_RUN.read_text()
    for tool in tool_specs() + tool_specs("gpt_only"):
        assert formed_by_consecutive_literals(tool["description"]), \
            f"描述文本与上游不符: {tool['name']}"
        for key in tool["inputSchema"]["properties"]:
            assert f"{key}=" in src_text, f"{tool['name']}.{key} 非上游参数"
    assert [t["name"] for t in tool_specs()] == [
        "robodojo_start", "pi05_infer", "robodojo_execute"]
    assert [t["name"] for t in tool_specs("gpt_only")] == ["robodojo_start", "robodojo_act"]


def test_gate_prompt_verbatim():
    if not UPSTREAM_GATE_PROMPT.exists():
        pytest.skip("upstream GPT-as-Policy clone missing")
    assert GATE_PROMPT == UPSTREAM_GATE_PROMPT.read_text(), "gate prompt 必须逐字同源"


def test_upstream_commit_pin():
    """上游 clone 必须处于契约所钉的 commit（防止漂移后黄金测试失义）。"""
    git = pathlib.Path("upstream/GPT-as-Policy/.git/HEAD")
    if not git.exists():
        pytest.skip("upstream clone missing")
    head = pathlib.Path("upstream/GPT-as-Policy/.git")
    resolved = (head / "HEAD").read_text().strip()
    if resolved.startswith("ref:"):
        resolved = (head / resolved.split(": ")[1]).read_text().strip()
    assert resolved == UPSTREAM_COMMIT


def test_gate_truth_table():
    def resp(mode, execution, intent):
        return {"mode": mode, "assessment": {
            "task_progress": {"verified_completed": ["a"], "currently_attempting": "b",
                              "remaining": ["c"]},
            "current_subgoal": "s", "execution_status": execution,
            "execution_evidence": "e", "expected_next_intent": "i",
            "predicted_next_intent": "p", "intent_status": intent,
            "intent_evidence": "v"}}
    req1 = {"step_id": 1}
    # takeover 需 failed ∨ misaligned
    assert validate_assessment(resp("edit", "failed", "aligned"), req1) == "execution_failure"
    assert validate_assessment(resp("eef", "progressing", "misaligned"), req1) == "wrong_intent"
    assert validate_assessment(resp("eef", "failed", "misaligned"), req1) == "both"
    # uncertainty 不是接管理由；student 段不触发
    with pytest.raises(ValueError, match="Takeover requires"):
        validate_assessment(resp("edit", "uncertain", "uncertain"), req1)
    assert validate_assessment(resp("student", "failed", "misaligned"), req1) == "none"
    assert validate_assessment(resp("stop", "progressing", "aligned"), req1) == "none"
    # not_started 只允许 step 0
    with pytest.raises(ValueError, match="not_started"):
        validate_assessment(resp("student", "not_started", "aligned"), req1)
    with pytest.raises(ValueError, match="not_started"):
        validate_assessment(resp("student", "progressing", "aligned"), {"step_id": 0})


def test_eef_decision_bounds():
    cur = {"position": [0.5, 0.0, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    ok = {"position": [0.54, 0.0, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
          "gripper_closed": False}
    assert validate_eef_target(ok, cur)["translation_m"] == pytest.approx(0.04)
    # 6cm 平移越界
    with pytest.raises(contract.ContractError, match="translation"):
        validate_eef_target({"position": [0.57, 0.0, 0.2],
                             "quaternion_wxyz": [1.0, 0, 0, 0],
                             "gripper_closed": True}, cur)
    # 0.4 rad 旋转越界（绕 z）
    q = [np.cos(0.2), 0, 0, np.sin(0.2)]
    with pytest.raises(contract.ContractError, match="rotation"):
        validate_eef_target({"position": [0.5, 0.0, 0.2], "quaternion_wxyz": q,
                             "gripper_closed": True}, cur)
    assert quat_angle_distance(q, cur["quaternion_wxyz"]) == pytest.approx(0.4)


def test_mode_steps_limits():
    validate_mode_steps("student", 15)
    validate_mode_steps("eef", 5)
    with pytest.raises(contract.ContractError):
        validate_mode_steps("student", 16)
    for mode in ("edit", "eef"):
        with pytest.raises(contract.ContractError):
            validate_mode_steps(mode, 6)
    with pytest.raises(contract.ContractError):
        validate_mode_steps("teleport", 3)


def test_apply_edit_and_gripper_mapping():
    pose = {"position": [0.1, 0.2, 0.3], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    edit = {"delta_position": [0.01, 0.0, 0.0],
            "delta_rotation_vector": [0.0, 0.0, 0.1], "gripper": "open"}
    new = apply_edit(pose, edit)
    assert new["position"] == pytest.approx([0.11, 0.2, 0.3])
    assert new["quaternion_wxyz"] == pytest.approx(
        [np.cos(0.05), 0, 0, np.sin(0.05)], abs=1e-12)
    assert EDIT_GRIPPER_TO_STUDENT["open"] == 1.0   # 学生值 1=开
    assert EDIT_GRIPPER_TO_STUDENT["closed"] == 0.0  # 学生值 0=闭
    assert EDIT_GRIPPER_TO_STUDENT["keep"] is None


def test_dls_clamp():
    d = dls_clamp_joint_step(np.array([0.03, -0.08, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert abs(d[0]) == pytest.approx(0.03) and abs(d[1]) == pytest.approx(DLS_MAX_PER_JOINT)


def test_observation_whitelist():
    pkt = build_observation_packet(step_id=3, images=["rgb"], instruction="put…",
                                   current_eef={"position": [0]}, current_state=[0.0]*14)
    assert set(pkt) <= OBSERVATION_KEYS
    with pytest.raises(contract.ContractError, match="non-whitelisted"):
        build_observation_packet(step_id=0, ground_truth_score=1.0)
    with pytest.raises(contract.ContractError, match="non-whitelisted"):
        build_observation_packet(step_id=0, layout_answer="layout_042")


def test_transport_missing_config_is_the_user_ask():
    a = AppServerTransport({})
    with pytest.raises(ConfigMissingError) as e:
        a.connect()
    assert set(e.value.missing) == {"app_server_cmd", "model", "provider", "effort"}
    b = ApiGatewayTransport({"endpoint": "https://x", "model": "m", "effort": "xhigh",
                             "api_key_env": "GPT Gateway no key"})
    with pytest.raises(ConfigMissingError) as e:
        b.connect()
    assert "environment variable GPT Gateway no key" in e.value.missing[0]


def test_dispatch_tool_call_semantics():
    calls = []

    def robodojo_start(task, output_dir):
        calls.append(task)
        return {"started": True}
    # 字符串参数解析 + 分发
    out = dispatch_tool_call("robodojo_start", json.dumps(
        {"task": "t", "output_dir": "/x"}), {"robodojo_start": robodojo_start})
    assert out == {"started": True} and calls == ["t"]
    # 未知工具拒绝（语义同上游：native 工具归 app-server）
    with pytest.raises(ValueError, match="Unknown host service"):
        dispatch_tool_call("shell", {}, {})
    # 非法 JSON 参数
    with pytest.raises(ValueError, match="JSON object"):
        dispatch_tool_call("robodojo_start", "{oops", {"robodojo_start": robodojo_start})


def test_input_validation_gates(review_p2_fixes=True):
    """审查 A-P2-1/2：NaN/非单位四元数/bool steps/gripper_closed 类型必须被拒。"""
    from gap_repro.agent import validate_target
    cur = {"position": [0.5, 0.0, 0.2], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}
    # NaN 位置拒绝
    with pytest.raises(contract.ContractError, match="non-finite"):
        validate_eef_target({"position": [float("nan"), 0, 0],
                             "quaternion_wxyz": [1, 0, 0, 0], "gripper_closed": True}, cur)
    # 非单位四元数拒绝（|‖q‖−1|>1e-4）
    with pytest.raises(contract.ContractError, match="unit quaternion"):
        validate_eef_target({"position": [0.5, 0, 0.2],
                             "quaternion_wxyz": [2.0, 0, 0, 0], "gripper_closed": True}, cur)
    # gripper_closed 非 bool 拒绝
    with pytest.raises(contract.ContractError, match="boolean"):
        validate_target({"position": [0.5, 0, 0.2],
                         "quaternion_wxyz": [1, 0, 0, 0], "gripper_closed": 1})
    # 缺键拒绝
    with pytest.raises(contract.ContractError, match="missing"):
        validate_target({"position": [0.5, 0, 0.2]})
    # apply_edit 拒绝 inf
    pose = {"position": [0.1, 0.2, 0.3], "quaternion_wxyz": [1.0, 0, 0, 0]}
    with pytest.raises(contract.ContractError, match="non-finite"):
        apply_edit(pose, {"delta_position": [float("inf"), 0, 0],
                          "delta_rotation_vector": [0.0, 0.0, 0.0], "gripper": "keep"})
    # bool steps 拒绝（True 是 int 子类）
    with pytest.raises(contract.ContractError):
        validate_mode_steps("student", True)
