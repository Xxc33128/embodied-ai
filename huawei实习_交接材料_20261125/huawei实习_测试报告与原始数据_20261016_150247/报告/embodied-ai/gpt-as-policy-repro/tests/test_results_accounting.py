"""W10 结果核算测试：§8 分母/配对/缺失/attempt 语义的构造 fixture。"""

from __future__ import annotations

import pytest

from gap_repro.results import (
    BUDGET_CENSORED,
    INFRA_INCOMPLETE,
    INVALID_LAYOUT,
    VALID_FAILURE,
    VALID_SUCCESS,
    EpisodeResult,
    freeze_campaign,
    pair_routes,
    select_attempt,
    summarize_route,
)


def ep(case, status, score="auto", method="direct", attempt=0, entered=True):
    if score == "auto":
        score = (1.0 if status == VALID_SUCCESS
                 else 0.0 if status == VALID_FAILURE else None)
    return EpisodeResult(campaign_id="c1", case_id=case, method=method,
                         route="app_server", status=status, native_score=score,
                         attempt_id=attempt, entered_policy_stage=entered)


class TestDenominators:
    def test_all_resolved_headline_available(self):
        results = [ep(f"c{i}", VALID_SUCCESS if i % 2 else VALID_FAILURE,
                      score=0.5 + 0.01 * i) for i in range(50)]
        s = summarize_route(results, expected_case_ids=[f"c{i}" for i in range(50)])
        assert s["headline_success_rate"] == 0.5
        assert s["resolved"] == 50 and s["unresolved"] == 0
        assert s["score_mean_full_panel"] is not None

    def test_invalid_layout_excluded_not_failure(self):
        results = [ep("c0", INVALID_LAYOUT)] + [ep(f"c{i}", VALID_FAILURE)
                                                for i in range(1, 50)]
        s = summarize_route(results, expected_case_ids=[f"c{i}" for i in range(50)])
        assert s["success_count"] == 0  # invalid ≠ failure，不计失败
        assert s["resolved_only_rate"] == 0.0  # 已解决 49 中全败
        assert s["unresolved_detail"]["c0"] == INVALID_LAYOUT

    def test_budget_censored_no_zero_fill(self):
        results = [ep("c0", VALID_SUCCESS, score=1.0),
                   ep("c1", BUDGET_CENSORED),
                   ep("c2", INFRA_INCOMPLETE)]
        s = summarize_route(results, expected_case_ids=["c0", "c1", "c2"])
        assert s["success_count"] == 1
        assert s["score_mean_full_panel"] is None  # 缺分不补零
        assert s["score_available_mean"] == 1.0 and s["n_scored"] == 1

    def test_bounds_bracket(self):
        results = [ep("c0", VALID_SUCCESS, score=1.0),
                   ep("c1", BUDGET_CENSORED),
                   ep("c2", INFRA_INCOMPLETE)]
        s = summarize_route(results, expected_case_ids=["c0", "c1", "c2"])
        lo, hi = s["bounds"]
        assert lo == pytest.approx(1 / 3)
        assert hi == pytest.approx(1.0)  # [S/N, (S+U)/N]


class TestPairing:
    def _routes(self):
        a = summarize_route([
            ep("x0", VALID_SUCCESS, score=1.0),
            ep("x1", VALID_FAILURE, score=0.0),
            ep("x2", VALID_SUCCESS, score=0.7),
            ep("x3", BUDGET_CENSORED),
        ], expected_case_ids=["x0", "x1", "x2", "x3"])
        b = summarize_route([
            ep("x0", VALID_FAILURE, score=0.0, method="hybrid"),
            ep("x1", VALID_FAILURE, score=0.1, method="hybrid"),
            ep("x2", BUDGET_CENSORED, method="hybrid"),
            ep("x3", VALID_SUCCESS, score=1.0, method="hybrid"),
        ], expected_case_ids=["x0", "x1", "x2", "x3"])
        return a, b

    def test_pairing_only_both_resolved(self):
        a, b = self._routes()
        p = pair_routes(a, b)
        assert p["paired_count"] == 2  # x2/x3 单边未解决不配对
        assert p["missing_from_pairing"]["a_resolved_b_missing"] == ["x2"]
        assert p["missing_from_pairing"]["b_resolved_a_missing"] == ["x3"]

    def test_pairing_direction_counts(self):
        a, b = self._routes()
        p = pair_routes(a, b)
        assert p["both_fail"] == 1  # x1 双败
        assert p["a_only"] == 1     # x0 a 成 b 败


class TestAttempts:
    def test_select_first_entered_policy_stage(self):
        attempts = [ep("c", INFRA_INCOMPLETE, attempt=0, entered=False),
                    ep("c", VALID_FAILURE, attempt=1)]
        assert select_attempt(attempts).attempt_id == 1

    def test_all_attempts_kept_not_success_picked(self):
        attempts = [ep("c", VALID_FAILURE, attempt=0, entered=True),
                    ep("c", VALID_SUCCESS, score=1.0, attempt=1, entered=True)]
        # §8.1.3：进入策略阶段后不得靠重跑挑好结果 → 首个进入者即最终
        assert select_attempt(attempts).attempt_id == 0


class TestValidation:
    def test_success_requires_score(self):
        with pytest.raises(ValueError):
            ep("c", VALID_SUCCESS, score=None)

    def test_unknown_status_rejected(self):
        with pytest.raises(ValueError):
            ep("c", "awesome")


def test_freeze_campaign_detects_input_change(tmp_path):
    f = tmp_path / "input.bin"
    f.write_bytes(b"v1")
    manifest = freeze_campaign({"campaign_id": "c1"}, {"data": str(f)})
    assert manifest["frozen_inputs"]["data"]["sha256"] == pytest.approx(
        __import__("hashlib").sha256(b"v1").hexdigest())
    f.write_bytes(b"v2")
    manifest2 = freeze_campaign({"campaign_id": "c1"}, {"data": str(f)})
    assert (manifest2["frozen_inputs"]["data"]["sha256"]
            != manifest["frozen_inputs"]["data"]["sha256"])
    assert manifest["frozen"] is True


def test_verify_frozen_campaign_detects_tamper(tmp_path):
    from gap_repro.results import verify_frozen_campaign
    f = tmp_path / "in.bin"
    f.write_bytes(b"data")
    manifest = freeze_campaign({"campaign_id": "c", "panel": 50}, {"data": str(f)})
    assert verify_frozen_campaign(manifest)["ok"] is True
    tampered = dict(manifest, panel=999)  # 篡改 campaign 内容
    assert verify_frozen_campaign(tampered)["ok"] is False


class TestFrozenPanelAndSensitivity:
    """T1（audit F01/F02）：冻结 case 分母、缺失 case、infra 敏感性、指纹配对。"""

    def test_infrastructure_is_not_a_success(self):
        rows = [ep("a", VALID_SUCCESS), ep("b", INFRA_INCOMPLETE)]
        s = summarize_route(rows, expected_case_ids=["a", "b"])
        assert s["bounds"] == [0.5, 1.0]
        assert s["sensitivity_infra_as_failure"] == 0.5  # infra 按失败计：1/2

    def test_missing_case_counts_as_unresolved(self):
        rows = [ep("a", VALID_SUCCESS)]
        s = summarize_route(rows, expected_case_ids=["a", "b"])
        assert s["missing_cases"] == ["b"]
        assert s["resolved"] == 1 and s["unresolved"] == 1
        assert s["resolved_only_rate"] == 1.0
        assert s["bounds"] == [0.5, 1.0]
        assert s["headline_success_rate"] is None

    def test_duplicate_expected_cases_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            summarize_route([], expected_case_ids=["a", "a"])

    def test_out_of_panel_case_rejected(self):
        with pytest.raises(ValueError, match="outside frozen panel"):
            summarize_route([ep("x", VALID_SUCCESS, score=1.0)],
                            expected_case_ids=["a"])

    def test_mixed_campaign_rejected(self):
        rows = [ep("a", VALID_SUCCESS),
                ep("a", VALID_SUCCESS, score=1.0)]
        rows[1].campaign_id = "other"
        with pytest.raises(ValueError, match="campaign"):
            summarize_route(rows, expected_case_ids=["a"])

    def test_sensitivity_partial_scope_labeled(self):
        # invalid 未解决 → infra 敏感性只能给部分口径并显式标注
        rows = [ep("a", VALID_SUCCESS), ep("b", INFRA_INCOMPLETE),
                ep("c", INVALID_LAYOUT)]
        s = summarize_route(rows, expected_case_ids=["a", "b", "c"])
        det = s["sensitivity_detail"]
        assert det["scope"] == "partial"
        assert det["numerator"] == 1 and det["denominator"] == 2
        assert det["infra_cases"] == ["b"]
        assert {"case": "c", "status": INVALID_LAYOUT} in [
            {"case": k, "status": v} for k, v in det["remaining_unresolved"].items()]
        assert s["sensitivity_infra_as_failure"] is None  # 部分口径不给单一数

    def test_sensitivity_full_scope_value(self):
        rows = [ep("a", VALID_SUCCESS), ep("b", VALID_FAILURE, score=0.0),
                ep("c", INFRA_INCOMPLETE)]
        s = summarize_route(rows, expected_case_ids=["a", "b", "c"])
        assert s["sensitivity_infra_as_failure"] == pytest.approx(1 / 3)
        assert s["sensitivity_detail"]["scope"] == "full"
        assert s["sensitivity_detail"]["numerator"] == 1
        assert s["sensitivity_detail"]["denominator"] == 3

    def test_attempts_ordered_by_attempt_id_not_log_order(self):
        rows = [ep("a", VALID_SUCCESS, score=1.0, attempt=1),
                ep("a", VALID_FAILURE, score=0.0, attempt=0)]
        s = summarize_route(rows, expected_case_ids=["a"])
        # §8.1.3 首个进入策略阶段的 attempt（按 attempt_id 序），不挑好结果
        assert s["per_case"]["a"]["status"] == VALID_FAILURE
        assert s["per_case"]["a"]["attempt_id"] == 0

    def test_conflicting_duplicate_attempt_rejected(self):
        rows = [ep("a", VALID_SUCCESS, score=1.0, attempt=0),
                ep("a", VALID_FAILURE, score=0.0, attempt=0)]
        with pytest.raises(ValueError, match="attempt_id 0"):
            summarize_route(rows, expected_case_ids=["a"])

    def test_identical_duplicate_attempt_deduped(self):
        rows = [ep("a", VALID_SUCCESS, score=1.0, attempt=0),
                ep("a", VALID_SUCCESS, score=1.0, attempt=0)]
        s = summarize_route(rows, expected_case_ids=["a"])
        assert s["attempt_counts"]["a"] == 1

    def test_per_case_keeps_fingerprint(self):
        rows = [ep("a", VALID_SUCCESS)]
        rows[0].init_fingerprint = "fp-a"
        s = summarize_route(rows, expected_case_ids=["a"])
        assert s["per_case"]["a"]["init_fingerprint"] == "fp-a"

    def test_pair_routes_reports_fingerprint_mismatch(self):
        a = summarize_route([ep("x0", VALID_SUCCESS)], expected_case_ids=["x0"])
        a["per_case"]["x0"]["init_fingerprint"] = "fpA"
        b = summarize_route([ep("x0", VALID_FAILURE, score=0.0, method="hybrid")],
                            expected_case_ids=["x0"])
        b["per_case"]["x0"]["init_fingerprint"] = "fpB"
        p = pair_routes(a, b)
        assert p["fingerprint_mismatch_pairs"] == [
            {"case_id": "x0", "a": "fpA", "b": "fpB"}]
        assert p["paired_fingerprint_strict_count"] == 0
        assert p["paired_count"] == 1  # 仍报告全部已解决配对，但差异显式
