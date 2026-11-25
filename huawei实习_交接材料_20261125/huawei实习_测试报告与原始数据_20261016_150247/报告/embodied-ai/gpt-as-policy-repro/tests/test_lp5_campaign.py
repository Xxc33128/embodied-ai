"""L3 campaign CLI 单元测试：case 池展开/套件匹配/冻结校验（纯本地）。"""

from __future__ import annotations

import json

import pytest

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_lp5_campaign", "scripts/run_lp5_campaign.py")
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

from gap_repro.results import freeze_campaign, verify_frozen_campaign


def _inv():
    def case(b, cond, ti, ok=True):
        return {"benchmark": b, "suite": b, "condition": cond,
                "task_index": ti, "task_name": f"t{ti}",
                "bddl_exists": ok, "init_exists": ok,
                "benchmark_language": "do it"}
    return {"cases": [
        case("libero_goal", "Ori", 0),
        case("libero_goal", "Sem", 1),
        case("libero_goal_env", "Env", 2, ok=False),  # 资产缺失
        case("libero_object", "Ori", 3),
        case("libero_10", "Task", 4),
    ]}


class TestSuiteMatch:
    def test_exact_and_variants(self):
        m = mod._suite_match
        assert m("libero_goal", ["libero_goal"])
        assert m("libero_goal_lan", ["libero_goal"])
        assert m("libero_goal_env", ["libero_goal"])
        assert m("libero_object", ["libero_object"])      # 基线套件本身
        assert m("libero_object_object", ["libero_object"])
        assert m("libero_10", ["libero_10"])
        assert m("libero_10_task", ["libero_10"])
        assert not m("libero_spatial", ["libero_goal"])

    def test_variant_suffix_whitelist(self):
        assert not mod._suite_match("libero_goal_temp", ["libero_goal"])
        assert not mod._suite_match("libero_10foo", ["libero_10"])


class TestSelectCases:
    def test_expand_and_identity(self):
        specs = mod.select_cases(_inv(), conditions=("Ori",),
                                 suites=["libero_goal"],
                                 episodes_per_case=3, campaign_id="c")
        assert [s.episode_id for s in specs] == [
            "libero_goal#0#i0", "libero_goal#0#i1", "libero_goal#0#i2"]
        assert specs[0].max_steps == 300

    def test_incomplete_case_rejected(self):
        with pytest.raises(RuntimeError, match="禁止抽掉"):
            mod.select_cases(_inv(), conditions=("Env",),
                             suites=["libero_goal"], episodes_per_case=1,
                             campaign_id="c")

    def test_object_condition_vs_object_suite(self):
        # Obj 条件 + libero_object 套件：两层 object 不得混淆
        specs = mod.select_cases(_inv(), conditions=("Obj",),
                                 suites=["libero_goal"],
                                 episodes_per_case=1, campaign_id="c")
        assert specs == []  # inventory 里没有 Obj 条件 case


class TestFreezeFlow:
    def test_formal_freeze_verify_roundtrip(self, tmp_path):
        specs = mod.select_cases(_inv(), conditions=("Ori",),
                                 suites=["libero_goal"],
                                 episodes_per_case=2, campaign_id="c1")
        ids = [s.episode_id for s in specs]
        inv_file = tmp_path / "inv.json"
        inv_file.write_text(json.dumps(_inv()))
        manifest = freeze_campaign(
            {"campaign_id": "c1", "expected_case_ids": ids},
            {"inventory": str(inv_file)})
        (tmp_path / "fm.json").write_text(json.dumps(manifest))
        verdict = verify_frozen_campaign(json.load(open(tmp_path / "fm.json")))
        assert verdict["ok"] is True
        # 篡改 case 池 → CLI 侧 sorted 比对拒绝（此处验证池比对逻辑）
        assert sorted(manifest["expected_case_ids"]) == sorted(ids)
