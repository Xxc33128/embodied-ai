"""W1 输入身份核查的契约测试（离线，全部使用临时 fixture）。

对应计划 W1 通过条件的四类反例：缺文件、内容变动、LFS 指针误当资产、错误 revision。
"""

import hashlib

import pytest

from gap_repro import inputs
from gap_repro.acceptance import decide_w1_ok
from gap_repro.inputs import (
    LfsPointerError,
    RevisionMismatchError,
    aggregate_identity,
    is_lfs_pointer,
    read_asset_bytes,
    verify_native_sources,
    verify_revision,
)


def _write(tmp_path, rel, content: bytes):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class TestNativeSourceVerification:
    def test_all_present_and_matching_passes(self, tmp_path):
        _write(tmp_path, "a.py", b"alpha")
        _write(tmp_path, "sub/b.py", b"beta")
        entries = [
            {"path": "a.py", "sha256": _sha(b"alpha")},
            {"path": "sub/b.py", "sha256": _sha(b"beta")},
        ]
        out = verify_native_sources(tmp_path, entries)
        assert out["passed"] == 2 and out["failed"] == 0

    def test_missing_file_is_recorded_not_skipped(self, tmp_path):
        _write(tmp_path, "a.py", b"alpha")
        entries = [{"path": "a.py", "sha256": _sha(b"alpha")},
                   {"path": "ghost.py", "sha256": _sha(b"x")}]
        out = verify_native_sources(tmp_path, entries)
        assert out["failed"] == 1
        assert out["results"][1]["status"] == "missing"
        assert out["results"][1]["match"] is False

    def test_content_drift_detected(self, tmp_path):
        _write(tmp_path, "a.py", b"alpha tampered")
        entries = [{"path": "a.py", "sha256": _sha(b"alpha")}]
        out = verify_native_sources(tmp_path, entries)
        assert out["failed"] == 1
        assert out["results"][0]["status"] == "ok" and out["results"][0]["match"] is False


class TestLfsPointerGuard:
    def test_pointer_bytes_recognized(self):
        ptr = b"version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\n"
        assert is_lfs_pointer(ptr) is True
        assert is_lfs_pointer(b"{\"json\": 1}") is False

    def test_pointer_file_rejected_as_asset(self, tmp_path):
        p = _write(tmp_path, "model.bin",
                   b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n")
        with pytest.raises(LfsPointerError):
            read_asset_bytes(p)

    def test_real_file_passes_guard(self, tmp_path):
        p = _write(tmp_path, "norm_stats.json", b"{\"mean\": 0.0}")
        assert read_asset_bytes(p) == b"{\"mean\": 0.0}"


class TestRevisionPin:
    def test_equal_revision_passes(self):
        verify_revision("ee67a146", "ee67a146")

    def test_wrong_revision_raises(self):
        with pytest.raises(RevisionMismatchError):
            verify_revision("91f76c28", "ee67a146")


class TestAggregateIdentity:
    def test_upstream_recipe_golden_value(self):
        # 上游配方：{relpath: sha256} 映射、json.dumps(sort_keys=True)（默认分隔符）的 sha256。
        # 金标值 = sha256('{"a.txt": "1111", "b.txt": "2222"}')，与实现独立预先算出。
        got = aggregate_identity([("b.txt", "2222"), ("a.txt", "1111")])
        assert got == "3fdf0ef34bcbf38bdc2617872773dc7fb5365664a14e7122b7d25c92f9867191"

    def test_input_order_does_not_matter(self):
        a = aggregate_identity([("x", "01"), ("a", "02")])
        b = aggregate_identity([("a", "02"), ("x", "01")])
        assert a == b

    def test_author_checkpoint_aggregate_reproduced_from_metadata(self):
        # 回归锚：用上轮记录的 17 文件元数据，必须精确复现作者侧
        # previous_load_sha256 = d15fb8bd…；此测试失败即配方被改动。
        pairs = [
            ("params/_METADATA", "f7b1dbf26ce9466c9d444c3a7f60dfe21767b3abc3380a455e841972eae4f274"),
            ("params/_sharding", "77b1fa6b2c8d2a16a94d19eb268bed916c6913565421fdf6312453ac6c4fc621"),
            ("params/array_metadatas/process_0", "2b650291cb689bb0504470bf03a071c40ae96e203667b89cead0c6adb2d6288f"),
            ("params/d/e9268d0868cc79dc08071adee3ba67f5", "9482acdbb1250cf3344fad3c8be7f587ba3770441198e7d60f357702486ac2c8"),
            ("params/manifest.ocdbt", "634fea47615fa13f4c0852dc13b05d897ee1694a473d9d0605d02b834fd2befe"),
            ("params/ocdbt.process_0/d/0a05b3b4870f64154af3641ef1c1eecf", "429161d8ab2ac9d13b72095d8b5be5ffba8f99302b12bb2cae149d1ee0a148b0"),
            ("params/ocdbt.process_0/d/1fbec215d6341f33f041c6145ee3756f", "cdf1415e6f30a9827e52e6d5442b5f819dadbaba7c8e941773290cfc2742f249"),
            ("params/ocdbt.process_0/d/3749fcdb2abf24595f8ca06ee208e773", "23ddd7f53b976bd27b1a62a83ad1499925a8a2e90a5b0545affd63c7bbd52bbc"),
            ("params/ocdbt.process_0/d/42eb48ef345d0e30b95bc9e2a194adc8", "41eb65bde7e3d16f63ac71ac80bc4547bdcf3107c41af70ac5174992bd38b709"),
            ("params/ocdbt.process_0/d/5dd44e9e6c20354d58016a470210b404", "d096237d5373dc120004174e89dc44a341fdffdcc7457d568d0b841700ff2ade"),
            ("params/ocdbt.process_0/d/a66e531837010ca0fb32ecd927335e53", "522d9057a70c021327d3c76132377aa0e6c323feb4dea4db0b8ccb16fc7980f4"),
            ("params/ocdbt.process_0/d/ce1637cf430319dc033526f7e32d833e", "93e3eb49cbbcae7b3e424c4e19a11262ee118355a4676a62681ec2e26fccbc16"),
            ("params/ocdbt.process_0/d/d7a5b27d8a8c4f63cc9dfe9d12d5e2d3", "437c63711361d678a44b06455d0124a50ec79d9ee0bafe2ada78e41d79360fa6"),
            ("params/ocdbt.process_0/d/d931aec2d1d586717a28915e6c710676", "1a8834dd2a575ce5f8bfa7be2fbd554cca261cdf997d928c5a719e77174b1d83"),
            ("params/ocdbt.process_0/d/fbf78c71d75f5d3cb7ef11cd9cde9ab0", "9c268e5dfc2248b4c75ae5ccbf2f23a196b2cd55ee798707c0bef5d4b3f6b653"),
            ("params/ocdbt.process_0/manifest.ocdbt", "b2117af3ce683fba15add35bdd5596df488e816ec4c5bcef1a82161a965115bd"),
            ("assets/arx_x5_sim/norm_stats.json", "ad7dea3e3d2bcdb348945fe03422ab1adccd03baf67318b1a1d153dfe8694db5"),
        ]
        assert aggregate_identity(pairs) == (
            "d15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b")


class TestSelectedCases:
    def test_scope_case_must_exist_in_panel(self):
        panel = {"cases": [{"case_id": "c1", "layout": {"path": "p", "sha256": "s"}}]}
        scope = {"cases": [{"case_id": "c2", "layout_sha256": "s"}]}
        with pytest.raises(KeyError):
            inputs.selected_cases(panel, scope)

    def test_scope_layout_hash_attached(self):
        panel = {"cases": [{"case_id": "c1", "layout": {"path": "L", "sha256": "AA"}},
                            {"case_id": "c2", "layout": {"path": "L2", "sha256": "BB"}}]}
        scope = {"cases": [{"case_id": "c2", "layout_sha256": "BB"}]}
        rows = inputs.selected_cases(panel, scope)
        assert len(rows) == 1 and rows[0]["scope_layout_sha256"] == "BB"


class TestDecideW1Ok:
    """端到端验收判定反例：错误版本、缺失 case、哈希不匹配都必须拦下。"""

    GOOD = dict(
        pins_ok={"a": True, "b": True, "c": True},
        submodule_pin="432f82b",
        expected_submodule_pin="432f82b",
        native_sources={"total": 169, "failed": 0},
        expected_source_total=169,
        layouts={"total": 50, "passed": 50, "prior_regression_consistent": True},
        expected_layout_total=50,
        trajectories={"total": 25, "passed": 25},
        checkpoint={"all_match_prior": True, "params_bytes_match": True},
        aggregate_reproduced=True,
    )

    def test_all_good_passes(self):
        ok, reasons = decide_w1_ok(**self.GOOD)
        assert ok is True and reasons == []

    def test_all_false_pins_fail_not_truthy_dict(self):
        # 回归：此前 `pins_ok and ...` 对非空 dict 恒真，全部 False 也判过。
        bad = dict(self.GOOD, pins_ok={"a": False, "b": False, "c": False})
        ok, reasons = decide_w1_ok(**bad)
        assert ok is False
        assert any("clone HEAD" in r for r in reasons)

    def test_wrong_submodule_pin_fails(self):
        ok, reasons = decide_w1_ok(**dict(self.GOOD, submodule_pin="deadbeef"))
        assert ok is False and any("submodule" in r for r in reasons)

    def test_source_count_shortfall_fails(self):
        bad = dict(self.GOOD, native_sources={"total": 75, "failed": 0})
        ok, reasons = decide_w1_ok(**bad)
        assert ok is False and any("169" in r for r in reasons)

    def test_missing_layout_case_fails(self):
        bad = dict(self.GOOD, layouts={"total": 49, "passed": 49,
                                       "prior_regression_consistent": True})
        ok, reasons = decide_w1_ok(**bad)
        assert ok is False and any("49" in r for r in reasons)

    def test_partial_layout_hash_failure_fails(self):
        bad = dict(self.GOOD, layouts={"total": 50, "passed": 49,
                                       "prior_regression_consistent": True})
        ok, reasons = decide_w1_ok(**bad)
        assert ok is False and any("passed=49" in r for r in reasons)

    def test_checkpoint_metadata_mismatch_fails(self):
        bad = dict(self.GOOD, checkpoint={"all_match_prior": False,
                                          "params_bytes_match": True})
        ok, reasons = decide_w1_ok(**bad)
        assert ok is False and any("checkpoint" in r for r in reasons)

    def test_aggregate_recipe_drift_fails(self):
        ok, reasons = decide_w1_ok(**dict(self.GOOD, aggregate_reproduced=False))
        assert ok is False and any("previous_load_sha256" in r for r in reasons)

    def test_trajectory_gap_fails(self):
        bad = dict(self.GOOD, trajectories={"total": 25, "passed": 24})
        ok, reasons = decide_w1_ok(**bad)
        assert ok is False and any("24/25" in r for r in reasons)

    def test_empty_pins_fail(self):
        ok, reasons = decide_w1_ok(**dict(self.GOOD, pins_ok={}))
        assert ok is False and any("空" in r for r in reasons)
