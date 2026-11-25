"""L5 Wilson 精确搜索测试（审计 F11 修正的数值钉定）。"""

from __future__ import annotations

import pytest

from gap_repro.stats.wilson import (
    Z_95,
    min_n_for_halfwidth,
    wilson_half_width,
    wilson_interval,
)


class TestWilsonInterval:
    def test_audit_f11_reference_k0_n20(self):
        lo, hi = wilson_interval(0, 20)
        assert hi == pytest.approx(0.1611, abs=5e-4)  # 审计 F11 例
        assert lo == 0.0

    def test_monotone_in_n(self):
        # 同 p̂ 下 N 越大半宽越窄
        assert wilson_half_width(5, 100) > wilson_half_width(50, 1000)

    def test_known_value_p05_n100(self):
        lo, hi = wilson_interval(5, 100)
        # 标准实现对照（z=1.96）：[0.0216, 0.1116] 邻域
        assert lo == pytest.approx(0.0217, abs=2e-3)
        assert hi == pytest.approx(0.1116, abs=2e-3)

    def test_validation(self):
        with pytest.raises(ValueError):
            wilson_interval(1, 0)
        with pytest.raises(ValueError):
            wilson_interval(11, 10)


class TestMinN:
    def test_low_success_rate_needs_more_than_wald(self):
        # 审计 F11 实质：p=0.01、w=0.05 时 Wald 大幅低估。
        # Wald: ceil(z²·0.01·0.99/0.05²) = ceil(15.37) = 16 → 半宽远超 0.05
        wald_n = 16
        assert wilson_half_width(0, wald_n) > 0.05
        result = min_n_for_halfwidth(0.01, 0.05, criterion="at-p")
        assert result["n"] > wald_n  # 精确 Wilson 要求更多
        assert result["half_width"] <= 0.05

    def test_worst_case_is_conservative(self):
        at_p = min_n_for_halfwidth(0.01, 0.05, criterion="at-p")["n"]
        worst = min_n_for_halfwidth(0.01, 0.05, criterion="worst-case")["n"]
        assert worst > at_p  # 全域达标（p̂=0.5）需要更多
        # Wald 近似给经典值 385（z²·0.25/w²=384.16→385）。
        # P2-1 修正：worst-case 必须全 k 扫描——半宽最大在 p̂≈0.40–0.45，
        # 只查 p̂=0.5 得 381 是假的保守（该 N 下 max_k hw=0.05021>0.05）。
        assert worst == 385
        assert max(wilson_half_width(k, worst) for k in range(worst + 1)) <= 0.05

    def test_trace_tail_kept(self):
        r = min_n_for_halfwidth(0.5, 0.05)
        assert r["search"][-1][0] == r["n"]
