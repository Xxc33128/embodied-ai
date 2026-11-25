"""W1 验收判定：把 runner 的通过条件收敛为可独立测试的纯函数。

背景（复核发现）：此前 runner 用 `pins_ok and ...` 判定，非空 dict 恒为真，
无法拦截"所有 pin 均不匹配"的失败。所有判定必须走本模块并对反例有测试。
"""

from __future__ import annotations


def decide_w1_ok(
    *,
    pins_ok: dict[str, bool],
    submodule_pin: str | None,
    expected_submodule_pin: str,
    native_sources: dict,
    expected_source_total: int,
    layouts: dict,
    expected_layout_total: int,
    trajectories: dict,
    checkpoint: dict,
    aggregate_reproduced: bool,
) -> tuple[bool, list[str]]:
    """返回 (是否通过, 失败原因列表)。任何一项不满足都给出具体原因。"""
    reasons: list[str] = []

    bad_pins = [k for k, v in pins_ok.items() if v is not True]
    if not pins_ok:
        reasons.append("pins_ok 为空：未执行任何 clone HEAD 校验")
    elif bad_pins:
        reasons.append(f"clone HEAD 与固定 commit 不符: {bad_pins}")

    if submodule_pin != expected_submodule_pin:
        reasons.append(
            f"RoboDojo 的 XPolicyLab submodule 指针 {submodule_pin!r} != 期望 {expected_submodule_pin!r}")

    if native_sources.get("total") != expected_source_total:
        reasons.append(
            f"源码清单数量 {native_sources.get('total')} != panel 记录 {expected_source_total}")
    if native_sources.get("failed", 1) != 0:
        reasons.append(f"源码哈希失败 {native_sources.get('failed')} 个")

    if layouts.get("total") != expected_layout_total:
        reasons.append(f"布局数量 {layouts.get('total')} != 所选 case 数 {expected_layout_total}")
    if layouts.get("passed") != layouts.get("total"):
        reasons.append(f"布局哈希未全过: passed={layouts.get('passed')}/{layouts.get('total')}")
    if layouts.get("prior_regression_consistent") is not True:
        reasons.append("布局与上轮记录回比不一致")

    if trajectories.get("passed") != trajectories.get("total"):
        reasons.append(
            f"轨迹引用未全过: passed={trajectories.get('passed')}/{trajectories.get('total')}")
    if trajectories.get("total", 0) <= 0:
        reasons.append("轨迹引用为 0：核查未覆盖支持演示")

    if checkpoint.get("all_match_prior") is not True:
        reasons.append("checkpoint 元数据与上轮记录存在不匹配")
    if checkpoint.get("params_bytes_match") is not True:
        reasons.append("params 16 文件字节数与原记录不一致")
    if aggregate_reproduced is not True:
        reasons.append("上游聚合配方未能复现作者侧 previous_load_sha256")

    return (len(reasons) == 0, reasons)
