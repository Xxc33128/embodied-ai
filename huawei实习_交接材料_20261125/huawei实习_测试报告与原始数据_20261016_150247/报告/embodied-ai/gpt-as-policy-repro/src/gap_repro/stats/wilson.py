"""L5：精确 Wilson 区间与最小样本量整数搜索（审计 F11 修正）。

审计结论：LP5 提案 v1 用 ceil(z²p(1-p)/w²)（Wald 正态近似）定每层样本量，
在极低成功率（LIBERO Task/Env 条件 p≈0.01）严重低估所需 N——Wald 半宽
与 Wilson 半宽在边界区发散。本模块实现精确 Wilson：
- wilson_interval(k, n)：标准 Wilson score interval，z 钉定双侧 95%。
- min_n_for_halfwidth：给定期望成功率 p 与目标半宽 w，整数搜索最小 N
  （criterion="at-p"：观测 p̂=p 处半宽达标，论文预期口径；
  "worst-case"：任意 p̂ 达标，检查 p̂=0.5 点，最保守）。

参照值（测试钉定，z=1.959963984540054）：
- k=0, n=20 → Wilson 上界 ≈ 0.1611（审计 F11 例）
- k=0, n=16 → Wilson 上界 ≈ 0.1936（p≈0.01 需求侧对照）
"""
import math

Z_95 = 1.959963984540054  # 双侧 95% 精确 z（norm.ppf(0.975)）


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple:
    """Wilson score interval。"""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= k <= n:
        raise ValueError(f"k out of range: {k}/{n}")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_half_width(k: int, n: int, z: float = Z_95) -> float:
    lo, hi = wilson_interval(k, n, z)
    p = k / n
    return max(p - lo, hi - p)


def min_n_for_halfwidth(p: float, w: float, *, z: float = Z_95,
                        criterion: str = "at-p", n_max: int = 100_000):
    """整数搜索最小 N 使 Wilson 半宽 ≤ w（返回 n/half_width/搜索尾部）。"""
    if not 0 <= p <= 1:
        raise ValueError("p out of range")
    if w <= 0:
        raise ValueError("w must be positive")
    if criterion not in ("at-p", "worst-case"):
        raise ValueError(f"unknown criterion {criterion!r}")
    trace = []
    for n in range(1, n_max + 1):
        if criterion == "worst-case":
            # P2-1 修正：Wilson 半宽（非对称 max(p−lo, hi−p)）全域最大在
            # p̂≈0.40–0.45 而非 0.5——必须全 k 扫描，只查 n//2 会低估 N
            hw = max(wilson_half_width(k, n, z) for k in range(n + 1))
        else:
            k = max(0, min(n, round(p * n)))
            hw = wilson_half_width(k, n, z)
        trace.append((n, round(hw, 6)))
        if hw <= w:
            return {"n": n, "half_width": hw, "search": trace[-10:]}
    raise RuntimeError(f"no N ≤ {n_max} achieves half-width {w}")
