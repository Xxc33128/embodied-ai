"""W11 场景资产查找测试：嵌套/扁平布局命中、缺失 fail-closed（审查 Y 系结论：
场景组装静默跳过缺失物体会造成"验证通过但物体全缺"的假阳性）。

另含 T4/F05：USD→OBJ 转换的变换正确性——以 pxr 的 Gf.Transform 为独立
oracle，经真实 convert_usdz_to_obj 产物验证（不是只测等价数学函数）。
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from gap_repro.sim.scene import find_obj_mesh

_CONVERTER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "w11_convert_scene_objects.py"


def _load_converter():
    pytest.importorskip("pxr")
    pytest.importorskip("trimesh")
    spec = importlib.util.spec_from_file_location("w11_convert_scene_objects",
                                                  _CONVERTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_usd(path: Path, transforms, points):
    """transforms: 由外到内的 Xform 列表，每项 {translate, rotate_z, scale}。"""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    parent = "/World"
    UsdGeom.Xform.Define(stage, parent)
    for i, ops in enumerate(transforms):
        node = f"{parent}/X{i}"
        x = UsdGeom.Xform.Define(stage, node)
        x.AddTranslateOp().Set(Gf.Vec3d(*ops.get("translate", (0.0, 0.0, 0.0))))
        if "rotate_z" in ops:
            x.AddRotateZOp().Set(float(ops["rotate_z"]))
        if "scale" in ops:
            x.AddScaleOp().Set(Gf.Vec3f(*ops["scale"]))
        parent = node
    mesh = UsdGeom.Mesh.Define(stage, f"{parent}/Mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    stage.Save()
    return stage


def _gf_world_vertices(usd_path: Path, points):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)][0]
    M = xf.GetLocalToWorldTransform(prim)
    return np.array([list(M.Transform(Gf.Vec3d(*p))) for p in points])


def _converted_vertices(converter, usd_path: Path, out_obj: Path, points):
    info = converter.convert_usdz_to_obj(usd_path, out_obj)
    assert info["n_verts"] > 0
    tm = __import__("trimesh").load(str(out_obj))
    return np.asarray(tm.vertices, dtype=np.float64), info


def _sorted_rows(a):
    return a[np.lexsort(a.T[::-1])]


TRI = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


def test_gf_transform_row_vector_oracle():
    """F05 oracle 本身：Gf 行向量语义 p @ M 与 M.Transform 一致（防约定漂移）。"""
    from pxr import Gf

    m = Gf.Matrix4d(1)
    m.SetTranslate(Gf.Vec3d(1, 2, 3))
    p = np.array([[0.0, 0.0, 0.0, 1.0]])
    expected = np.asarray(m.Transform(Gf.Vec3d(0, 0, 0)))
    assert np.allclose((p @ np.asarray(m))[0, :3], expected)


def test_converter_parent_translation(tmp_path):
    converter = _load_converter()
    usd = tmp_path / "t.usda"
    _make_usd(usd, [{"translate": (1.0, 2.0, 3.0)}], TRI)
    verts, _ = _converted_vertices(converter, usd, tmp_path / "t.obj", TRI)
    assert np.allclose(verts.min(axis=0), [1.0, 2.0, 3.0], atol=1e-6)
    assert np.allclose(verts.max(axis=0), [2.0, 3.0, 3.0], atol=1e-6)
    assert np.allclose(_sorted_rows(verts),
                       _sorted_rows(_gf_world_vertices(usd, TRI)), atol=1e-5)


def test_converter_rotation_and_translation(tmp_path):
    converter = _load_converter()
    usd = tmp_path / "r.usda"
    ops = {"translate": (0.5, -1.0, 2.0), "rotate_z": 90.0}
    _make_usd(usd, [ops], TRI)
    verts, _ = _converted_vertices(converter, usd, tmp_path / "r.obj", TRI)
    assert np.allclose(_sorted_rows(verts),
                       _sorted_rows(_gf_world_vertices(usd, TRI)), atol=1e-5)


def test_converter_nonuniform_scale(tmp_path):
    converter = _load_converter()
    usd = tmp_path / "s.usda"
    _make_usd(usd, [{"scale": (2.0, 0.5, 1.5)}], TRI)
    verts, _ = _converted_vertices(converter, usd, tmp_path / "s.obj", TRI)
    assert np.allclose(_sorted_rows(verts),
                       _sorted_rows(_gf_world_vertices(usd, TRI)), atol=1e-5)


def test_converter_nested_transforms(tmp_path):
    converter = _load_converter()
    usd = tmp_path / "n.usda"
    _make_usd(usd, [{"translate": (1.0, 2.0, 3.0)},
                    {"rotate_z": 90.0, "scale": (1.0, 2.0, 0.5)}], TRI)
    verts, _ = _converted_vertices(converter, usd, tmp_path / "n.obj", TRI)
    assert np.allclose(_sorted_rows(verts),
                       _sorted_rows(_gf_world_vertices(usd, TRI)), atol=1e-5)


def _mk_tree(root: Path, nested: bool):
    for section, cat, idx in (("Rigid", "mouse", 4), ("Geometry", "drawer", 0)):
        if nested:
            d = root / section / cat / f"{idx:05d}"
        else:
            d = root
        d.mkdir(parents=True, exist_ok=True)
        (d / "object.obj" if nested else d / f"{section}_{cat}_{idx:05d}_object.obj").touch()


def test_find_nested(tmp_path):
    _mk_tree(tmp_path, nested=True)
    assert find_obj_mesh(tmp_path, "Rigid", "mouse", 4) is not None
    assert find_obj_mesh(tmp_path, "Geometry", "drawer", 0) is not None


def test_find_flat(tmp_path):
    _mk_tree(tmp_path, nested=False)
    assert find_obj_mesh(tmp_path, "Rigid", "mouse", 4) is not None
    assert find_obj_mesh(tmp_path, "Geometry", "drawer", 0) is not None


def test_missing_returns_none(tmp_path):
    assert find_obj_mesh(tmp_path, "Rigid", "nonexistent", 9) is None
