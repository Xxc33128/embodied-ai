#!/usr/bin/env python3
"""W11 场景全景渲染：双 X5 + 真实 OBJ 对象 + 桌面，自由视角 + 三相机。"""
import sys, os
os.environ["MUJOCO_GL"] = "osmesa"
sys.path.insert(0, "/workspace/repo/src")
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image
from gap_repro.sim import scene, robots

urdf = Path("/workspace/data/hf_cache/Assets/Robots/x5/X5A.urdf")
model, data, _manifest = scene.build_scene_mjcf(urdf, "organize_table_0",
                                                obj_root="/workspace/data/w11_scene_obj")
for side in ("left", "right"):
    for i, j in enumerate(robots.ARM_JOINTS):
        data.qpos[model.joint(f"{side}_{j}").qposadr[0]] = [0.3, -0.4, 0.5, -0.9, 0.4, 0.2][i]
mujoco.mj_forward(model, data)
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720

out_dir = Path("/workspace/data/w11_renders")
out_dir.mkdir(exist_ok=True)

# 全景自由视角：稍高、正对桌面，双臂入画
r = mujoco.Renderer(model, 720, 1280)
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.0, -0.35, 0.55]
cam.distance = 1.6
cam.azimuth = 180
cam.elevation = -25
r.update_scene(data, camera=cam)
Image.fromarray(r.render()).save(str(out_dir / "overview.png"))
print("RENDER overview 1280x720")

# 俯视桌面：看清物体摆放
cam.lookat[:] = [0.0, -0.05, 0.78]
cam.distance = 1.1
cam.azimuth = 180
cam.elevation = -80
r.update_scene(data, camera=cam)
Image.fromarray(r.render()).save(str(out_dir / "topdown.png"))
print("RENDER topdown 1280x720")
r.close()

# 三相机（渲染 venv 下再存一次，light 已在场景内）
r2 = mujoco.Renderer(model, 480, 640)
cam2 = mujoco.MjvCamera()
cam2.type = mujoco.mjtCamera.mjCAMERA_FIXED
for cid in range(model.ncam):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cid)
    cam2.fixedcamid = cid
    r2.update_scene(data, camera=cam2)
    Image.fromarray(r2.render()).save(str(out_dir / f"{name}.png"))
    print(f"RENDER {name}")
r2.close()

print(f"TOTAL_BODIES {model.nbody} TOTAL_MESHES {model.nmesh}")
print("W11_OVERVIEW_OK")
