"""T4d：原三相机视觉观测（cam_head / cam_left_wrist / cam_right_wrist）。

契约（E1：docs/acceptance/w5-session-call-inventory.md）：
`raw['vision'][<相机>]['color']` 为 HxWx4 uint8；MuJoCo Renderer 出 RGB，
此处补 alpha=255 并对齐原相机名（此前为 left_wrist_cam/right_wrist_cam，
且 include_vision=True 直接 BLOCKED）。

后端不可用（无 Renderer/GL）时显式 BLOCKED——不静默返回空图。
"""

from __future__ import annotations

import numpy as np

VISION_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


class VisionRenderer:
    def __init__(self, model, width: int = 640, height: int = 480):
        import mujoco

        self._model = model
        self.width = int(width)
        self.height = int(height)
        try:
            self._renderer = mujoco.Renderer(model, self.height, self.width)
        except Exception as e:  # 后端/驱动缺失 → fail-closed
            raise RuntimeError(f"BLOCKED: MuJoCo 渲染后端不可用: {e}") from e
        self._camera_ids = {}
        for key in VISION_KEYS:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, key)
            if cid < 0:
                raise RuntimeError(f"BLOCKED: 相机 {key} 不在模型中")
            self._camera_ids[key] = int(cid)

    def render(self, data) -> dict:
        vision = {}
        for key, cid in self._camera_ids.items():
            self._renderer.update_scene(data, camera=cid)
            rgb = self._renderer.render()
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise RuntimeError(f"{key}: unexpected render shape {rgb.shape}")
            alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
            vision[key] = {
                "color": np.concatenate([rgb, alpha], axis=2),
                "resolution": (self.width, self.height),
                "camera_id": cid,
            }
        return vision

    def close(self):
        self._renderer.close()
