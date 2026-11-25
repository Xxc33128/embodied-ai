# =====================================================================
# Kaggle T4 ×2 侦察 + Isaac 安装 notebook（GPT-6 Astra 复现 Phase 2）
#
# 用法：Kaggle 新建 Notebook → Settings: Accelerator = GPU T4 ×2,
# Internet = ON（需手机验证账号）。然后把本文件按 `# CELL n` 注释
# 分段逐格粘贴运行。CELL 1-2 是纯侦察（只读），跑完先看结果再继续。
# 死线规则：若 CELL 5 的 BOOT PROOF 反复失败（Vulkan 无解/崩溃循环），
# 停止恋战，回报侦察输出，切降级方案 A（MuJoCo embodiment）。
# =====================================================================

# CELL 1 —— 环境侦察（只读，~1 min）
import subprocess, sys, shutil, os

def run(cmd, timeout=120):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    print(f"$ {cmd}\n{r.stdout.strip()[:3000]}")
    if r.returncode != 0 and r.stderr.strip():
        print(f"[stderr] {r.stderr.strip()[:1000]}")
    return r.stdout

run("nvidia-smi")                                   # GPU 型号/驱动/CUDA
run("df -h / /kaggle/working /kaggle/tmp 2>/dev/null || df -h")  # 磁盘布局
run("python --version && pip --version")
run("ls /usr/share/vulkan/icd.d/ 2>/dev/null || echo 'no vulkan ICD dir'")
run("which vulkaninfo || echo 'vulkaninfo not installed'")
run("cat /usr/share/vulkan/icd.d/*.json 2>/dev/null | head -20")
run("free -g | head -2; nproc")

# CELL 2 —— Vulkan/OpenGL 运行库（Isaac 必需；含 libGLU，boot 文档点名）
run("apt-get update -qq && apt-get install -y -qq libvulkan1 vulkan-tools "
    "mesa-vulkan-drivers libglu1-mesa libgl1 libxrandr2 libxinerama1 libxcursor1",
    timeout=600)
run("vulkaninfo --summary 2>/dev/null | head -40 || echo 'vulkaninfo failed'")

# CELL 3 —— 独立 venv + Isaac 安装（~20-40 min，装到 /kaggle/tmp 不占输出配额）
# 版本组合 = 我们云 3060 实证跑通的组合：isaacsim 5.0.0.0 (PyPI) + IsaacLab v2.3.2。
# 不用 Kaggle 预装 torch 的环境，避免版本冲突；isaacsim 自带配套 torch。
# 若 5.0.0.0 在 T4 上渲染崩（T4 仅勉强达最低线），回退试 isaacsim==4.5.0.0
#   （4.5 需加 --extra-index-url https://pypi.nvidia.com）+ IsaacLab v1.x。
import os
os.environ["PIP_CACHE_DIR"] = "/kaggle/tmp/pipcache"
run("python -m venv /kaggle/tmp/venv && /kaggle/tmp/venv/bin/pip install -q -U pip")
run("/kaggle/tmp/venv/bin/pip install isaacsim==5.0.0.0", timeout=3600)
run("git clone --depth 1 --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git /kaggle/tmp/IsaacLab")
run("cd /kaggle/tmp/IsaacLab && /kaggle/tmp/venv/bin/pip install "
    "-e ./source/isaaclab -e ./source/isaaclab_assets -e ./source/isaaclab_tasks", timeout=1800)
run("/kaggle/tmp/venv/bin/pip install 'inspect-robots[rerun]==0.58.0' "
    "inspect-robots-agent inspect-robots-isaacsim", timeout=600)
run("/kaggle/tmp/venv/bin/python -c 'import isaacsim, isaaclab, inspect_robots_isaacsim; print(\"imports OK\")'")

# CELL 4 —— 环境打包（可选）：把 venv 存成 Kaggle Dataset，下次会话秒级恢复
# 先跑 CELL 3 成功后执行；输出 ~10GB，需开 Kaggle Dataset（私有）。
# run("du -sh /kaggle/tmp/venv /kaggle/tmp/IsaacLab")

# CELL 5 —— BOOT PROOF（决定性闸门：Isaac 能否在 T4 headless 启动，~3-10 min）
boot_proof = '''
import time
from inspect_robots import Embodiment
from inspect_robots_isaacsim import IsaacSimEmbodiment

emb = IsaacSimEmbodiment(headless=True)
assert isinstance(emb, Embodiment)
print(f"[adapter] name={emb.info.name} action_dim={emb.info.action_space.dim} "
      f"sim={emb.info.is_simulated} caps={sorted(emb.info.capabilities)}", flush=True)
t0 = time.perf_counter()
print("[boot] launching Isaac Sim SimulationApp (headless)...", flush=True)
app = emb._ensure_app()
dt = time.perf_counter() - t0
running = getattr(app, "is_running", lambda: None)()
print(f"[boot] SimulationApp live in {dt:.1f}s  is_running={running}", flush=True)
for _ in range(3):
    app.update()
print(f"BOOT PROOF: SUCCESS  (started + stepped in {dt:.1f}s)", flush=True)
emb.close()
'''
with open("/kaggle/tmp/boot_proof.py", "w") as f:
    f.write(boot_proof)
run("/kaggle/tmp/venv/bin/python /kaggle/tmp/boot_proof.py", timeout=900)

# CELL 6 —— LiftCube 任务冒烟（reset + 数步 + 相机出图验证，~5-15 min）
# 成功标准：obs.state 有 5 个字段、obs.images 至少 1 张且非全黑、无异常。
smoke = '''
import numpy as np
from inspect_robots_isaacsim import IsaacSimEmbodiment
from inspect_robots.scene import Scene

emb = IsaacSimEmbodiment(
    task_id="Isaac-Lift-Cube-Franka-v0",
    cameras=[("base_rgb", 224, 224)],
    headless=True,
    device="cuda:0",
)
obs = emb.reset(Scene(id="smoke", instruction="lift the cube"), seed=0)
print("state keys:", sorted(obs.state))
for k, im in obs.images.items():
    a = np.asarray(im)
    print(f"image {k}: shape={a.shape} min={a.min()} max={a.max()} mean={a.mean():.1f}")
act_dim = emb.info.action_space.shape[0]
import inspect_robots.types as T
for i in range(5):
    step = emb.step(T.Action(data=np.zeros(act_dim)))
    if i == 0:
        print("step result keys:", type(step).__name__, "reward:", step.reward)
print("SMOKE: SUCCESS")
emb.close()
'''
with open("/kaggle/tmp/smoke.py", "w") as f:
    f.write(smoke)
run("/kaggle/tmp/venv/bin/python /kaggle/tmp/smoke.py", timeout=1800)

# CELL 7 ——（CELL 5/6 成功后）mock LLM 全链路（把 Mac 上验证过的 run_mock_agent.py
# 原样粘贴成 cell，用 /kaggle/tmp/venv 的解释器逻辑跑——即把文件内容贴进来执行）。
# 目的：确认 agent loop 在 Kaggle 的 Isaac 环境里同样成立（零 API 费用）。
