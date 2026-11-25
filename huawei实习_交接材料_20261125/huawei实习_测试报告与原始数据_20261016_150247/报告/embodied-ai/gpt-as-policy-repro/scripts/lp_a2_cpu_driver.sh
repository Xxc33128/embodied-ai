#!/bin/bash
# LP-A2 CPU 臂无人值守驱动（在 <实验服务器> 宿主机上跑，docker exec 编排）。
# 纪律：只编排我们自己的容器与文件；不触碰共享盘。
set -u
D=/data_nv0/gpt-as-policy-repro/data/lp_a2
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 1) 等 paper 两集（客户端已在跑）
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_paper_faithful_torch_cpu_init0.json; do sleep 60; done
log paper_init0_done
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_paper_faithful_torch_cpu_init1.json; do sleep 60; done
log paper_init1_done

# 2) 生成钉扎布局并跑 pinned 集
docker exec gap-repro python3 /workspace/repo/scripts/lp_a2_make_pin.py \
  /workspace/data/lp_a2/ep_paper_faithful_torch_cpu_init0.json /workspace/data/lp_a2/pin_ref.json
docker exec -d gap-sim bash -c "source /opt/miniconda3/bin/activate libero_pro && cd /workspace/repo && MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config python3 scripts/lp_a2_closed_loop.py scene_pinned torch_cpu 0 /workspace/data/lp_a2/pin_ref.json > /workspace/data/lp_a2/client_cpu_pinned.log 2>&1"
log pinned_launched
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_scene_pinned_torch_cpu_init0.json; do sleep 60; done
log CPU_ARM_ALL_DONE
