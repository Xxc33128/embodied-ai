#!/bin/bash
# LP-A2 NPU 臂无人值守驱动（<实验服务器> 宿主机）。前置：CPU 臂已完成（pin_ref.json 在）。
# NPU 纪律：实验用卡 1（ASCEND_RT_VISIBLE_DEVICES=1）；跑前暂停 pi05 服务（卡0），
# 跑完恢复并 health 验证（坑 C1：常驻进程与在线算子编译竞争）。
set -u
D=/data_nv0/gpt-as-policy-repro/data/lp_a2
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 0) 等 CPU 臂完成
while ! grep -q CPU_ARM_ALL_DONE "$D/driver_cpu.log" 2>/dev/null; do sleep 120; done
log cpu_arm_done_observed

# 1) 暂停 pi05 服务（我们的副本进程，纪律 §8）
docker exec gap-repro bash -c "pkill -f pi05_service.py || true"
sleep 8
if curl -s -m 5 http://localhost:8642/health >/dev/null 2>&1; then
  log "ERROR service still alive, abort"; exit 1
fi
log service_paused

# 2) 起 NPU policy server（卡 1）
docker exec gap-repro bash -c "rm -f /workspace/data/lp_a2/server_ready_npu"
docker exec -d gap-repro bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && export ASCEND_RT_VISIBLE_DEVICES=1 && cd /workspace/repo && python3 scripts/lp_a2_policy_server.py npu > /workspace/data/lp_a2/server_npu.log 2>&1"
while ! docker exec gap-repro test -f /workspace/data/lp_a2/server_ready_npu; do
  sleep 30
  docker exec gap-repro bash -c "grep -i 'error\|Traceback' /workspace/data/lp_a2/server_npu.log" | head -2 && break
done
docker exec gap-repro test -f /workspace/data/lp_a2/server_ready_npu || { log "ERROR npu server not ready"; docker exec gap-repro tail -20 /workspace/data/lp_a2/server_npu.log; exit 1; }
log npu_server_ready

# 3) paper 两集 → pinned 一集（pin_ref.json 复用 CPU 参考布局，跨设备同钉扎）
docker exec -d gap-sim bash -c "source /opt/miniconda3/bin/activate libero_pro && cd /workspace/repo && MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config python3 scripts/lp_a2_closed_loop.py paper_faithful torch_npu 0,1 > /workspace/data/lp_a2/client_npu.log 2>&1"
log npu_paper_launched
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_paper_faithful_torch_npu_init1.json; do sleep 60; done
log npu_paper_done
docker exec -d gap-sim bash -c "source /opt/miniconda3/bin/activate libero_pro && cd /workspace/repo && MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config python3 scripts/lp_a2_closed_loop.py scene_pinned torch_npu 0 /workspace/data/lp_a2/pin_ref.json > /workspace/data/lp_a2/client_npu_pinned.log 2>&1"
log npu_pinned_launched
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_scene_pinned_torch_npu_init0.json; do sleep 60; done
log npu_pinned_done

# 4) 停 NPU server，恢复 pi05 服务并验证
docker exec gap-repro bash -c "pkill -f lp_a2_policy_server || true"
sleep 5
docker exec -d gap-repro bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd /data/pi05_hybrid/scripts && python3 pi05_service.py npu 8642 > /workspace/data/pi05_service.log 2>&1"
for i in $(seq 1 20); do
  sleep 15
  h=$(curl -s -m 5 http://localhost:8642/health || true)
  if echo "$h" | grep -q '"ok"'; then log "service_restored health=$h"; break; fi
done
curl -s -m 5 http://localhost:8642/health || log "ERROR service not healthy"
echo NPU_ARM_ALL_DONE
