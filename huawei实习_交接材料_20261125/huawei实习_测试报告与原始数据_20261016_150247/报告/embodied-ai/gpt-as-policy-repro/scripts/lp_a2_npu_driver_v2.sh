#!/bin/bash
# LP-A2 NPU 臂恢复驱动 v2（v1 的 grep|head 管道退出码恒 0 导致提前退出）。
# 前置：pi05 服务已被 v1 暂停；NPU server 进程在构建中。本驱动接管到底，
# 任何失败路径都恢复 pi05 服务再退出。
set -u
D=/data_nv0/gpt-as-policy-repro/data/lp_a2
log() { echo "[$(date +%H:%M:%S)] $*"; }
restore_service() {
  docker exec gap-repro bash -c "pgrep -f lp_a2_policy_server >/dev/null && pkill -f lp_a2_policy_server || true"
  sleep 5
  docker exec -d gap-repro bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd /data/pi05_hybrid/scripts && python3 pi05_service.py npu 8642 > /workspace/data/pi05_service.log 2>&1"
  for i in $(seq 1 20); do
    sleep 15
    h=$(curl -s -m 5 http://localhost:8642/health || true)
    if echo "$h" | grep -q '"ok"'; then log "service_restored health=$h"; return 0; fi
  done
  log "ERROR service restore failed"
  return 1
}

# 1) 等 NPU server 就绪（v1 死于此处的 bash 管道 bug；这里只看 marker 文件）
for i in $(seq 1 40); do
  if docker exec gap-repro test -f /workspace/data/lp_a2/server_ready_npu; then break; fi
  if ! docker exec gap-repro pgrep -f lp_a2_policy_server >/dev/null; then
    log "ERROR npu server process died"; docker exec gap-repro tail -20 /workspace/data/lp_a2/server_npu.log; restore_service; exit 1
  fi
  sleep 30
done
docker exec gap-repro test -f /workspace/data/lp_a2/server_ready_npu || { log "ERROR npu server not ready after 20min"; docker exec gap-repro tail -20 /workspace/data/lp_a2/server_npu.log; restore_service; exit 1; }
log npu_server_ready

# 2) paper 两集
docker exec -d gap-sim bash -c "source /opt/miniconda3/bin/activate libero_pro && cd /workspace/repo && MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config python3 scripts/lp_a2_closed_loop.py paper_faithful torch_npu 0,1 > /workspace/data/lp_a2/client_npu.log 2>&1"
log npu_paper_launched
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_paper_faithful_torch_npu_init1.json; do sleep 60; done
log npu_paper_done

# 3) pinned 一集（复用 CPU 参考布局 pin_ref.json）
docker exec -d gap-sim bash -c "source /opt/miniconda3/bin/activate libero_pro && cd /workspace/repo && MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config python3 scripts/lp_a2_closed_loop.py scene_pinned torch_npu 0 /workspace/data/lp_a2/pin_ref.json > /workspace/data/lp_a2/client_npu_pinned.log 2>&1"
log npu_pinned_launched
while ! docker exec gap-sim test -f /workspace/data/lp_a2/ep_scene_pinned_torch_npu_init0.json; do sleep 60; done
log npu_pinned_done

# 4) 停 NPU server，恢复 pi05 服务并验证（任何退出路径都先恢复）
restore_service
rc=$?
log NPU_ARM_ALL_DONE
exit $rc
