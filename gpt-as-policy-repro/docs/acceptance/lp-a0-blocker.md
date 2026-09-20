# LP-A0 NPU 侧阻塞点交接（2026-09-18）

## 已完成
- CPU 侧双跑：确定性底噪已产出（进程日志 `/data_nv0/gpt-as-policy-repro/data/lp_a0_result.log`，
  A0 行含 cpu_determinism 与 cpu_sample；NPU 抛错前 CPU 部分完成）。
- 权重级等价已闭合（LP2：811 张量逐位）。

## 阻塞点（可复现，非瞬态）
- pi05 模型 NPU 前向在 siglip 视觉塔触发 GE 图编译，`GEInitialize failed`
  + `OpCompileProcessor init failed`，进程挂死。**小算子探针（matmul/conv1d）
  在同卡新进程可过** → 单算子 ACL 路径正常，GE 图编译路径失败。
- 与卡无关（卡 1/2 同样）；与常驻服务无关（停服务后复现）；fresh cache 无效；
  torch.compile 已关（eager）。robosuite 服务（卡 0）启动期 GE 初始化成功过，
  提示可能是**新进程的 GE 初始化在当前宿主状态下失败**（宿主级，或其他租户进程竞争）。
- 证据：`data/lp_a0_result.log`（每次重跑覆盖，最后状态为 siglip 塍 traceback）、
  容器内 `/root/ascend/log/run/plog/plog-28641_20260918031605957.log` 等近期 plog
  （GE 失败计数为 0——失败信息经 stdout ReportCallError 输出，GE 子进程 plog 待定位）。

## 下一步候选（按优先级）
1. 用 `ASCEND_GLOBAL_LOG_LEVEL=0`（DEBUG）重跑，从 plog 抓 GE init 失败前的第一条
   根因行（当前 plog 未覆盖 GE 子进程日志，需开 `ASCEND_SUB_PROCESS_LOG` 类开关）。
2. 改用 ascend-docker-runtime 规范容器（`--runtime=ascend -e ASCEND_VISIBLE_DEVICES=1`）
   替代 privileged 手动设备映射，对齐宿主钩子的 GE 初始化路径。
3. 若 GE 在 privileged 容器存在已知限制：联系服务器管理员确认宿主 GE 编译守护状态；
   或临时以 JAX CPU 为唯一参考完成 LP-A0 的 CPU 内对比，NPU 侧等宿主修复（如实披露）。
4. pi05 服务已恢复运行（8642 health ok，卡 0），后续 NPU 试验优先用卡 1–7。

## 恢复状态
- pi05 服务：`{"ok": true, "device": "npu"}`（已重启验证）。
- lp_a0 脚本：`scripts/lp_a0_numerical_check.py`（compile 关闭、卡可切换）。
