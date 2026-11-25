# Mac 接入与实验运行

## 复现哪一个版本

昨天的四组实验使用 `experiments/20260920/` 中的开发 runner，而不是仅运行 `scripts/run_lp5_gpt_episode.py`。每组保留自己的任务和预算，避免悄悄改变历史实验。

| 目录 | 场景 | 控制预算 |
|---|---|---|
| pilot_object5_20260920 | 番茄酱入篮，Ori | 280 |
| pro_pos_object5_20260920 | 番茄酱入篮，Pos | 280 |
| pro_pos_spatial0_600_20260920 | 黑碗放盘，Pos | 600 |
| pro_pos_goal0_600_20260920 | 打开抽屉，Pos | 600 |

每组含 `run_episode.py`、`recorded_runtime.py` 和 `finalize_pilot.py`。后两组的 600 步是开发实验预算，不能当作原套件默认协议。所有脚本都是历史快照，不是通用生产启动器。

## 先运行一个 Student 回合

先按 Weights 页启动指向 `repro_colleague_001/pi_student` 的服务。然后在仿真容器中：

```bash
docker exec -it gap-sim bash
source /opt/miniconda3/bin/activate libero_pro
cd /workspace/repo
export PYTHONPATH=/workspace/repo/src
export MUJOCO_GL=osmesa
export LIBERO_CONFIG_PATH=/workspace/data/libero_config
# 本次 root 应只含刚启动的 pi_student，不能含旧的 episode 输出
python experiments/20260920/pilot_object5_20260920/run_episode.py   --method student --root /workspace/data/repro_colleague_001
```

策略服务会创建父目录。启动 runner 前确认本次 root 没有 student/result.json 或旧 events.jsonl；若已有，请换一个新的 root，并同步修改策略服务的交换目录。

runner 完成 reset 后写入 `student/READY`，随后等待 `GO`。在另一个终端检查 READY 后启动：

```bash
docker exec gap-sim test -f /workspace/data/repro_colleague_001/student/READY
docker exec gap-sim touch /workspace/data/repro_colleague_001/GO
```

结束后检查 `student/result.json` 的 status、success 与异常信息，不能仅凭脚本退出码判成功。`events.jsonl`、frames 和 config.json 保存本回合证据。

`finalize_pilot.py` 的输入目录是写死的历史目录；复跑新目录时先复制脚本并调整 `base`，不要直接运行。录像按 20 fps 合并两个视角，省略模型等待时间。

## Direct／Hybrid 的 Mac 接入

实际历史拓扑为：服务器 runner → stdio_socket.py → Unix socket 转发 → Mac 模型会话。昨天记录的模型设置为 `gpt-6-astra`、provider `openai`、effort `medium`；使用者需要自己的模型权限。

服务器配置示例（单层 JSON，不是 gpt_access.example.json 的外层模板）：

```json
{
  "app_server_cmd": ["python", "/workspace/repo/tools/mac_bridge/stdio_socket.py", "/workspace/data/mac_bridge/codex.sock"],
  "model": "gpt-6-astra",
  "provider": "openai",
  "effort": "medium"
}
```

Mac 端现已补写 `tools/mac_bridge/listen.py`。先按 [Mac 监听程序](https://github.com/Xxc33128/embodied-ai/wiki/Mac-bridge) 启动监听、建立 SSH 转发并完成连接检查，再执行下面的命令。该程序已验证传输，尚未用它完成完整模型回合。

```bash
# 在与 Student 相同的 gap-sim 环境中；每次使用新的 root
python experiments/20260920/pilot_object5_20260920/run_episode.py   --method direct --root /workspace/data/repro_direct_001   --config /workspace/data/mac_bridge/gpt.local.json
```

Hybrid 将 method 改为 `hybrid`，并启动使用 `<本次root>/pi_hybrid` 的 NPU 策略服务。两者同样在各自 READY 后等待 GO。先逐个运行，不直接复制历史并发服务管理脚本，它们含实验专用的停止／恢复服务逻辑。

## 如何判断交付完成

第二位使用者至少完成：有效双视角观测 → Student 真实 NPU 推理与动作执行 → Mac 真实模型工具调用 → 一个 Direct 或 Hybrid 回合 → 结果和视频可追溯。本文发布时尚未完成第二位使用者环境的验收，不能据本机单测推断这一点。


[代码目录](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro/) · [Wiki 首页](https://github.com/Xxc33128/embodied-ai/wiki)
