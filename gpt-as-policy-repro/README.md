# GPT-as-Policy 复现工程（MuJoCo / 昇腾迁移）

**当前执行范围（2026-09-19 用户确认）：仅 LIBERO-PRO。** 以[LIBERO-PRO 单线执行计划 v2.0](docs/superpowers/plans/2026-09-19-repro-next-execution.md)为当前执行入口。RoboDojo / RoboLab 退出后续范围，已有代码与证据保留；旧总计划仅作历史技术参考。

## 路径决策（对计划 §6.2 的一处偏离，已确认理由）

计划 §6.2 原拟把适配代码放在 `embodied-ai/gpt-as-policy-repro/`；实际执行以本独立目录为工程根：用户在创建本目录后于其中启动执行，且大型上游 clone、权重与数据不宜进公开仓库。脱敏后的验收记录与结果索引后续按周工作流拷入 `embodied-ai/`。其余 §6.2 目录边界照原计划执行。

## 布局

| 路径 | 职责 |
|---|---|
| `configs/upstream.lock.json` | 原输入身份（commit/revision/哈希/聚合配方），W1 产物 |
| `src/gap_repro/inputs.py` | 内容哈希、panel/scope 解析、HF 固定 revision 取数、LFS 指针拒绝 |
| `tests/test_input_identity.py` | W1 契约测试（缺文件/内容变动/LFS 指针/错误 revision 四类反例） |
| `scripts/run_w1_verification.py` | W1 全量核查 runner，产出验收证据 |
| `scripts/download_checkpoint.sh` | checkpoint 17 文件断点续传下载 + 逐字节验证 |
| `docs/acceptance/` | 验收证据与 W0 运行条件记录 |
| `upstream/`（gitignored） | 三仓库固定 commit clone（GPT-as-Policy / RoboDojo / XPolicyLab） |
| `data/`（gitignored） | HF 缓存、checkpoint 下载；正式大量数据最终放执行机数据目录 |

## 当前下一步（2026-09-19，范围收敛）

- 交付：LIBERO-PRO 四套件 × 六条件 × Student-only / Direct / Hybrid，完整矩阵为72格。
- 队列：L0运行基线 → L1条件/case清单 → L2学生与动作适配 → L3批量runner → L4 GPT工具循环 → L5冻结 → L6正式评测。
- 共享统计/ACK修复已有提交；本次与agent契约合计58项测试通过，仍须LIBERO集成验收。
- LP-A0/A1/A2既有证据按原覆盖范围复用，goal/task0通过不表示24条件格全部通过。
- 不再推进旧T0–T12中的RoboDojo专有收尾或RoboLab审计；仅保留LIBERO需要的共享组件。
- 详细文件、依赖和验收标准见[LIBERO-PRO 单线执行计划 v2.0](docs/superpowers/plans/2026-09-19-repro-next-execution.md)。

## 历史状态快照（2026-09-19，第三轮执行；不作当前队列）

- **W1 全链闭合**：169/169 原源码、50/50 布局、9 轨迹内容级通过；原 JAX checkpoint
  17 文件共享盘逐字节校验（17/17 OK，聚合 d15fb8bd… 内容级成立）；资产闭包
  355 文件/3.74GB、0 未解析。
- **服务器容器与推理服务已部署**：<实验服务器> 容器 `gap-repro`（镜像 `pi05-hybrid:v1`，
  同事交付包克隆到 `/data_nv0` 我们的副本、按原路径挂载，单卡 0，`--privileged`），
  π₀.₅ 服务 8642 端口 `/health` 通过，冒烟推理 `(50,14)` 成功。详见
  `docs/acceptance/runtime.md`。
- **推理权源（用户 2026-09-17 决定）**：使用同事适配版本 `robodojo_pi05_pt`（指南 v1）；
  共享盘另有 `robodojo_pi05_pt_v2`，指南与脚本均用 v1，差异待同事确认，不擅自切换。
  原始 checkpoint 的内容身份按 `configs/upstream.lock.json` 锁定。
- **LIBERO-PRO 面板推进**：LP0 源锁定、LP1 老栈环境、LP2 转换（811 张量逐位保真）、
  LP-A0 三腿数值等价（8/8 draws 过预注册阈值）均已验收；LP-A1 P1（fixture 漂移
  指纹盲区）已修复（三层指纹协议 + 归因隔离，审查 PASS）；**LP-A2 闭环设备替换
  PASS**——6/6 episode 双设备全成功、pinned 首块差 0.0022 过预注册阈值、零成功率
  翻转（NPU 稳态 0.52s/chunk）。对抗审查报告 17 份入库（root 9 + docs/review/ 8；LP-A2 PASS− 已闭环，LP3 契约层+LP5 提案审查 PASS− 修正闭环）。
- **LP3 契约层（模型无关部分）已完成**：与钉定上游 8f3d362b 逐字节对等的
  schema/工具表/gate/动作界，路线 A·B 传输接口就绪——**只差模型接入信息**。
- **LP5 冻结提案已出**（2026-09-19-LP5冻结提案.md）：学生行三档 1,786 集
  （Env 降 0.10 → 1,410），5 个决策点待拍板。
- **下一动作**：用户提供 LP3 模型接入（路线 A/B 配置清单见 agent/client.py 的
  ConfigMissingError）+ LP5 提案 5 个决策点；W2 归属协调（详见交接文档 §7）。
- **未完成**：W2–W3 收尾、W5 收尾、W6–W12（主计划线）；LP3/LP5（面板线）；
  W0 目标机部分字段待补。挂账见交接文档 §6。
- **完成口径纪律（audit F12）**：模块已实现 ≠ 单测通过 ≠ 目标机集成 ≠ 正式验收。
  四态逐包状态见 `docs/acceptance/status-matrix.json`（唯一权威进度口径）。

## 历史工程命令（按需参考；当前LIBERO入口见执行计划L0）

```bash
# Mac 契约测试（已验证入口；未设环境变量时 libero 组跳过且不污染其他模块）
MUJOCO_GL=glfw GAP_REPRO_DATA="$PWD/data" ./.venv/bin/python -m pytest tests/ -q --tb=short -rxX
# Linux RoboDojo 目标机（gap-repro 内）与 LIBERO 老栈（gap-sim 内）分别用各自环境，
# 不共用一个 Python：robodojo 用项目 venv + torch_npu；libero 用 conda libero_pro。
./.venv/bin/python scripts/run_w1_verification.py # W1 核查（需网络，产出验收证据）
./.venv/bin/python scripts/check_runtime.py --profile mac   # 组件/资产/部署哈希体检
# 服务器（<实验服务器>）：容器已起、服务已在 8642；重启方式见 docs/acceptance/runtime.md
ssh <npu-host> 'curl -s http://localhost:8642/health'     # 健康检查
./.venv/bin/python scripts/pi05_smoke_infer.py <host> 8642  # 50×14 冒烟推理
bash scripts/download_checkpoint.sh /path/to/data  # 仅当需从 HF 另下权重时（已由共享盘校验替代）
```

## 权重与参考推理部署（2026-09-17 更新）

- Mac 保留代码、配置、哈希清单和轻量验收证据，不要求下载完整 π₀.₅ 权重。
- NPU 服务器的数据盘保存固定 revision 的原始权重及后续转换权重；不将权重放进 Git，也不把 NPU 设备显存当作下载目录。
- W6 的 JAX CPU / PyTorch CPU 参考推理可以在服务器 CPU 上执行，W7 的 torch_npu 推理使用 NPU。CPU 架构、依赖和内存能否满足参考推理由 W0 实测确认。
- 权重内容校验已完成：RoboDojo 原 JAX ckpt 17/17（共享盘逐字节）；pi05_libero 官方权重 16 对象 md5+crc32c 对 GCS 元数据核验。
