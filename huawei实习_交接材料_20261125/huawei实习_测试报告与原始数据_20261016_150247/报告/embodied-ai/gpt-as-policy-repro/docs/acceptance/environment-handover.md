> **2026-09-21 复现更正：** 此页保留旧部署记录。新同事请先看 [Wiki](https://github.com/Xxc33128/embodied-ai/wiki)；并非只缺 GPT 接入信息，Mac 监听程序已补写，NPU 镜像及数据仍需按新指南核对。

# 环境交付：LIBERO-PRO 单线（2026-09-20）

> 用户目标口径：环境完全搭好，实验用户自己跑；唯一外部依赖 = GPT 接入信息。
> 本文是运行入口的唯一权威；进度口径见 `status-matrix.json`。

## 1. 已就绪清单

| 组件 | 状态 | 位置 |
|---|---|---|
| 240 case 池（4 套件 × 6 条件，全部 BDDL+50 init） | ✅ 240/240，冒烟 240/240 OK | gap-sim `…/LIBERO-PRO/libero/libero/{bddl,init}_files/`；清单 `docs/acceptance/l1-case-inventory.json` |
| Env 条件资产生成 | ✅ files=80 missing=0，vacuous=5 登记 | manifest `docs/acceptance/l1-env-gen-manifest.json` |
| 学生链路（CaseSpec→policy→runner→campaign CLI） | ✅ 264 tests，双容器同步 | gap-sim `/workspace/repo/src/gap_repro/libero/` |
| 策略服务器（noise-optional 正式契约） | ✅ 语法验证，待启动 | gap-repro `/workspace/repo/scripts/lp_a2_policy_server.py` |
| GPT 契约层（单臂 schema/门/Direct+Hybrid 桥/双路线传输） | ✅ 假传输全路径验证 | `src/gap_repro/libero/{agent_contract,agent_loop,gpt_bridge}.py`、`agent/client.py` |
| LP5 每格样本量（精确 Wilson） | ✅ 2,813 集/24 格全可行 | `docs/acceptance/lp5-recompute.json` |
| 统计/ACK/结果核算（T1–T3 修复） | ✅ | `src/gap_repro/{results,transport}.py` |

## 2. 运行入口

### 2.1 策略服务器（先启；gap-repro 容器）
```bash
# NPU 纪律：跑前暂停卡0服务、推理用卡1–7、跑后恢复并 health 验证
ssh <实验服务器>
docker exec -d gap-repro bash -c "source /opt/miniconda3/bin/activate <env> \
  && cd /workspace/repo && MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config \
  CUDA_VISIBLE_DEVICES=1 python3 scripts/lp_a2_policy_server.py npu \
  > /workspace/data/lp5_server.log 2>&1"
# 就绪标记：/workspace/data/lp_a2/server_ready_npu
```
正式协议下客户端请求**不带 noise 字段** → openpi 内部采样（官方一致）；
显式 noise 仅保留给设备对照复验（LP-A2 语义不变）。

### 2.2 学生开发冒烟（单集管线验证；gap-sim 容器）
```bash
docker exec -it gap-sim bash  # 容器内：
source /opt/miniconda3/bin/activate libero_pro && cd /workspace/repo
MUJOCO_GL=osmesa LIBERO_CONFIG_PATH=/workspace/data/libero_config \
python3 scripts/run_lp5_campaign.py \
  --inventory /workspace/data/l1_case_inventory.json \
  --campaign-id dev_smoke --mode development \
  --episodes-per-case 2 --conditions Ori,Env --suites libero_goal \
  --req-dir /workspace/data/lp5/req --resp-dir /workspace/data/lp5/resp \
  --out /workspace/data/lp5/dev_smoke.json
```
development 模式不入正式 n。
**集成证明已做**（chunk 级，`docs/acceptance/lp5-integration-probe.json`）：
观测→npz→服务器真实推理（internal 采样）→chunk 校验→环境执行全链路通过；
注意 CPU 推理 ≈213 s/chunk，全集闭环请用 NPU（正式排期按 LP-A2 纪律）。
`--req-dir/--resp-dir` 必须与服务器 `LP_POLICY_EXCHANGE_DIR` 一致
（默认 `/workspace/data/lp_a2`；lp5 campaign 用 `/workspace/data/lp5`）。**正式采集**加 `--mode formal --freeze-manifest …`
（先由 `freeze_campaign` 生成 manifest——冻结生产端是 L5 冻结动作，见 §4）。

### 2.3 GPT 联调（接入信息到位后）
```bash
cp configs/gpt_access.example.json configs/gpt_access.json  # 填值；凭据只写环境变量
python3 scripts/run_lp5_gpt_episode.py --method hybrid --route app_server \
  --config configs/gpt_access.json \
  --inventory /workspace/data/l1_case_inventory.json \
  --benchmark libero_goal --condition Ori --task-index 0 --init-index 0 \
  --out /workspace/data/lp5/gpt_dev/hybrid_goal_Ori_t0_i0.json
```
缺配置项会打印精确清单（ConfigMissingError）。route B（HTTP 网关）的
`ingest_response` 包络字段需在首次联调时按真实网关返回钉定（代码处已注明）。

### 2.4 全量逐 case 冒烟（已跑过，重跑入口备查）
`scripts/l1_smoke_all_cases.py`（幂等可续；结果 `l1-smoke-results.json` 240/240）。

## 3. 冻结口径（采集前不得单方面更改）

1. **指令 = benchmark 属性**（`task.language`，文件名转写；evaluate.py 同源）。
   BDDL `:language` 仅档案对照——Ori 本身 30/40 馈送≠BDDL（冒烟证据），
   这使 **Sem=Ori 管线等价重跑**成立。
2. **动作**：7D 归一化 delta；学生路径零裁剪（实测 −1.003 原样下发）；
   实测传递 0.011 m / 0.085 rad 每单位每步（probe v2/v3）；旋转 delta
   **世界系**（probe v3 纯度 0.9975）；决策界 0.055 m / 0.425 rad。
3. **预算**：spatial 220 / object 280 / goal 300 / 10 520（openpi 逐字）；
   wait10 在 reset_to 内；超预算 = valid_failure（非 censored）。
4. **噪声**：正式 = 无显式 noise（openpi 内部采样）；LP-A2 显式 NumPy
   noise 仅设备对照。
5. **统计**：分母 = 冻结清单；失败如实五态；精确 Wilson（非 Wald）。

## 4. 待用户决策（阻塞正式冻结，不阻塞开发）

1. **LP5 分层总量**：精确 Wilson 后 A882+B1166+C765 = **2,813 集**
   （旧 Wald 1,786 低估 57%）。接受 / 调档 / 论文同口径 500×格？
2. **Env 四格**：@0.07（765）还是 @0.10（省 ≈400 集）？
3. **goal·Task 保底**：p=0 无先验，重算给检测口径 n=73（0 成功时上界
   ≤0.05）；提案旧保底 20（上界 0.161）。取哪档？
4. **Sem 条件口径**：按论文管线（Sem=Ori 等价重跑，推荐——数字可比）
   还是将改写句送达模型（偏离论文协议）？
5. **GPT 行预算**：n≥30/格 最低门槛 vs 更高；保底格建议 goal·Ori/Sem/Task。
6. GPT 接入信息（route A app-server_cmd/model/provider/effort 或
   route B endpoint/api_key_env/model/effort）。
7. libero_10·Env 的 5 个 vacuous case：保留（推荐，论文管线一致）或剔除。

## 5. 已知开放项（不阻塞用户开跑学生行）

- LP-A0 未解码动作对比：explicitly open，闭环路径 = L6 预检三设备复验
  （status-matrix `lp_a0_open_item`）。
- route A/B 真实协议联调（需凭据）。
- L1 对抗审查进行中，结论与修复见后续 commit。
- `freeze_campaign` 生产端脚本在 L5 冻结动作时补（results.py 机制已备）。
