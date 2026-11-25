# RPent 复现（npunew 节点 · GLM-5.3 大脑 · Object Swap 20 集）

> 执行：2026-09-24（单日完成：环境搭建 → 冒烟 → 20 集评测 → 分析），ZCode（GLM-5.3 驱动）+ 用户。
> 参照系：[Maxwell-AI-lab/rpent-replication](https://github.com/Maxwell-AI-lab/rpent-replication)（同事 2026-09-22/23 在 aura-7 节点完成，Object Swap 20/20 = 100%）。上游报告细节不在此重复，仅对照。
> 证据分级按工作区规范 E1–E4 标注。

## 0. 一句话结论

**跨节点复现成立**：在另一台昇腾节点（npunew）上从零搭建 RPent 全栈（GLM-5.3 planner + SAM3 + 冻结 π0.5 + MuJoCo/osmesa + 记忆语料），Object Swap 20 集 **18/20 = 90%**，与上游 20/20 无统计显著差异（Fisher 精确检验双侧 p≈0.49）；LLM 用量轮廓几乎逐项复刻（**prompt cache 命中率两边同为 90.4%**）。

## 1. 口径与背景

- 上游复现主张：记忆引导 agent 循环把冻结 π0.5（Object Swap 单独 17%）拉到 90%+ 档（官方 Astra 档 99%）。
- 本次目的：验证该复现**本身**的可复现性——不同节点、不同镜像（版本一致）、RPent 代码新 HEAD、全新安装。
- 评测协议与上游完全一致：libero_object_swap × task0-9 × seed0/1 = 20 集，官方终止谓词判定，`--max-turns 100 --planner-timeout-s 5400`，VLA/SAM3 外部常驻服务，记忆只读。[E2]

## 2. 结果

### 2.1 战绩

| 方法 | Object Swap（20 集） |
|---|---|
| 冻结 π0.5 单独（官方） | 17% |
| RPent + Qwen3.6-27B（官方） | 84% |
| RPent + GPT-5.5（官方） | 91% |
| RPent + GPT-6 Astra（官方） | 99% |
| RPent + GLM-5.3 @ aura-7（上游复现） | 100%（20/20） |
| **RPent + GLM-5.3 @ npunew（本次）** | **90%（18/20）** |

- 18/20 的 Wilson 95% CI 为 70%–97%，与上游 20/20 差异不显著（Fisher 双侧 p≈0.49）。
- 档位解读：本次样点与 GPT-5.5（91%）持平，处于上游自述"真实档位 90–99% 区间"内，佐证 GLM-5.3 落在 5.5–Astra 档。[E3]

### 2.2 逐集明细（analyze_perf2.py，节点 `/data/rpent2/` 可重跑）

| 集 | 结果 | 时长 s | 请求 | ctx max | ctx avg | 新增 prefill | cache% | decode |
|---|---|---|---|---|---|---|---|---|
| t0_s0 | ✅ | 580 | 23 | 69k | 51k | 131k | 88.8 | 17.3k |
| t0_s1 | ✅ | 333 | 18 | 58k | 46k | 76k | 90.8 | 11.0k |
| t1_s0 | ✅ | 935 | 38 | 77k | 55k | 156k | 92.5 | 19.8k |
| t1_s1 | ✅ | 930 | 35 | 54k | 37k | 161k | 87.5 | 19.7k |
| t2_s0 | ✅ | 526 | 25 | 57k | 42k | 108k | 89.7 | 18.8k |
| **t2_s1** | ❌ | 2818 | 100 | 149k | 94k | 628k | 93.3 | 65.1k |
| t3_s0 | ✅ | 714 | 31 | 72k | 56k | 150k | 91.3 | 18.6k |
| **t3_s1** | ❌ | 3840 | 100 | 165k | 99k | 498k | 94.9 | 76.2k |
| t4_s0 | ✅ | 339 | 18 | 56k | 45k | 75k | 90.7 | 8.4k |
| t4_s1 | ✅ | 411 | 20 | 41k | 28k | 69k | 87.8 | 13.8k |
| t5_s0 | ✅ | 497 | 19 | 61k | 44k | 104k | 87.6 | 16.1k |
| t5_s1 | ✅ | 528 | 24 | 67k | 50k | 118k | 90.3 | 17.8k |
| t6_s0 | ✅ | 1348 | 58 | 123k | 73k | 323k | 92.3 | 57.6k |
| t6_s1 | ✅ | 590 | 27 | 68k | 51k | 173k | 87.4 | 15.3k |
| t7_s0 | ✅ | 350 | 14 | 56k | 42k | 77k | 86.9 | 9.9k |
| t7_s1 | ✅ | 569 | 27 | 70k | 49k | 133k | 89.9 | 18.9k |
| t8_s0 | ✅ | 651 | 34 | 68k | 49k | 121k | 92.7 | 16.3k |
| t8_s1 | ✅ | 713 | 33 | 73k | 52k | 134k | 92.1 | 19.3k |
| t9_s0 | ✅ | 605 | 25 | 84k | 61k | 155k | 89.9 | 22.0k |
| t9_s1 | ✅ | 527 | 19 | 69k | 52k | 77k | 92.2 | 16.7k |

### 2.3 均值对照（上游 20 集 vs 本次 20 集）

| 指标 | 上游 aura-7 | 本次 npunew |
|---|---|---|
| 成功率 | 100% | 90% |
| 单集时长 | 919 s | 890 s |
| 请求/集 | 31 | 34 |
| 单请求上下文 avg | 74k | 76k |
| 每集新增 prefill | 157k | 173k |
| **cache 命中率** | **90.4%** | **90.4%** |
| 每集 decode | 22.7k | 23.9k |
| 平均 turn 周期 | 29.1 s | 23.7 s |

LLM 侧行为轮廓逐项复刻；turn 周期略快（npunew 出网到 bigmodel 更干净）。[E1]

### 2.4 两败分析

- t2_s1（2818s/100 轮/ctx 149k/decode 65k）与 t3_s1（3840s/100 轮/ctx 165k/decode 76k）均为 **max_turns 打满**：GLM 走"手工原语"慢路径（back_project/segment/move 反复重试）未收敛。
- 与上游总结的失败前兆特征（ctx>150k + decode 3× + 请求 2.5×）完全吻合；上游同两集均解出（469s/639s）——差异属 GLM-5.3 非确定性采样方差，非环境问题。[E1]
- 上游两集苦战（t4_s1 2833s、t6_s1 1654s）在本次反而速胜（411s/590s），进一步印证方差双向。[E1]

## 3. 环境与上游差异清单 [E2]

| 项 | 上游 aura-7 | 本次 npunew |
|---|---|---|
| 硬件 | 8×910B3（专用） | 8×910B3（共享：避让他任务，VLA→dev5、SAM3→dev6） |
| 镜像 | k3-train:cann852-v14 | 本地 pi05-hybrid:v1（py3.11.15 / torch 2.7.1+cpu / torch_npu 2.7.1.post2，版本一致） |
| RPent 代码 | 9/22 HEAD + 手工 sed | 9/24 HEAD 全新 clone + 补丁（布局有漂移，见 §4.2） |
| 权重/资产 | Mac 分块推送（HK 出口断网） | **零迁移**：两节点共享 /data SFS，π0.5(7.0G)/sam3.pt(3.45G, 字节与上游校验一致)/liberopro-assets(622M) 直用 |
| 网络 | HK 出口对 HF/PyPI/GitHub 大面积不通 | 出网干净（HF/GitHub/modelscope 直连），无需任何代理方案 |
| 大脑 | GLM-5.3（bigmodel Anthropic 口） | 同（同 key 体系） |

同 seed 冒烟集（t2_s0）SAM3 首次定位输出 (-0.183, -0.080)，与上游报告记录 (-0.184, -0.080) 毫米级一致——仿真确定性核验通过。[E1]

## 4. 移植新发现（超出上游报告的部分）

### 4.1 `get_device_properties` major=None 全局 shim（上游缺口）

上游报告提到"get_device_properties None shim"但**未随仓库提交补丁**。本次实际踩坑并补齐：torch_npu 的 transfer_to_npu 使 `torch.cuda.is_available()` 为 True，但 `get_device_properties(0).major` 为 None；sam3 共 5 处能力门判定（model_misc.get_sdpa_settings 导入期、sam3_multiplex_base 模块级、model_builder._setup_tf32 构建期等）逐一崩。逐处改不可维护，最终方案：在 sam3_server 的 NPU 兼容块后注入全局代理（major=None 时返回 major=6 的保守对象，全部能力门走 math/兼容路径，与非 CUDA 分支语义一致）。两份补丁：`patch_sdpa_none.py` + `patch_props_shim.py`（节点 `/data/rpent2/patches/`，本地 `rpent-repo/` 有同份；建议回馈上游仓库）。[E1]

### 4.2 RPent 主线代码漂移（9/22 → 9/24 HEAD）

- `robots/libero/robot_spec.py` env_overrides 仍硬编码 `"MUJOCO_GL": "egl"`——上游 sed 同款修法（改继承父环境），但注意文件需 `import os`（本次首打时踩坑）。
- `robots/libero/env_server.py` 已改为 `setdefault`（父环境导出 osmesa 即可，无需再 sed）。
- 记忆语料（`memory/libero`）**不在 git 内**：需从 HF `RLinf/RPent-memory` 下载或从同事 clone 复制（1.8M）。
- liberopro pip 包**不含 scenes 资产**：需 `liberopro-download-assets`（读 `LIBERO_PRO_ASSET_PATH`，把包内 assets 软链到资产树），否则 env.reset 直接 FileNotFoundError。[E1]

### 4.3 运维记录（诚实性）

- t9_s1 首次运行被监控端误杀（容器 UTC 与宿主机 +8 时差误读为"挂死"），按断点协议重跑一次并成功（527s）；重跑前后该集无其他干预。
- Mac↔节点 SSH 中断约 2 小时（Clash Verge 被切到全局模式导致 SSH 走代理节点不通；切回规则模式即恢复）。期间评测在容器内无人值守正常推进，`.done` 断点机制验证有效。

## 5. 复现步骤索引（工具类，不展开）

- 节点侧：`/data/rpent2/scripts-npunew/`（vla_boot / sam3_boot_npu / first_run / eval20_final / rerun_t9s1，均自上游脚本适配 ROOT 与卡号）、`/data/rpent2/patches/`（上游 4 补丁 + 本次 2 新增）、`/data/rpent2/status.sh`（进度一览）。
- 本地存档：`rpent-repo/`（上游仓库 clone + scripts-npunew + eval20-npunew 轻产物：20 集 mp4/run.log/recipe/analyze 输出，5MB）。
- 容器：`rpent-npu`（pi05-hybrid:v1，--privileged + ascend runtime + 64G shm + /data 挂载）。

## 6. 局限

- 20 集小样本，90% 点估计区间宽（Wilson 95% CI 70%–97%）；仅 1 套件（与上游局限一致，未测 libero_goal/Flash Mode/探索记忆）。
- 单节点单次运行；共享卡环境（dev5/6 虽无计算争用，但非专用）。
- GLM-5.3 采样非确定性：同 seed 不同运行结果可异（本次两败 vs 上游同集全胜为直接证据），单次 20 集不足以精确估计成功率差。

## 7. 结论

1. **RPent 复现的可复现性成立**：跨节点、跨镜像（同版本）、跨代码 HEAD、资产零迁移，成功率达 90%，LLM 用量轮廓逐项复刻（cache 命中率精确一致）。
2. **GLM-5.3 档位再证**：与 GPT-5.5（91%）持平样点，落在 5.5–Astra 档内；agent 循环对冻结 π0.5 的增益（17% → 90%+）继续成立。
3. **工程坑补丁可移植 + 增量**：上游三大 NPU 死锁补丁在新节点直接生效；本次新增的 props/major shim 为上游仓库缺口，已可回馈。
4. 失败模式与上游总结的失败前兆特征一致，"在线早停"信号在本环境同样可用。
