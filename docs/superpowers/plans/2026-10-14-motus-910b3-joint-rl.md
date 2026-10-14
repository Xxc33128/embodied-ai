# Motus 与 910B3 联合端到端 RL 执行方案

> v1.0 · 2026-10-14 · 仅方案；所有任务未执行，本文新增模块和命令接口均尚不存在。
>
> **For agentic workers:** REQUIRED SUB-SKILL: 实施时使用 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans`，按任务及门槛推进。勾选只依据实际产物和验证结果，不依据计划文本。

**Goal:** 在 910B3 上完成 task reward 同时更新 Motus VLM、video、action 指定参数的闭环，并用四臂对照区分联合更新收益与工程正确性。

**Architecture:** Motus 内部三模态融合；视频确定性 Euler 递推、动作 Gaussian 随机转移；更新时对固定动作链可微重放，首版使用 fresh on-policy REINFORCE。物理仿真与渲染运行在 CPU，完整模型训练运行在 NPU。

**Tech Stack:** PyTorch＋torch_npu、FSDP/HCCL、Motus、CPU MuJoCo/LIBERO/OSMesa、独立 Python 训练进程与环境进程；首版不接入 vLLM 文本生成服务或多框架调度。

**Spec:** [实验预注册与设计 §2](../../../weeks/2026_1012-1018_RPent多模型性能分析/2026-10-14-Motus与910B3联合端到端RL实验预注册.md#2-已确认设计与范围)。以下实现选择受该设计约束。

## 1. 全局约束

- 所有模型训练从第一版起就在 NPU 上进行。CPU 用于物理仿真、渲染、数据处理和小张量数值参考。
- 同一个 task-RL loss 必须使以下三个参数集合发生更新：VLM 内部 LoRA、参与控制的 video/WAN 参数、action 参数。Understanding adapter/expert 单独记录。
- 首版使用 REINFORCE＋前序批次 baseline，每批 fresh rollout 只进行一次累计后的 optimizer step。
- 第一版采用部分参数联合更新，不要求所有参数全量训练。
- 同卡分时 rollout/update；首版无跨物理动作块预测 KV cache、无 learned reward/imagined rollout、无多 epoch 旧数据 RL。
- 参考源码 clone 只读；未来实现使用独立 checkout，固定基准，不修改同事 fork。公开记录仓仅使用 `personal`，不推 `upstream`；本次不 push。
- 只使用独立实验中新采集或公开演示数据，不读取既有凭据、原始会话、受保护日志或带私发标记文档。公共 manifest 不含地址、身份或机器路径。
- 主机驱动/固件、现有容器与服务不改动。资源只用已分配卡；当前设备快照不能当作预留。
- E1=本次固定源码核查，E2=配置/设备快照，E3=文献/设计。本文所有拟新增接口、命令、阈值与预算均为 E3，尚未实跑。

## 2. 固定基准与执行目录

| 仓库 | 固定提交 | 用途 |
|---|---|---|
| Motus | `f771216802b8a1601599422f12088bee3c068c14` | 唯一模型实现起点；本轮核查工作树干净 |
| RLinf | `1c9eed00a60009d9b04337b85e4cc84d474e405e` | Ascend 环境、CPU LIBERO、flow RL 组织参考 |
| verl | `0d3f56a8980a55bfbbf1214ea54fa1b32ca1405c` | Qwen3-VL NPU/FSDP2 配置参考，不直接复用 token loss |

实施时在用户选定的独立实验位置建立 `source/Motus`、`data`、`weights`、`runs`、`evidence`，不把大权重/轨迹存入文档仓库。下文所有源码路径均相对 **该独立 Motus checkout**。每次新运行创建新的 run ID，已有输出拒绝覆盖；中断恢复只从本实验的版本化 checkpoint 读取。

`source_lock.json` 记录上述 commit、实现 commit、LIBERO/robosuite/MuJoCo 版本、BDDL 与权重文件 SHA256、模型库 revision、镜像 digest、Python/torch/torch_npu/CANN/HCCL/transformers/PEFT 版本。镜像 digest 和硬件拓扑必须在 G0 实测填写；在 G0 通过前不允许启动后续命令，缺失是阻断条件而非可忽略默认值。

## 3. 必须修改的现有代码

行号均以固定 Motus 提交为准；修改后依靠函数名和测试定位，不依赖旧行号。

| 文件与定位 | E1/E2 当前问题 | 拟改动与验收 |
|---|---|---|
| `models/motus.py`：`MotusConfig`、初始化 508/518/526、529 | CUDA placement；无条件冻结 Qwen | 显式 device/trainable config；在包装前注入并启用 VLM 内部 LoRA；冻结参数集合可审计 |
| 同文件 `UndModule.extract_und_features` 292–330 | language model 在 `no_grad()`；当前返回值无有效位 mask | 返回特征与有效位；允许 VLM 目标参数梯度；VAE/T5 冻结保持独立 |
| 同文件 `process_joint_attention` 206–280、253 | 三路双向 attention；understanding padding 被统一长度掩盖 | 组合三路有效位，禁止 padded key/value 影响有效动作；同样输入加 padding 后输出不变 |
| `bak/wan/modules/attention.py` 54/112；`bak/wan/modules/model.py` self/cross attention | 直接 CUDA FlashAttention，卸载扩展不会自动回退 | 增加显式 NPU 训练 backend；保留非 causal、缩放与 padding 语义；前反向对照 |
| `bak/wan/modules/model.py` 40–76、407 | float64/complex RoPE 与 CUDA autocast 路径 | 实数 sin/cos 等价实现，保留时间/归一化必要 FP32；按 shape 验证，不把静态出现 complex 当实测失败 |
| `models/motus.py` `training_step` 763、`inference_step` 906 | 两处重复三模态循环；现有方法不适合作 FSDP 外层入口 | 抽出共同单步 velocity 计算；新增 `forward(mode=...)`，训练统一经包装后的 forward |
| 同文件 `MotusConfig.__post_init__` 86、grid初始化 616–625 | H由帧数与ratio推导；grid按构造时batch_size展开 | 显式H=8与future_frames=4、ratio=2相互校验；grid从实际patch输出和当前B动态构造，覆盖B=1/尾批/dummy |
| `models/action_expert.py` 240、301；`models/und_expert.py` 91 | 四维 raw QKV Parameter＋einsum；部分 CUDA autocast | device 参数化；raw QKV 使用单独低秩增量，不能被 Linear-only LoRA 漏掉 |
| `models/motus.py` `load_pretrain_weights` 713–760；`configs/robotwin.yaml` 3–4 | 原14D/latent-action IO 与 LIBERO7D语义不同；原loader跳过整个action input_encoder/decoder | 新7D action IO、9D proprio IO；基于实际key/shape逐项迁移allowlist，禁止无报告的 `strict=False` |
| `data/lerobot/lerobot_dataset.py` 692–718、796–804 | 可用 action 填 state，并用 action min/max 归一化 state | 新 LIBERO loader 禁止该分支；独立 action/state stats，只由 train split 计算 |
| `train/train.py` 293–294、443–495、599–608 | unwrap 后调用方法；创建 config 未透传 frozen；旧 Accelerate/DeepSpeed 配方 | 保留旧 SFT入口兼容；新增联合训练入口用外层 FSDP.forward；不沿用 unwrap 训练调用 |

官方源码入口：[Motus 模型](https://github.com/StevenBSF/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/motus.py)、[WAN attention](https://github.com/StevenBSF/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/bak/wan/modules/attention.py)、[LeRobot loader](https://github.com/StevenBSF/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/data/lerobot/lerobot_dataset.py)、[训练入口](https://github.com/StevenBSF/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/train/train.py)。这些代码存在不等于新路径已可运行。

## 4. 新增文件与接口边界

| 拟新增文件 | 单一职责 |
|---|---|
| `motus_rl/contracts.py` | `Observation`、`Condition`、`SamplerConfig`、`ChunkRecord`、`EpisodeRecord` 字段与版本校验 |
| `motus_rl/backends.py` | NPU device/autocast、attention backend、实数 RoPE、分布式初始化 |
| `motus_rl/adapters.py` | LoRA/四维低秩参数化、唯一参数归属、冻结白名单、梯度与更新清单 |
| `motus_rl/model.py` | `JointPolicy.forward`：`sample`、`replay`、`sft` 三模式；所有可训练模块包含于同一包装模型 |
| `motus_rl/sampler.py` | 因果 Euler＋Gaussian；在模型 forward 内调用，不持有跨轮可训练 cache |
| `motus_rl/trajectory.py` | 原子落盘、校验和、版本/shape 验证、完整动作链、预算与恢复去重 |
| `motus_rl/env_adapter.py` | CPU LIBERO 生命周期、7D控制、9D proprio、两相机、物理步计数与奖励 |
| `motus_rl/dataset.py` | 演示转换、时间对齐、分组切分、独立归一化和 load report |
| `motus_rl/losses.py` | 完整链 log-prob、物理步 return、前序 baseline、REINFORCE、分项 SFT |
| `motus_rl/trainer.py` | T2先实现SFT所需分片加载/launcher；T5再补分时rollout/update、梯度累计、保存恢复 |
| `motus_rl/cli.py`、`motus_rl/gates.py` | `preflight`、`verify`、`prepare`、`sft`、`sampler-check`、`smoke`、`train`、`evaluate`；自动收集门槛证据 |
| `configs/libero_joint_rl_npu.yaml` | 下文候选协议与所有显式开关；不含私有路径或网络身份 |
| `tests/joint_rl/test_*.py` | 后述 T0–T7 的契约测试 |

核心调用约定：

```python
# 实施时的接口契约；不是当前已存在的 API。
# Observation: 两相机RGB、9D proprio、instruction、真实物理步号。
# Condition: 冻结VAE得到的当前帧latent、Qwen输入、固定T5 embedding及masks。
# SamplerConfig: K、sigma、tau_grid、H、D、精度、预处理版本。
# ChunkRecord: §7 的完整样本及版本信息。

class JointPolicy:
    def forward(self, mode, *, condition=None, sampler=None,
                record=None, sft_batch=None):
        if mode == "sample":
            return self.sample_chunk(condition, sampler)  # ChunkRecord
        if mode == "replay":
            return self.replay_chunk(record)  # logp[B], audit metrics
        if mode == "sft":
            return self.supervised_losses(sft_batch)  # action/video loss
        raise ValueError(f"unsupported mode: {mode}")
```

`sample_chunk` 和 `replay_chunk` 内部都调用同一个 `predict_joint_velocity(video_latent, action_latent, t_video, t_action, state, vlm_features, t5_embeddings, masks)`，返回两种 velocity。VLM 在整个 chunk forward 内只计算一次，循环的 understanding token 从同一初始特征开始，不把上一 denoise step 的修改继续缓存。`wrapped_model(mode="replay", record=record)` 是 FSDP 的训练入口；禁止 `unwrap_model(...).training_step(...)` 或在 wrapper 外保留可微 VLM 特征。

## 5. 首版配置与参数组

以下为候选默认值，在 dev/G1–G3 校准后冻结为一个版本；任何变化产生新协议版本和新 SFT/采样基线，不在正式对照中自适应挑选。

```yaml
experiment:
  version: motus-910b3-joint-rl-v1
  backend: npu
  algorithm: reinforce
  auxiliary_in_main_comparison: false
model:
  initialization: motus_stage2
  precision: bf16
  canonical_parameter_dtype: float32
  fsdp_param_dtype: bfloat16
  fsdp_reduce_dtype: float32
  latent_and_logprob_dtype: float32
  lora_rank: 8
  lora_alpha: 8
  lora_dropout: 0.0
  train_qwen_visual: false
  train_qwen_language_lora: true
  train_video_lora: true
  train_action_adapters: true
  train_understanding_adapters: true
  cross_chunk_cache: false
data:
  task_suite: libero_spatial
  task_index: 0
  action_dim: 7
  state_dim: 9
  camera_order: [agentview_image, robot0_eye_in_hand_image]
  camera_hw: [128, 128]
  wan_frame_hw: [128, 256]
  future_video_frames: 4
  video_action_freq_ratio: 2
  video_frame_physics_offsets: [0, 2, 4, 6, 8]
sampler:
  steps: 10
  tau_start: 1.0
  tau_end: 0.0
  sigma: 0.02
  horizon: 8
  execute_horizon: 8
  cfg_scale: 1.0
  dropout: 0.0
optimization:
  episodes_per_batch: 32
  updates_per_batch: 1
  learning_rate: 0.000001
  weight_decay: 0.0
  grad_clip_norm: 0.5
  gamma_physical_step: 1.0
  baseline: previous_batch_ema_per_task
  baseline_ema_decay: 0.9
  microbatch_chunks_per_rank: 1
  normalize_loss_by_path_dimension: true
budget:
  max_episode_steps: 200
  physical_steps_per_seed_per_arm: 25600
  train_seeds: [11, 23, 37]
  eval_initial_states: 100
```

`normalize_loss_by_path_dimension` 只把 **整体 loss** 乘固定 `1/(K×H×D)`；日志中的 log-prob 和后续 ratio 仍使用原始求和值。本例系数为 `1/560`，不是把平均 log-prob 当作路径概率。

配置映射必须显式：原 `num_video_frames=4` 表示未来帧数，输入clip实际为当前帧加4帧，共5帧；`video_action_freq_ratio=2` 得到 `action_chunk_size=8`，与sampler H一致。新入口独立接收H并验证三者关系。VAE的latent帧数、patch grid与有效空间尺寸从实际编码结果推导，不能把像素帧数直接当latent序列长度。

T5首版保留基座固定512-token上下文及 `context_lens=None` 的语义：预处理统一截断/补零到512，所有512位置参与原WAN cross-attention。Condition分别记录T5固定长度协议和Qwen实际有效位；不能声称已有T5 padding屏蔽。若未来引入T5真实length mask，需单独版本和SFT基线。Qwen/MoT可变长度padding必须按§3屏蔽。

参数组：

1. **VLM**：Qwen language attention/MLP 的 Linear LoRA。视觉 encoder 首版冻结，正文只能称“VLM 内部语言侧参数参与 RL”；若扩展视觉侧 LoRA，单独记录资源和梯度。
2. **World**：WAN attention/FFN 的 LoRA，以及经 G3 证实影响后续动作的 video head LoRA。VAE/T5 保持冻结。
3. **Action**：action expert Linear LoRA＋新7D IO；四维 `wan_action_qkv` 使用 `ΔW[k,n,d,e]=Σr A[d,r]B[k,n,r,e]` 的低秩增量。
4. **Understanding**：adapter/FFN LoRA；四维 `wan_und_qkv` 同样单独参数化。B/C/D均训练这组，名称和参数集合完全一致。

以上Linear LoRA均排除T2的冻结FP32计算island；不能用“匹配所有Linear”的规则重新注入这些模块。

按参数对象 identity 去重后产生 manifest：canonical name、shape、numel、group、requires_grad、optimizer membership。共享引用只能注入/包装一次。LoRA 常用零初始化会使部分因子首步梯度为零，验收要求每个预声明组有有效 task 梯度，而非每个张量都非零；另做非零初始化的小模型梯度检查。

10 步是现有配置的候选，50 步作确定性 dev 参考；原函数签名50步与配置10步不能混用。对 `sigma∈{0.01,0.02,0.05}` 做固定 dev 初态扰动检查，选最小能产生有效探索且不过度破坏基线的值；选定后所有臂一致。若10步不合格，先用50步协议或做有明确新版本的采样蒸馏；不能为省显存直接宣称2步可用。

## 6. 联合路径的数学与数值契约

Motus 的目标 velocity 为 `noise − clean`。定义下降时间轴 `tau[k]=1-k/K`，因此 `dt<0`。初始 `zv` 在第一次velocity计算前就替换当前观测条件帧；rollout/replay顺序一致。每步从同一 `(zv,za)` 起点得到 `vv,va`：

```python
# 位于可微 replay forward 内的核心关系；action_chain为固定采样值。
dt = tau[k + 1] - tau[k]
assert dt < 0
vv, va = predict_joint_velocity(zv, action_chain[k].detach(), tau[k])
mu = action_chain[k].detach().float() + dt * va.float()
logp_step = normal_log_prob(action_chain[k + 1].detach(), mu, sigma)
zv = zv + dt * vv.float()          # 不detach；用于下一个动作分布
zv = replace_condition_frame(zv, observed_frame_latent)
```

这里 `predict_joint_velocity` 的其余参数由同一 chunk 的 Condition 提供，`normal_log_prob` 按步骤、H、D全部求和；`replace_condition_frame` 使用非原地拼接/掩码替换，避免破坏 autograd 版本。采样时 `za_next=mu+sigma*epsilon`；重算时绝不根据保存的 `epsilon` 和新均值生成新 action chain。

完整 `logp = -0.5 * sum(((za_next-mu)/sigma)**2 + 2*log(sigma) + log(2*pi))`，sum覆盖K、H、D，sigma为固定标量标准差。不得把sigma误作方差，或把密度求和换成均值。

外生 `xi_video,xi_action` 的分布与参数无关，初始概率项在梯度和旧新 ratio 中抵消。定义 augmented action 为这些初始噪声及整条 action chain，则条件路径概率可计算。video 递推是内部确定性计算，必须从同一外生噪声按新参数重算，不需要额外的 video stochastic density。此方案改变了原始 deterministic sampler，不宣称保持原边缘分布。

为避免 BF16 把小 Gaussian 扰动量化为不同的采样机制，初始噪声、latent 状态、均值、随机加法和 log-prob 保持 FP32；大模型矩阵计算使用 BF16。所有未登记随机源关闭，重算使用与 rollout 相同的 module train/eval 行为。记录实际 tensor 值或稳定编码后的张量，不只记 RNG seed。

环境执行采用固定的反归一化和 action-space 限幅映射。保存未裁剪链，Gaussian density 不作用于限幅后的值；同时保存 `action_proposed` 与 `action_executed` 以及限幅比例。若模型输出包含可训练后处理，其概率定义须重新审查，首版禁止。

提前结束时仅累计实际物理步奖励，但 **完整 raw action chain 的概率项仍保留**：未执行尾部可能经 joint attention 影响已执行动作。不能简单按执行前缀裁掉 log-prob 维度。

## 7. 轨迹、return 和更新规则

`ChunkRecord` 至少保存以下字段，带 schema version 和 checksum：

| 类别 | 字段 |
|---|---|
| 身份 | run/task/episode/chunk ID、reset state ID、policy/adapter/normalizer/preprocessor/sampler 版本 |
| 观测 | 当前两路真实图像或无损引用、9D state、instruction、实际控制步号、Qwen/T5处理配置、条件mask |
| 生成 | `xi_video`、`xi_action`、`raw_action_chain[K+1,H,7]`、tau、sigma、完整 `old_logp`；调试批另存每步mu/logp/video摘要 |
| 执行 | proposed/executed actions、实际执行步数m、逐步reward、terminated/truncated/budget_truncated、success、下一真实观测 |
| 学习 | baseline版本、完整return、advantage、参数白名单hash、采集时模型精度与train/eval状态 |

video chain 可用于同权重重放对照，不能在新参数训练时当固定条件。共享真实图像按内容去重；不把同一张图复制到每个去噪步骤。

首版有限时域任务：首次成功的物理步 reward=1并结束；其他步0；200步任务时间上限记0并结束。总交互预算耗尽导致的未完成episode标记 `budget_truncated`，只计交互成本，不当作200步任务失败、不用于Monte Carlo更新。基础设施错误也记无效样本；恢复后只在剩余硬预算内重新采集，不补采到超过预算。最后一批只使用完整且版本一致的episode，按实际数量归一化；没有完整episode则跳过更新。

分别记录 `raw_attempted_steps`（包括中断、预算尾部和丢弃批）与 `valid_training_steps`（实际进入优化的完整episode步数）。25,600硬上限约束前者；有效数据量和丢弃原因均公开报告。基础设施错误导致丢弃步数超过硬预算的5%时，停止该run并标记基础设施失败；修复后可新建完整run，但失败run及其成本保留，不能混作一次成功训练。

```python
def chunk_returns(chunk_rewards, executed_lengths, gamma):
    # chunk_rewards[t]包含该chunk真正执行的逐物理步奖励
    result = [0.0] * len(chunk_rewards)
    future = 0.0
    for t in reversed(range(len(chunk_rewards))):
        assert len(chunk_rewards[t]) == executed_lengths[t]
        local = sum((gamma ** j) * r for j, r in enumerate(chunk_rewards[t]))
        future = local + gamma ** executed_lengths[t] * future
        result[t] = future
    return result
```

首版 `gamma=1`，目标为episode成功概率。若将来改为小于1，actor objective同时乘 `gamma**chunk_start_physical_step`，保持episode-start discounted目标；不能只修改return递推。

每批32个episode采样期间参数不变。baseline使用前一批按任务维护的return EMA，初值0；当前批计算advantage时固定baseline，更新完成后才更新EMA。对每episode内部各chunk求和，再除以batch episode数；梯度累计不得按可变chunk数改变episode权重。

```python
# 概念式：replayed_logp与G、baseline均按episode/chunk对齐。
# sg表示停止梯度；phase_scale=1/(K*H*D)固定，不能进logp指数。
actor_loss = -phase_scale * sum(
    sum(sg(G - baseline) * logp for G, logp in episode)
    for episode in batch
) / len(batch)
```

每批最多 `optimizer.step()` 一次，之后丢弃该批的RL优化资格；实际更新后才递增policy版本。全批advantage精确为零时标记 `no_signal` 并跳过optimizer/scheduler step，参数、optimizer step计数与policy版本保持不变，防止已有Adam动量导致额外更新；baseline仍在批结束后更新。另用全新optimizer检查零优势梯度。非零baseline下全失败批可能仍有有效负advantage，不能仅凭“本批无成功”跳过。首版无critic、无KL reference、无多epoch PPO，因而不要求常驻旧模型；`old_logp`仍保存用于一致性验收与后续扩展。

## 8. T0 资源与软件锁定（G0）

**Files:** 新建 `motus_rl/cli.py` 的 preflight入口、`tests/joint_rl/test_preflight.py`；产物 `evidence/preflight.json`、`source_lock.json`，只存于新实验目录。

**Consumes:** 固定源码、已分配卡号、可读的官方驱动/容器兼容表。**Produces:** 经过人工复核的 `preflight_pass=true` 与 immutable软件/权重清单。

- [ ] 只读核查已分配卡、HBM、拓扑、CPU/RAM/磁盘、主机驱动/固件；不操作其他进程。
- [ ] 创建独立训练环境。优先验证 RLinf 参考的 CANN9.0/910b 路线；原服务CANN8.5.2不升级。若主机不兼容9.0，记录阻断并重新选择官方兼容组合，不能以替换Python包绕过。
- [ ] 固定 PyTorch/torch_npu 匹配版本、HCCL 与 transformers。Motus pin 的 `transformers==5.0.0rc0` 和 Qwen DeepStack内部API需单独验证；已有Qwen3-VL训练示例不能代替本版本兼容性。
- [ ] 测试无NPU、版本缺失、设备不可用、输出已存在时 preflight拒绝继续；通过后写入锁定清单。
- [ ] 保存环境变更diff；提交实现代码，不提交权重/原始节点快照。

拟定验证命令：`python -m pytest tests/joint_rl/test_preflight.py -q`。预期：缺项必须失败，完整fixture成功；真实 preflight证据与fixture测试分开。

## 9. T1 NPU backend 与数值基线（G1）

**Files:** 修改§3的device/attention/RoPE路径；新建 `motus_rl/backends.py`、`tests/joint_rl/test_attention.py`、`test_rope.py`、`test_numerics.py`。

**Consumes:** G0软件锁。**Produces:** `attention(q,k,v,valid_mask)` 的NPU前反向；实数RoPE；`numerics_protocol.json`。

- [ ] 先写小张量CPU FP32参考测试：非causal三路attention，带padding与不同文本长度；确认masked token改变不影响有效输出。
- [ ] 替换真正的WAN self/cross调用点；先验证数学等价路径，再引入支持训练反向的NPU融合实现，不能误用仅推理算子。
- [ ] 用实数旋转复现原RoPE，比较输出、输入梯度和权重梯度；保留time embedding/AdaLN必要FP32。
- [ ] 1卡NPU运行固定小张量的FP32/BF16检查与10次重复计算，覆盖实际head dim、mask及多模态拼接shape。
- [ ] 冻结容差协议；正式测试失败后不得扩大阈值。提交backend改动及测试。

定义 `nrmse=||candidate-reference||₂/(||reference||₂+eps)`，同时报告最大绝对误差和gradient cosine。算法梯度CPU float64的中心差分使用`epsilon=1e-5`、`rtol=1e-4,atol=1e-6`。NPU算子候选上限为：FP32 nrmse≤1e-3、BF16 nrmse≤5e-2，非近零梯度cosine≥0.99；这些是本实验容许误差上限，不是硬件官方指标。先依据10次重复噪声确定更严格的shape专用阈值，再冻结；超过上限则修实现/精度，不能据实测任意放宽。

完整链logp重放另设维度尺度：报告`abs(delta_logp)/(KHD)`和`abs(delta_logp)`，正式门槛要求两者分别≤1e-5和1e-2，且`abs(expm1(delta_logp))≤0.011`。若BF16重复运行达不到，保持FP32概率、关闭非确定性路径或阻断G3；不能只因每维误差小而忽略联合ratio偏差。

## 10. T2 可导模型与参数白名单（G1/G3前置）

**Files:** 修改 `models/motus.py`、两expert文件；新建 `motus_rl/adapters.py`、`motus_rl/model.py`、`motus_rl/trainer.py` 的最小分片加载/launcher、`tests/joint_rl/test_joint_forward.py`、`test_parameter_groups.py`、`test_sharded_sft.py`。

**Consumes:** backend、现有Motus权重布局。**Produces:** §4包装forward、四组唯一trainable manifest。

- [ ] 写参数冻结/no_grad、mask、重复参数引用测试，使旧代码暴露预期缺口。
- [ ] 配置真正控制冻结；VLM内部目标LoRA前向可导，`lm_head`/VAE/T5不进入optimizer。
- [ ] 抽取单步joint velocity共同实现，SFT与采样使用相同attention、mask、时间与head逻辑。
- [ ] 实现四维QKV低秩增量并验证r=0/禁用adapter时与基座输出一致；按对象ID去重注入和参数组。
- [ ] 在小模型中检查padding不变性、三个主组可达性与冻结负对照；记录未使用参数，不能靠`find_unused_parameters`掩盖梯度阻断。提交模型接口。
- [ ] 提前实现SFT所需的FSDP/HCCL加载和forward/backward：CPU构造后注入并去重adapter、固定包装边界、使用包装forward、在正确阶段创建optimizer。首版候选FSDP1 `FULL_SHARD,use_orig_params=True`；只使用G0已分配卡。若改FSDP2，先完成同一契约测试并记录协议版本。
- [ ] 第一正确性版仅对 `JointPolicy` 顶层FSDP包装，禁止直接auto-wrap现有WAN block：joint代码通过属性访问norm/QKV/FFN，未调用该block.forward，嵌套包装可能绕过all-gather。VLM/WAN/action/understanding由JointPolicy的canonical registry各拥有一次；helper不再重复注册相同子模块。raw QKV增量归所属expert，包装后重新核验唯一参数和optimizer归属。后续若需逐层分片，先抽取完整三模态 `JointLayer.forward` 再以它为嵌套边界，并通过同样数值测试。
- [ ] 包装前主体canonical参数统一FP32（含LoRA），避免同一flat parameter混合原始BF16/FP32；候选 `MixedPrecision(param_dtype=bf16, reduce_dtype=float32, buffer_dtype=None)`。需保留FP32权重计算的time embedding/projection等小模块先列完整清单，首版冻结并通过FSDP的ignored module/state机制复制保留；它们不属于trainable白名单。禁止盲目把全模型 `.bfloat16()`。探针核对canonical、all-gather后、实际算子输入、gradient及optimizer state的dtype；若任何ignored参数可训练，必须失败，不能漏做跨rank同步。NPU上该混合精度组合通过G2前不能视为可用。
- [ ] FP32 island初始清单为 `action_expert.time_embedding`、`action_expert.time_projection`、`video_model.wan_model.time_embedding`、`video_model.wan_model.time_projection`；按实际禁用autocast且接收FP32输入的有权重调用逐项复核。仅需FP32统计的normalization不能一概ignored。测试必须证明：所有受管理canonical参数为FP32、所有ignored参数冻结且未注入LoRA、所有trainable参数同时被FSDP管理并进入optimizer。`use_orig_params=True`不会免除flat parameter统一dtype要求〔E1，[PyTorch 2.7.1检查](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/distributed/fsdp/_flat_param.py#L727)〕。
- [ ] 用小模型验证单/双卡SFT梯度及更新；再在2卡完整模型上验证一次SFT前反向、保存/加载和峰值，必要时申请4卡。此处不要求RL链已完成；后续G4增加RL重放、累计与恢复验收，避免SFT依赖尚不存在的G4。

契约测试示例：

```python
def test_parameter_manifest(policy):
    manifest = policy.parameter_manifest()
    ids = [row["object_id"] for row in manifest]
    assert len(ids) == len(set(ids))
    for group in ("vlm", "world", "action", "understanding"):
        assert any(row["group"] == group and row["trainable"] for row in manifest)
    assert not any(row["trainable"] and row["role"] in
                   ("vae", "t5", "unused_lm_head") for row in manifest)
```

## 11. T3 LIBERO接口、数据和联合SFT（G2）

**Files:** 新建 `motus_rl/env_adapter.py`、`dataset.py`；修改Stage2加载入口；新增 `configs/libero_joint_rl_npu.yaml`、`tests/joint_rl/test_libero_contract.py`、`test_data_split.py`、`test_checkpoint_mapping.py`。

**Consumes:** 固定LIBERO版本与公开demonstrations、T2模型接口和最小分片执行器。**Produces:** `dataset_manifest.json`、`normalization.json`、`weight_load_report.json`、可用SFT checkpoint。

- [ ] 固定 LIBERO-Spatial task0的实际任务名/BDDL；初态按完整状态hash和来源episode分组分配train/dev/eval。禁止相邻帧、同初态重跑跨split。
- [ ] CPU OSMesa依次验证两相机方向、RGB范围、平移单轴、旋转分量与夹爪符号。动作顺序由controller实现确认；正常动作长度必须为7。
- [ ] state固定为7个关节位置＋2个夹爪位置；环境字段缺失或shape不同直接失败。state/action分别计算train split统计，不用action代state。
- [ ] 首版SFT只保留拥有完整8步动作及当前/动作后第2/4/6/8步5帧的clip；丢弃不足长度的尾clip并分split报告数量，不补末帧。验证VAE时间压缩、patch grid和动作时间戳；加入B=1/尾batch/dummy形状测试。未来支持短clip时必须另做VAE感受野到latent mask映射和独立action mask，不把像素frame mask直接广播至latent loss。此规则只用于演示SFT；RL的提前结束仍按完整生成链处理。
- [ ] Stage2加载时精确跳过新action/state IO及shape不兼容项，逐个列出原因；未预声明的missing/unexpected key均失败。原RoboTwin14D权重不直接截断。
- [ ] 在NPU做联合SFT暖启动：三个主组及understanding均可训练，学习率候选1e-5、microbatch1、global batch8、最多2000更新步，每200步dev检查；此预算是停止上限而非收敛保证。
- [ ] 在20个dev初态检查确定性50步与10步采样。至少5次成功且不过度饱和才进入RL预试验；不足则修接口/数据或扩大SFT预算并生成新版本。不得从eval集挑任务或参数。
- [ ] dev校准sigma；若随机基线相对相同K的确定性基线下降超过10个百分点或异常限幅超过1%，不进入正式比较，调整噪声/暖启动并重新冻结协议。20集判据仅作工程筛查。
- [ ] 在首次RL前冻结H2的held-out完整clip清单、每clip四个固定噪声draw及均匀时间点（由独立固定seed生成）。动作输入为held-out真实动作按固定噪声和时间加噪的值；观测/VLM条件、video/action噪声、两路时间和mask对所有checkpoint一致，不使用各模型自行生成的动作。video velocity MSE仅在未来预测latent位置计算，排除当前观测条件帧及padding；action单独报告。按有效维度、episode两级聚合，不能把训练数据或大量相关帧冒充独立held-out episode。

新归一化采用每维train split分位数范围（p1/p99）并显式处理常量维，state使用独立均值/标准差；具体公式与夹爪离散映射写入normalizer版本。控制器原有动作缩放只能执行一次。所有新IO从固定seed初始化并在SFT训练。

## 12. T4 路径采样、重放与纯RL梯度（G3）

**Files:** 新建 `motus_rl/contracts.py`、`sampler.py`、`trajectory.py`、`losses.py`、`tests/joint_rl/test_path_replay.py`、`test_score_gradient.py`、`test_returns.py`、`test_gradient_routes.py`。

**Consumes:** SFTcheckpoint、固定sampler/normalizer。**Produces:**可重放ChunkRecord、纯任务loss的三模块梯度证据。

- [ ] 实现§6–§7的因果采样与重放；先用3个标量参数分别模拟VLM、video、action验证，再接完整模型。
- [ ] 固定同一个action chain比较rollout/replay每步mu、video状态、logp；篡改policy/sampler/normalizer版本必须被拒绝。
- [ ] 对固定样本logp做中心差分；在有解析期望的Gaussian控制玩具上，多次采样验证score-function估计的均值和置信区间，不能只检查梯度非零。
- [ ] 关闭BC/FM，检查四组optimizer成员、有限梯度及更新delta。VLM特征detach后检查VLM内部LoRA梯度消失；跨步video状态detach后检查仅经该递推边影响动作的video head梯度消失，不能要求仍通过同一步joint attention影响动作的全部world参数梯度消失。另做world组冻结后的参数不变对照。
- [ ] 测试提前成功、200步截断、预算尾部截断、零advantage、raw chain与clipped执行映射；完整路径项保留，reward只来自实际步。
- [ ] 将数值、mask、梯度与probability验收结果写入G3 evidence；提交算法实现。

```python
def test_physical_discount():
    assert chunk_returns([[0.0, 1.0], [1.0]], [2, 1], 0.5) == [0.75, 1.0]

def test_single_sample_logprob_grad(logp_at, theta, analytic):
    eps = 1e-5
    numeric = (logp_at(theta + eps) - logp_at(theta - eps)) / (2 * eps)
    assert abs(numeric - analytic) <= 1e-6 + 1e-4 * abs(numeric)
```

`logp_at`在同一噪声和固定action链上计算；Monte Carlo期望梯度测试是另一个测试，不用这个局部差分替代。video head的梯度检查至少K≥2，以覆盖视频递推影响后续动作的路径；末步单独不影响动作的video输出不要求task梯度。

## 13. T5 FSDP、完整更新与恢复（G4）

**Files:** 新建 `motus_rl/trainer.py`、checkpoint逻辑；`tests/joint_rl/test_distributed_update.py`、`test_resume.py`、`test_memory_lifecycle.py`。

**Consumes:** G3单模型算法、参数manifest、已分配资源。**Produces:**外层包装联合forward、一次真实更新、恢复一致性与每rank资源报告。

- [ ] 复用T2已验证的加载、包装和optimizer构造顺序；新增完整RL依赖图的累计与恢复。沿用同一个FSDP实现，不在同一结果里混用FSDP1/2。
- [ ] 所有更新走 `wrapped_model(mode="replay",...)`；采样也使用一致的包装权重版本。真实观测的冻结VAE特征可缓存，VLM可微特征与视频预测不能缓存为脱图条件。
- [ ] 每rank1个chunk microbatch，按global episode总数归一化、完整batch累计后至多一次step。默认跨rank梯度取均值时，本地loss为 `L_rank = -world_size * phase_scale / global_episode_count * sum(local_advantage * local_logp)`；若明确使用SUM reduction则去掉world_size因子。各rank执行相同collective次数；用dummy零权重chunk对齐尾部，不把dummy计入episode数。单/多卡等价测试覆盖不等长episode与dummy尾部。
- [ ] 用activation checkpointing重算完整video依赖图，不截断梯度；反传后释放graph。按当前窗口逐步增加2/4/8卡，不能用detach作为OOM处理。
- [ ] 保存base身份、adapter、optimizer、scheduler、各rank RNG、global step、policy_version、baseline状态、数据游标与累计物理步。恢复必须拒绝配置或normalizer不匹配。
- [ ] 只在batch边界提交checkpoint；独立原子成本日志逐chunk记录实际步数。batch中断时未提交批不更新，恢复从剩余硬预算采集新批，不重复消费已完成batch，不把丢失优化资格的步数从成本中抹去。另测试完整环境snapshot/restore，但首版不依赖任意时刻中断后的无缝物理继续。
- [ ] 提交分布式与恢复实现，保存memory证据。

G4数值比较采用固定global batch与相同记录，比较小模型单卡与2/4卡梯度及更新；全模型单卡装不下时不能声称做了全模型单卡对照。至少连续10次固定shape更新无OOM/NaN；稳定阶段reserved峰值不得呈持续增长，最后5轮相对前5轮增长超过5%需解释或阻断。每卡峰值不超过可分配HBM的85%，余量用于波动和保存；这些是工程预留阈值，不是吞吐指标。

G4反复使用固定记录仅作数值/内存工程探针，使用单独checkpoint副本与固定测试advantage，不作为on-policy学习结果；其权重不得进入G5/G6。正式各臂仍从同一个SFT checkpoint重新加载。

## 14. T6 闭环烟测与T7四臂对照（G5/G6）

**Files:** 完成 `motus_rl/cli.py`、`trainer.py`、`evaluate`入口；新增 `tests/joint_rl/test_rollout_lifecycle.py`、`test_budget.py`、`test_eval_manifest.py`。

**Consumes:** G0–G4通过、冻结SFT与采样协议。**Produces:**烟测证据；之后才产出四臂统计。

- [ ] 1seed、32条训练episode完成一次更新，再做1条更新后验证rollout（不参与优化、成本单列），合计33条episode上限。训练batch须有有效任务信号；若全零advantage则报告G5无信号、不得用辅助loss代替通过。检查验证rollout使用新policy版本、invalid episode单列、预算不重复。
- [ ] 从同一SFT checkpoint创建A/B/C/D；B/C/D三个独立训练seeds。各臂使用相同网络forward与固定sampler，只有trainable白名单不同。
- [ ] 执行预注册的25,600实际物理步硬预算；记录有效训练步、优化次数、有效/无信号batch数，最后不足32条完整episode的批按实际数量归一化。各臂不强求优化次数相同；不得因某臂成功快而减少其交互预算。
- [ ] 使用100个冻结未见初态评估，所有臂匹配动作随机seed；评估全程无optimizer update，成功predicate不进入策略输入。
- [ ] 报每seed成功率、paired结果、跨seed离散度、恢复/失败类型、NPU-hours、实际物理步、CPU渲染等待、峰值HBM、step时间。
- [ ] 按T3冻结的held-out监督评估输入计算video/action velocity MSE；video仅计算未来预测latent位置，按有效维度归一化、先episode内平均再episode间平均。H2以D/A的video MSE比值为主，逐seed报告，超过1.10视为该固定指标超出预设可接受退化范围；不是世界建模增强证明。提交脱敏结果与完整协议，不筛选只展示最佳seed。

25,600步是首轮探索预算。若多数episode跑满200步，它仅覆盖约128集、4次batch更新；无收益不能据此排除联合RL，也不能声称已收敛。G6先回答工程成本和初步优化方向，确认性效果实验需据实测更新曲线/吞吐另行预注册更大步数预算及至少5个训练seeds。

## 15. 资源申请与估算

资源事实来自[公开设备快照](../../../weeks/2026_1012-1018_RPent多模型性能分析/910b3-baseline32/environment-observed-20261013.json)。原服务软件为CANN8.5.2、torch2.7.1+cpu、torch_npu2.7.1.post2；这是旧服务身份，不是新训练兼容性证明。RLinf的CANN9.0路线仅为候选。

| 阶段 | NPU申请 | CPU/RAM/磁盘建议 | 目的 |
|---|---|---|---|
| 算子/玩具检查 | 1张已分配910B3 | 8 CPU核、32GiB RAM、20GiB空间 | 小张量与数据探针；不要求完整8B装入 |
| 完整SFT/联合更新 | 2张起步，失败后按证据申请4张 | 16–32 CPU核、256GiB RAM、至少300GiB空闲NVMe作为申请起点 | 分片加载、公开权重、预处理、轨迹和checkpoint；须实测host峰值 |
| 正式目标配置 | 必要时4/8张，明确独占时段 | CPU worker从2起测，不默认堆满；磁盘配额按首批轨迹外推 | 同卡分时rollout/update，优先正确性与完整预算 |

上述host资源是规划值，不是已经确认的节点配置或理论最低值。G0若不足，先降低并发和分批预处理/加载；不能改研究输入或自动挪用其他任务资源。

Motus参数表约为5.00B＋2.13B＋0.6415B＋0.2535B＝8.025B〔E3，官方近似值〕。BF16主体权重约16.05GB＝14.95GiB。理想2/4/8卡完全分片时仅该项约7.47/3.74/1.87GiB每卡，**不表示总训练内存**。

设分片主体的冻结参数`Pf`、训练参数`Pt`。本版候选使用FP32 canonical参数与梯度、FP32 Adam m/v，理想静态状态为 `M_state = 4*Pf + 16*Pt` bytes；FSDP生成的BF16权重、all-gather和额外buffer另算，不再把FP32 canonical又重复计作一份master。主体全部FP32存储约32.10GB＝29.90GiB（均摊前），忽略此项会低估常驻状态。加载后按唯一numel计数，不能由rank猜LoRA参数量。VAE/T5、ignored FP32小模块及其他复制模块另计。若实际optimizer/梯度dtype不同，按实测tensor清单重算，不套用该公式。

`M_peak(rank) = sharded_state + replicated_state + live_allgather + activations + video_replay_graph + workspace/comm/allocator`。

首版top-level FSDP在forward期间可能all-gather整个主体，峰值需计入约14.95GiB的完整BF16主体权重，而不能只计均摊项；混合冻结/可训练参数还可能放大梯度buffer。增加卡数未必解决这种峰值，G2/G4若被它阻断，需先实现经forward调用的JointLayer分片并复验，不能直接按卡数线性外推。

本方案video重放有跨去噪步的参数依赖，activation checkpointing可以重算，不能把这些状态全detach后仍声称同一目标。每个候选配置分别测加载、rollout、replay forward、backward、optimizer、checkpoint的allocated/reserved峰值。原始chain存CPU/磁盘，搬运开销另记。

暂不承诺小时数或steps/s。G5测得中位episode采样、每chunk重放、更新和评估成本后，使用`NPU-hours=Σ(占用卡数×占用秒数)/3600`据实申请G6时段，包含空等与checkpoint；CPU环境步与NPU吞吐分别报告。

## 16. 实施后的命令与门槛顺序

以下命令是 **T0–T7需要实现的CLI验收接口**，当前不能直接运行。本轮不创建训练脚本、不执行这些命令。CLI必须校验上游门槛文件、source/config hash及输出目录不存在；缺项返回非零退出码。

`verify`负责运行对应测试并收集真实报告；不得由人手填pass。每个输出目录包含原子写入的 `gate_pass.json`，字段至少为gate ID、status、source/config/weight/data hash、upstream证据hash、测试名与结果、产物相对路径及校验和、时间和已分配资源。失败也写报告，但没有通过标记。代码、输入协议或初始化权重改变后，旧证据自动失效，需从受影响的最早门槛复验。G3/G4必须覆盖B/C/D三套预声明参数白名单，保存各自manifest hash；`--arm`只选择已经验证的清单，不能用临时开关绕过证据校验。

| 唯一生产入口 | 输出 | 检查内容 |
|---|---|---|
| `preflight` | `evidence/g0/` | T0真实软件/资源检查与source lock |
| `verify --gate g1` | `evidence/g1/` | T1 attention/RoPE/mask数值与固定容差 |
| `verify --gate g2` | `evidence/g2/` | T2/T3模型、最小分片SFT测试、数据/权重映射、SFT dev和采样审计 |
| `verify --gate g3` | `evidence/g3/` | T4概率重放、梯度路径、return/预算边界测试 |
| `verify --gate g4` | `evidence/g4/` | T5完整RL更新探针、单/多卡等价、恢复和内存 |
| `smoke --evidence-output` | `evidence/g5/` | T6真实任务32训练episode＋1更新后验证rollout |
| `verify --gate g6` | `evidence/g6/` | T7全部run list、评估、预算和统计完整性；不以D获胜为通过条件 |

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
python -m motus_rl.cli preflight --config configs/libero_joint_rl_npu.yaml --output evidence/g0
python -m motus_rl.cli verify --gate g1 --config configs/libero_joint_rl_npu.yaml --evidence evidence/g0 --output evidence/g1
python -m motus_rl.cli prepare --config configs/libero_joint_rl_npu.yaml --evidence evidence/g1 --output data/libero_v1
python -m motus_rl.cli sft --config configs/libero_joint_rl_npu.yaml --data data/libero_v1 --evidence evidence/g1 --output runs/sft_v1
python -m motus_rl.cli sampler-check --config configs/libero_joint_rl_npu.yaml --checkpoint runs/sft_v1 --data data/libero_v1 --evidence evidence/g1 --output evidence/sampler_v1
python -m motus_rl.cli verify --gate g2 --config configs/libero_joint_rl_npu.yaml --checkpoint runs/sft_v1 --data data/libero_v1 --sampler-audit evidence/sampler_v1 --evidence evidence/g1 --output evidence/g2
python -m motus_rl.cli verify --gate g3 --config configs/libero_joint_rl_npu.yaml --checkpoint runs/sft_v1 --evidence evidence/g2 --output evidence/g3
python -m motus_rl.cli verify --gate g4 --config configs/libero_joint_rl_npu.yaml --checkpoint runs/sft_v1 --evidence evidence/g3 --output evidence/g4
python -m motus_rl.cli smoke --config configs/libero_joint_rl_npu.yaml --checkpoint runs/sft_v1 --data data/libero_v1 --evidence evidence/g4 --output runs/smoke_v1 --evidence-output evidence/g5
python -m motus_rl.cli train --config configs/libero_joint_rl_npu.yaml --checkpoint runs/sft_v1 --data data/libero_v1 --arm D --seed 11 --evidence evidence/g5 --output runs/D_seed11_v1
python -m motus_rl.cli evaluate --config configs/libero_joint_rl_npu.yaml --run runs/D_seed11_v1 --manifest data/libero_v1/eval_manifest.json --output evidence/D_seed11_eval
```

`sft`在第一次优化前强制运行T2的加载/参数/分片前反向测试，失败不进入SFT循环。分布式启动参数由G0已分配卡数与T2的launcher封装处理；不能凭示例默认访问全部设备。G2引用冻结后的SFT checkpoint与采样协议，不能使自身成为SFT启动的前置条件。

上述train/evaluate展示D/seed11一项。`prepare`同时生成预注册的 `run_list.json`：B/C/D×{11,23,37}共9个训练run，以及A/B/C/D×3个评估随机seed共12个评估run；A通过evaluate的 `--checkpoint runs/sft_v1 --arm A --seed ...` 接口直接评估。逐项完成后运行 `python -m motus_rl.cli verify --gate g6 --config configs/libero_joint_rl_npu.yaml --run-list data/libero_v1/run_list.json --evidence evidence/g5 --output evidence/g6`。失败run保留，禁止同名覆盖；正式各臂从SFT重新开始，不继承G4/G5更新后的权重。

## 17. 验收证据与最终交付

| 证据文件 | 必含内容 | 支持的结论 |
|---|---|---|
| `source_lock.json`、`preflight.json` | 软件/权重/卡分配及兼容检查 | 执行环境明确 |
| `numerics_protocol.json`、`numerics_results.json` | 固定输入、容差、前反向误差、重复噪声 | 已测shape数值可用 |
| `weight_load_report.json`、`dataset_manifest.json` | 迁移allowlist、动作/相机/时间轴与split | 目标任务适配可追溯 |
| `parameter_manifest.json`、`rl_gradient_audit.json` | 四组唯一参数、RL-only梯度、负对照、更新delta | 三模块联合任务梯度成立 |
| `replay_audit.json` | raw chain、video重算、old/new logp、ratio | 采样与概率重算一致 |
| `memory.jsonl`、`resume_audit.json` | 每rank阶段峰值、保存恢复、版本/预算去重 | 指定配置训练工程可用 |
| `eval_results.jsonl`、`comparison.json` | 全部臂/seed/初态及失败、区间、资源账 | 仅支持实际完成范围的效果判断 |

NPU实跑证据保存在独立实验位置；原始设备身份/路径不入公开仓。发表时只同步脱敏汇总、配置、源码commit、必要统计和批准的视觉样例。不能把本方案代码块的语法检查写成模型训练通过。

## 18. 文献与扩展边界

- [Stochastic Computation Graphs](https://arxiv.org/abs/1506.05254)：混合确定性计算与随机节点的梯度依据。
- [ReinFlow](https://arxiv.org/html/2505.22094v2)：路径概率与flow策略梯度；不是Motus/NPU兼容性证据。
- [πRL](https://arxiv.org/abs/2510.25889)：Flow-Noise/Flow-SDE参考；第一版固定Gaussian构造不冒充其原配方。
- [Motus2](https://arxiv.org/html/2608.30237v2)：其MBRL冻结video骨干，只更新动作相关参数；不能作为三模块joint RL已验证的依据。
- [Ascend FSDP官方样例](https://ascend.github.io/docs/sources/pytorch/examples.html)：NPU分布式基础支持；具体Motus共享引用、混合冻结和反向仍须测试。

G6后才考虑PPO多epoch、DiffusionNFT、learned noise、视频辅助损失、imagined rollout和异步调度。每一项改变单独版本、独立对照，不纳入第一版完成条件。
