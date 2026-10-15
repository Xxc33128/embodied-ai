# Motus 联合训练的 VLM 与 Wan 模型选型核查

> v1.0 · 2026-10-15 · 公开资料与固定源码核查。适用目标为 910B3 上的 VLM、video/WAN、action 联合 task-RL；不以视频观感或通用问答榜单替代控制实验。

**Wan 已有 3.0 服务，2026 年也新增了开放权重的角色动画与音乐舞蹈分支；目前确认可下载、适合继续评估通用视频骨干的主线仍为 Wan2.2。VLM 候选应补齐已开放的 Qwen3.8-27B。首轮保留 Motus 原生组合的依据是已有机器人预训练与内部融合接口，更新模型是否更适合联合 RL 仍要用共同协议比较。**〔产品与能力文档/E3；权重目录与配置/E2；接口/E1；选型判断/E3〕

研究目标与既有预算见[实验预注册](2026-10-14-Motus与910B3联合端到端RL实验预注册.md)，实现门槛见[联合 RL 执行方案](../../docs/superpowers/plans/2026-10-14-motus-910b3-joint-rl.md)。本核查提供候选与适配依据，不修改其中已经固定的实验臂和输入协议。

## 1 证据口径与检索范围

- **E1 源码**：固定提交的实现、接口或训练脚本；只支持“代码中存在该路径”，不等于在目标设备运行成功。
- **E2 配置与快照**：官方模型配置、公开权重目录及 revision；不等于完整训练兼容性。
- **E3 文档与推断**：作者论文、厂商公布的评测、产品说明，以及明确标记的实验设计判断；公布分数不是本工作区复现结果。

Wan 检索覆盖 Wan-Video 官方源码组织、Wan-AI 官方模型目录、作者链接的 Wan-Move、Wan-Streamer，以及阿里云产品文档。VLM 覆盖与当前方案相关的 Qwen3-VL、Qwen3.5 至 3.8、SpatioLM、GR3D 和 Qwen-3D；不把它写成全市场穷尽排名。公开 revision 与 Wan-AI 27 个模型的目录快照保存在 [source-manifest.json](motus-model-selection-20261015/source-manifest.json)。`last_modified` 是更新时间，不能当作首次发布日期。〔E2/E3〕

## 2 Wan 新版本与开放状态

| 模型或分支 | 已确认的发布物与用途 | 对本实验的判断 |
|---|---|---|
| Wan2.1 T2V 1.3B、T2V/I2V 14B | 官方开放权重与实现；较早的视频生成主线。[官方仓库](https://github.com/Wan-Video/Wan2.1)〔E3；目录/E2〕 | 可参考训练和 NPU 迁移经验；替换 Motus 仍需重新适配。〔E3〕 |
| Wan2.2-TI2V-5B | 5B dense，兼容文本/图像条件；官方公告于 2025-07-28 开放权重与推理代码。[固定模型说明](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/README.md)〔E3；权重/E2〕 | 当前 Motus 基础视频架构，但应加载 Motus 领域适配权重，而非随意换回原始 Wan。〔E1/E3〕 |
| Wan2.2-I2V-A14B、T2V-A14B | 开放权重；高噪声和低噪声两个专家，合计约 27B、每步约 14B 活跃。[官方架构说明](https://github.com/Wan-Video/Wan2.2/blob/1ea34ff48f87168174e12956e200b1d908b1c5ff/README.md#introduction-of-wan22)〔E3〕 | 更大容量候选；需要改 joint attention、专家切换与权重映射，不能视为 5B 的同接口升级。〔E1/E2/E3〕 |
| Wan2.2-S2V-14B、Animate-14B | 官方目录中有开放权重，主要面向音频驱动或角色动画。[官方目录](https://huggingface.co/Wan-AI)〔E2/E3〕 | 条件与任务发生变化，不作为首轮机器人视频骨干优先候选。〔E3〕 |
| Wan-Dancer-14B | 官方模型卡公告 2026-07-13 发布权重和推理代码，面向音乐到长时舞蹈视频。[官方模型卡](https://huggingface.co/Wan-AI/Wan-Dancer-14B)〔E3；权重/E2〕 | 开放的新分支；舞蹈节奏与音乐条件不构成机器人动作条件建模的证据。〔E3〕 |
| Wan-Animate-2 及蒸馏版 | 官方公告 2026-08-07 发布基础及蒸馏权重、推理代码；直接使用 driving video 做角色动画。蒸馏示例为 10 步、无 CFG。[官方仓库](https://github.com/Wan-Video/Wan-Animate-2/blob/3ad2fef7d61d6200c9c653e0fe47be7616b323f3/README.md)〔E3；权重/E2〕 | 有新的开放模型，但新增驱动视频条件且重设计 DiT。其少步结果不能迁移成 Motus 少步联合控制已可用。〔E3〕 |
| Wan-Move-14B-480P | 作者仓库于 2025-12-10 公告开放权重及推理代码，基于 Wan2.1-I2V-14B 做点级运动轨迹控制。[作者仓库](https://github.com/ali-vilab/Wan-Move)〔E3〕 | 运动条件研究参考；图像点轨迹不等于机器人 7D 控制量，需学习或设计映射。〔E3〕 |
| Wan2.5、Wan2.6、Wan2.7 | 厂商提供视频生成 API，覆盖音画同步、参考生成和编辑等产品能力。[官方产品目录](https://www.alibabacloud.com/help/en/model-studio/video-generate-edit-model/)〔E3〕 | 本次官方权重目录中未查到对应可下载训练权重，暂不进入本地联合反向候选。〔E2/E3〕 |
| Wan3.0-video | 厂商博客日期 2026-08-06；支持多模态输入和最长 30 秒视频。模型说明明确列出 fine-tuning unsupported。[发布说明](https://modelstudio.alibabacloud.com/intl/blog/wan3-ai-video-generation-model/) · [模型能力](https://www.alibabacloud.com/help/en/model-studio/wan3-0-video)〔E3〕 | 已确认服务发布，未确认官方本地训练权重。API 返回视频不能提供本方案所需的内部 task-RL 梯度路径。〔E3〕 |
| Wan-Streamer v0.3 | 作者于 2026-07-16 介绍流式音视频交互与持续 world context。[论文](https://arxiv.org/abs/2607.15038)〔E3〕 | 独立架构研究方向；论文里的 world 不等于已经具备机器人 action-conditioned dynamics。本次未确认可直接用于 Motus 的权重与训练入口。〔E3〕 |

“未查到”限定于本次官方发布渠道和日期，不代表厂商永久不开放。官方 Wan-AI 模型目录有 27 个条目，尚无 2.5、2.6、2.7 或 3.0 的模型 ID；角色动画和舞蹈新权重则有明确条目。〔E2，[公开目录 API](https://huggingface.co/api/models?author=Wan-AI&limit=100)〕

视频生成的审美、音画同步、角色一致性和最长时长，与机器人控制所需的动作条件响应、接触后果和短时动力学是不同指标。现有这些发布资料不足以判断新版本能否提高本实验成功率。〔E3，适用边界〕

## 3 Motus 为什么不能直接更换 Wan

固定源码为 `thu-ml/Motus@f771216802b8a1601599422f12088bee3c068c14`。原方案使用的同一提交仍可作为基准；下列链接统一指向当前上游名称。〔E1〕

### 3.1 维度和层数已有耦合

| 官方配置 | TI2V-5B | I2V-A14B 每个专家 |
|---|---:|---:|
| hidden dimension | 3072 | 5120 |
| attention heads | 24 | 40 |
| Transformer layers | 30 | 40 |
| DiT input channels | 48 | 36 |
| DiT output channels | 48 | 16 |

以上为 E2 配置实证：[5B 配置](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/blob/921dbaf3f1674a56f47e83fb80a34bac8a8f203e/config.json)、[A14B 高噪声配置](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/blob/206a9ee1b7bfaaf8f7e4d81335650533490646a3/high_noise_model/config.json)、[低噪声配置](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B/blob/206a9ee1b7bfaaf8f7e4d81335650533490646a3/low_noise_model/config.json)。这些 channel 不是机器人动作维度。

A14B-I2V 的 36 个输入 channel 由 16 个噪声视频 latent、16 个条件视频 latent 和 4 个 mask channel 拼接而成，其 VAE latent 本身为 16 通道。[噪声与 mask](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/bak/wan/image2video.py#L280) · [DiT 输入拼接](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/bak/wan/modules/model.py#L565) · [VAE z_dim](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/bak/wan/modules/vae2_1.py#L622)〔E1〕

Motus 中的时间投影显式按 `(6, 3072)` 展开；joint attention 使用 WAN 的 head 数与 head dimension 重排 action/understanding QKV；三路专家按共同 `layer_idx` 循环。因此，换 A14B 会涉及参数 shape、层间对齐和已经学习的跨模态连接，不能只修改 checkpoint 路径。〔E1，代码；E3，替换判断〕

源码定位：[时间投影](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/motus.py#L163)、[三路 attention](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/motus.py#L229)、[三专家循环](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/motus.py#L853)。

5B 使用的 VAE stride 为 `(4,16,16)`，A14B 为 `(4,8,8)`，二者 patch 均为 `(1,2,2)`；相同帧数和像素尺寸下，后者的空间 token 数约为前者 4 倍。这是配置推导，不是显存或吞吐实测倍率。[5B 配置](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/bak/wan/configs/wan_ti2v_5B.py#L16) · [A14B 配置](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/bak/wan/configs/wan_i2v_A14B.py#L16)〔E2/E3〕

Motus 还按 `H//32,W//32` 和 `1+future_frames//4` 构造 grid；换 VAE/patch 后须重验视频编码、时间对齐、条件帧和 loss mask。A14B 的高低噪声专家切换也必须进入联合参数清单与重放协议。[grid 源码](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/motus.py#L616)〔E1；设计要求/E3〕

### 3.2 要保留机器人领域的初始化来源

Motus 提供视频领域预训练 Stage 1、latent-action 预训练 Stage 2 和 RoboTwin Stage 3；当前计划采用 Stage 2 共享权重，再适配 LIBERO。官方原始 Wan 视频权重与经过这些阶段训练的 Motus 参数不是同一个初始化。[固定权重说明](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/README.md#model-checkpoints)〔E3〕

换用新的视频底座可能同时丢失原有的机器人视频先验与跨专家对齐。只有经过相应适配与共同任务评估，才能区分新底座容量收益和初始化差异。不能用“新版电影感更好”代替这项验证。〔E3〕

### 3.3 权重加载必须逐项验收

当前 WAN loader 使用 `strict=False`；遇到异常时捕获错误、输出使用随机初始化的警告，随后仍返回模型。由此，“进程启动成功”不能证明领域预训练权重完整加载。[loader 283–323 行](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/wan_model.py#L283)〔E1〕

实现时必须分别锁定架构 config 与权重 revision/文件 hash，生成逐 key/shape 加载报告，对未声明的缺失或异常直接失败，并验证 Stage 2 到新 7D IO 的迁移。本文只核查公开目录与代码，未下载或加载 checkpoint，因此实际权重覆盖率仍是 G2 验收项。〔E3〕

## 4 VLM 候选补充与能力边界

空间 QA、2D grounding 和度量 3D grounding 分别考察关系判断、图像定位和三维几何输出。ERQA、VSI-Bench、EmbSpatialBench、RefSpatialBench 分数不能混成一个“空间能力”总分；不同论文的输入、prompt、thinking、工具调用和评估协议也可能不同。〔E3〕

| 候选 | 已确认的能力与发布物 | 进入联合训练前的主要问题 |
|---|---|---|
| Qwen3-VL-2B-Instruct | Motus 原生 VLM，现有模型直接提取内部特征〔E1〕；作者参数表约 2.13B。[Motus 参数表](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/README.md#overview)〔E3〕 | 原实现冻结 Qwen 且语言前向有 `no_grad()`，须开放内部 LoRA 和梯度。〔E1〕 |
| Qwen3-VL-4B / 8B | 官方开放权重，同家族涵盖空间与 grounding 评测。[技术报告](https://arxiv.org/html/2511.21631v1) · [8B 权重](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)〔E3；目录/E2〕 | 可以作为规模对照，但 adapter shape 与特征分布仍须重新适配。〔E3〕 |
| Qwen3.5-4B / 9B | 官方 EmbSpatialBench 81.3/83.0，RefSpatialBench 54.6/58.5，ERQA 54.0/55.5。[官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)〔E3〕 | 内部已采用 Gated DeltaNet 与 attention 混合架构，不能复用 Qwen3-VL 专用内部调用而不测试。〔E2/E3〕 |
| Qwen3.6-27B / 35B-A3B | 官方 27B EmbSpatialBench 84.6、RefSpatialBench 70.0、ERQA 62.5；35B-A3B 同表分别为 84.3、64.3、61.8。[官方模型卡](https://huggingface.co/Qwen/Qwen3.6-27B)〔E3〕 | 更大容量与混合架构增加训练成本；MoE 的活跃参数数不能当成全部权重存储量。〔E3〕 |
| Qwen3.7-Max / Plus | 已确认厂商服务发布；本次未确认对应官方可下载 checkpoint。[Qwen3.7 官方博客](https://qwen.ai/blog?id=qwen3.7&locale=en)〔E3〕 | API 能力参考，未满足本地 hidden-feature 与参数反传的候选条件。〔E3〕 |
| Qwen3.8-27B | 官方开放 BF16 权重，revision 已记录；同一模型卡 ERQA 65.5，对照 Qwen3.6-27B 为 62.5。未公布该版本的 EmbSpatial/RefSpatial 分数。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)〔E2/E3〕 | 应纳入较新开放模型候选；ERQA 提升不能推成三维定位更好。27B 全模型与 Motus 联合运行的峰值须测。〔E3〕 |
| SpatioLM Understanding，InternVL3.5-8B / SenseNovaSI-8B | 同研究 VSI-Bench 分别从 56.1→65.0、68.7→71.6；推理不要求额外 3D 输入。[作者结果](https://xiaomi-research.github.io/spatio-lm/) · [官方 Understanding checkpoint](https://huggingface.co/xiaomi-research/SpatioLM-Understanding-InternVL3.5)〔E3；权重/E2〕 | 原方法冻结 VLM 并增加空间模块。若只训练外加模块，不满足本实验 VLM 本体 RL 定义；须另设内部 LoRA。〔E3〕 |
| GR3D-8B | 面向显式/隐式 2D grounding 与单目 3D box，基于 NVILA-Lite，并采用相机内参归一化。[论文](https://arxiv.org/html/2605.30307v1)〔E3〕 | [作者项目](https://anjiecheng.me/gr3d/) 仍标 Code/Weights Coming Soon，暂列为几何监督和接口研究参考。〔E3〕 |
| Qwen-3D 3B / 7B | CMU 作者基于 Qwen2.5-VL 改造，使用多视角 RGB、深度与 camera intrinsics/pose。[论文](https://arxiv.org/html/2608.02980v1)〔E3〕；实际 `.pth` 权重已发布，许可证 CC-BY-NC-4.0。[作者权重](https://huggingface.co/katefgroup/Qwen-3D)〔E2/E3〕 | 需要改变当前纯 RGB 观测协议，且许可证与 Qwen 官方 Apache-2.0 发布物不同；不能与原方案直接混作同输入对照。〔E3〕 |

SpatioLM 的 71.6 属于 SenseNovaSI 底座路线，不属于 InternVL3.5 checkpoint；Qwen-3D 也不能当作 Qwen3.8 的另一名称。〔E2/E3〕

SpatioLM 论文另有 LIBERO 结果，但使用独立的 2B VLA：离散 VLA0 从 79.2% 到 91.0%，连续 OFT 从 95.3% 到 96.3%。这些不是两个 8B Understanding checkpoint 或 Motus 联合 RL 的成绩，不能与 VSI-Bench 71.6 拼接归属到同一个模型。[论文 §4.4](https://arxiv.org/html/2608.01899v1#S4.SS4)〔E3〕

Motus 使用 Qwen3-VL 专用的视觉 token、位置编码和 DeepStack 内部接口，再把最后一层特征送入 understanding adapter。更换 VLM 时，不仅要检查维度，也要检查多图输入、padding 有效位、位置编码、视觉特征注入和梯度路径。[特征提取源码](https://github.com/thu-ml/Motus/blob/f771216802b8a1601599422f12088bee3c068c14/models/motus.py#L292)〔E1；重验要求/E3〕

许多公布分数依赖文字生成、thinking 或工具使用；本实验使用内部特征控制动作。答题收益能否变成控制收益，须在相同适配流程后另测。〔E3〕

## 5 训练与 NPU 支持的证据强度

| 已核查入口 | 实际支持的结论 | 尚不能支持的结论 |
|---|---|---|
| DiffSynth-Studio 固定提交下 [Wan2.2-TI2V-5B LoRA 脚本](https://github.com/modelscope/DiffSynth-Studio/blob/208d3abad800f09bd0acd18a134013e2ebd22c17/examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh) | 有 5B 视频训练示例，包含 q/k/v/o、FFN LoRA 与 input_image。〔E1〕 | 不能由此判断其 NPU 路径、Motus joint attention 和 task-RL 重放均可用。〔E3〕 |
| 同提交下 [Wan2.1-T2V-14B NPU 训练脚本](https://github.com/modelscope/DiffSynth-Studio/blob/208d3abad800f09bd0acd18a134013e2ebd22c17/examples/wanvideo/model_training/special/npu_training/Wan2.1-T2V-14B-NPU.sh) | 有设置 NPU 相关选项并启动 dit 训练的维护者示例。〔E1〕 | 目标模型为 Wan2.1，不能与上一行拼成 Wan2.2/Motus 在 910B3 联合训练已验证。〔E3〕 |
| ms-swift 固定提交下 [Qwen3.8-27B FSDP2 LoRA 脚本](https://github.com/modelscope/ms-swift/blob/191313a4f5bf60361ec57c0cb2116e12de1b894d/examples/ascend/train/qwen3_8/qwen3_8_27b_lora_fsdp2.sh) | 脚本注明 16 卡 Atlas 800T A3，使用纯文本 alpaca 中英与 self-cognition，LoRA rank 8、2048 tokens、50 steps。〔E1/E2〕 | 不覆盖 910B3，也不覆盖图像条件下 VLM 特征接入 WAN 的联合 RL。默认冻结 vision/aligner，不能声称训练了视觉主干。[冻结默认值](https://github.com/modelscope/ms-swift/blob/191313a4f5bf60361ec57c0cb2116e12de1b894d/swift/arguments/tuner_args.py#L122)〔E1/E3〕 |
| 原预注册引用的 [verl Qwen3-VL NPU FSDP2 GRPO 配置](https://github.com/volcengine/verl/blob/0d3f56a8980a55bfbbf1214ea54fa1b32ca1405c/tests/special_npu/run_qwen3_vl_8b_Instruct_fsdp2_npu.sh) | 可参考 VLM 的 NPU 训练和分布式组织。〔E2〕 | token 概率不能替代连续动作概率，也不能覆盖 Motus 的内部图。〔E3〕 |

Qwen3.8 示例由 [PR #9920](https://github.com/modelscope/ms-swift/pull/9920) 于 2026-08-18 合入。作者报告的栈为 torch 2.7.1、torch_npu 2.7.1.post4、Triton-Ascend 3.2.1、CANN 9.1，并固定 FLA 提交；这些条件需要与分配的 910B3 环境重新匹配，不能据示例升级现有主机或服务。〔E3〕

原生 GPU 推理、NPU 推理、LoRA 训练脚本和三模块联合 task-RL 是不同验收层级。当前方案仍需逐项完成 G0–G4，再进行闭环烟测；不能因某个组件有样例就跳过联合前反向、路径重放与完整更新测试。〔E3〕

## 6 选型比较应采用的共同标准

以下为后续初筛与实验设计建议，均属 E3；本次只完成资料和源码核查。

1. **发布物可用性**：确认官方权重、revision、许可证、config 和 tokenizer/processor；识别 API wrapper、量化副本和推理蒸馏模型，不能只看仓库名称。
2. **模型内部可训练性**：在真实图像条件下检查目标 LoRA 的 task 梯度、optimizer membership 和更新 delta；通过 detach、冻结与零优势负对照验证路径。VLM/WAN 中用于辅助展示但不影响动作的参数不能冒充控制路径。
3. **动作和视频语义**：固定两相机、7D 动作、9D state、8 步动作块，以及当前帧加物理步 2/4/6/8 的视频对齐；底座改变需要新的显式权重映射和归一化版本。
4. **NPU 数值和成本**：分别测 attention/位置编码、前向、反向、概率重放、optimizer、checkpoint 的精度与峰值；记录每个阶段的 NPU-hours 和 CPU 环境等待。视频 API 的生成时长不用于估算本地梯度成本。
5. **模型质量**：VLM 增加未见指令、指代、空间关系和 grounding 检查；视频侧保留冻结 held-out velocity MSE，并检查动作变化是否带来正确的预测变化；最终比较未见初态及多任务成功率。单任务成功率不能证明通用 VLM 或世界模型能力全面提升。
6. **公平归因**：每种底座内部都从自己的共同暖启动 checkpoint 比较 B/C/D。换底座造成的预训练差异、适配预算、模型规模和采样步数必须单列，不把它们归因成联合 RL 算法收益。

### 6.1 优先级建议

- **工程基线**：先保留 Qwen3-VL-2B＋Motus Stage 2 中的 Wan2.2-5B 组合，用已有预注册验证联合 task 梯度及 NPU 闭环。这是可追溯的基准，不是最优能力断言。〔E3〕
- **VLM 初筛**：Qwen3-VL-4B 作为同家族规模候选，Qwen3.5-4B 作为新架构小模型候选；Qwen3.8-27B 先作为能力与 NPU 前反向候选，SpatioLM Understanding 作为空间特化候选。先筛选，再决定哪些值得重新暖启动。〔E3〕
- **Wan 扩展**：如需更大通用视频骨干，优先研究有明确权重的 Wan2.2-I2V-A14B；先评估层数、双专家、latent 接口与机器人预训练代价。Animate-2、Dancer、Move 分别保留为驱动条件、长时一致性和轨迹控制参考。〔E3〕
- **服务模型**：Wan2.5–3.0 及本次仅确认 API 的 Qwen3.7 不纳入直接本地联合反向比较。后续若出现官方权重，应重新检查同一组条件。〔E3〕

### 6.2 对原计划的影响

现有基线选择有明确的融合与初始化依据，但不能写成已经完成新模型优劣比较。更换任一底座都应产生新的协议版本、SFT checkpoint 和门槛证据；需要重做受影响的最早验收，随后重建该底座内部四臂对照。原有 G0–G6 尚未执行，所以上述候选目前均没有本工作区的 910B3 联合训练性能结论。〔E3〕

## 7 当前结论与仍需实测的事项

| 问题 | 当前结论 |
|---|---|
| Wan 是否有更新模型 | 有，已确认 Wan3.0 服务及 2026 年开放的 Dancer、Animate-2 分支。〔E2/E3〕 |
| 能否把 Wan3.0 接进本方案一起反向训练 | 当前确认发布物不满足内部可训练权重条件。〔E2/E3〕 |
| 当前 5B 是否一定比新模型好 | 没有这项对照证据；它的优势是保留 Motus 已有领域预训练与融合接口。〔E1/E3〕 |
| Qwen3.8 是否应纳入调研 | 应纳入，官方开放权重和新能力指标均已确认；精确 grounding 与联合 RL 收益仍待测。〔E2/E3〕 |
| 910B3 能否稳定完成候选联合更新 | 尚须本实验的软件锁、算子数值、完整图反向、分片更新与恢复验收。〔E3〕 |
| 本次产物的用途 | 用于决定初筛范围和形成新版本实验依据，不是训练完成或收益证明。〔E3〕 |
