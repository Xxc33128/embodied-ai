# 2026-10-26 ~ 11-01 具身蒸馏调研

## 本周目标

- 从精读 [arXiv 2609.24682](https://arxiv.org/abs/2609.24682)（世界模型表征蒸馏进紧凑 VLA）出发，完成具身智能知识蒸馏六条线的并行调研，产出方法地图与经验清单 → [具身蒸馏调研报告.md](具身蒸馏调研报告.md)
- 其余目标待补（周目录名可随本周主线演化调整）

## 按日结论

| 日期 | 结论 | 证据 |
|---|---|---|
| 10-26 | 六线 17+2 篇蒸馏论文并行调研成稿：latent 蒸馏优于 pixel/logits、动作导向表征优于语义、on-policy+per-token reverse-KL 为效率上限、少步化必须带分布锚、速度收益主要来自动作解码方式而非骨干规模；勘误 RLDG 归属（Berkeley Levine 组、方向为专才→通才） | E3（论文转引，未独立复核） |
| 10-30 | 大小脑联合后训练与 RL 调研：区分 VLM＋Action 联合 RL、Reasoning–Action 轨迹优化、World SFT＋Actor RL；优先 Z-1／π_RL／SyVLA／DeepThinkVLA／WAM-RL，补 Motus2 与长程信用分配；提出冻结／有限解冻／世界在线适配对照 | E3（论文与官方文档）；Motus／SyVLA 定点源码 E1 |
| 10-30（补） | 扩线补充与核验（第二册）：SaiVLA-0／InternVLA·M1 结构线＋EXIMO／Psi-R1／LaST-R1／ReCoVLA 联合 RL 增量＋WAM×RL 谱系（WoVR+／VLA-RFT／DRILL／DR-APO／测试时）＋RLinf×Ascend CANN 与 verl-vla 基建线索；标记 MIND-2 引用待复核 | E3（检索索引级；部分 curl 实抓）；本地 clone／PDF E1／E2 |
| 10-30（第三册） | 四路全网原文核验（网络抓取未阻断）：解决抽查清单 #8/#10（τ0-VLA、Helix 2.5）；纠错 15 个流传名（VERSE／dexGRPO／AnyDual／TGRoK3.0／WBT-1／RAPO／R1-02 查无此文，iRe-VLA／AutoVLA／π*0.6／DeepThink-with-Confidence／Dream-1 归属纠错）；新增六线（双系统 SFT 教训／flow-RL 方法栈／GRPO 退化变体／TEMPO 解耦 RL／世界模型新增／reward 谱系）＋LIBERO 数字总表＋Motus 四 arms 映射与纯任务奖励坑清单；重判：双端同奖励联合 RL 公开配方仍缺位，产业界零公开，NVIDIA WAM 大一统是并列空白 | E3（子调研打开原文页）；转述项标 E4 |
| 10-30（重点精读，v2） | 按领域问题组织具身 RL 重点报告：Policy RL、优势条件化经验学习、轻量真机在线 RL、VLM／Action 联合、推理对齐、World／Policy 协同及长程信用；重点正文加入 RECAP、RLT、SyVLA，不以特定项目适配性筛选；复核梯度、消融、数据与预算 | E3（原文与分析）；RLinf／SyVLA 锁定源码 E1 |
| 10-30（图解与训练） | 补11张 Mermaid 架构／训练流程图及十项研究训练详表，展开数据、Reward、Loss、更新范围、Horizon、超参、评测与算力；分列论文和公开配置，标注 DeepThinkVLA KL 差异及其他披露缺项 | E3（论文与图示）；发布源码／配置 E1／E2 |

## 过程文档索引

- [具身蒸馏调研报告.md](具身蒸馏调研报告.md)——六条线分线笔记 + 横向对比表 + 配方/坑清单 + 与 Motus/DreamLab/GR00T NPU 部署的结合点
- [2026-10-30-具身大小脑联合后训练与RL调研.md](2026-10-30-具身大小脑联合后训练与RL调研.md)——实际更新模块／梯度与奖励边界、优先阅读清单、官方实现入口及研究对照建议
- [2026-10-30-具身大小脑联合后训练与RL调研-扩线补充与核验.md](2026-10-30-具身大小脑联合后训练与RL调研-扩线补充与核验.md)——双脑／三脑结构线、VLM×VLA 联合 RL 增量、WAM×RL 谱系扩展（测试时／工业／评测）、基建×昇腾、底表资源；与 v1.0 互补合读〔E3；本地项 E1／E2〕
- [联合后训练调研-分线笔记/line-C-WAM与世界模型RL.md](联合后训练调研-分线笔记/line-C-WAM与世界模型RL.md)——C 线（WAM／世界模型×RL）249 行实抓笔记（逐条 URL、数字、扫描清单与 910B 粗判）；第二册 §3 的详细底表
- [2026-10-30-具身大小脑联合后训练与RL调研-第三册.md](2026-10-30-具身大小脑联合后训练与RL调研-第三册.md)——全网原文核验与新增线：抽查清单 #8/#10 解决、15 个流传名纠错表、双系统联合 SFT 工程教训（OpenHelix）、flow 头 RL 方法栈（π_RL/RECAP/RLT/EXPO）、GRPO 退化与 credit assignment 变体、TEMPO 解耦 RL、世界模型×动作新增（UniWAM/DUST/Prophet/Dream-1）、reward 谱系、RLinf/verl-vla 核验、LIBERO 数字总表、Motus 四 arms 文献映射〔E3；转述 E4〕
- [2026-10-30-具身后训练重点研究-RL方法与证据.md](2026-10-30-具身后训练重点研究-RL方法与证据.md)——领域视角的重点研究精读：实际 RL 更新、方法贡献、实验与消融、代码状态、证据局限及研究路线比较；v2 重排选篇与解读口径，后续引用优先查看本报告相应章节〔E1／E3〕
- [2026-10-30-具身RL图解与训练详表.md](2026-10-30-具身RL图解与训练详表.md)——讲解用 Mermaid 图及十项研究的训练细节；图中区分冻结／训练和数据／更新路径，训练表区分文献、发布配置与未披露项〔E1／E2／E3〕

## 遗留 → 下周

- 延伸阅读四项（Refined Policy Distillation、Fast ECoT/ECoT-Lite、综述 §3.6/§5.2、REPA 原文）见报告 §7
- 关键数字引用进正式报告前需抽查复核（当前全 E3）
- 补充分册 §7 抽查清单（SaiVLA-0／EXIMO／Psi-R1／LaST-R1／ReCoVLA／DR-APO／DR-WM 命名等）逐项复核后再进正式材料（#8／#10 已由第三册 §1 结项）
