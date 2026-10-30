# 分线 C：WAM / 世界模型 × RL 的联合后训练（调研笔记）

> 执行：2026-10-30。检索方式：web_search + bash curl（arXiv abs/HTML、arXiv API、CVF OpenAccess、项目页）。
> 【方法备注】本会话 web_fetch 工具被沙箱全部拒绝（域名解析为非公网 IP），改用 curl 直连抓取，均成功；以下均为实际抓取内容，未抓到的明确标"未核"。数字附来源 URL；仅来源于摘要的标注"（摘要）"；证据分级 E3（文献/文档，未复现）。
> 联合方式标签：**L1** 同一损失同训；**L2** 交替/两阶段（含迭代 co-evolution）；**L3** 单向信号（WM→policy 或 policy→WM，单边冻结/单向）；**L4** 仅推理期协同（测试时搜索/选择，权重不变）。
> Motus 对齐口径：Motus = VLM + Wan2.2 视频分支 + 动作专家；目标 = **同一任务奖励同时更新 VLM 与 World/Action 参数**。

---

## A. 必查种子（7 篇）

### A1. WAM-RL / WA-RL: World-Action Model Reinforcement Learning with Reconstruction Rewards and Online Video SFT
- 链接：arXiv [2606.17906](https://arxiv.org/abs/2606.17906)；CVPRW 2026 版改名 **WA-RL**，[CVF OpenAccess](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Qian_WA-RL_World-Action_Model_Reinforcement_Learning_with_Reconstruction_Rewards_and_Online_CVPRW_2026_paper.html)（Proceedings of CVPR Workshops 2026, pp. 4478-4485）
- 机构+时间：北京大学（多媒体信息处理国家重点实验室）+ 东北大学 + 清华；2026-06（arXiv v1）/ 2026-06（CVPRW）
- WAM 角色：**训练器**——首个明确"把 RL 引入 World-Action 范式"的工作，WM 与 actor 联合在线优化
- 联合标签：**L2**（机制上双轨：WM 走在线视频自监督微调 + KL，actor 走 RL；证据：论文 §3.2–3.4，双轨分别更新，KL 约束 WM 潜空间对齐）
- RL 细节：actor 用 policy gradient（REINFORCE 式，`∇J=E[∇logπ·A]`），奖励 = **重建奖励** `r_t=sim(想象的未来帧, 实际执行帧)`，尝试了 pixel-MSE / optical flow / DINOv2 / V-JEPA2 四种相似度，pixel-MSE 最好；WM 用成功轨迹做 online video SFT（flow matching/重建损失）+ 潜特征 KL（对预训练冻结副本的 Gaussian 近似）；基座 = Genie Envisioner-ACT（DiT 视频生成器 + actor 读中间潜特征）；8×A800、混合在线 RL+视频微调共 8 小时
- 关键数字：LIBERO-Object 68%→**82%**（actor-only RL 78%）；RLBench Water Plants 19%→**22%**（πRL 18%）；纯 actor 优化只在短程任务涨、长程任务无增益——**联合优化 WM 对长程关键**（[全文](https://arxiv.org/html/2606.17906v1)）
- 代码/项目页：未找到（无代码链接，局限项）
- 局限：KL 正则限制 WM 适应幅度；重建奖励对比度有限（成功/失败区分弱）；实验规模小（仅 2 个任务集）
- Motus 关联度：**最高**。与"同一任务奖励更新 VLM 与世界/动作分支"的目标最接近的公开对照：它已做到"WM 与 actor 共同演化"，但 WM 更新信号是**视频自监督 + 成功轨迹**（非任务奖励梯度）；奖励是"想象-执行一致性"（重建）而非任务成功奖励，且未更新 VLM 语义分支

### A2. τ₀-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation
- 链接：arXiv [2608.16885](https://arxiv.org/abs/2608.16885)；项目页 [tau0-vla.github.io](https://tau0-vla.github.io/)
- 机构+时间：上海创智学院（Shanghai Innovation Institute）+ Agibot Finch + 香港中文大学；2026-08-17
- WAM 角色：**测试时计算/搜索**（WM 预测候选子任务的终态图像，供 value model 打分做 beam search）
- 联合标签：**L4**（推理期协同；WM 单独用子任务首尾帧监督训练，不参与 RL；证据 §V-C）
- RL 细节：无 RL。高层"propose–predict–evaluate"：proposal(Qwen3.5-9B 微调) 生成候选子任务 → WM(基于 Step1X-Edit) 预测终态 → value model 五级序数打分[0.05,0.95] → beam search（宽度 B×深度 D）→ reflective model 定稿；低层 VLA 执行；训练数据 40,115 小时异构真机数据
- 关键数字：闭环真机 TTC vs Plan Once：Make Milk Tea 5/10→**7/10**；Book Organization 6/10→**9/10**（progress 66.67%→93.33%）；Clean Room 5/10→7/10；OOD 子任务预测准确率 TTC **74.0%** vs Plan Once 50.0% vs Best-of-N 57.5%（[全文](https://arxiv.org/html/2608.16885v1)）
- 代码/项目页：tau0-vla.github.io（项目页）
- 局限：TTC 算力开销随 beam 规模上升（收益饱和）；每步只预测单张 head-camera 图像
- Motus 关联度：中——"世界模型作为测试时验证器"路线；若 Motus 要在 910B 上做推理期 search，可参考其 beam+value 打分设计，但不涉及参数更新

### A3. How Much Can Best-of-N Help a World Action Model?（Best-of-N for WAM）
- 链接：arXiv [2610.37196](https://arxiv.org/abs/2610.37196)
- 机构+时间：清华 + Everwise-Tech + 复旦（另：清华深圳研究院+北大）；2026-10-27
- WAM 角色：**测试时选择/验证器审计**——测量"可选择性 headroom"并落地 fail-gated selector
- 联合标签：**L4**（推理期选择，策略与 WM 权重冻结；证据 §5 全部为 fix-policy 实验）
- RL 细节：非 RL 训练；但借鉴 RL 的 off-policy 评估思想（held-out oracle、common random numbers、方差分解 τ²）；4 个 WAM（Cosmos Policy 2B、Flex-π、Fast-WAM、Streaming-WAM）× 2 基准（RoboCasa、LIBERO-Plus）、10 万+ 分支 rollout、预注册检验
- 关键数字：标准单次执行估计的 headroom 被随机性污染（17 次续跑下可选择性占 24%–66%）；结构化动作扰动（沿机器人轴偏移 0.1/0.5κ）产生 4.5–16 点真实 headroom（67–95% 通过 held-out 检验）；想象未来可线性探测候选质量（ρ≈0.16–0.54），而 Cosmos 的 value head 用不上（AUROC 0.90 但决策内 |ρ|≤0.05）；fail-gated selector 闭环成功率 76.9%→**81.6%**（训练任务预留集）、62.6%→**69.8%**（未见任务）；每集仅多花 ~17.5 次 imagination pass（[全文](https://arxiv.org/html/2610.37196v1)）
- 代码/项目页：HeadroomBench 将公开（论文承诺）
- 局限：selector 依赖 Cosmos Policy 的 action-conditioned 想象；结论是测量性的，跨模型泛化未知；对 Motus 类"无专用 value head"模型需自建读出
- Motus 关联度：中高——直接点名 Motus2 "policy+simulator+value 单网络"；其"想象通道含候选优劣信息、value head 丢信息"的结论对 Motus 价值头设计是重要警示；测试时选择不更新权重

### A4. Towards VLA-Dreamer: Refining VLA Behavior Using World Models
- 链接：arXiv [2609.31313](https://arxiv.org/abs/2609.31313)
- 机构+时间：汉堡大学 Knowledge Technology（Wermter 组）；2026-09-25
- WAM 角色：**训练器 + 模拟器**（提案；尚未实验）——在 VLA 视觉编码器 embedding 空间训 world model（JEPA 式，非像素），再在 WM 内用 RL 微调 VLA
- 联合标签：**L3** 规划（WM 冻结、VLA 内 RL；概念论文自述：WM 训练后冻结，VLA 在其中跑 PPO；证据 §2.4）
- RL 细节（提案）：PPO（选 π0-FAST/OpenVLA 因 token 概率便于 RL）；奖励 = 当前 embedding 与 goal image embedding 的负 MSE（稠密）；冻结视觉编码器防漂移、强正则、限制想象时域防误差累积
- 关键数字：概念论文，**尚无实验数字**（摘要+方法）
- 代码/项目页与局限：未见代码/项目页；依赖"VLA 视觉 embedding 足以预测未来"的未验证假设；reward 非单调风险；drift 限制想象时域
- Motus 关联度：中——提供"embedding-space WM + PPO"低算力思路（WM 是 MLP 级轻模型）；但其 WM 更新不含视频分支（纯 embedding 预测头），与 Motus 的 Wan2.2 视频分支路线不同

### A5. VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators
- 链接：arXiv [2510.00406](https://arxiv.org/abs/2510.00406)；项目页 [vla-rft.github.io](https://vla-rft.github.io/)
- 机构+时间：西湖大学 + 浙大 + OpenHelix Team + 复旦 + 郑州大学 + 北邮等；2025-10-01
- WAM 角色：**模拟器 + 奖励来源**（世界模型作为可控 simulator，提供 trajectory-level "verified rewards"）
- 联合标签：**L3**（WM 预训练后冻结供 RL；只更新 VLA；证据：文中"Update policy πθ and sigma net with GRPO"，WM 无 RL 更新）
- RL 细节：GRPO + SDE 策略（保流量头小权重 MSE 辅助）；奖励三种设计对比——最终用"同一 WM 内渲染 policy 轨迹 vs 参考轨迹的负 MAE+LPIPS"（Type-3，+4.5 点）；WM 为 138M LLaMA 轻量自回归模型（LIBERO 预训练）
- 关键数字：LIBERO 平均 **91.1%**（+4.5 vs base）；<400 微调步 vs 监督基线 150K 步（0.4K vs 150K）；扰动设定 Goal/复合扰动 +6% 以上（[全文](https://arxiv.org/html/2510.00406v1)）
- 代码/项目页：vla-rft.github.io（项目页）
- 局限：WM 冻结、误差会限制上限；奖励依赖参考轨迹存在；LIBERO 为主
- Motus 关联度：高（方法学）——"同一生成空间内比轨迹"的 reward 设计（避免生成质量偏差）可移植到 Motus 视频分支；但 WM 不更新，属"外挂模拟器"

### A6. Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- 链接：arXiv [2603.16666](https://arxiv.org/abs/2603.16666)；项目页 [yuantianyuan01.github.io/FastWAM](https://yuantianyuan01.github.io/FastWAM/)
- 机构+时间：清华 IIIS + Galaxea AI；2026-03（v2 03-23）
- WAM 角色：**训练期视频共训（representation shaping）**；测试期不做未来想象
- 联合标签：**无 RL**；对 Motus 的价值在于"L1 训练 vs 测试时想象"的消融结论
- RL 细节：无。受控对比：Fast-WAM（保留视频共训、去测试时生成）vs Joint（视频+动作联合去噪）vs IDM（先视频后动作）vs 去视频共训
- 关键数字：RoboTwin 2.0 **91.8%**（Motus 预训练 87.8%、无预训练 77.3%；LingBot-VA 92.2%/80.6%）；LIBERO 平均 **97.6%**（LingBot-VA 98.5%、Motus 97.7%）；190ms、比 imagine-then-execute 快 4×+；**去掉视频共训**：RoboTwin 掉到 83.8%、LIBERO 掉到 93.5%、真机叠毛巾掉到 10% 成功率——视频共训是主因、测试时想象贡献小（[全文](https://arxiv.org/html/2603.16666v2)）
- 代码/项目页：项目页（无公开代码证据）
- 局限：结论限于其设置；真机仅 towel-fold 单任务
- Motus 关联度：**极高**——直接以 Motus 为基线并指出"Motus 的视频共训+预训练收益大部分可在不做测试时想象时保留"；是 Motus 视频分支设计的最强对照数据点

### A7. LingBot-VA（Causal World Modeling for Robot Control）
- 链接：arXiv [2601.21998](https://arxiv.org/abs/2601.21998)；GitHub [robbyant/lingbot-va](https://github.com/robbyant/lingbot-va)；HF checkpoints robbyant/lingbot-va
- 机构+时间：Robbyant 团队（GitHub 组织 robbyant）；2026-01（v2 03-22）
- WAM 角色：**统一自回归模型**（帧预测+动作联合训练；测试时闭环 rollout + 异步推理）
- 联合标签：**L1**（共享潜空间内视觉+动作 token，MoT 架构，帧预测与策略同时学；无 RL）
- RL 细节：无。机制：MoT（Mixture-of-Transformers）+ UniDiffuser 式共享潜空间；closed-loop rollout（每步用真观测刷新）；异步流水线（动作预测与执行并行，2× 加速）；部分去噪（动作从中间噪声状态解码）
- 关键数字：RoboTwin 2.0 平均 **92.9%**(Easy)/91.6%(Hard)；LIBERO 平均 **98.5%**（LIBERO-Long 98.5%）；真机 6 任务超 π0.5 20%+；**50 条示范即可 post-training 部署**；低数据（10 条）比 π0.5 progress +15.6%（[全文](https://arxiv.org/html/2601.21998v2)）
- 代码/项目页与局限：GitHub + HF ckpt + 官网 technology.robbyant.com/lingbot-va；无 RL；10.9B 参数（GAWM 统计），推理重（4.5s/query 量级，GAWM 评估）
- Motus 关联度：高——"视频-动作联合 + 闭环 + 异步"的规模化路线；其"50 条示范即可适配"的数据效率为 Motus 后训练配方提供参照；无 RL 更新

---

## B. 扫描清单（补漏）

### B1. RynnWorld-Latent: A Cross-Embodiment World Model Conditioned on Universal Latent Actions
- 链接：arXiv [2610.36791](https://arxiv.org/abs/2610.36791)；GitHub [alibaba-damo-academy/RynnWorld-Latent](https://github.com/alibaba-damo-academy/RynnWorld-Latent)；HF/modelscope 同款 ckpt
- 机构+时间：阿里 DAMO Academy + 香港具身智能实验室（CUHK）+ Hupan Lab；2026-10-27
- WAM 角色：**数据生成器**（两条路线：digital teleoperation + latent-action synthesis，产出 action-video 演示供策略 mid-training）
- 联合标签：**L3**（WM 生成数据 → 策略 SFT/中间训练；无 RL，WM 不因任务奖励更新）
- RL 细节：无。553.5M 帧、23 数据集预训练；RynnLAM 3D-grounded latent action 作共享条件接口；合成数据分两条 route 生成
- 关键数字：合成 mid-training 提升策略：π0 LIBERO 94.4%→**96.1%**、LIBERO-Plus 53.6%→**79.4%**；π0.5 96.9%→97.8%/84.4%→84.8%；真机 Plate-Placement 82.86%→91.43%（π0.5）；视频预测 PSNR/SSIM/LPIPS 优于 DreamDojo-14B 等（[全文](https://arxiv.org/html/2610.36791v1)）
- 局限：合成数据是 augmentation 而非替代（低于 Real only 上界）；真机仅 35 trials/任务；不隔离数据量 vs 算力贡献
- Motus 关联度：中——"世界模型产数据"路线可作 Motus 的 data engine 组件；但与"RL 更新世界分支"无关

### B2. World Dreamer v5-lite: A Multi-Robot World Action Model with One Dreamer
- 链接：arXiv [2610.37518](https://arxiv.org/abs/2610.37518)
- 机构+时间：World Agents 团队；2026-10-27
- WAM 角色：**多机器人 WAM**（共享世界状态 + 每机器人独立动作流；VLM 提供场景上下文）
- 联合标签：无 RL；L1（视频编码器特征构建共享 world state；masked-view 协同训练）
- 关键数字：RoboFactory **74.0%**（+23.7 点 vs 最强基线）；4 个真机任务平均 **78.8%**（+13.8 点）；RoboCasa365 58.08%；预训练 >10,000 小时轨迹、20 种仿真机器人
- 局限：无 RL；多机硬件假设强
- Motus 关联度：低-中——多机扩展场景参考

### B3. GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation
- 链接：arXiv [2610.37091](https://arxiv.org/abs/2610.37091)；代码 [github.com/gaoxiang12/ga-wm](https://github.com/gaoxiang12/ga-wm)
- 机构+时间：Genisom AI（北京）；2026-10-27（RA-L letter 风格）
- WAM 角色：**紧凑潜空间 WM**（冻结 DINOv3 特征上小预测器，预测两时域 token；动作头直接从当前+预测 token 回归）
- 联合标签：L1（动作损失反传进预测器，预测器同时受未来特征损失；测试时也用预测 token 条件化动作——训练+推理双用；无 RL）
- 关键数字：**<100M 参数**（77M/38M 版本）：RoboTwin 2.0 93.94%/93.10%（超十亿级 WAM）；LIBERO 95.6%；真机 92% vs π0.5 78%；延迟 8.8ms/7.1ms（最快）；WM 消融：短训练下 RoboTwin 90.7%→**18.8%**、1/4 LIBERO 数据 -9.5 点（[全文](https://arxiv.org/html/2610.37091v1)）
- 局限：闭任务集、不支持语言指令；无 RL
- Motus 关联度：中——"小型潜空间 WM + 冻结编码器"的算力友好设计，可作为 Motus 在 910B 上做消融的低成本对照；不涉及 RL 更新

### B4. Trace-WAM: Learning to Act via 3D Motion Prediction for World Action Models
- 链接：arXiv [2610.36840](https://arxiv.org/abs/2610.36840)；项目页 [trace-wam.github.io](https://trace-wam.github.io/)
- 机构+时间：香港理工 + 高德/阿里 + 清华 + CASIA NLPR + 北京科技大学；2026-10-27
- WAM 角色：**训练期辅助监督**（未来 3D 运动预测作为 WAM 的辅助目标；含深度/对应点；Tri-MoT：video/motion/action 三专家）
- 联合标签：L1（三目标联合训练；动作专家不消费运动预测，推理时仍单次前向）；无 RL
- 关键数字：LIBERO 98.7%（Fast-WAM 97.6%）；LIBERO-Plus 零样本 **84.5%**（超 Fast-WAM 18.3 点）；RoboCasa-GR1 few-shot 63.6%（+18.8）、full 80.3%；真机 Piper 85.0% vs Fast-WAM 63.3%；基于 Fast-WAM 代码 + **Wan2.2-5B** 视频骨干、16/32×H20（[全文](https://arxiv.org/html/2610.36840v1)）
- 局限：无 RL；依赖 3D 点跟踪标注（仿真深度/估计深度）
- Motus 关联度：高（技术栈重叠）——同样基于 Wan2.2 路线，用"运动/几何辅助目标"提升 OOD；为 Motus 视频分支的辅助损失设计提供实证（深度保留 +2.1 点等）

### B5. DR-WM: From Pixels to Executable Physics（扫描清单"DR-WM-1.0/DEEP Robotics"）
- 链接：arXiv [2610.17630](https://arxiv.org/abs/2610.17630)
- 机构+时间：作者 Ruo Chen, Jiahua Pang, …, Chao Li；2026-10-13。**注意**：未检索到"DR-WM-1.0"确切命名；本文数据集使用 DEEPRobotics M20 与 Lite3 机器人，但论文未标注公司 affiliation（待核）
- WAM 角色：**行动后果预测器（推理期选择）**：四足机器人室外地形；PhyMap 预测触地先验→物理仿真标定→视频模型生成未来帧→VLM 选动作
- 联合标签：L4（推理期协同；物理标定+视频生成耦合，无 RL）
- 关键数字：90 个场景中，**8B VLM + DR-WM 在 6 项指标中 5 项超过裸 32B VLM**（摘要）；物理指标超 geometry-only 与 uniform-physics 基线
- 局限：四足导航域、非操作域；无 RL；命名待核
- Motus 关联度：低——"物理一致 WM 用于动作筛选"参考

### B6. WMPO: World Model-based Policy Optimization for VLA Models（WMPO 类）
- 链接：arXiv [2511.09515](https://arxiv.org/abs/2511.09515)；项目页 [wm-po.github.io](https://wm-po.github.io)
- 机构+时间：香港科技大学（HKUST）；2025-11-12
- WAM 角色：**模拟器 + 奖励来源**（像素级 WM 做"Imagination 内 on-policy RL"，不碰真机）
- 联合标签：**L3**（WM/奖励模型先训后冻结；GRPO 只更新策略；证据 §3：noisy-frame conditioning + frame-level action control 的 OpenSora 系 WM + 轻量 reward model）
- RL 细节：GRPO（可同初始态重复 rollout——真机做不到、WM 天然支持）；奖励模型预测成败（F1>0.95 抗 reward hacking）
- 关键数字（全文）：Mimicgen 4 任务平均：P=128 时 WMPO **47.1%** vs DPO 37.3 / GRPO 33.2；P=1280 时 **57.6%**；真机"Insert square into stick" **70%** vs DPO 60% / base 53%
- 局限：WM 用 OpenSora 系像素模型，视觉失真与帧-动作错位需专门处理；不更新 WM
- Motus 关联度：高（最近邻路线）——"WM 内 GRPO"是 Motus 若要 RL 的直接起点；但其 WM 冻结、奖励为独立模型，属外挂

### B7. WoVR / WoVR+: World Models as Reliable Simulators for Post-Training VLA Policies with RL
- 链接：arXiv [2602.13977](https://arxiv.org/abs/2602.13977)（WoVR）；[2610.17922](https://arxiv.org/abs/2610.17922)（WoVR+，2026-10-13）；项目页 [wovr-rlinf.github.io](https://wovr-rlinf.github.io/)；RLinf 生态 [github.com/RLinf/RLinf](https://github.com/RLinf/RLinf)
- 机构+时间：中科院自动化所 + 中关村学院 + 清华 + UCAS + Infinigence AI；2026-02（v2）/ WoVR+ 2026-10
- WAM 角色：**模拟器**（hallucination-aware；WoVR+ 扩到 VLA 界面级多视角+本体感觉）
- 联合标签：**L2**（两阶段 policy–simulator co-evolution：先用 base policy 1500 条轨迹训 WM → 想象力内 RL → 用进化后策略的 1000 条新轨迹再精炼 WM；WoVR+ 加入 stage-local 想象训练）
- RL 细节：GRPO（clipped objective）+ 幻觉感知：Keyframe-Initialized Rollouts、可控 action-conditioned WM、WoVR+ 的 chunk-wise Flow-SDE 探索与 stage-specific 奖励
- 关键数字：WoVR：全轨迹 SFT 设定下平均 **95.9%**（WMPO 92.4%、GRPO 89.7%）；WoVR+：较 SFT +21.09%（RoboTwin 2 任务）、真机 6 任务平均 39.2%→**55.8%**、长程 phone-packaging 30%→**75%**（[全文](https://arxiv.org/html/2610.17922v1)）
- 局限：预定义 stage 与数据支持的 stage-entry；chunk 级相关性未细化
- Motus 关联度：**高（工程路径）**——RLinf 生态（工作区已有 RLinf 源码）+ "两阶段 WM 精炼"与 Motus 联合训练目标直接相关；WM 更新仍为 SFT 数据回流、非任务奖励梯度

### B8. 相邻/历史参照：DreamVLA / UniVLA / DreamGen(GR00T-Dreams)
- DreamVLA（[2507.04447](https://arxiv.org/abs/2507.04447)）：动态区域引导的"世界知识预测（深度+语义+动态）"+逆动力学；无 RL；VLA 侧综合世界知识（摘要）
- UniVLA（[2505.06111](https://arxiv.org/abs/2505.06111)，RSS 2025）：DINO 特征空间任务中心 latent action，跨具身数据；无 RL；代码 OpenDriveLab/UniVLA（摘要）
- DreamGen/GR00T-Dreams（[2505.12705](https://arxiv.org/abs/2505.12705)，NVIDIA GEAR）：视频世界模型产"神经轨迹"（latent action/IDM 恢复伪动作）；人形 22 种新行为、仅需单任务单环境遥操数据；DreamGen Bench；无 RL（摘要）

## C. 扩展发现（RL×世界模型，本轮新增优先读）

### C1. DRILL: Training World Models to Improve Policies, Not Predict Pixels
- 链接：arXiv [2610.30814](https://arxiv.org/abs/2610.30814)（NeurIPS 2026）
- 机构+时间：腾讯混元 + 北大；2026-10-22
- WAM 角色：**训练器**——**唯一一条"用 RL 目标梯度更新世界模型"的明确工作**：bilevel 框架，外层更新 WM 参数 φ 以最大化内层 GRPO 更新后的策略回报
- 联合标签：**L1 逼近**（外层目标是策略回报对 WM 的元梯度；两种实例：DRILL-VWI 闭式代理=按策略访问度/GRPO 优势/模型置信度重加权预测误差；DRILL-IMG 全元梯度=截断 Hessian-向量积）
- RL 细节：内层 GRPO（组相对优势 stop-gradient）；外层 AdamW；G=8；"像素 MSE 更低≠策略梯度更好"（WM 经策略效用训练后像素 MSE 高 17% 但下游策略更好）
- 关键数字：DRILL-IMG 平均 SR 比 WMPO **+13.1 点**、比 RLVR-World **+7.9 点**（5 个操作模拟器、2 个 VLA 骨干）；与 RLVR 预训练互补（[全文](https://arxiv.org/html/2610.30814v1)）
- 局限：方法较理论化（元梯度截断近似）；评测为模拟器；代码未见
- Motus 关联度：**最高（理论路线）**——"WM 由下游策略回报驱动"正是 Motus"同任务奖励更新 World 参数"的严格形式；可作 Motus 联合优化的目标函数参照；其结论（按策略访问度重加权）可直接试验

### C2. World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy
- 链接：arXiv [2602.06508](https://arxiv.org/abs/2602.06508)；项目页 [showlab.github.io/World-VLA-Loop](https://showlab.github.io/World-VLA-Loop/)
- 机构+时间：新加坡国立大学 Show Lab；2026-02（v2）
- WAM 角色：**模拟器 + 奖励来源**（视频 WM 联合预测未来帧与二值奖励；四阶段闭环：SANS 数据 → WM 预训练（视频+奖励联合监督）→ WM 内 GRPO → 部署收集新数据回流扩 SANS → 迭代）
- 联合标签：**L2**（每轮 RL 后收集新策略的成功/接近成功数据 → 精炼 WM → 再 RL；"迭代联合优化"）
- RL 细节：GRPO（沿用 SimpleVLA-RL 代码，把物理模拟器替换为学习 WM；组大小 8；请求-响应架构）
- 关键数字：WM 视觉对齐 88.5%、奖励对齐 87.25%；LIBERO 平均 +12.7%；真机 +23.4%/+13.3%；两轮联合优化后真机成功率 +36.7%/+26.6%（两个任务）（[全文](https://arxiv.org/html/2602.06508v2)）
- 局限：近成功样本依赖遥操采集；WM 奖励需 SANS 监督
- Motus 关联度：高——"WM 与策略交替联合提升"的完整工程闭环样本；WM 更新=数据回流 SFT，非奖励梯度

### C3. GigaBrain-0.5M*: a VLA That Learns From World Model-Based Reinforcement Learning（RAMP）
- 链接：arXiv [2602.12099](https://arxiv.org/abs/2602.12099)；项目页 [gigabrain05m.github.io](https://gigabrain05m.github.io)
- 机构+时间：GigaAI（GigaBrain Team）；2026-02（v2）（与 GigaBrainChallenge CVPRW 同生态）
- WAM 角色：**价值/未来状态条件化训练器**（WM 联合预测未来状态 + value；策略以 z、value 为条件）
- 联合标签：**L2**（四阶段迭代：①WM 预训练预测价值+未来状态 ②策略条件化微调 ③真机 HIL rollout 采集 ④WM 与策略用 rollout 数据联合精炼；理论上证明 RECAP=忽略 z 的退化特例）
- RL 细节：RAMP（"奖励条件化"式 RL，非策略梯度；对比 AWR/RECAP；稠密未来 latent 条件替代稀疏 advantage）
- 关键数字：较 RECAP 基线在 Box Packing / Espresso Preparation / Laundry Folding 提升约 **30 点**；RoboChallenge 平均 51.67%（中间版本 GigaBrain-0.1 排名第一，超 π0.5 42.67% 约 9 点）；真实长程任务无中断完成（[全文](https://arxiv.org/html/2602.12099v2)）
- 局限：真机评测体系内部自评为主；"联合精炼"对 WM 的更新细节未完全公开；代码未见
- Motus 关联度：高——"WM 价值+未来状态作策略条件 + 迭代联合精炼"与 Motus 的"同奖励更新多分支"目标同构；其"未来 latent 比稀疏 advantage 信息增益大"的论证对 Motus 奖励设计有用

### C4. World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training
- 链接：arXiv [2509.24948](https://arxiv.org/abs/2509.24948)；代码 [github.com/amap-cvlab/world-env](https://github.com/amap-cvlab/world-env)
- 机构+时间：中山大学 + 高德/阿里；2025-09-29（v6）
- WAM 角色：**模拟器 + 奖励（VLM instant reflector）**：几何感知特征注入（VGGT+CLIP）的扩散 WM；LLaVA+reward head 连续奖励并可即时终止
- 联合标签：**L3**（WM 冻结；RLOO+PPO 只更新 VLA [LoRA r32]；48h/8×H20）
- 关键数字：每任务仅 5 条示范；LIBERO 全 suites 超 Octo（用 10% 数据）；相对 RIPT-VLA 可比且可部署真机（[全文](https://arxiv.org/html/2509.24948v6)）
- 局限：WM 数据需增广自我探索；终止阈值 η 需调
- Motus 关联度：中——"WM 内 RL + 连续奖励防 advantage 退化"的工程细节可借鉴

### C5. Motus / Motus2（与目标方法的关系锚点）
- Motus（[2512.13030](https://arxiv.org/abs/2512.13030)）：MoT 三专家（理解/视频生成/动作）+ UniDiffuser 式调度器，支持 VAM/VLA/逆动力学/视频生成/视频-动作联合预测多模式切换；光流 latent action；仿真 +15% vs X-VLA、+45% vs π0.5；真机 +11~48%（摘要）
- Motus2（[2608.30237](https://arxiv.org/abs/2608.30237)）：**自演化**——同一网络三接口：policy（world–action）/simulator（action-conditioned WM）/evaluator（value，VLAC 相对进度监督含失败负样本）；MBRL 用 **DiffusionNFT** 把分支价值转为流匹配策略更新；**只更新动作通路，共享视频骨干与 evaluator 冻结**；计划/评估：Motus2 65.0% → +Planning 67.5% → +MBRL **72.5%** → +MBRL+Planning **75.0%**（Put Phone/Multi-Finger 平均）；异步 MBRL（Ray 宏异步，仿 AcceRL）（[全文](https://arxiv.org/html/2608.30237v2)）
- Motus2 局限：视频骨干不参与 RL 更新（自演化只到动作层）；评测任务少（20 rollouts）
- **对 Motus 目标的自检**：公开的 Motus2 已实现"单奖励信号→动作参数"闭环，但**尚未实现"同奖励更新世界/视频分支"**——这正是本线调研确认的公开空白（DRILL 是唯一从 RL 目标回传 WM 的方法，且未含视频分支+VLM 同训）

### C6. 其他 RL×WM 相关（一轮扫描，简记）
- RISE（[2602.11075](https://arxiv.org/abs/2602.11075)，RSS 2026，OpenDriveLab）：组合式 WM（多视角动态模型 + progress value model）产 advantage，闭环自我改进；真机 +35%~+45%（摘要）
- World-Gymnast（[2602.02454](https://arxiv.org/abs/2602.02454)）：在 action-conditioned 视频 WM 中 RL 微调 VLA、VLM 给奖励；Bridge 上超 SFT 最多 18×、超软件模拟器 2×；支持在线迭代 WM+策略改进（摘要）
- MERL / Trust-Calibrated（[2610.16517](https://arxiv.org/abs/2610.16517)）：**在线更新**的模拟器（WM+奖励代理头）+ trust 校准控制想象数据比例；LIBERO-PRO 79.7%、真机 80.0%（摘要）
- VLA-MBPO（[2603.20607](https://arxiv.org/abs/2603.20607)）：UMM 世界模型 + 多视角一致性 + chunk 级分支 rollout（摘要）；Sword（[2605.07288](https://arxiv.org/abs/2605.07288)）：风格鲁棒 WM（结构引导风格增广 + 动态潜引导），作 VLA post-training 模拟器优于 WoVR 基线（摘要）
- EchoArena（[CVPRW 2026, pp.4486-4494](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Wang_EchoArena_Learning_World_Models_for_Reliable_VLA_Policy_Evaluation_CVPRW_2026_paper.html)）：**evaluator-centric WM**，以"策略排序与真机一致性"评估 WM（与 WAM-RL 同组：[Xiaowei Chi/Zezhong Qian/Shanghang Zhang]）
- ARM（[CVPRW 2026, pp.4468-4477](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Mao_ARM_Advantage_Reward_Modeling_for_Long-Horizon_Manipulation_CVPRW_2026_paper.html)）：相对优势奖励建模（三态标注）→ 离线 RL，长程任务 99.4%（摘要）；RoboWM-Bench（[pp.4455-4460](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Jiang_RoboWM-Bench_A_Benchmark_for_Evaluating_World_Models_in_Robotic_Manipulation_CVPRW_2026_paper.html)）：把生成行为转成可执行动作验证的 WM 基准（摘要）
- WAM 综述（[2609.16074](https://arxiv.org/abs/2609.16074)）：WAM 与经典 WM/model-based RL 的关系梳理（19 页，摘要）；2026-10 WAM 新文波（标题级，arXiv API）：OSWAM(2610.35850)、TurboWAM(2610.33941)、UniFlowWAM(2610.30991)、PrismWAM(2610.26564)、TrajWAM(2610.25763)、CW-WAM(2610.30117)、SmolWAM(2610.13740) 等

## D. 小结：RL×WAM 的三种角色（哪些真的联合更新视频/世界分支）

**① 真正把 RL/任务信号用于更新"世界分支"的：只有两个半**
- **DRILL（2610.30814）**：唯一显式用"策略回报的外层梯度"更新 WM 参数（bilevel；DRILL-VWI/IMG）——但未含视频生成分支+VLM 同训，也未开源
- **WAM-RL/WA-RL（2606.17906）**：WM 通过"在线视频 SFT + KL"随交互更新（数据驱动，非奖励梯度）+ actor 重建奖励 RL——**联合优化叙事最强，但 WM 信号不是任务奖励**
- **Motus2（2608.30237）**：同网络三接口 + MBRL，但**视频骨干与 evaluator 冻结**，自演化实际只更新动作通路（半个）
- 次强（L2 交替）：WoVR/WoVR+（WM 数据回流精炼）、World-VLA-Loop（迭代闭环）、GigaBrain RAMP（联合精炼）、MERL（在线更新模拟器）

**② 只把 WM 当外挂（WM 冻结/离线训练，只更新策略）：**
- WMPO（WM+reward model 冻结，GRPO）、VLA-RFT（138M WM 冻结）、World-Env（WM 冻结，RLOO+PPO）、RISE、World-Gymnast、VLA-MBPO、VLA-Dreamer（提案级）
- 测试时计算/选择（**L4**）：τ₀-VLA（beam+value）、Best-of-N（fail-gated selector）、DR-WM（物理 WM 选动作）——权重完全不变

**③ "视频-动作联合 + RL"的交集现状：**
- 公开工作中，"视频-动作联合训练"（Fast-WAM、LingBot-VA、Trace-WAM、GAWM、RynnWorld、Motus）与"RL 更新"基本是**两拨人**：联合训练派做大规模监督/自监督，RL 派把 WM 当冻结模拟器。"同任务奖励同时更新视频分支+动作分支+VLM"的组合**目前只有 DRILL 半只脚 + WAM-RL 的 SFT 版**——这是 Motus 方向可主张的空位（E3）

## E. 存疑 / 受限项

1. **工具受限**：web_fetch 在本会话被沙箱拒绝（所有域名→"非公网 IP"），改用 bash `curl` 直连抓取（arXiv API/CVF 均通）。抓取失败备选中 `ar5iv/export/browse-export` 未测（curl 已够用）。
2. **"DR-WM-1.0（DEEP Robotics）"**：arXiv 全库检索（`all:"DR-WM-1.0"`）为空；仅有 DR-WM（2610.17630），使用 DEEPRobotics M20/Lite3 数据但未列公司 affiliation——该条目的"名称/归属"待核。
3. **WAM-RL 无代码/项目页**；其 CVPRW 版（WA-RL）与 arXiv 版同摘要、同作者，页码 pp.4478-4485。
4. **大量数字来自论文 HTML/摘要（E3）**，未复现；Best-of-N headroom 结论限于 4 个测试 WAM×2 基准；VLA-Dreamer 为概念论文（无实验），其"embedding 足以预测未来"仍是假设。
5. **GigaBrain-0.5M\* 的 ~30 点提升为单来源（项目自评）**，"联合精炼"对 WM 更新细节未公开；Fast-WAM/Trace-WAM/GAWM 对 Motus 的对比数字来自其各自第三方重述（Fast-WAM 表 1 引 Motus 87.8/77.3、LingBot-VA 92.2/80.6），未查 Motus 原文校对。

## F. 对 910B 可落地性的粗判（E3，粗判）

- **最可落地（小模型）**：GAWM（<100M、8.8ms、RTX PRO 5000 45-1749 GPU-h 级训练）→ 910B 训练/推理都现实；VLA-RFT（138M WM + OpenVLA-OFT 级）→ WM 本身小，瓶颈在 VLA 骨干。
- **需要适配工程量**：WMPO / WoVR+ / World-Env 类"WM 内 RL"框架——若走 **RLinf**（工作区已有源码；WoVR 官方集成 RLinf，HF 集合 `RLinf/rlinf-wovr`），可先核 RLinf 的昇腾支持再选型；WM 为扩散/OpenSora 级，需 torch_npu 适配。
- **重（慎选）**：WAM-RL（8×A800 8h、DiT 视频分支+KL）、LingBot-VA（10.9B）、Motus/Motus2（Wan2.2 骨干、11.9B 级、异步流水线）、τ₀-VLA（9B proposal + WM beam search，推理展开大）、Best-of-N（每集 +17.5 imagination passes，延迟敏感）。
- **建议路径（供 Lead 参考）**：在 910B 上先复现"小 WM（VLA-RFT/GAWM 式）+ GRPO"打通"WM 内 RL"管道 → 再做 WAM-RL 式"actor-only vs 联合更新"消融（其结论：长程任务必须联合）→ 最后尝试 DRILL-VWI 式"按策略访问度重加权 WM 损失"（无需二阶导，工程量可控），这正是向"同任务奖励更新世界分支"递进的三步。Motus 自身可先做 **Fast-WAM 式测试时去想象的消融**与 **Trace-WAM 式辅助目标**，成本低于全量联合 RL。
- 硬件注意：以上均为 NVIDIA 上的报告值，910B 实测需重新测延迟/显存；本判断不含实测（无节点数据）。

> 备注：本文件为分线 C 笔记（Lead 汇总素材），所有链接均为抓取当日可访问版本；"（摘要）"= 仅 arXiv API/摘要页来源；其余为 HTML 全文抓取。未核项已在 §E 列明。
