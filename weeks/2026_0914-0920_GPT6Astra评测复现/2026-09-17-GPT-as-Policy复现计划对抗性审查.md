# GPT-as-Policy 复现计划：独立 agent 对抗性审查

> v1.0 · 2026-09-17。被审版本：`fb48ba5` 中的[迁移计划 v1.0](2026-09-17-GPT-as-Policy在MuJoCo与昇腾上的复现计划.md)。
>
> 用户要求开新 agent 做对抗性审查；独立 reviewer 只读检查，主 agent 对关键源码和输入哈希再复核。本轮仅归档发现，**未修改被审计划或实施迁移**，未运行付费模型/NPU 实验。
>
> 证据等级：E1=源码实证；E2=配置、哈希实证；E3=影响判断与修订建议。以下三项均为已定位的规格缺口，不是已发生的运行失败。

## 审查结论

**有条件通过：发现 3 项 P2 可执行规格缺口，应在正式实现/评测前补齐。** 没有发现必须改用相似任务、替换原权重或删除困难任务的证据；这也不等于已证明全部任务可高保真迁移。

审查对象包括原任务范围、原始输入证据、MuJoCo 迁移边界、NPU 数值路径、agent 接入语义、评测与异常统计。保留原十任务的方向成立；cloth、渲染和 NPU 等价性仍需要实施证据。

## F1 [P2] 支持臂资产与 250 Hz 物理步时序未明确纳入验收

定位：[计划第 121 行](2026-09-17-GPT-as-Policy在MuJoCo与昇腾上的复现计划.md)，关联第 80、84、123、147 行。

**事实（E1/E2）：** `imitate_sorting_sequence` 和 `make_kong` 除双 ARX X5 外，还有第三台 Franka 支持臂。支持轨迹包含机械臂、夹爪的位置与速度，在主策略每次动作内部逐物理步推进。原 `dt=0.004 s`，即物理 250 Hz；主策略/观测 25 Hz，即每控制动作 10 个物理子步。

证据（均固定原 RoboDojo commit `ee67a1468510da7624a089164402359f2afc72c8`）：

- [第三台 Franka 的配置](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env_cfg/robot/dual_x5_and_franka_competition.yml#L18)：18–27 行。
- [支持轨迹字段](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/task/RoboDojo/tasks/imitate_sorting_sequence.py#L47)：47–70 行。
- [逐物理步回放与主臂插值](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/src/eval_client/eval_env.py#L500)：500–557 行。
- [物理步长](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env_cfg/sim/sim_config.yml#L2)与[25 Hz 观测配置](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/env_cfg/arx_x5.yml#L10)。

**影响（E3）：** 计划已经提及支持臂，不能说完全遗漏；但资产工作包只明确列出 X5，控制契约只明确列出 25 Hz。照此实施可能遗漏 Franka 驱动或错误回放支持轨迹，改变演示、弃牌接触和原有效性判断。

**最小修订（E3）：** A 工作包加入 Franka 资产、初态、驱动、夹爪与关节映射；C 和控制验收加入 250 Hz/10 子步、原插值公式、支持轨迹采样及队列耗尽时序，并验证演示结束后的原生有效性条件。第三臂仍由原支持逻辑驱动，不能误加到 GPT 的 14 维策略动作中。

## F2 [P2] 50-case 的无效运行、重试和最终 attempt 选取尚未冻结

定位：计划第 158、164 行。

**事实（E1）：** 原实现区分普通任务失败、无效布局、预算截断、不完整运行及经裁定的基础设施失败。支持臂演示失效可将环境标为 unstable；此时原 session 输出 `valid_for_success_rate=False`，成功与分数为 `null`。原 Direct 的部分 idle timeout 经裁定计作失败，但没有补造 native score。

证据：

- [支持臂演示失效检查](https://github.com/RoboDojo-Benchmark/RoboDojo/blob/ee67a1468510da7624a089164402359f2afc72c8/task/RoboDojo/tasks/imitate_sorting_sequence.py#L90)：90–117 行。
- [原结果资格、状态和空分数](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/robodojo_server/session.py#L178)：178–189 行。
- [固定分母及 idle timeout 裁定](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/paired_evaluation.py#L135)：135–145 行。

**影响（E3）：** “保留全部 attempt、分别记账”不能决定最终用哪一次运行。自动重试至成功、选择最佳 attempt 或剔除无效 case，都会改变比较口径。迁移后支持轨迹失效使这个问题更值得提前处理；目前没有运行证据证明其发生率会上升。

**最小修订（E3）：** 冻结终局状态、允许重试的原因、次数与最终 attempt 选取规则；保留原 50 个 case 身份，不自动补新 case，不按成功优先选取。invalid/incomplete 与普通失败分开；有效配对不足时报告缺项和有效数量。若沿用上游基础设施失败裁定，明确触发标准与 native score 保持空值的规则。

## F3 [P2] 正式 NPU 运行的随机噪声序列未定义

定位：计划第 137–140 行。

**事实（E1）：** 原服务声明每个 episode 使用全新 JAX seed0。固定 OpenPI 在 JAX 路径初始化 `jax.random.key(0)`，每次 infer 分裂 key，用该 key 生成 `(batch, horizon, internal_action_dim)` 噪声；PyTorch 路径忽略传入的 `rng`。所以数值验收注入同一噪声，不等于正式运行使用 `torch.manual_seed(0)` 便保持原随机序列。

证据：

- [原服务 RNG 身份](https://github.com/anonymous-report-421/GPT-as-Policy/blob/8f3d362b077d8efb77e2a7274d5b2c20e2243846/hybrid_rollout/robodojo/pi05_server/server.py#L29)。
- [固定 OpenPI 的 RNG 与 noise 注入](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/policies/policy.py#L41)：41、65、85、94 行；68、98–103 行已有 `noise` 参数接口。
- [JAX 动作采样噪声](https://github.com/XPolicyLab/XPolicyLab/blob/432f82b1758c5b1202e42a3dfe014546dbc50871/policy/Pi_05/openpi/src/openpi/models/pi0.py#L230)：230–231 行。

**影响（E3）：** 数值转换通过后，候选动作仍可能因随机流改变而变化。若继续只记录 `policy_rng_seed=0` 而不披露后端序列变化，会夸大与原策略的一致性。

**最小修订（E3）：** 正式运行也在 CPU 端按原 JAX 版本、PRNG 配置、key 分裂与形状生成噪声，再注入 PyTorch/NPU；每个 episode 重置，按 `inference_index` 留存噪声哈希，重试不意外推进 RNG。若改变随机流，作为显式协议变化记录，不能声称采样身份等价。噪声形状依据模型内部动作维度，不直接用解码后的 14 维。

## 已披露风险：不重复计为新缺陷

- **柔性物理：** 原 cloth 的粒子材料、拉伸/弯折/剪切、自碰撞与 MuJoCo 表示的映射需要验证。MuJoCo 有柔性对象支持不代表原参数可直接等价迁移。计划已承认该风险。（E1 原 garment 实现；E3 工程判断）
- **NPU 模型：** 原配置是 `Pi0Config(pi05=True)`；昇腾 LeRobot/mock 样例不能代替原 checkpoint 的证明。计划已安排分层核验，没有据此宣布迁移成功。（E1/E3）
- **RoboLab：** 已核查的公开 `skill/run.py` 仅见 start/infer/execute 的 Hybrid 入口；原 Direct 实现及完整案例身份仍待取得或证明。计划第 113 行已披露，不能另算“隐瞒缺项”；重建入口时仍须说明不是已保留的原实现。（E1 已查范围；E3 边界判断）
- **Mac/模型服务：** 上游采用持久 Codex app-server agent。计划明确裸 API 不自动等价，实际模型、推理设置、工具和记忆语义仍需运行验证。（E1/E3）

## 主 agent 的独立复核

1. 对暂存的 50 个布局和 75 个源文件重新读取实际字节并计算 SHA256，全部与原 panel 匹配。75 是 panel 中 169 个源文件的子集；原计划“抽查”表述正确。（E2）
2. 根据原 `checkpoint_identity` 的 `params + assets` 算法，以固定 HF revision 的大文件 LFS SHA256 元数据及小文件实际下载哈希，计算 17 个文件的聚合哈希：

   `d15fb8bd1d29cb30b69f01b71c66596cb0293c1a8a94111b343c1580dd3e3e5b`

   与 `SOURCE.json` 的 `previous_load_sha256` 完全一致。见[权重身份元数据核查](2026-09-17-GPT-as-Policy权重身份元数据核查.json)。这加强了候选权重身份的证据；大文件内容未下载，不能称全量文件完整性、加载或推理通过。（E2）
3. 复读支持臂配置/250 Hz 子步、结果资格/空分数、JAX/PyTorch RNG 分支，确认上述三项反馈有源码依据。（E1）

未发现需要修正的方向：固定原案例、区分抽查与全量审计、元数据验证边界、复用原 FK/DLS、拒绝相似 Demo、评分真值隔离、不承诺重现原成功率。

本轮保留原计划 v1.0 作为被审基线。三项发现的状态均为“待修订规格”，没有把审查意见写成已完成修复或已通过实验。
