# GPU＋Isaac 与 MuJoCo＋LIBERO：两条路线怎么选

> v1.1 · 2026-09-20  
> 当前继续以 GPU＋Isaac 为主线，MuJoCo＋LIBERO 保留为候选。

## GPU＋Isaac：便于继续做现有双臂实验

GPU 负责 Isaac 仿真，NPU 负责 π0.5 推理，Mac 调用云端 GPT。

**优势是能延续现有工作。** 当前研究的是双臂四瓶入桶，沿用这条路线，可以在同一任务上比较 GPT 决策间隔、π0.5 执行方式和混合策略，不需要先换场景、重新建立基线。

**不足是部署和通信较复杂。** 仿真与推理分在两台服务器，图像和动作需要跨机传输，再加上 Mac 的 GPT 调用，排查耗时需要同时检查多条链路。实验也依赖独立的 GPU 资源。通信究竟占多少时间，仍需实测。

## MuJoCo＋LIBERO：便于在一台服务器上扩展测试

NPU 服务器的 CPU 运行 MuJoCo 仿真，NPU 运行 π0.5；需要云端决策时再通过 Mac 调用模型。

**优势是部署集中，现成任务较多。** 仿真和推理可以放在同一台服务器，省去独立 GPU 仿真节点及其跨机传输。LIBERO 提供任务、初始状态和评分方法，适合增加测试场景，观察不同策略在位置、物体等变化下的表现。

**不足是与当前双臂任务有差异。** 我们现有的 LIBERO 环境是单臂，机器人、动作接口和模型权重都与双臂方案不同，成绩不能直接比较。CPU 软件渲染也可能较慢，因此部署简化并不保证整轮运行更快。

## 目前的取舍

如果重点是改进现有双臂任务，继续用 GPU＋Isaac 更直接；如果重点是减少部署依赖、扩大任务测试，MuJoCo＋LIBERO 更有吸引力。

目前先保留 GPU＋Isaac 主线，用 LIBERO 做补充测试，最终选择待定。还需区分：**MuJoCo 并不局限于单臂，单臂是当前 LIBERO 环境的配置。** 如果希望同机部署又保留双臂任务，可以进一步检查已有的 RoboDojo 双臂 MuJoCo 移植。

---

依据：部署取舍为工程判断【E3】；LIBERO 接口及任务范围见[策略代码](../../gpt-as-policy-repro/src/gap_repro/libero/policy.py)与[任务记录](../../gpt-as-policy-repro/docs/acceptance/l0-l1-baseline-and-cases.md)【E1/E2】；双臂 MuJoCo 移植见 [RoboDojo 分支](https://github.com/versatile-ai/embodied-ai/tree/3f45991828784efbd84491b996fc3fe4b65484b7/simulation/mujoco)【E1】。
