# MJWarp 架构讲解讲稿（对齐 v1.3 报告）

> 配合阅读：`MuJoCo_Warp架构与昇腾NPU全量迁移B路线报告.md` v1.3
> 时长：30–35 分钟讲解 + 10 分钟 Q&A
> 听众假设：懂 MuJoCo / RL 基础，不一定懂 GPU 编程与昇腾
> 讲解原则：只讲已验证结论；凡报告中标“待P0/待复核”处，口头必须带“待实测”，不报绝对数字

---

## 0. 开场（2 分钟）

大家好，今天讲 MJWarp 的架构。大家手里这份报告是 v1.3，标题已经改了——叫“物理核心子集迁移昇腾可行性研究”，不是“全量迁移”。

先说清楚今天的目标和非目标：

- 目标：讲明白 MJWarp 为什么能大批量跑，它的数据和执行跟经典 MuJoCo有什么本质不同。
- 非目标：今天不等出 NPU 移植结论。NPU 部分只在最后给一句话展望，详细可行性看报告 §11–§19。

报告页眉有三行拆分，大家先记住，后面不再重复：上游仓库总量、外审值 5.28 万行 / 约 296 个 kernel 定义，这是总量；G1 实际跑到哪些，还要做 event trace；我们 PoC 要重写多少，待定。三者不要混。

---

## 1. 生态定位：三套实现（5 分钟，对应报告 §2）

大家看报告 §2 那张图。

同样一份 MJCF，今天有三条路：

1. 经典 MuJoCo：C 写的，CPU 上跑，默认 float64，延迟最低。做实时控制、MPC，还是它。
2. MJX-JAX：JAX 重写，pytrees 固定 shape，靠 `vmap/pmap` + XLA 上 GPU/TPU，可微。
3. MJWarp：Warp 重写，SoA + batch，上 NVIDIA GPU，不可微，对 PyTorch 友好。

关键一句话：`mjwarp.step` 是 `mj_step` 的重实现，不是把 C 代码编译到 GPU。模型语义一致，代码路径不同，所以长轨迹不保证一致。这个结论后面 §9 还会回扣。

听众常问“哪个更快”：不要直接答数字。按官方定位说——单步延迟经典最优，批量吞吐 MJWarp 占优，具体倍数要同机 benchmark，报告里已经把旧的绝对数字删了。

过渡：既然是重实现，它重在哪里？看分层。

---

## 2. 总体分层（4 分钟，对应 §3）

大家看 §3 的五层图，我在白板上再画一遍，自底向上：

- L1 硬件：NVIDIA SM + HBM，SIMT warp32，SharedMem，Atomics。
- L2 算子：30 不是 30，是外审值约 296 个 kernel 定义。注意定义数不等于运行时 launch 数，运行时到底 launch 多少次，要看 event trace。
- L3 调度：Warp Runtime + CUDA Graph + 多卡 ScopedDevice。
- L4 API：`Model/Data/step/forward`，`put_model/make_data`。
- L5 应用：Brax、MJX API、IsaacLab、mjlab、Playground。

源码结构注意：报告 v1.2 那版文件列表已经作废。v1.3 按外审改了，现在集中在 `smooth.py / collision_driver.py / collision_convex.py / island.py / sleep.py / block_cholesky.py` 等，P0 要拿锁定 commit `7e4afee` 的 `ls` 贴进来。今天不要背文件名，就说“以 ls 为准”。

过渡：分层里最值得展开的是 L4 的数据结构，这也是 GPU 化的核心。

---

## 3. 数据结构：为 GPU 重生（7 分钟，对应 §4）

这是今天最重要的一节。经典 `mjModel` 是 C 结构体 + 指针 + 变长，GPU 吃不了。MJWarp 做了四件事，注意措辞——是“运行时预分配、容量固定、shape 稳定”，不是编译时定长。

第一，SoA。原来一个 Body 存 pos + quat，现在拆成 `body_pos` 一列、`body_quat` 一列。报告里有个细节 v1.3 刚改：`wp.array2d[vec3f]` 逻辑 shape 是 `(nworld, nbody)`，每个元素本身就是 vec3，不要说成 `(nworld, nbody, 3)`。

第二，batch 前导维。几乎所有可变字段多个 `[*]` 维，默认 batch=1 广播全 world，也可以按字段设不同 batch 做域随机化，语义是 `field[world % batch]`。这里纠正一个旧说法：不是“零成本”，取模和额外带宽仍有开销，只是省了逐步 H2D 更新。

第三，稀疏化。`efc.J` 这类只存非零 + colind，报告里 Aloha 的例子稠密 408MB 变稀疏 84MB，就是这么来的。

第四，静态容量。`nworld / nconmax / njmax / nccdmax / nvmax` 在 `make_data` 时固化，超了记 `Data.overflow` bitmask，没有 malloc，没有变长。

Model vs Data 一句话：Model 只读常驻，`put_model` 一次 H2D；Data 可变每 world 独立，`make_data` 分配全部 buffer + solver workspace。

过渡：数据摆好了，怎么执行？看编译流水线。

---

## 4. 编译与执行（5 分钟，对应 §5）

Warp 是 Python 内嵌 DSL，`@wp.kernel` JIT 到 CUDA，缓存复用。举例注意不要再用 kinematics 讲原子——kinematics 不写 `ncon`。报告 v1.3 换成了 narrowphase 的示意：`tid` 是线性索引，自己解码成 `(world, pair)`，算出 contact 后用 `atomic_add` 预约写入位置。

然后是重点纠错，也是外审抓的：CUDA Graph 是什么？

- Eager：每步逐个 launch 到同一 stream，stream 内异步顺序执行，不是每 kernel 间 synchronize。开销在 launch 本身。
- Graph：`ScopedCapture` 把 launch 序列捕获成 CUDA Graph，后面 `capture_launch` 重放。注意——是捕获 / 重放，不是从多个 kernel 融合成一个大 kernel。今天谁再说 fusion，直接纠正。

多 GPU 一句话：`ScopedDevice` 每卡独立 Model/Data/graph，手动分发，没有自动 pmap。

最后一句回扣吞吐：Data 常驻 HBM，PyTorch 经互操作零拷贝拿，省了经典每步的 host-device 搬运。NPU 如果退化成每步 NPU-CPU-NPU 搬运，优势就没了。

过渡：有了数据和调度，一步 `step` 到底干了什么？

---

## 5. 前向流水线（8 分钟，对应 §6）

大家看 §6 的表，顺序以 event trace 为准，我按逻辑讲，真实切分 P0 再核。

1. Kinematics：`xpos/xquat/xmat`，crb，bias。
2. Tendon / Transmission / Actuator：稀疏装配。
3. Broadphase：AABB/SAP 粗筛，受 sleep 过滤，`nconmax` 定内存。
4. Narrowphase：三条路径必须分开说——MuJoCo C 是经典 CPU 实现；MJX-JAX 是 branchless SAT 等；MJWarp 是 primitive + GJK/EPA。不要再合写 SAT/GJK。
5. Flex：experimental，人形刚体一期可裁；hfield 是地形刚需，不跟着裁。
6. make_efc：contact/limit/equality 转成 efc，`njmax` 硬截断。
7. Solver Setup + 8. Iterate：v1.3 改了，不再默认稠密。sparse/dense/compact 多路径，G1 走哪条要 trace，Cube 值不值得用后面再定。EarlyExit 收敛就跳，所以 iterations 对 MJWarp 不像对 MJX 那么敏感。
8. Integrate：Euler，IMPLICITFAST 不支持。
9. Sensor + Sleep 更新：sleep 不是尾阶段，是贯穿 broadphase 和 solver 的。

Compact Solver 单独说一句：大 DoF 把 active 岛 compact 到固定 tile 再做 Cholesky，避免发散。Other memory 经常超过 Model/Data 本身，拿 `--memory` 看。

过渡：流水线讲完，就能回答为什么能大批量。

---

## 6. 为什么能大批量（5 分钟，对应 §7）

五条，趋势对，不报绝对值：

1. 环境独立 SPMD：world 间无通信，`tid` 解码，world 越多并行度越高。
2. 容量固定 shape 稳定：一次编译，无动态分支。
3. 图捕获重放：省 launch 开销，不是 fusion。
4. 合并访问 SoA：连续访存，显著高于 AoS，具体待 ncu。
5. 设备常驻零拷贝：`host-device` 每步搬运省掉。

再泼一盆冷水：单环境 MJWarp 延迟通常不如经典，批量后总吞吐才反超。报告里旧的毫秒数和百分比已经删了，今天也不要报。

---

## 7. 性能与一致性（3 分钟，对应 §8–§9）

性能只讲方法：`--measure_alloc --overflow_behavior=error --memory --event_trace` 调 `nconmax/njmax/nccdmax/nvmax`；`warn_overflow` 的 printf 会串行化，`d.overflow.numpy()` 是 D2H（注意方向，报告 v1.3 刚改），不要每步同步。

一致性讲三层：XML 语义大体一致；单步接近不逐位；长轨迹接触密集必分叉。α 表已经移到附录 C，它是经典 vs PhysX 的背景，不是 MJWarp-NPU 门禁，今天不要拿它下结论。

---

## 8. 收尾 + NPU 一句话（2 分钟）

MJWarp 的批量是体系化设计的结果：SoA + 容量固定 + 图捕获重放 + 零拷贝，不是单点优化。

NPU 一句话：GPU 是同 kernel 跑多 world，NPU 是算子切 tile 搬运计算，直译 `tid/atomic` 跑不动；B 拆纯 NPU 与带 fallback 的 B′；solver 先 trace 再定 Cube；后端五选一 P0 比选。细节看报告后半，不展开。

谢谢，大家看 Q&A。

---

## Q&A 预判（备用）

1. 问：到底多少 kernel / 多少行？答：仓库总量外审值 5.28 万行 / 296 定义，G1 子集和重写量都要 P0 trace 后给，今天不报。
2. 问：Graph 能加速多少？答：省 launch 开销，不是 fusion，具体看 event trace。
3. 问：NPU 能直接跑 Warp 吗？答：不能，编程模型不对，要重写；走 B3 子集 PoC 还是 B1，看 P0。
4. 问：精度能对齐吗？答：只做集合比对 + 分布统计，不做长轨迹逐位；附录 α 不作门禁。
5. 问：CCD 能回退 CPU 吗？答：B 不允许，B′ 允许但要量化每次同步损失。

---

## 讲者自查（开讲前 1 分钟）

- 没说“8000行/30 kernel/1–2数量级/5k/10k”旧数字？
- Graph 说成捕获/重放而非 fusion？
- 碰撞说成三分法而非 SAT/GJK 合写？
- overflow 说成 D2H？
- shape 说成 `(nworld,nbody)` + vec3 元素？
- 生态判断带“本次检索尚未确认、P0复核”？
- 超纲问题统一收敛到“待P0”？
