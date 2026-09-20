# W0 运行条件与两种接入的能力清单（阶段性记录）

> 状态：开发机（本 Mac）部分已核，目标执行机（昇腾 NPU 服务器）未接入，相关项 **外部缺失**。
> 依据计划 §6.3 W0；本文件只记录非敏感配置，凭据与接入信息不入库。

## 已满足（开发机，2026-09-17 实测）

| 项 | 结果 |
|---|---|
| 硬件/OS | macOS 26.6.2 arm64，磁盘可用 83GB |
| 工具链 | git 2.51.0；系统 python 3.13.1；pytest 9.0.2 |
| 网络 | GitHub 与 HuggingFace 连通（间歇不稳，pip 曾超时一次后重试成功） |
| 上游固定源 | 三仓库 shallow clone 于固定 commit，HEAD 与计划 §2 一致（见 upstream.lock.json） |
| 专用依赖环境 | `.venv/`（本工程内，gitignored），依赖安装不污染系统 |
| W1 工具链 | `src/gap_repro/inputs.py` 纯标准库；12 项契约测试全绿（四类反例覆盖） |

本机 MuJoCo / JAX 探测：见下节（若标记完成则已有实测记录）。

## 本机 MuJoCo / JAX CPU 探测（2026-09-17 实测，仅可运行性证据）

| 项 | 结果 |
|---|---|
| macOS arm64 wheel 安装 | mujoco 3.13.0 + jax 0.11.1（Python 3.13 venv；PyPI 间歇超时，重试成功） |
| MuJoCo CPU 步进 + 有效 RGB | 通过：free-joint box 50 步 qpos 有限；`mj_Renderer` 64×64×3 有效图像（mean≈120.8, std≈34.9） |
| JAX CPU PRNG | 可运行且确定性成立（plain 与 jit 逐位一致），backend=cpu；**默认实现是 Threefry**（此前误记为 Philox，已更正） |

**非基线声明：** 本机 jax 0.11.1 的 PRNG 输出**不得**作为 W7 原策略噪声基准。
上游 openpi 固定 `jax[cuda12]==0.5.3`（E1：`policy/Pi_05/openpi/pyproject.toml` L17，
uv.lock jax/jaxlib 均为 0.5.3）。W7 必须在服务器锁定 jax==0.5.3 环境下按原
PRNG 配置重新生成基线噪声并与原代码逐字节对比；本机首 4 值
`[1.0040143, -0.9063372, -0.7481722, -1.1713669]` 仅为 jax 0.11.1/Threefry 的
可运行性记录。本机探测不替代目标机 W0 验收。

## 运行时依赖 pin（开发机已装 / 服务器待 W0 锁定）

| 依赖 | 开发机（.venv） | 上游固定（E1） | 服务器（W0 待定） |
|---|---|---|---|
| jax / jaxlib | 0.11.1（仅可运行性探测） | 0.5.3（openpi pyproject L17） | W0 锁定 0.5.3 |
| mujoco | 3.13.0 | 原链路为 Isaac，无 mujoco pin | W0 实测选择并锁定 |
| pytest | 9.0.2（requirements-dev.txt） | — | 随工程 requirements-dev.txt |

## 执行机接入

用户已提供 NPU 服务器的 SSH 主机别名（2026-09-17）。别名与接入细节按计划
不入库；BatchMode 实测**可达**（key 认证生效）。已确认（W0 部分，待完整实测补充）：

- CPU 架构 **aarch64**（Huawei Cloud EulerOS 2.0），192 逻辑核；Docker 27.2，
  **ascend runtime 为默认运行时**；8×910B3（64GB HBM）健康空闲；host CANN 8.5.2。
- 服务器到 PyPI / HuggingFace / hf-mirror 均连通。

## 服务器容器与 π₀.₅ 推理服务（2026-09-17 部署）

**容器 `gap-repro`**（运行中，在 <实验服务器>）：镜像 `pi05-hybrid:v1`（同事交付包，
`docker load` 自共享盘 tar，18.3GB）；`--privileged --network host --shm-size 64g`；
`ASCEND_VISIBLE_DEVICES=0`（只用 0 号卡）；挂载：工程 repo→`/workspace/repo`、
数据盘→`/workspace/data`、我们的 pi05 副本→`/data/pi05_hybrid`（同事脚本的
硬编码路径，读我们自己的副本，不动共享盘原件）、host driver→ro。

实测要点：
- 必须按指南 `--privileged`：仅靠 ascend runtime 设备注入缺 `devmm_svm`/`hisi_hdc`，
  torch_npu 报 npu_count=0（drvErr 87）；privileged 后 8 卡可见，卡上 matmul 实算通过。
  容器内默认仅调用 device 0。
- python 必须先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`。
- 镜像内环境（同事交付，实测读出）：Python 3.11.15、PyTorch 2.7.1+cpu、
  torch_npu 2.7.1.post2、CANN 8.5.2、MuJoCo 3.11.0、transformers 4.53.2+openpi 补丁。
- **π₀.₅ 推理服务已跑通**：`pi05_service.py npu 8642`（权重 6.8G 加载约 4 分钟），
  `/health` → `{"ok": true, "device": "npu"}`；冒烟推理合成观测 → `(50,14)` 有限值，
  首次 11.2s（含预热），稳态 **661ms/chunk**（指南称 ~570ms，同量级）。
- 部署修复一条：PaligemmaTokenizer 需要
  `/data/pi05_hybrid/tokenizer/paligemma-3b-pt-224/tokenizer.model`，已从共享盘
  拷入我们副本（我们的副本缺它会去 gs:// 下载并因无 gcsfs 崩溃）。
- 依据用户 2026-09-17 决定：推理使用同事适配版本（robodojo_pi05_pt，指南 v1），
  已克隆进我们的容器；共享盘上另有 `robodojo_pi05_pt_v2`，指南/脚本均用 v1，
  差异待同事确认，不擅自切换。
- jax/mujoco 的 W0 服务器 pin：MuJoCo 3.11.0（镜像自带，实测可用）；**jax==0.5.3 +
  jaxlib==0.5.3 已装入 gap-repro 容器（cpu 后端，2026-09-17）**，默认 PRNG 实现
  `threefry2x32`，`PRNGKey(0)`→normal(4) 首 4 值
  `[1.622642159461975, 2.0252647399902344, -0.4335944354534149, -0.07861734926700592]`
  作为 W7 的服务器端版本锚点（原噪声流仍须按 openpi 原 key 分裂次序生成比对）。

## 容器 `gap-sim`（第二个容器，LIBERO-PRO 仿真专用，2026-09-17 新起）

- 镜像 `ubuntu:22.04`，**无 NPU 设备、无 privileged**（纯 CPU，与其他容器零算力冲突）；
  host 网络（自身不开服务端口，仅作客户端访问推理服务）；挂载同 gap-repro
  （repo→`/workspace/repo`、data→`/workspace/data`）。
- 职责：LIBERO-PRO 老栈环境（conda py3.8.13 + torch 1.11 CPU 适配）+ LIBERO/PRO
  仿真与渲染。LP1 构建日志 `data/libero_env_build.log`，阶段化脚本
  `scripts/setup_libero_env.sh`（失败不阻断，用于定位旧栈在 aarch64 的断点）。
- 已知最大风险：robosuite 1.4.0→mujoco-py→MuJoCo 2.1.0 官方二进制仅 x86_64。
  **（2026-09-17 已证伪：robosuite 1.4.0 依赖 DM `mujoco>=2.3.0` bindings，有 aarch64
  wheel；实际断点是一连串小坑，全部修复。）LP1 依赖已全部装齐**；此前仅测 robosuite/bddl 导入的冒烟不充分（对抗审查 P1），
  已修复 libero 导入（预生成 config.yaml 规避交互式 input）并使 `import libero.libero`
  + benchmark 初始化通过。环境：python 3.8.13、torch 1.11.0(CPU)、
  robosuite 1.4.0、mujoco 3.2.3、robomimic 0.2.0、bddl 1.0.1、transformers 4.21.1、
  numpy 1.22.4、h5py 3.11.0。
  平台适配清单（详见脚本注释）：conda ToS 非交互接受；conda/pip 走 TUNA 镜像
  （同包同版本）；git 强制 HTTP/1.1 + 浅克隆 + SHA 校验，GitHub 弱网改由 Mac 克隆
  后 SSH 推送；大 wheel 用 curl 断点续传 + sha256 预取（fetch_big_wheels.py，
  本网络大文件单流 HTTP 必坏）；apt 补 pkg-config/libhdf5-dev/cmake/libegl-dev；
  egl_probe 需 setuptools==65.7.0 + --no-build-isolation 预装。
  遗留风险：mujoco 3.2.3 与 robosuite 1.4.0 官方验证域（2.3.x）有版本差，
  env 创建/step 冒烟在 LP1 验收实测，必要时钉 mujoco==2.3.7。
- **LP1 环境子项验收（2026-09-17；session 适配器与 reset 指纹测试未做，见下）**：`scripts/libero_env_smoke.py` 在 gap-sim 实测通过——
  LIBERO-Goal 首任务 env 创建 6.2s / reset+init 6.3s / step 0.62s（含 OSMesa 渲染），
  agentview+wrist 256×256×3 有效，robot0_joint_pos/eef/gripper/object-state 结构完整，
  dummy 动作不误终止。mujoco 3.2.3 × robosuite 1.4.0 兼容性实测排除（预留钉 2.3.7 退路）。
  产能口径（E3）：单 env ~4min/ep（300 步仿真+60 chunk 推理），7,200 集需 8–16 并发。
- **LP1 剩余未闭合**：session 适配器（src/gap_repro 对 libero 零引用）、reset 指纹
  一致性测试、robomimic 在推理路径是否可省（egl_probe 已装）的 import 链验证。
- **LP-A0 状态（2026-09-18）**：Torch 段通过（开发诊断级：CPU 双跑 determinism=0；
  注入同一噪声后 NPU vs CPU max_abs=3.565e-3）。对抗审查 P1 撤回 PASS 标签：
  JAX leg 与预注册阈值未完成——正式运行按 `configs/validation.json` 冻结阈值执行
  （8 draws：2 合成 + 6 真实帧；三腿 jax_cpu/torch_cpu/torch_npu；JAX 生成噪声注入）。
  GE/TBE 失败根因=/root/atc_data 缓存损坏（复位解决）+ 与常驻服务编译竞争
  （正式跑暂停服务，完成后恢复，8642 health ok 已验证）。
- 来源锁定见 `configs/libero_pro.lock.json`。

## W1 资产引用闭包（2026-09-17）

- 规则来源（E1）：`layout_manager.py` —— 物体
  `Assets/Object/RoboDojo/{section}/{cat}/{idx:05d}/object.usdz|usd`（cluttered→Clutter/），
  Room→`Assets/Room/{default}/` 首个 .usd，Table→`Assets/Material/{default}/` 首个 .mdl，
  Background→`Assets/Background/{category_name}`，衣物材质含 `$Robodojo_ASSETS` 前缀变量。
- **50 case 全部解析成功，闭包 355 文件 / 3.74GB，0 个未解析**：
  rigid 198、robot_franka 95、robot_x5 23、geometry 13、room 7、table 10、
  garment 4+4（含材质）、background 1。表见 `docs/acceptance/w1-asset-closure.json`
  （引用→文件→哈希→使用 case）。Franka 仅入排序/麻将支持臂 case 的使用面。
- **闭包文件已全部下载并逐字节验证：355/355 OK**（gap-repro 容器
  `/workspace/data/hf_cache`，3.7GB）。过程记录：两下载进程曾并发写同一 .part
  互相干扰导致 10 余个失败，改单进程续传后收敛；最后 2 个大文件
  （toy_car/00002、watch/00007，43/46MB）服务器侧持续超时，改由 Mac 断点续传
  下载、哈希匹配后 rsync 入位。**W1 全部输入核查闭合**（169 源码 + 50 布局 +
  9 轨迹 + 17 权重文件 + 355 资产，全部内容级）。

## W4 子步调度与执行器（2026-09-17 验收通过）

- `sim/control.py`：25Hz ACK → 10×250Hz 子步序列，复刻原 eval_env 语义——
  前 8 子步 alpha=(i+1)/9 插值、后 2 子步保持；夹爪每插值步 clip 到 scale 后配 mimic；
  支持臂队列每子步最多消费 1 项、耗尽即断、剩余跨步保留（0/1/9/10/11 全测）。
- 渐进夹爪（MetaControl 层，20% 行程/步）实现于 25Hz 目标级，插值层不重复限幅。
- 执行器：position 伺服 kp/kv=原 stiffness/damping（臂 4400/40，夹爪 2300/100），
  forcerange ±100，夹爪 ctrlrange=scale；mimic 的 joint8 不单独驱动（nu=14）。
- 容器内全量 **43/43 测试通过**。遗留 W5：velocity 钳制、软限位对齐、自碰撞接触验证。权重下载与参考推理在服务器执行，Mac 不下载完整权重。

## 外部缺失（需目标执行机 / 授权后填写）

- [ ] 目标机 CPU 架构、内存、磁盘、实际卡数/单卡容量、驱动、固件、CANN 版本。
- [ ] 目标机 Python、PyTorch、torch_npu 官方支持组合选择并锁定（`configs/runtime.lock.json`）。
- [ ] 目标机 CPU MuJoCo + OSMesa/EGL 产生有效 RGB；JAX CPU 跑通原 PRNG。
- [ ] 路线 A（Codex app-server）：实际模型身份、xhigh、图像输入、持久会话、工具调用、用量字段、限流行为。
- [ ] 路线 B（API/网关）：同上各项；两条路线差异表。
- [ ] 付费验证：本轮不启动（计划 §6.1）。

## 通过条件对照

计划 W0 通过条件 = 明确可运行的依赖组合与两路线差异。当前：开发机部分满足；
目标机与两路线核验 **blocked（等待执行环境与授权）**，不阻塞 W1–W5 的资产/任务移植，
按计划阻塞 W9/W11 相关路线的验收。

## 对抗性审查响应（2026-09-17，docs/review/2026-09-17-工程对抗性审查.md）

红级 R1–R7 全部修复（tests 60/60 容器内验证）：谓词按原文重写（15°/无 abs、
严格 <、2D 凸包+z_min、pre_state episode 基准、逐分量 0.15+20°）、双臂根
四元数改 wxyz 直通（R6，原实现绕 X 错为绕 Z）、渐进夹爪移至 250Hz 子步层
（R7）、garage per-env 阈值（Y1）、functional point 显式 NotImplementedError
（Y2）、transition 分单调+未成功封顶 75（Y4）、check_list 每 episode 登记一次
（Y5）。fixture 全部按原文语义重建（鼠标 -90° yaw、垫面上、XY 移出判失败）。

仍未闭合（如实登记，Y7）：W1 轨迹格式检查/USD 递归解析/50 布局 reset 指纹/
指令来源；W2 Franka FK 与布局初态；W4 触发时刻与 unstable 标志留痕；W5
get_obs 完整 raw 结构与 unstable_envs 实义；九任务判据镜像；W3 渲染与衣物。

## W3 渲染后端解锁（2026-09-18）
- 配方（gap-repro 容器内实测 OSMESA_RENDER_OK）：
  1. venv 安装官方 mujoco==3.11.0（镜像内置为定制构建，无 Renderer）
  2. apt-get update && apt-get install -y libosmesa6-dev
  3. MUJOCO_GL=osmesa + MUJOCO_RENDERER 使用官方 wheel
- 环境已 docker commit 为 gap-repro:v2-render（含 osmesa 库 + render_venv）。
- 残留：三相机 key/分辨率/内参与原 vision 契约的对齐（W3 后续）；
  服务进程（镜像内置定制 mujoco）与渲染 venv 分离运行。

## LP-A1 reset 指纹与 fixture 漂移归因（2026-09-18 初验，2026-09-19 P1 修复闭合）

- **session 适配器**：`src/gap_repro/sim/libero_session.py`（gap-sim 容器、conda
  `libero_pro`、`MUJOCO_GL=osmesa`）。配对初态原样使用；成功判定
  `env.env._check_success()`；wait_steps=10 对齐 openpi 客户端 E1。
- **指纹协议（P1 修复后三层）**：movables 硬门（joint_pos/eef_pos/gripper_qpos/
  object-state，逐位）＋ `fixture_pose_digest()`（场景布局）＋ `fingerprint_full()`
  （两者拼接，LP-A2 配对身份）。
- **fixture 漂移（审查 P1，已证实并入档）**：robosuite `set_init_state` 只恢复
  time/qpos/qvel；fixture `body_pos/body_quat` 是模型场，每 reset 在 BDDL region
  内重采样。实测（libero_goal task0，各 3 次 reset）：scene 指纹全不同；相对
  rep0 最大漂移 stove ~12mm / wine_rack ~15mm / cabinet ~9mm（z 恒 0，region
  2cm×2cm 包络内）。协议内在（与 openpi/LIBERO-PRO 论文协议同构），不修复、
  显式入档；LP-A2 配对噪声模型必须计入。
- **图像差归因（同状态双渲染对照分离）**：渲染器逐位稳定（预注册判据双相机通过）
  ——agentview 地板 mean_abs 0.0019、腕部 0.04–0.09；跨 reset diff（agentview
  ~2.9、腕部 ~5.0）为地板的 1500×/60×+，fixture 重采样主导。注意：3 个穿插
  dummy 步在腕部视野即有 ~1.0 的真实视觉变化（首版探针曾误归因渲染器，已修正
  捕获顺序）。v1 lock 的"OSMesa 非确定性 mean≈3"归因已撤回更正。图像容差
  （mean≤5.0、frac_gt30≤0.10）=观测极值×~1.5，n=4 开发诊断样本，正式样本量按
  LP5 精度反推。
- **证据**：`lp-a1-fingerprint-evidence.json`（v1，md5 580812e5…）；
  `lp-a1-fixture-drift-evidence.json`（md5 76d40dd7…，服务器双侧一致）；
  `lp-a1-fixture-drift-run.log`（md5 49a4b7df…）；
  `lp-a1-tests-gap-sim.log`（9/9 通过，md5 见文件）。
- **测试**：`tests/test_libero_session.py` 9/9 于 gap-sim（2026-09-19），含
  fixture 1mm 微扰反盲回归、漂移 region 包络门、同状态双渲染归因哨兵。

## LP-A2 闭环设备替换（2026-09-19，PASS）

- **设计**（预注册 `configs/lp_a2_validation.json`，先于任何运行冻结）：libero_goal
  task0 × {paper_faithful(init 0,1), scene_pinned(init 0)} × {torch_cpu, torch_npu}；
  官方 openpi 客户端契约 E1 逐项对齐；噪声按 (arm,init,replan_idx) 稳定播种显式
  注入两设备；scene_pinned 为 fixture 钉扎的工程消融（偏离论文协议，不进论文口径）。
- **结果**：6/6 episode 双设备全成功（钉扎机制由 pin selftest+指纹敏感性回归支撑，
  本数据本身不可判别钉扎生效——pin_ref 恰等于自然布局）；scene_pinned 首块动作差 max 0.00215 /
  mean 0.00053 → 过预注册阈值（0.03/0.005，13.9×/9.4× 余量）；paper 臂 3/3 成功
  相等、零翻转；同种子同 reset 序列下 fixture 布局跨设备复现（init0 对场景指纹
  逐位一致），init1 对布局不同（审查 P2-1 修正归因：CPU init1 因两次 harness 崩溃独立
  重启=新进程第 2 次 reset 自然布局，NPU init1 为单进程第 3 次 reset；5/6 集
  位姿逐位等于第 2 次 reset 布局为决定性证据）→ 首块差升至 0.072（paper 臂按
  预注册不设阈值，归因记录）。发散步（‖Δ‖∞>0.1）：82/90/49，双侧均收敛成功。
- **吞吐**：NPU 稳态 0.52–0.56s/chunk；CPU 230s/chunk（aarch64 bf16 参考内核
  单线程）。LP-A0 的 76.6s/25.7s per-draw 与本轮差异大，记为 ops 观察。
- **NPU 纪律**：卡 1（ASCEND_RT_VISIBLE_DEVICES=1）；pi05 服务暂停窗口约 14 分钟，
  恢复后 /health ok。
- **途中修复的 harness 缺陷（全部留痕 commit）**：噪声形状 (10,7)→(10,32) 勘误；
  客户端请求非原子写竞态；np.savez 后缀陷阱×2；npu 驱动 v1 grep|head 管道 bug
  （v2 接管并保证失败路径恢复服务）。
- **证据**：`docs/acceptance/lp-a2-episodes/`（6 集 json+npz、pin_ref.json、
  server/driver 日志；与服务器逐文件 md5 全等）；`lp-a2-compare.json`。

- **对抗审查**：2026-09-19-LP-A2对抗性审查.md 判定 PASS−（结论成立）：P2-1
  init1 归因改写、P2-2 补登记 E1 偏差（渲染 224 vs 256+resize、seed 0 vs 7，
  对设备配对零影响）；P3 择要已修。无需重跑实验。

## 测试入口与部署身份（T0，2026-09-19，audit F12 响应）

- **Mac 已验证命令**（121 passed / 6 xfailed / 1 xpassed / 1 skipped）：
  `MUJOCO_GL=glfw GAP_REPRO_DATA="$PWD/data" ./.venv/bin/python -m pytest tests/ -q --tb=short -rxX`
- 未设环境变量的默认入口：libero 组 skip；需要 MuJoCo+资产的测试显式 BLOCKED
  （fail-closed，非静默 skip）。libero_session 的 MUJOCO_GL 副作用已移出导入期
  （导入不再污染同进程其他模块）。
- Linux 目标机按栈分环境：RoboDojo（gap-repro 项目 venv + torch_npu）与 LIBERO
  （gap-sim conda libero_pro + MUJOCO_GL=osmesa）不共用一个 Python。
- 部署身份：`scripts/check_runtime.py --deploy-manifest` 输出源码文件 SHA256，
  与目标机同路径文件对账；`--profile mac|libero|robodojo` 输出组件/资产/blocked。
- **四态完成口径**（模块已实现 ≠ 单测通过 ≠ 目标机集成 ≠ 正式验收）：
  逐包状态见 `docs/acceptance/status-matrix.json`（唯一权威进度口径）；
  xfail/xpass/skip 逐项登记，不算验收。

## T4 场景物理与三相机（2026-09-19，audit F04/F05/F07 视觉部分）

- **T4a USD 变换修复**：转换器改为行向量 `p_local @ M`（原 `p @ M.T` 丢父节点
  平移/旋转）。oracle 测试经真实 `convert_usdz_to_obj` 与 `Gf.Transform` 对照
  （平移/旋转/缩放/嵌套）。本地 120 个源资产扫描：raw-affected 2 个（均为
  collision-only prim），转换器实际收录集合 0 个受影响；候选重建产物在
  `data/w11_scene_obj_v2`（gitignored，旧产物未动），bbox 对 USD oracle 全等。
  证据：`docs/acceptance/w11-usd-transform-rebuild.json`。旧 `w11_scene_obj`
  产物与两种算法都对不上（schema/bbox 来历不同），服务器需用修复版重转生产
  源树后复核。单位审计：全部 Z-up / metersPerUnit=1.0。
- **T4b/c 场景物理**：`build_scene_mjcf` 输出 (model, data, manifest)。Rigid →
  freejoint 自由刚体；Geometry → 静态且默认不碰撞（可用 physics.collision 打
  开）；质量 `min(mass,0.5)`/缺省 0.5/≤0→0.05、摩擦优先 layout `friction`，
  均带来源标签；桌面尺寸取 layout Table（1.4×1.1×0.05，旧硬编码 0.5×0.35 错）。
  机器人失重由逐 body gravcomp 实现（全局重力开启供物体下落；300 步轨迹与
  全局关重力一致，漂移同为 0.0722）。每个 take_action 后从同一 MjData 同步
  物体位姿再评分；reset 写回冻结物体位姿并清 qvel。
- **T4d 三相机**：`get_obs(include_vision=True)` 经 `VisionRenderer` 输出
  `cam_head`/`cam_left_wrist`/`cam_right_wrist` 的 HxWx4 uint8；后端不可用显式
  BLOCKED。头部相机位姿来自 `camera_config.yml`（pos [0,-0.41,1.308]，
  ori [30,0,0] 度）；腕部相机外参暂为 URDF camera link 帧占位，robot_config.yml
  到位后复核（robots.CAMERA_MANIFEST 记 diff）。
- **Mac 全套**：`MUJOCO_GL=glfw GAP_REPRO_DATA="$PWD/data" ./.venv/bin/python -m pytest tests/ -q`
  → 167 passed / 6 xfailed / 1 xpassed / 1 skipped（含 T4 新增
  test_scene_physics.py 10 项、test_cameras.py 7 项）。
- **未达标项（T4 完成门未闭）**：真实 case 的 reset→三相机→学生推理→动作→
  物理→评分→日志闭环要等 T6（policy）与 T7（runner）；50 case 资产闭包与碰撞
  代理（visual 凸包 vs 原 convexDecomposition）逐项核查未做完。
