# humanoid-gym / humanoid-lab 双仓库调研与仿真演示 · 会话总结

> 撰写时间：2026-08-26
> 会话：`--E--humanoid-gym--` 2026-08-26（含复盘 8-25 的 humanoid-gym 讨论）
> 磁盘路径随时效：E:\humanoid-gym（克隆）、E:\humanoid-lab（克隆）、E:\huawei（本文+历史环境记录）
> 用途：保留本次会话全部有价值的结论，供后续"跑 humanoid-lab / 录视频 / 训练"决策选用

---

## 一、本次会话干了什么（时间线）

1. `pi update --extensions` 更新扩展（顺带更新了 pi-wechat-assistant）
2. 找回 8-25 上一段 humanoid-gym 讨论（session 在 `--E----` 目录），回顾了：
   - 克隆 roboterax/humanoid-gym；Isaac Gym Preview 4 与 RTX 5060 (sm_120) 冲突结论
   - sim2sim 机制三板斧拆解（模型对齐/控制器外置/观测逐位复刻）——见 `交接文档` 补充
3. **WSL 里跑通了 humanoid-gym 的 sim2sim 演示（XBot-L 走路）** ✅
4. **排查了电脑"内存溢出/死机"危机**（结论见第五节）
5. **调研 roboterax 组织新仓库 humanoid-lab**：README 里两个 Coming Soon（perceptive locomotion / dex hand）未发布，但 2026-03 发布了 Isaac Lab 版 humanoid-lab；克隆到本地并通读全部核心代码
6. **盘点本地环境**，给出跑 humanoid-lab 的缺口清单（第四节）

---

## 二、两个仓库档案

### 2.1 humanoid-gym（`E:\humanoid-gym`，roboterax/humanoid-gym，2074⭐）
- 基于 **Isaac Gym Preview 4**（2023 停更闭源库）+ legged_gym/rsl_rl 结构
- **只能走路**（速度指令追踪 vx∈[-0.3,0.6]/vy±0.3/dyaw±0.3 的训练范围，但 demo 只会 vx=0.4 前进；无键盘控制）
- 资源里只有 **XBot-L 一种机器人**（上次总结说"5 种"是错的，已更正）
- 附带示例权重 `logs/XBot_ppo/exported/policies/policy_example.pt`
- **训练在本机跑不了**：Isaac Gym 闭源二进制只编译到 sm_80 无 PTX，RTX 5060 (sm_120) 必报 no kernel image；无 workaround
- sim2sim 机制：URDF↔MJCF 手工对齐 + PD 控制器写在用户代码（仿真器只积分）+ 47×15 帧观测逐位复刻
- **我们已跑通的✓**：WSL + MuJoCo 离屏渲染，18.2s 视频（见第三节）

### 2.2 humanoid-lab（`E:\humanoid-lab`，roboterax/humanoid-lab，40⭐，MIT，2026-03-27 发布）
- 基于 **IsaacLab 2.3.2 + IsaacSim 5.1**（不需要 Isaac Gym！sm_120 无坑）
- 机器人：**ROMEGA L7 29 DOF**（l7_29dof_neck_fixed.urdf，含手部）
- **两大任务**：
  - `locomotion`：速度指令行走（vx∈[-0.5,1.0]/vy±0.5/dyaw±1.0，带 gait 相位参数 0.8s 周期）
  - `mimic`（单段动作模仿）：加载 mocap 动作 npz，全身跟踪
- **mimic 核心机制（高质量实现）**：
  - `MotionCommand`：npz 含 29 关节轨迹 + 全部 body 的 pos/quat/lin_vel/ang_vel
  - anchor 锚点机制：以 pelvis 为锚，朝向 yaw + Z 对齐到机器人当前状态 → 相对位姿奖励，循环动作稳
  - **自适应课程采样**：动作按时间分 bin，失败多的 bin 权重上调 + 非因果核平滑，自动聚焦难段
  - 奖励：anchor 位姿/速度 + 相对 body 位姿 + 关节极限 + 非预期接触（脚踝/手腕）惩罚
- **`DelayedImplicitActuatorCfg`**：电机延迟建模（min_delay=0/max_delay=4ms）+ 真实电机型号（6508/5005/10520/9015/15017/6008）惯量/刚度/阻尼参数化 —— sim2real 关键，humanoid-gym 没有
- **完整闭环**：CSV mocap → l7_csv_to_npz.py（GMR 重定向后处理）→ replay_l7_npz.py 回放验证 → train.py PPO（rsl_rl 3.0.1，官方默认 8192 envs）→ exporter.py 导出 ONNX+元数据 → ROS2 部署（era_rl_controller_node，sim2sim/real 双模式，含 3 秒渐进入位防抖）
- **自带资产**：`deploy/policy/`（loco_model.onnx + dance_7/dance_9 + long_motion_1 的 onnx+npz）、`deploy/deploy/l7_29dof_neck_fixed/`（MJCF + 全部 STL 网格）、`motions/`（dance_7/9、long_motion_1 npz）
- 关节序转换（训练↔sim2sim↔真机三种顺序不同，`convert_joint_order`）

---

## 三、已产出的成果物（可直接复用）

| 路径 | 内容 |
|---|---|
| `E:\tmp\hg_wsl\out_v2\xbot_walk.mp4` | **XBot-L 走路演示**（1280x720 / 18.2s / 30fps，确定性相机跟随，机器人清晰） |
| `E:\tmp\hg_wsl\out_v2\frame0.png / frame_end.png` | 起止帧静图 |
| `E:\tmp\hg_wsl\out_test\` | 1s 取景测试版 |
| `E:\tmp\hg_wsl\*.py` | 可复用脚本：hg_shim.py（isaacgym 规避）、smoke.py（无渲染验证）、record.py（离屏录视频）、analyze/framing/color/detect_motion/extract_frames.py（画面分析）|
| WSL `~/.venv_sim` 内 | mujoco-python-viewer 0.1.4 + imageio + pillow（已装好，供后续复用） |

record.py 关键要点（复刻 sim2sim 演示的标准方案）：
- 离屏渲染：`mujoco.Renderer(model, H, W)`（EGL，WSL 可用）
- 相机：**必须用 free camera（camera=-1）+ 每帧改 `renderer.scene.camera[0]` 的 pos/forward/up 手动跟随**；仓库自带的 trackcom 相机不可靠（之前整段视频看不到机器人）
- 模型兼容画板（mujoco 3.2.x）：① `<flag sensornoise>` 已移除需剔除 ② patched XML 必须放原模型同目录（meshdir 是相对路径）③ 枚举名 `mjOBJ_CAMERA`（不是 mjOBJ_CAM）④ 离屏缓冲要 `<visual><global offwidth/offheight>`（默认 EGL 只有 640）
- 视频用 cv2 mp4v 写 mp4（Windows 可播）；检查画面用边缘/颜色统计 + 视觉子代理复核

---

## 四、本地环境现状 & 跑 humanoid-lab 的缺口

### 已有（冻结验证过的）
- **`E:\isaaclab_env51`**：Python 3.11.9 + IsaacSim **5.1.0.0**（全套）+ IsaacLab **0.48.0**（源码 `-e` 装在 `E:\IsaacLab23`，git tag **v2.3.1**）+ isaaclab_rl 0.4.4 / isaaclab_mimic 1.0.15 + torch **2.7.0+cu128**（sm_120 ✓）+ numpy 1.26.0 + onnx 1.22 + mujoco 3.11 + wandb/tensordict/scipy/tensorboard + psutil 5.9.8
- NVIDIA 驱动 **580.88**（Isaac Sim 5.1 适配，**永远不要升级**；虚拟显示适配器 Parsec/MuMu 保持禁用）
- WSL `.venv_sim`：mujoco **3.2.7** + torch 2.13.0+cu130 + cv2 + glfw + scipy（跑 MuJoCo 演示用）

### 缺口（就 3 项）
| # | 缺口 | 命令 |
|---|---|---|
| 1 | **rsl-rl-lib 没装**（train.py 硬性要求 ≥3.0.1） | `E:\isaaclab_env51\Scripts\python -m pip install rsl-rl-lib==3.0.1`（清华源）|
| 2 | **onnxruntime 没装**（部署推理用；录视频也建议装到 .venv_sim） | `pip install onnxruntime`（两处环境）|
| 3 | **era_okcc_humanoid_lab 扩展未安装** | `pip install -e E:\humanoid-lab\source\era_okcc_humanoid_lab`（依赖仅 psutil）|

### 两个注意点
1. **版本微差**：IsaacLab 本地是 v2.3.1，humanoid-lab 按 2.3.2 开发 → 先直接试，大概率兼容；**别升级 E:\IsaacLab23**（GR00T 对齐项目冻结环境共用同一次 `-e` 安装）
2. **硬件纪律**（16GB 内存/8GB 显存，全是血泪）：`--num_envs` 默认 8192 必须降到 **1024~2048**（GR00T 实测 1024 甜点、2048 临界）；num_envs 必须 4 的倍数；headless 必须 `--/renderer/activeApi=d3d12` + `OMNI_KIT_ACCEPT_EULA=yes`；跑前查空闲内存（<3GB 停手）、杀残留 Kit

---

## 五、电脑"内存溢出/死机"调查结论（2026-08-26 11:32 事件）

### 证据
- Event 6008：11:32:33 异常关机；Event 41 BugcheckCode=0（**非蓝屏**，硬死机强制重启）；无 minidump、无 WHEA、无 GPU TDR
- **近 60 天异常关机 28 次**（8/24 一天崩 3 次）—— 慢性问题，非本次 sim 代码导致
- 机器：15.2GB 内存 + 19GB 页面文件（D 盘）；WSL `.wslconfig` 只有 swapFile 行（WSL 内存无上限，可吃 50%）
- 系统环境：Parsec + ToDesk 虚拟显示适配器 + NVIDIA Overlay（8/24 有 overlay 崩溃报告）

### 结论：不是 sim2sim 代码的责任（纯 CPU 任务，WSL 峰值 2-3GB）；是系统慢性不稳定 + 资源紧
### 建议（按优先级）
1. 禁用/卸载 **Parsec、ToDesk 虚拟显示器驱动**（远程桌面虚拟显示是反复死机经典元凶）
2. `.wslconfig` 加内存上限：`[wsl2] memory=4GB swap=8GB`（改后 `wsl --shutdown`）
3. 跑一次 Windows 内存诊断（mdsched.exe）排除内存条
4. 清 C 盘（当前仅 13.3GB 空闲）

---

## 六、本次会话新增的经验（工具/仿真层，补充《防踩坑手册》）

1. **wsl.exe 传参坑**：`wsl -- bash -c '...$VAR...'` 里的 `$` 变量会被 Windows 参数层吞掉 → **改 `bash -s < script.sh`（stdin 管道）** 稳定可靠
2. **Git Bash 路径转换**：以 `/` 开头的路径被改成 `E:/portable-git/...` → 前缀加 `MSYS_NO_PATHCONV=1`
3. **uv 不在 PATH**：`~/.local/bin/uv`，非登录 shell 直接用全路径；装大包用清华源 `--index-url https://pypi.tuna.tsinghua.edu.cn/simple` 快得多（mujoco-python-viewer 300s 超时 → 镜像 4s）
4. **WSL 后台进程**：`bash -s` 会话退出后后台子进程会死，别用 `&` 裸起，用 nohup/前台+长超时
5. **mujoco 版本差异**：3.2 移除了 `sensornoise`、`mjOBJ_CAM`→`mjOBJ_CAMERA`；3.11 viewer 段错误（历史已验），**WSL 稳定组合 mujoco 3.2.7**
6. **视觉核查**：主模型不支持图片 → 可用 `deepseek/deepseek-v4-flash-vision-exp` 子代理看截图复核（本次确认了机器人可见性）；或让用户自己看
7. **pip 装包注意**：pip install 前确认 venv 有没有 pip 模块（uv 建的裸 venv 无 pip，用 `uv pip install --python <venv>/bin/python`）

---

## 七、待办 / 下一步选项（等用户拍板）

- [ ] **路线 A（快）**：拆 3 个缺口前，先用仓库自带预训练 ONNX + MuJoCo 在 WSL 录 **L7 走路/跳舞视频**（只需 .venv_sim 装 onnxruntime + 复刻 rl_interfaces 逻辑）
- [ ] **路线 B（完整）**：补 3 个缺口 → `play.py` 在 Isaac Sim 加载预训练权重看效果 → 评估自训（num_envs 1024）
- [ ] 真机/ROS2 部署（需要 ROS2 Humble + pynput，当前 WSL 未装 ROS2，暂缓）
- [ ] （遗留自 huawei 项目）两个泄露的 GitHub PAT 待用户吊销——**仍未处理**
- [ ] 系统稳定性：是否执行第五节四项建议由用户决定

---

## 八、关键路径速查

```
humanoid-gym repo:      E:\humanoid-gym（XBot-L，Isaac Gym 老框架）
humanoid-lab repo:      E:\humanoid-lab（L7，IsaacLab 2.3.x 新框架）
预训练策略:             E:\humanoid-lab\deploy\policy\{locomotion,dance_7,dance_9,long_motion_1}\
MuJoCo 模型:            E:\humanoid-lab\deploy\deploy\l7_29dof_neck_fixed\
动作数据:               E:\humanoid-lab\motions\（dance_7/9 npz + csv）
Isaac 主环境:           E:\isaaclab_env51（Py3.11 + IsaacSim 5.1 + isaaclab 2.3.1）
IsaacLab 源码:          E:\IsaacLab23（tag v2.3.1，GR00T 冻结环境，勿动）
WSL 仿真环境:           ~/projects/GR00T-WholeBodyControl/.venv_sim（mujoco 3.2.7）
演示视频产出:           E:\tmp\hg_wsl\out_v2\（xbot_walk.mp4）
可复用脚本:             E:\tmp\hg_wsl\（hg_shim.py / smoke.py / record.py 等）
历史环境记录:           E:\huawei\（交接文档 / 排障复盘 / 防踩坑手册）
```