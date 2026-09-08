# GR00T-WholeBodyControl 环境搭建 · 完整交接文档

> 更新日期：2026-08-18
> 用途：新接手 agent 的完整上下文（目标 / 现状 / 经验 / 下一步）

---

## 一、原始目标

用户在 **Windows 笔记本**（AMD Ryzen 9 7845HX + AMD Radeon 610M 核显 + **NVIDIA RTX 5060 Laptop 8GB**，Blackwell sm_120）上搭建 **GR00T-WholeBodyControl**（NVIDIA 的 G1 人形机器人全身控制项目）环境，目标是 **demo 级别**：
1. ✅ MuJoCo 仿真跑通（已完成！）
2. ⬜ Isaac Lab 训练/评估（进行中，卡在版本兼容）
3. ⬜ 真机部署 / VR 遥操作（未开始）

用户要求**所有大文件装在 E 盘**（C 盘几乎满）。

---

## 二、当前状态总览

| 模块 | 状态 | 位置 |
|---|---|---|
| WSL2 + Ubuntu 22.04 | ✅ 完成 | `E:\WSL\Ubuntu-22.04\ext4.vhdx` |
| MuJoCo 仿真（G1 机器人跑动作） | ✅ **跑通**（deploy.sh sim + run_sim_loop.py） | WSL `~/projects/GR00T-WholeBodyControl` |
| TensorRT 10.13 + C++ 部署程序 | ✅ 编译成功 | WSL `~/TensorRT` |
| **Isaac Lab 训练/评估（WSL 路径）** | ❌ **放弃**（Vulkan 不可用） | — |
| **Isaac Lab 训练/评估（Windows 原生路径）** | 🔶 **进行中**，torch 已解决；卡在 Kit 106.5 渲染器 scenedb 崩溃（见第三节） | `E:\isaaclab_env` (venv) + `E:\IsaacLab` |

---

## 三、当前卡点（接手者必读）

### 正在进行的方案：Windows 原生 Isaac Sim 4.5 + Isaac Lab 2.1.1

**已完成：**
- [x] venv 创建：`E:\isaaclab_env`（Python 3.10.8，来自 D:\py\python）
- [x] `isaacsim[all,extscache]==4.5.0` 安装成功（50 个组件，含 extscache 3GB）
- [x] **（2026-08-19）torch 降级 2.7.1+cu128 成功**，验证全过：`2.7.1+cu128 / cuda=True / RTX 5060 Laptop GPU / jit OK`，GPU 实际计算正常
- [x] numpy 回滚 1.26.4（--force-reinstall 曾连带顶到 2.2.6）；`pip check` 干净
- [x] Git 2.55 安装到 `E:\Git`（Windows 之前没装 git）
- [x] IsaacLab v2.1.1 克隆到 `E:\IsaacLab`（--depth 1），`pip install -e source\isaaclab` 成功（isaaclab 0.41.3）
- [x] 补装缺失依赖：requests、h5py；flatdict 构建报 pkg_resources 错 → `--no-build-isolation` 解决
- [x] **修复 VC 运行时遮蔽问题**（详见下面卡点 #2）——启动期 osqp/qdldl 崩溃已根除
- [x] Isaac Sim 应用可完整启动到 `app ready`（扩展全部加载、PhysX 初始化、CUDA device 0）

**✅ 卡点 #1 已解决：torch nightly 不兼容（2026-08-19）**
- 降级 torch==2.7.1+cu128 + torchvision==0.22.1+cu128（cu128 通道、保留 torch.jit、满足 isaacsim 的 torch>=2.5.1）
- 验证：`E:\huawei\verify_env.py` 全过（含 GPU 实际计算）

**✅ 卡点 #2 已解决：启动时第三方 pyd 崩溃（osqp/qdldl access violation）**
- 现象：启动加载 wheeled_robots 扩展的 osqp/qdldl 原生模块必崩；事件查看器（事件 1000）faulting module = `site-packages\omni\MSVCP140.dll`
- **根因：Isaac Sim 自带的 VC++ 运行时太旧**（14.29/VS2019 + Win10-1607 的 ucrtbase），Kit 把 omni 目录加入 DLL 搜索路径，遮蔽了系统新运行时（14.51）；VS2022 编译的新 pyd 混用旧运行时 → 崩溃
- **修复：已把 `E:\isaaclab_env\Lib\site-packages\omni\` 下 8 个 VC DLL（msvcp140*.dll / vcruntime140*.dll / ucrtbase.dll）替换为 System32 新版**。原件备份在 `E:\isaaclab_env\vcrt_backup_omni\`。VC 运行时向下兼容，勿还原！

**❌ 卡点 #3（当前未解决）：Kit 106.5 渲染器 scenedb 确定性崩溃**
- 现象：`app ready` 后 `rtx.scenedb.plugin.dll + 0xd6d4b` access violation（事件 1000），100% 复现，偏移完全固定
- 所有形态都崩：GUI / headless / Storm 渲染器 / IsaacLab experience / isaacsim 默认 experience
- 已试无效：TLAS 实例上限（`--/rtx-transient/scenedb/maxInstancesLimit=1048576` + `forceMaxTLASInstancesLimit`，TLAS 缓冲从 7.5GB→469MB 仍崩）、`--/rtx/scenedb/cudaInterop/enabled=false`、`--/app/content/emptyStageOnStart=false`、注册表强制 python.exe 用 NVIDIA GPU（HKCU UserGpuPreferences GpuPreference=2）
- 判断：Kit 106.5（Isaac Sim 4.5）渲染栈与 RTX 5060 Laptop（Blackwell GB207）+ 驱动 610.88 组合不兼容，进程内配置无法绕过

**两条前进路线（二选一，接手者先问用户）：**

**路线 A（便宜，先试）：回退 NVIDIA 驱动**到 Isaac Sim 4.5 时代的验证版本（约 572.xx，2025 年中）
- 当前 610.88 太新，新驱动 Vulkan 行为可能与 Kit 106.5 冲突；RTX 5060 Laptop 需 ≥572 驱动（Blackwell 支持自 572 起），回退范围安全
- 步骤：NVIDIA 官网下载 572.xx/576.xx → 自定义安装勾选"执行清洁安装" → 重跑 `E:\huawei\smoke_test.py`
- 成功标准：输出 `SMOKE_OK` 且事件查看器无新崩溃。成功则 4.5 全保留，GR00T 训练代码（IsaacLab 2.1 API）零改动

**路线 B（升级，A 无效则走）：Isaac Sim 5.1 + Python 3.11 + IsaacLab 2.3.2**
- 已探明：isaacsim-core 5.0/5.1 = cp311（pypi.nvidia.com，Windows 侧可通）；isaaclab 2.3.2/3.0.0 也在 pypi.nvidia.com；Kit 107 是 Blackwell 时代产物
- 步骤：装 Python 3.11 到 E 盘 → 新 venv `E:\isaaclab_env51` → torch（按 5.1 文档 pin，预计 2.7.x cu128，与现有 wheel 缓存兼容）→ `pip install isaacsim[all]==5.1.0.0 --extra-index-url https://pypi.nvidia.com` → `pip install isaaclab==2.3.2 --extra-index-url https://pypi.nvidia.com`
- 代价：数 GB 下载；GR00T 训练代码按 IsaacLab 2.1 写的，2.3 API 有差异需适配（本来也是项目集成阶段要踩的坑）

**现成验证/诊断脚本（都在 E:\huawei\）：**
- `smoke_test.py` —— SimulationApp GUI 启动冒烟；`test_headless.py` —— headless+物理 100 步
- `verify_env.py` —— torch/cuda/jit/numpy 体检；`get_crash_events.ps1` —— 读事件 1000 拿崩溃模块+偏移
- Kit 日志：`E:\isaaclab_env\Lib\site-packages\omni\logs\Kit\`

**项目集成提示（后置）：** GR00T 项目代码跑在 WSL（Linux），Windows 版 Isaac Sim 需通过 ROS bridge / 文件共享对接；train_agent_trl.py 是 Linux 向的，Windows 上要适配

---

## 四、版本兼容矩阵（血泪经验，接手者必读）

### 核心矛盾三角
- **A. Isaac Sim 4.5** 要：Python 3.10 + numpy<2.0 + torch.jit（torch ≤ 2.9 左右）
- **B. RTX 5060 (sm_120)** 要：torch ≥ 2.7 且 cu128（cu126 不行！）
- **C. Isaac Lab** 要：与 isaacsim 匹配

### 已知事实
| 组件 | 约束 |
|---|---|
| isaacsim 4.5 pip 包 | 只有 **cp310** wheel（无 cp311） |
| isaaclab 2.2+/2.3 pip 包 | 只有 **cp311** wheel（无 cp310）→ WSL 路径必须用 **isaaclab 2.1.0** |
| isaaclab 2.1.x | 兼容 isaacsim 4.5，Python 3.10 |
| torch cu126 | **不支持 sm_120**（RTX 50 系报 no kernel image） |
| torch 2.7.1+cu128 | ✅ 支持 sm_120 + 保留 torch.jit（甜点版本） |
| torch 2.12.0.dev nightly cu128 | 支持 sm_120 但 **c10.dll 坏 + torch.jit 移除** ❌ |
| torch 2.13.0 | 与旧 NCCL 冲突（WSL Linux 环境实测） |
| numpy 2.x | **isaacsim 4.5 不兼容**（broadcast_to 报错）→ 必须 numpy 1.26.4 |
| RTX 5060 驱动 | 596.36，CUDA 13.2，Windows 侧一切正常 |

### WSL 路径为何失败（不要再走）
WSL2 内 Vulkan 只有 llvmpipe（缺 Mesa dzn，Ubuntu 22.04 官方源/PAA 均无），Isaac Sim GPU 渲染无法初始化。OpenGL 走 D3D12 正常，但 **Vulkan 是 Isaac Sim 的硬依赖**。修复路径（装 dzn）已确认死路（kisak/oibaf PPA 对 jammy 停更）。**WSL 保留作为 ROS/开发/部署环境，训练走 Windows 原生。**

---

## 五、关键环境信息速查

### Windows 侧
```
Python 3.10.8 (D:\py\python, py launcher 可用)
venv: E:\isaaclab_env
pip 缓存: E:\pip_cache（setx 已设 PIP_CACHE_DIR/TMP/TEMP）
磁盘: C 盘几乎满（0GB！），D/E 盘有空间
代理: Clash Verge（TUN 模式；mixed-port 7897，但 Clash 可能未运行，连不上时先检查）
网络: download.pytorch.org 直连可通（下载过 2.8GB wheel）；pypi.nvidia.com 可通；files.pythonhosted.org 被墙 → pip 用清华源
```

### WSL 侧（Ubuntu 22.04，用户 xerxes_ubuntu）
```
项目: ~/projects/GR00T-WholeBodyControl（复制自 E:\huawei\GR00T-WholeBodyControl）
仿真环境: .venv_sim（mujoco 3.2.7 + torch 2.7.1+cu128 + unitree_sdk2py）
训练环境: .venv_train（Python 3.10 + isaacsim 4.5.0 + isaaclab 2.1.0 + torch 2.7.1+cu128 + numpy 1.26.4）——可留作参考/备胎
TensorRT: ~/TensorRT（10.13.0.35，TensorRT_ROOT 已设）
镜像: UV_DEFAULT_INDEX=清华 / HF_ENDPOINT=hf-mirror（已写入 ~/.bashrc）
LD_LIBRARY_PATH=/usr/lib/wsl/lib（已写入 ~/.bashrc）
关键修复（已应用到 WSL 副本）:
  - unitree_sdk2py channel.py 幂等修复（DDS 重复初始化）
  - channel_config.py multicast=false
  - 符号链接修复（libddsc.so.0 等）
  - 所有 .sh 转 LF
```

### 常用命令
```bash
# MuJoCo 仿真（两个终端）
# 终端1:
cd ~/projects/GR00T-WholeBodyControl && source .venv_sim/bin/activate
python gear_sonic/scripts/run_sim_loop.py
# 终端2:
cd ~/projects/GR00T-WholeBodyControl/gear_sonic_deploy && source scripts/setup_env.sh
bash deploy.sh sim   # y → ] 启动策略 → MuJoCo窗口按9 → T播放
```

---

## 六、踩坑经验总结（按主题）

### 1. bat 闪退
- 原因：UTF-8 编码 + GBK 代码页冲突 → bat 存 **GBK/ANSI + CRLF**，ps1 存 **UTF-8 BOM**

### 2. 网络（国内环境）
- `files.pythonhosted.org` 被墙 → 一律用**清华源** `https://pypi.tuna.tsinghua.edu.cn/simple`
- HuggingFace → `HF_ENDPOINT=https://hf-mirror.com`
- PyTorch cu128 → `https://download.pytorch.org/whl/cu128`（**直连可通**！）或清华 pytorch-wheels
- 代理坑：Clash TUN 会劫持 WSL 流量导致黑洞；**WSL 里别依赖 TUN**，用镜像直连；代理环境变量残留会弄坏 apt/pip（`unset http_proxy https_proxy`）

### 3. WSL 图形
- WSLg 窗口黑屏/点不开 → `wsl --update` + `wsl --shutdown` 修复过一次
- WSL 内 Vulkan 只有 llvmpipe（缺 dzn）→ **Isaac Sim 在 WSL 跑不了，别浪费时间**
- OpenGL 选择 GPU：`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`（混合显卡默认选 AMD 核显）

### 4. Git LFS / 符号链接
- `.lfsconfig` 的 `fetchexclude=motionbricks/out/**` 是故意的，主流程不需要那 2.2GB ckpt
- Windows 克隆的仓库符号链接变文本文件（libddsc.so.0 10 字节）→ 手动 `ln -s` 修复
- .sh 脚本 CRLF 报 `$'\r'` → `sed -i 's/\r$//'`

### 5. 版本兼容（最痛的）
- **先查版本矩阵再动手**，别装最新版！
- pip/uv 装大包前先 `--dry-run` / 查 ABI tag（`cp310` vs `cp311`）
- 混合显卡 + 新 GPU 架构（sm_120）是双重坑：驱动、torch、CUDA 都要配套

---

## 七、接手者下一步行动清单

1. ~~torch 降级 2.7.1+cu128~~ ✅ 已完成（2026-08-19）
2. ~~装 Isaac Lab 2.1.1~~ ✅ 已完成（E:\IsaacLab，isaaclab 0.41.3）
3. ~~VC 运行时遮蔽导致的 osqp/qdldl 崩溃~~ ✅ 已修复（omni 目录 VC DLL 已换新，备份在 vcrt_backup_omni）
4. ~~scenedb 崩溃（卡点 #3）~~ ✅ 已解决（2026-08-19 隔离测试）：**根因 = NVIDIA 驱动 610.88**，回退到 **580.88**（Isaac Sim 5.1 官方测试驱动）+ 禁用 Parsec/MuMu 虚拟显示适配器 + 清理 C 盘 CrashDumps(10.6GB) 后，smoke_isaac51.py 原样复跑通过（app ready 32s / 100 物理步 OK）。Isaac Sim 5.1 + Py3.11 + IsaacLab 2.3 路线恢复
5. 渲染器通了之后：跑最小测试 create_empty.py → Ant headless 训练（8GB 显存，`--num_envs 256` 起步）
6. **注意**：项目代码（train_agent_trl.py / eval_agent_trl.py）是 Linux 向的，Windows 原生 Isaac Sim 对接项目代码可能需要：
   - 把项目仓库 clone/复制到 Windows（E:\ 下，用 E:\Git\cmd\git.exe）
   - 处理路径、`pip install -e gear_sonic/[training]`（注意 WSL 里装过，Windows 要重装）
   - Hydra 配置、sample_data、sonic_release checkpoint 都要在 Windows 侧有一份

---

## 八、备注

- 用户是中文交流，实操型，跟随命令执行，报错会完整粘贴
- 用户对"循环踩坑"很敏感，**动手前先给版本矩阵和理由**
- 用户明确要 demo 级别，8GB 显存不要开大 num_envs
- 所有新下载的大文件优先放 E 盘

---

## 九、对话全历程回顾（详细版，2026-08-17 ~ 08-18）

### 阶段 1：bat 闪退修复（问题：编码）
- **现象**：用户"双击 bat 闪退"，文件夹里有两个 WSL 安装 bat
- **诊断**：hexdump 发现 bat 是 UTF-8 无 BOM；系统 ACP/OEMCP=936 (GBK)；实测运行出现 `'+' is not recognized`、`'ypass' is not recognized` 等碎片化解析错误——cmd 按 GBK 解析 UTF-8 中文导致行错乱，语法错误直接跳过结尾 pause 退出
- **修复**：两个 bat 转 GBK+CRLF、删掉 `chcp 65001`；ps1 转 UTF-8 with BOM（Windows PowerShell 5.1 必须 BOM 才能读中文）
- **教训**：Windows 中文系统写 bat 用 ANSI(GBK)，写 ps1 用 UTF-8 BOM

### 阶段 2：WSL 安装到 E 盘（问题：磁盘）
- 用户要求"装到 e 盘"
- 重写 install_wsl_ubuntu.ps1：`wsl --install -d Ubuntu-22.04 --location E:\WSL\Ubuntu-22.04`（WSL 2.0.14+ 支持），swap 也配置 E 盘（`.wslconfig` swapFile，注意要双反斜杠转义），旧版 WSL 自动降级 export/import 迁移
- 用户实测安装成功，用户 xerxes_ubuntu，验证 ext4.vhdx 在 E 盘 ✅

### 阶段 3：Ubuntu 内部环境配置（问题：脚本/网络）
- `setup_wsl_ubuntu.sh`：装工具、uv、复制项目到 WSL
- 遇坑 1：脚本在 `git lfs install` 卡住——仓库自带 pre-push hook，git-lfs 拒绝覆盖返回非零，`set -e` 中断 → 改 `git lfs install --force`
- 遇坑 2：`git lfs pull` 无反应——最初以为是 origin 引用缺失，让用户 `git fetch origin`；后来发现仓库 `.lfsconfig` 里 `fetchexclude=motionbricks/out/**` 是**故意**的，4 个 ckpt（2.2GB）主流程不需要 → 结论：不用管
- 遇坑 3：`uv` 命令找不到 → PATH 没持久化，写进 ~/.bashrc
- 遇坑 4：Python 包下载 `Connection reset` → 清华源 + hf-mirror；又遇到代理残留 `tunnel error` → unset 代理变量
- 遇坑 5：Clash TUN 与 WSL2 的恩怨——镜像网络模式 + TUN 导致 WSL 流量黑洞（DNS 变 fake-ip、连接全超时），最终**回退 NAT 模式**，WSL 内一律直连+镜像；Clash 7897 端口经常没监听

### 阶段 4：部署栈（TensorRT + C++ 编译）
- ONNX 模型下载成功（encoder 48M + decoder 40M + planner 739M，hf-mirror）
- TensorRT 10.13.0.35 下载（用户 E 盘）、解压、拷入 WSL、`install_deps.sh`（装 CUDA/ONNX Runtime/just）、`just build` 编译成功
- 遇坑：符号链接残骸 `libddsc.so.0: file too short`（Windows 克隆的符号链接是 10 字节文本）→ `rm` + `ln -s` 修复（x86_64 + aarch64 两处）

### 阶段 5：MuJoCo 仿真跑通（第一座里程碑）
- 遇坑 1：DDS `create domain error`——`run_sim_loop.py` 里 ChannelFactoryInitialize 被调用两次（SimWrapper + base_sim），同 domain id 重复创建报 PRECONDITION_NOT_MET → 给 `unitree_sdk2py/core/channel.py` 的 `ChannelFactory.Init` 加幂等判断；另把 DDS XML 的 multicast 改 false
- 遇坑 2：WSLg 窗口黑屏点不开——glxgears 80FPS 说明渲染正常但窗口显示不出来 → `wsl --update` + `wsl --shutdown` 修复
- 遇坑 3：mujoco 3.11.0 viewer 段错误 → 降级 mujoco 3.2.7
- **最终：deploy.sh sim + run_sim_loop.py 两个终端，`]` 启动、`9` 落地、`T` 播放，G1 机器人动起来** ✅
- 期间还修了 bat/ps1/sh 的编码、写了 `环境搭建记录.md`

### 阶段 6：Isaac Lab 训练环境（版本战争，WSL 路径）
- 用户要求训练 demo；check_environment.py 提示缺 Isaac Lab 等
- **版本三角矛盾**：
  1. isaacsim 4.5.0 pip 包只有 **cp310** wheel
  2. isaaclab 2.2.0/2.3.0 只有 **cp311** wheel → 被迫用 isaaclab 2.1.0（cp310）
  3. RTX 5060 (sm_120) 需要 **cu128** torch（cu126 报 no kernel image）
- 过程：建 .venv_train (3.10) → isaacsim 4.5.0 ✅ → flatdict 构建失败（setuptools 84 移除了 pkg_resources）→ 降 setuptools<81 ✅ → isaaclab 2.1.0 ✅ → torch 2.13（NCCL 冲突 ❌）→ torch 2.7.1+cu126（sm_120 不支持 ❌）→ **torch 2.7.1+cu128（清华镜像没有 → 官方源成功 ✅）** → numpy 2.2.6 被带上来导致 isaacsim `broadcast_to` 崩 → 降 numpy 1.26.4 → wandb/protobuf 冲突 → protobuf 升级
- **最终死路**：`vkCreateInstance failed: ERROR_INCOMPATIBLE_DRIVER`、`activeGpu index 0 is higher than available GPUs`——WSL 内 Vulkan 只有 llvmpipe
- 排查：`/usr/lib/wsl/lib` 无 libnvidia-vulkan → 一度误判驱动缺组件；外部咨询纠正：**WSL Vulkan 的正确路径是 Mesa dzn（Vulkan→D3D12），不是 NVIDIA Linux ICD**；但 Ubuntu 22.04 的 Mesa 23.2 无 dzn，kisak/oibaf PPA 对 jammy 已停更 → **dzn 死路确认**
- 附带发现：`MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` 让 OpenGL 走 RTX 5060（混合显卡默认选 AMD 核显）
- 结论：**WSL 内跑 Isaac Sim 不可行**（Isaac Sim 的 RTX 渲染器需要原生 Vulkan/RTX，dzn 也不行，且有已知 SIGSEGV bug）；WSL 保留作 ROS/开发/部署环境

### 阶段 7：Windows 原生 Isaac Sim（交接时的当前状态）
- 转向官方支持路径：Windows 原生跑 Isaac Sim 4.5 + Isaac Lab 2.1.1
- venv 建在 E:\isaaclab_env（Python 3.10.8 来自 D:\py\python，没装 conda）
- **torch nightly 2.12.0.dev+cu128**：`cuda.is_available()=True`，RTX 5060 可见 ✅（sm_120 通道确认）
- **isaacsim[all,extscache]==4.5.0** 安装成功（50 组件，extscache 3GB）
- 遇坑：C 盘满导致 pip 下载 OSError → `PIP_CACHE_DIR/TMP/TEMP` 重定向 E 盘（setx 永久）
- **当时的卡点**：isaacsim.exe 窗口空白未响应，日志 `c10.dll WinError 1114` + `module 'torch' has no attribute 'jit'` → **nightly 2.12 与 Isaac Sim 4.5 不兼容**
- VC++ 运行库已确认齐全（14.51），排除缺失
- **当时拟定的方案**：torch 降级 **2.7.1+cu128**（WSL 里实测支持 sm_120 且保留 torch.jit），验证 `jit OK` 后重启 isaacsim，接着装 Isaac Lab 2.1.1（clone + checkout v2.1.1 + pip install -e），跑 create_empty.py 最小测试，最后 Ant headless 训练（num_envs 从 256 起）
- **后续实际走向（见第十节）**：Isaac Sim 4.5 的 scenedb 崩溃另经多轮隔离测试，最终定位为 NVIDIA 驱动 610.88 不兼容 → 回退 580.88 + 禁用虚拟显示适配器后，路线升级为 **Isaac Sim 5.1 + Py3.11 + IsaacLab 2.3**（E:\isaaclab_env51）
- **隐藏大坑预警（仍有效）**：项目训练代码（train_agent_trl.py / eval_agent_trl.py、gear_sonic[training]）是 Linux 向的，Windows 原生 Isaac Sim 需要把项目复制到 Windows、重装依赖、适配路径

### 方法论总结（用户两次督促后）
1. 用户质疑"循环踩坑"→ 整理版本兼容矩阵，确认根因是三角矛盾而非盲目试错
2. 开对抗性审查子 agent（因卡住中断，但外部咨询完成了关键纠偏：dzn 路径、nightly 推荐、RTX 50 系换 torch）
3. 最重要的教训：**涉及 GPU/框架/仿真器三方时，先查官方版本矩阵再动手；稳定版 > 最新版；一个环境一个 venv，别混装**

---

## 十、2026-08-19 隔离测试记录（scenedb 崩溃最终定位）

- **现象**：Isaac Sim 4.5(Kit 106.5) 与 5.1(Kit 107) 在 RTX 渲染器初始化 `rtx.scenedb.plugin.dll!carbOnPluginStartup` 处确定性访问冲突（0xC0000005），与 VC 运行时无关（4.5 的 osqp/qdldl 是另一个独立问题，已单独修复）
- **假设梳理**：驱动 610.88 vs 文档测试驱动 596.36/580.88；Parsec/MuMu 虚拟显示适配器；AMD 610M 混合输出；C 盘空间不足
- **一次性变更**（2026-08-19 16:00-17:00）
  1. 禁用虚拟显示适配器：`Disable-PnpDevice` `ROOT\DISPLAY\0000`(Parsec) + `ROOT\DISPLAY\0001`(MuMu)，停 MuMu 服务（脚本：E:\huawei\disable_virtual_displays.ps1）
  2. 清理 `C:\Users\Xerxes\AppData\Local\CrashDumps`（10.6GB，C 盘 1.42→11.76GB）
  3. 下载并**清洁安装 580.88 笔记本 DCH WHQL**（849.5MB，cn CDN），GUI 安装，移除 NVIDIA App + PhysX 系统软件，路径用系统默认（E:\NVIDIA 自定义路径会报"无法创建 folder"，属 launcher 自身毛病，无权限问题）；launcher 静默模式首次退出码 8/0xE0E00019 且不写日志，GUI 模式正常
  4. 重启后原样复跑 `E:\isaaclab_env51\Scripts\python.exe E:\huawei\smoke_isaac51.py`（首次需 EULA：喂 `y` 或设 `OMNI_KIT_ACCEPT_EULA=yes`）
- **结果**：PASS（报告：E:\huawei\artifacts\phase3r2_isolation_report.txt）
- **经验固化**：
  - 这台机器 Isaac Sim 路线必须用 **580.88**（或 NVIDIA 官方对该版本测试过的驱动），610.88 不可用；不要随意升级驱动
  - 驱动装完之后临时目录/解压一律走 E 盘（C 盘常年紧张，定期清 CrashDumps）
  - 首次跑 Isaac Sim 需要接受 EULA，交互/CI 用 `OMNI_KIT_ACCEPT_EULA=yes`
- **当前状态**：Phase 4 进行中——`git clone --branch v2.3.1 IsaacLab -> E:\IsaacLab23`，之后 `isaaclab.bat -i none`，跑 `scripts\tutorials\00_sim\create_empty.py`（Gate 2）
