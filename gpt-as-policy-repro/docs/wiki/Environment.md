# 环境与资产

## 已有相同环境时

先克隆代码；服务器中将 `gpt-as-policy-repro` 挂载为 `/workspace/repo`，数据目录挂载为 `/workspace/data`。不要把整个 embodied-ai 仓库直接挂到 `/workspace/repo`，否则 src、scripts 的路径会错。

```bash
git clone https://github.com/Xxc33128/embodied-ai.git
cd embodied-ai/gpt-as-policy-repro
git rev-parse HEAD
```

实际使用两个容器：CPU 仿真容器 `gap-sim`，推理容器 `gap-repro`。同事若已有相同镜像和挂载，先检查下面的版本，不必覆盖现有环境。

| 环境 | 已核对版本 |
|---|---|
| gap-sim | Python 3.8.13；torch 1.11.0；MuJoCo 3.2.3；robosuite 1.4.0；numpy 1.22.4；imageio 2.35.1；imageio-ffmpeg 0.5.1 |
| gap-repro | Python 3.11.15；torch 2.7.1+cpu；torch_npu 2.7.1.post2；transformers 4.53.2；JAX/JAXlib 0.5.3；numpy 2.3.5 |

版本来自目标机读取；NPU 推理还依赖镜像内的 CANN、驱动与 transformers 适配，同样的 pip 版本并不保证代码完全相同。

## CPU 仿真环境

已有脚本为 `scripts/setup_libero_env.sh`，目标是 Linux aarch64。它会安装系统依赖和 Conda 环境，也有修改全局 Git 配置、接受 Conda 渠道条款等步骤；重建前先阅读，不要在共享环境直接盲跑。

该脚本是历史诊断脚本：部分安装失败后继续，而且浅克隆上游默认分支后检查固定 SHA。若上游 HEAD 已变化，应先取得锁定提交，不能忽略 SHA 检查。

LIBERO-PRO 锁定版本：`eafdb809426b13153aa1e4c42d6601844217dfec`。

```bash
# gap-sim 中，已安装好 libero_pro 环境之后
source /opt/miniconda3/bin/activate libero_pro
cd /workspace/repo
export PYTHONPATH=/workspace/repo/src
export MUJOCO_GL=osmesa
export LIBERO_CONFIG_PATH=/workspace/data/libero_config
python scripts/libero_env_smoke.py
```

`requirements-dev.txt` 只有 pytest，不是运行环境依赖表。Python 3.8 仿真环境也不要直接安装其中面向新 Python 的 pytest 版本。

## 场景资产

已部署环境需要保留：

- `/workspace/repo/upstream/LIBERO-PRO`，含机器人及场景资源；
- `/workspace/data/libero_pro_assets` 中的 BDDL 与初态文件；
- `/workspace/data/libero_config/config.yaml`；
- `/workspace/data/l1_case_inventory.json`；
- 已生成的 Env 条件资产。

`deploy_libero_pro_assets.sh` 只负责把已有资产接入环境，不会下载完整数据。来源与锁定信息见 `configs/libero_pro.lock.json`。Env 条件生成入口为 `l1_generate_env_condition.py`，已有相同数据时不必重新生成。

现有清单覆盖 240 个任务—条件组合；环境基础检查不等于全部策略评测通过。Sem 输入与部分 Env 条件的已知限制见仓库 `docs/acceptance/l0-l1-baseline-and-cases.md`。


[代码目录](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro/) · [Wiki 首页](https://github.com/Xxc33128/embodied-ai/wiki)
