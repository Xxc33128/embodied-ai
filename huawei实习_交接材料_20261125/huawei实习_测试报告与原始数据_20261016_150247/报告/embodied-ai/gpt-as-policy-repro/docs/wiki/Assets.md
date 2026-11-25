# 下载 LIBERO 仿真文件

使用两个来源即可：基础资源取固定版本的 LIBERO-PRO，补充任务与初始状态下载本项目的归档包。无需从我们的 NPU 服务器复制场景目录，也无需下载完整训练数据集。

- [基础代码与机器人／物体资源](https://github.com/Zxy-MLlab/LIBERO-PRO/tree/eafdb809426b13153aa1e4c42d6601844217dfec)
- [本项目场景补充包](https://github.com/Xxc33128/embodied-ai/releases/tag/repro-assets-20260921)

补充包约 1.6 MB（压缩后），包含 390 个已部署的上游任务／初态文件，以及 80 个本项目生成的 Env 文件。全部 470 个文件均与来源服务器逐文件核对 SHA256，80 个 Env 文件还与原实验清单一致。包内保留来源、许可和校验清单，不含模型权重或账号配置。

## 安装到新的仿真环境

以下命令在 `gap-sim` 中执行，仓库子目录已挂到 `/workspace/repo`，独立数据目录已挂到 `/workspace/data`。使用新环境；已有实验环境不要直接覆盖，应先比较版本和文件清单。

### 1. 获取固定版本基础资源

```bash
cd /workspace/repo
mkdir -p upstream
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git upstream/LIBERO-PRO
git -C upstream/LIBERO-PRO checkout --detach eafdb809426b13153aa1e4c42d6601844217dfec
git -C upstream/LIBERO-PRO rev-parse HEAD
```

如果已经克隆并处于该提交，跳过此步。不要用其他版本替代后忽略后面的检查。

### 2. 下载并检查补充包

```bash
cd /workspace/data
curl -fL --retry 3 -o libero-pro-assets-20260921.tar.gz   https://github.com/Xxc33128/embodied-ai/releases/download/repro-assets-20260921/libero-pro-assets-20260921.tar.gz
printf '%s  %s\n'   2556dd85edc362e18c228aa6df41bd571b200b48631bd939ec662fc9dd1e7455   libero-pro-assets-20260921.tar.gz | sha256sum -c -
```

只有校验成功才解压：

```bash
tar -xzf libero-pro-assets-20260921.tar.gz
cd libero-pro-assets-20260921
sha256sum -c SHA256SUMS
```

### 3. 接入场景目录

下面针对尚无 `libero_pro_assets` 的新 data 目录。已有该目录时先核对，不要在里面再建同名子链接。

```bash
cd /workspace/data
ln -s /workspace/data/libero-pro-assets-20260921/libero_pro_assets libero_pro_assets
cd /workspace/repo
bash scripts/deploy_libero_pro_assets.sh
cp -R /workspace/data/libero-pro-assets-20260921/generated_env/bddl_files/. upstream/LIBERO-PRO/libero/libero/bddl_files/
cp -R /workspace/data/libero-pro-assets-20260921/generated_env/init_files/. upstream/LIBERO-PRO/libero/libero/init_files/
```

部署脚本保留基础任务目录，并链接补充变体；最后两条命令安装归档的 Env 文件，不重新随机生成初始状态。

### 4. 设置路径并检查环境

先按 Environment 页准备 Python 依赖，再执行：

```bash
source /opt/miniconda3/bin/activate libero_pro
cd /workspace/repo
python -m pip install --no-deps -e upstream/LIBERO-PRO
export PYTHONPATH=/workspace/repo/src
export MUJOCO_GL=osmesa
export LIBERO_CONFIG_PATH=/workspace/data/libero_config
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'CONFIG'
from pathlib import Path
import yaml
root = Path('/workspace/repo/upstream/LIBERO-PRO/libero/libero')
cfg = Path('/workspace/data/libero_config/config.yaml')
if not cfg.exists():
    cfg.write_text(yaml.safe_dump({
        'benchmark_root': str(root), 'bddl_files': str(root/'bddl_files'),
        'init_states': str(root/'init_files'), 'assets': str(root/'assets'),
        'datasets': str(root.parent/'datasets')
    }))
CONFIG
python scripts/libero_env_smoke.py
python scripts/l1_case_inventory.py
```

再按 Experiments 页跑一个 Student 回合。文件下载与校验通过，不等于已完成该服务器上的仿真／NPU 验收。

## 来源

上游补充数据来自 [zhouxueyang/LIBERO-Pro](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)，数据许可为 CC BY 4.0；基础代码为 MIT。原始下载未记录 Hugging Face revision，因此本包以文件 SHA256 固定身份。Env 生成设置与清单在仓库 `scripts/l1_generate_env_condition.py`、`docs/acceptance/l1-env-gen-manifest.json`。
