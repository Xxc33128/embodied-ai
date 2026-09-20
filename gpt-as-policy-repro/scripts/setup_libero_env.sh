#!/usr/bin/env bash
# LP1：LIBERO-PRO 老栈仿真环境搭建（在 gap-sim 容器内执行，aarch64）。
# 阶段化执行：任一阶段失败落日志后继续，便于定位旧栈在 ARM 上的断点。
# 版本策略（计划 v0.2 §4 LP1）：物理/观测/评分相关组件按 fork 声明精确锁定；
# 平台差异（官方 cu113 torch → PyPI aarch64 CPU torch）为已记录的最小适配。
# 平台适配（已记录）：conda ToS 非交互接受；git 强制 HTTP/1.1 + safe.directory
#（本网络 HTTP/2 流会随机中断，与 HF 下载问题同源）。
set -uo pipefail
LOG=/workspace/data/libero_env_build.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "=== LP1 build start $(date -Is) ==="

export DEBIAN_FRONTEND=noninteractive
echo "--- stage 1: apt deps ---"
apt-get update -qq
apt-get install -y -qq git wget curl build-essential patchelf pkg-config \
  libgl1 libglew-dev libosmesa6-dev libglfw3 libegl1 libegl-dev libgles-dev \
  libglvnd-dev unzip ca-certificates libhdf5-dev cmake

echo "--- stage 2: miniconda + python 3.8.13 env ---"
if [ ! -x /opt/miniconda3/bin/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/miniconda3
fi
# 新版 conda 需非交互接受默认渠道 ToS
/opt/miniconda3/bin/conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/main \
  --channel https://repo.anaconda.com/pkgs/r || true
if [ ! -d /opt/miniconda3/envs/libero_pro ]; then
  # 平台适配：repo.anaconda.com 包下载在本网络间歇失败，优先清华 TUNA 渠道镜像
  # （渠道镜像不改变包版本），失败回退官方源。
  /opt/miniconda3/bin/conda create -y -n libero_pro python=3.8.13 \
    --override-channels \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
    -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r \
  || /opt/miniconda3/bin/conda create -y -n libero_pro python=3.8.13
fi
# shellcheck disable=SC1091
source /opt/miniconda3/bin/activate libero_pro
python --version || { echo "STAGE2_ENV_FAIL"; exit 1; }

echo "--- stage 3: clone canonical fork @ locked SHA ---"
git config --global http.version HTTP/1.1
git config --global --add safe.directory /workspace/repo
git config --global --add safe.directory /workspace/repo/upstream/LIBERO-PRO
mkdir -p /workspace/repo/upstream && cd /workspace/repo/upstream
if [ ! -d LIBERO-PRO/.git ]; then
  # 平台适配：全量克隆(~360MB 含历史)在本网络反复中断；锁定提交即 master HEAD，
  # 浅克隆只传快照，体积小一个量级。克隆后必须校验 HEAD == 锁定 SHA。
  for i in 1 2 3 4 5; do
    rm -rf LIBERO-PRO
    git clone -q --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO.git && break
    echo "clone_retry_$i"; sleep 15
  done
fi
cd LIBERO-PRO || { echo "STAGE3_CLONE_FAIL"; exit 1; }
HEAD_SHA=$(git rev-parse HEAD)
echo "checked_out=$HEAD_SHA"
if [ "$HEAD_SHA" != "eafdb809426b13153aa1e4c42d6601844217dfec" ]; then
  echo "STAGE3_SHA_MISMATCH: $HEAD_SHA"; exit 1
fi

echo "--- stage 4: torch 1.11.0（官方 cu113 → PyPI aarch64 CPU，已记录的最小适配）---"
# 平台适配：大 wheel 单流 HTTP 在本网络必坏，改用 fetch_big_wheels.py 预取
# （curl 断点续传 + sha256 校验），pip 只装本地文件并从源拉小依赖。
if ls /workspace/data/wheels/torch-1.11.0*.whl >/dev/null 2>&1; then
  pip install -q /workspace/data/wheels/torch-1.11.0*.whl \
      /workspace/data/wheels/torchvision-*.whl /workspace/data/wheels/torchaudio-*.whl \
    || echo "STAGE4_TORCH_FAIL"
else
  pip install -q -i https://pypi.tuna.tsinghua.edu.cn/simple \
      torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 \
    || pip install -q torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 \
    || echo "STAGE4_TORCH_FAIL"
fi
python -c "import torch; print('torch', torch.__version__)" || true

echo "--- stage 5: requirements（robosuite 1.4.0 → robomimic→egl_probe 需旧 setuptools）---"
# 平台适配：egl_probe sdist 在 setuptools>=67 下因 glad 多顶层包判定报错，
# 先钉 setuptools 65 并以 --no-build-isolation 预装 egl_probe。
pip install -q "setuptools==65.7.0" wheel || echo "STAGE5_SETUPTOOLS_FAIL"
pip install -q --no-build-isolation egl_probe==1.0.2 || echo "STAGE5_EGLPROBE_FAIL"
if ls /workspace/data/wheels/opencv_python-*.whl >/dev/null 2>&1; then
  pip install -q /workspace/data/wheels/opencv_python-*.whl \
      /workspace/data/wheels/matplotlib-*.whl /workspace/data/wheels/[Pp]illow-*.whl \
    || echo "STAGE5_BIGWHEEL_FAIL"
fi
pip install -r requirements.txt \
  || pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || echo "STAGE5_REQ_FAIL"

echo "--- stage 6: pip install -e . ---"
pip install -e . \
  || pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple \
  || echo "STAGE6_EDITABLE_FAIL"

echo "--- stage 7: smoke imports ---"
# 平台适配：libero import 时若缺 config.yaml 会触发交互式 input()（非交互直接
# EOFError，对抗审查 P1）。预生成默认 config.yaml（与 get_default_path_dict 一致），
# 显式指定 LIBERO_CONFIG_PATH，不改上游源码。
export LIBERO_CONFIG_PATH=/workspace/data/libero_config
mkdir -p "$LIBERO_CONFIG_PATH"
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
python - <<'PY'
import os, yaml
root = "/workspace/repo/upstream/LIBERO-PRO/libero/libero"
d = {
    "benchmark_root": root,
    "bddl_files": os.path.join(root, "bddl_files"),
    "init_states": os.path.join(root, "init_files"),
    "datasets": os.path.join(root, "..", "datasets"),
    "assets": os.path.join(root, "assets"),
}
cfg = os.path.join(os.environ["LIBERO_CONFIG_PATH"], "config.yaml")
with open(cfg, "w") as f:
    yaml.dump(d, f)
print("config written:", cfg)
PY
fi
python - <<'PY' || echo "STAGE7_IMPORT_FAIL"
import robosuite
import bddl
import libero
import libero.libero
from libero.libero import benchmark
print("robosuite", robosuite.__version__)
print("bddl ok; libero import ok")
PY

echo "=== LP1 build end $(date -Is) ==="
