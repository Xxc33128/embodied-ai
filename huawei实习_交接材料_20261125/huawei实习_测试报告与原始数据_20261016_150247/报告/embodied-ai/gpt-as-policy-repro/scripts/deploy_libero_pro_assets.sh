#!/usr/bin/env bash
# L0 部署：LIBERO-PRO 官方配套资产接入 gap-sim 容器栈（fork README 步骤的
# 符号链接变体——不覆盖钉定 clone 的既有文件，只新增条件变体目录）。
# 前置：宿主 data 盘已有 data/libero_pro_assets/{bddl_files,init_files}
# （下载脚本产物，SHA256SUMS 部分对账 + 本地快照 manifest）。
# 用法：在 gap-sim 容器内以 root 执行本脚本（repo 挂载于 /workspace/repo，
# 资产挂载于 /workspace/data/libero_pro_assets）。
set -euo pipefail
SRC=/workspace/data/libero_pro_assets
PKG=/workspace/repo/upstream/LIBERO-PRO/libero/libero
for d in "$SRC"/bddl_files/libero_*; do
  name=$(basename "$d")
  case "$name" in
    libero_goal|libero_spatial|libero_10|libero_object) ;;  # 基线不动
  esac
  target="$PKG/bddl_files/$name"
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "SKIP (real dir exists): $target"; continue
  fi
  [ -L "$target" ] || ln -s "$d" "$target"
  echo "LINKED bddl: $target"
done
for d in "$SRC"/init_files/libero_*; do
  name=$(basename "$d")
  target="$PKG/init_files/$name"
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "SKIP (real dir exists): $target"; continue
  fi
  [ -L "$target" ] || ln -s "$d" "$target"
  echo "LINKED init: $target"
done
echo DEPLOY_DONE
