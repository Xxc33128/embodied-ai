# 在另一台 NPU 服务器复现

同事在自己的 NPU 上推理。两台服务器只在准备权重和适配文件时通信，不共享推理服务，也不需要重新训练或转换。

## 准备顺序

1. 从本仓库克隆实验代码。
2. 按 [场景下载](https://github.com/Xxc33128/embodied-ai/wiki/Assets) 获取固定版本 LIBERO-PRO 和本项目补充包。
3. 从已有服务器复制转换后的 π0.5 权重，核对 NPU 软件环境。
4. 先运行 Student，再按 [Mac 监听程序](https://github.com/Xxc33128/embodied-ai/wiki/Mac-bridge) 接入 GPT，运行 Direct／Hybrid。

## 复制 π0.5 权重

`pi05_libero_pt` 约 6.8 GiB。必须复制整个目录，包含 `assets` 归一化统计；不能只复制 `model.safetensors`。

在同事服务器运行。`SOURCE` 为可连接来源服务器的 SSH 别名；源路径由来源机维护者提供，账号须有读取权限。目标使用新的目录，双方需安装 rsync。

```bash
SOURCE=source-npu
SOURCE_DATA=/absolute/source/data
DEST_DATA="$HOME/gap-repro-data"
mkdir -p "$DEST_DATA/checkpoints/pi05_libero_pt"
rsync -aP "$SOURCE:$SOURCE_DATA/checkpoints/pi05_libero_pt/"   "$DEST_DATA/checkpoints/pi05_libero_pt/"
```

中断后重新运行同一命令即可。传完按内容校验，正常情况输出为空、退出码为 0：

```bash
rsync -rnc --itemize-changes   "$SOURCE:$SOURCE_DATA/checkpoints/pi05_libero_pt/"   "$DEST_DATA/checkpoints/pi05_libero_pt/"
```

此检查不报告目标多出的旧文件，因此应使用新目录。之后将 `DEST_DATA` 挂载为容器 `/workspace/data`，推理脚本即可使用这些权重。

## NPU 镜像与适配文件是什么

镜像 `pi05-hybrid:v1` 是装好 Python、torch_npu、CANN 等用户态软件的容器环境。模型权重是它运行时加载的参数文件，两者不同。同事已有相同可运行环境时直接复用，无须重复传镜像。

当前来源容器的 `/data/pi05_hybrid` 是单独挂载的目录，其中适配后的 openpi 约 3.3 MiB，tokenizer 约 21 MiB。它们不在镜像中；同事没有相同副本时还需复制：

```bash
# SOURCE_RUNTIME 填来源宿主机上对应 /data/pi05_hybrid 的目录
SOURCE_RUNTIME=/absolute/source/pi05_hybrid
DEST_RUNTIME="$HOME/gap-repro-runtime"
mkdir -p "$DEST_RUNTIME/openpi" "$DEST_RUNTIME/tokenizer"
rsync -aLP --exclude=.git --exclude=__pycache__ --exclude=.venv   "$SOURCE:$SOURCE_RUNTIME/openpi/" "$DEST_RUNTIME/openpi/"
rsync -aLP "$SOURCE:$SOURCE_RUNTIME/tokenizer/" "$DEST_RUNTIME/tokenizer/"
```

将 `DEST_RUNTIME` 挂载为 NPU 容器的 `/data/pi05_hybrid`。`-L` 将软链接指向的文件复制为实体文件，避免保留原服务器路径。可像权重一样追加 `-rLnc --itemize-changes` 做内容校验。

维护者可在来源宿主机查看挂载对应关系：

```bash
docker inspect gap-repro --format '{{range .Mounts}}{{println .Source "->" .Destination}}{{end}}'
```

如果同事还没有运行环境，需要维护者交付可运行镜像，或补齐构建流程。本仓尚无覆盖全部 NPU 适配的 Dockerfile。镜像不包含宿主机 NPU 驱动、挂载卷，也不自动包含运行中容器未固化的修改；不能只凭硬件型号相同就认定环境一致。

## 运行前核对

实验代码挂到 `/workspace/repo`，data 挂到 `/workspace/data`，适配目录挂到 `/data/pi05_hybrid`。按 Environment 页核对软件版本及本机设备／驱动，再按 Weights 页启动策略服务。

这里是文件交付和运行步骤；尚未在同事的新服务器完成实际安装与回合验收。现有策略服务采用本机共享目录请求／响应，没有新增跨服务器推理 API。
