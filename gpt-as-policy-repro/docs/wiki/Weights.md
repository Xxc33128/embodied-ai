# 权重转换与 NPU 推理

## 先区分两套权重

本指南运行 LIBERO，使用 `pi05_libero`。RoboDojo 双臂权重 `robodojo_pi05_pt` 的动作接口不同，不能替换使用。

`scripts/download_checkpoint.sh` 下载的是 RoboDojo checkpoint，**不是 LIBERO 权重下载器**。

LIBERO 原始权重来源：`gs://openpi-assets/checkpoints/pi05_libero`。完整性核查入口是 `scripts/verify_pi05_libero_crc32c.py`，既有记录见 `configs/libero_pro.lock.json`。本仓不包含大体积权重；相同环境优先复用已验证的副本。

已有可通信的来源 NPU 时，推荐直接复制整个 `pi05_libero_pt` 目录到同事服务器，含 `assets`；无需重新转换。方法见 [服务器迁移](https://github.com/Xxc33128/embodied-ai/wiki/Transfer)。

## 转换脚本

本轮归档 `vendor/openpi-converter/convert_jax_model_to_pytorch.py`，来源为 openpi 提交 `215abfb217dbac7d5f1273282331b9b1866c0479`。完整 openpi 及其依赖仍需安装，单独复制转换脚本不够。

以下是已记录的转换命令，在推理容器中执行。本轮只核对命令与源码，没有重新执行转换：

```bash
PYTHONPATH=/data/pi05_hybrid/openpi/src python   /workspace/repo/vendor/openpi-converter/convert_jax_model_to_pytorch.py   --checkpoint_dir /workspace/data/checkpoints/pi05_libero   --config_name pi05_libero   --output_path /workspace/data/checkpoints/pi05_libero_pt
```

已有转换权重时跳过转换，避免覆盖它。运行还需要转换目录中的 assets／归一化统计，以及 `/data/pi05_hybrid/tokenizer/paligemma-3b-pt-224/tokenizer.model`。

目标机的额外源码改动已记录为 `patches/openpi-runtime-20260921.patch`。先 `git apply --check`，再判断是否需要应用；相同环境可能已经含这些修改。该补丁不包含镜像全部 NPU 适配，也不替代镜像。

## 启动策略服务

在独立终端进入推理容器，使用已有授权的空闲 NPU；设备分配由环境维护者确认，不执行历史脚本中的停止其他服务操作。

```bash
docker exec -it gap-repro bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/repo
export LP_POLICY_EXCHANGE_DIR=/workspace/data/repro_colleague_001/pi_student
python scripts/lp_a2_policy_server.py npu
```

等待日志 `LP_A2_SERVER_READY device=npu`。进程实际启动并打印该行后再启动客户端，不能只相信上次留下的 ready 文件。

服务固定读取 `/workspace/data/checkpoints/pi05_libero_pt`，输出 LIBERO 的 10×7 动作，客户端执行前 5 步。它用共享卷中的 req／resp 交换文件；这不是 RoboDojo 的 HTTP 8642 服务。

Student 与 Hybrid 必须使用各自的交换目录。最简单的方式是顺序运行，上一服务结束后再换目录；不要让多个 worker 消费同一请求队列。


[代码目录](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro/) · [Wiki 首页](https://github.com/Xxc33128/embodied-ai/wiki)
