# 完整性检查

检查日期：2026-09-21。范围为 CPU MuJoCo／LIBERO-PRO＋NPU π0.5＋Mac GPT。

**结论：已有环境可以复用，但仓库还不是从空机器一键安装的完整交付包。** 本轮没有启动模型实验、重建容器或重做权重转换。

| 项目 | 核对结果 |
|---|---|
| 基础代码 | 公开仓库 src 与服务器对应文件 SHA256 一致 |
| 昨天的实验代码 | 原来未上传；本轮补入 experiments/20260920，含四组 runner、录制修正及视频整理脚本 |
| CPU 环境安装 | setup_libero_env.sh 已有，但属于诊断式脚本，部分失败后仍继续；退出成功不等于安装完整 |
| NPU 环境 | 依赖交付镜像 pi05-hybrid:v1 与适配后的 openpi；没有完整 Dockerfile |
| 权重转换 | 本轮归档固定版本的官方转换脚本和许可证；依赖完整 openpi，不是独立转换器 |
| 权重与场景资产 | 不在 Git；需复用已部署数据或按锁定来源恢复 |
| Mac GPT 接入 | 服务器端 stdio_socket.py 已归档；Mac 端监听桥源代码本轮未找到，不能声称新 Mac 可直接复现 |
| 配置与账号 | 只提供示例；每位使用者配置自己的模型权限和连接信息 |

## 本轮补入

- `experiments/20260920/`：服务器四组开发实验原始脚本与 SHA256 清单。录制层修正涉及请求身份、动作预算及候选动作可见性，不能省略后声称复现同一实验。
- `tools/mac_bridge/stdio_socket.py`：服务器端 Unix socket 转发。
- `vendor/openpi-converter/`：openpi 固定提交的 JAX→PyTorch 转换脚本，附 Apache-2.0 许可证。
- `patches/openpi-runtime-20260921.patch`：目标机 tokenizer 本地回退与图像布局补丁。包含旧 resize 补丁的内容，不要重复应用。

## 验证结果

3 项录制修正测试、25 项传输／GPT bridge／学生策略测试通过；全部新增 Python 文件通过语法检查。补丁对锁定 openpi 源码的 `git apply --check` 通过。没有据此声称目标机完整复现实验通过。

## 仍需交付的项目

1. Mac 端监听桥及其启动方式；仅服务器端桥接脚本不够。
2. 同版本 NPU 镜像、适配后的 openpi 和 tokenizer 文件；这些需由环境维护者提供。
3. 原始／转换权重、LIBERO 资产及实际部署路径。

历史 README 中“全部工程”“唯一外部依赖为 GPT 信息”等措辞不能作为完整性保证。后续补齐以上项目后，仍应在第二位使用者的环境完成一个全新回合。


[代码目录](https://github.com/Xxc33128/embodied-ai/tree/main/gpt-as-policy-repro/) · [Wiki 首页](https://github.com/Xxc33128/embodied-ai/wiki)
