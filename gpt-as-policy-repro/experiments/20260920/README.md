# 2026-09-20 开发实验代码快照

四个目录来自服务器当日实验 code/。源码 SHA256 见 source-manifest.json。本轮未改其动作语义，未复制含机器地址的服务管理和登录配置。

这是历史开发 runner：任务与部分路径写死，部分版本不拒绝覆盖旧输出；复跑必须使用新目录。finalize_pilot.py 的 base 写死为历史目录，先复制并调整后才可用于新输出。

Mac 服务器端桥在 tools/mac_bridge/；Mac 监听端仍未找到。完整使用步骤见项目 Wiki 的 Experiments 页。基础 scripts/run_lp5_gpt_episode.py 不等价于这些含录制修正的 runner。
