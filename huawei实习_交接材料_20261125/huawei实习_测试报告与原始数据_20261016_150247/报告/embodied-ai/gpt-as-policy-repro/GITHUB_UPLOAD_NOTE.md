> **2026-09-21 更正：** 当前复现以 [Wiki](https://github.com/Xxc33128/embodied-ai/wiki) 为准。下面是历史上传记录，不是完整安装指南。`requirements-dev.txt` 只有测试依赖；`download_checkpoint.sh` 下载 RoboDojo 权重，不能用来恢复 LIBERO 权重。Mac 监听桥和 NPU 镜像还未随仓库交付。

# 上传说明（2026-09-20）——在新电脑上恢复完整工程

本文件夹 = 工程主仓 `GPT-as-Policy-repro`（本机路径）在 commit `4bb2b06` 的
完整 git 追踪树，**剔除**了以下内容（原因与恢复方式见 §2/§3）：
`organize_objects.tar.gz`（326MB，超 GitHub 单文件 100MB 硬限）、
`Rigid/`（349MB）、`Geometry/`（19MB）（三者均为已退出范围的 W11
RoboDojo 场景资产，LIBERO 线不使用）；`upstream/` 与 `data/` 本就
gitignored（见工程根 .gitignore）。

**运行入口（唯一权威）**：`docs/acceptance/environment-handover.md`。
执行计划：`docs/superpowers/plans/2026-09-19-repro-next-execution.md`。

## 1. 本仓内容

- `src/gap_repro/`：全部工程代码（LIBERO session/动作契约/学生链路/
  runner/GPT 桥与传输/统计核算），264 项单测（`tests/`）。
- `docs/acceptance/`：验收证据（240/240 case 清单、生成 manifest、
  冒烟结果、动作探针 v1–v3、集成证明、环境交付文档）。
- `configs/`：全部 lock（上游钉定、策略服务器、GPT 接入模板）。
- `scripts/`：L1 生成/冒烟、campaign CLI、GPT episode CLI、策略服务器。
- 根目录 `2026-09-*.md`：历次对抗性审查与提案档案。

## 2. 在新电脑上恢复（开发者机；不需要 GPU）

```bash
git clone https://github.com/Xxc33128/embodied-ai && cd embodied-ai/gpt-as-policy-repro
python3 -m venv .venv && ./.venv/bin/pip install -r requirements-dev.txt
# 上游钉定克隆（原 upstream/，gitignored）——按 lock 恢复：
python3 -c "import json; print(json.dumps(json.load(open('configs/upstream.lock.json'))['pinned_sources'], indent=1))"
# 按 pinned_sources 里的 commit 逐一 git clone + checkout（GPT-as-Policy
# 为同事交付包，仅本机可获取者需从原交付渠道取，勿公开传播）
```

## 3. 实验机资产（不在 GitHub，也不宜公开）

实验实际跑在**实验服务器**的 gap-sim（环境）/ gap-repro（π₀.₅ 学生策略）
容器内；新电脑只需能 SSH 到该服务器（别名/地址不入库，凭据自带）：

- 240 case 池（BDDL+init）：已在服务器
  `…/LIBERO-PRO/libero/libero/{bddl,init}_files/`，无须重下。
- π₀.₅ checkpoint：服务器 `/workspace/data/checkpoints/pi05_libero_pt`。
- HF 资产（如需重建）：`zhouxueyang/LIBERO-Pro` 固定 revision，
  部署脚本 `scripts/deploy_libero_pro_assets.sh`。
- LIBERO 权重请按 Wiki Weights 页恢复；`scripts/download_checkpoint.sh` 仅用于 RoboDojo。

## 4. GPT 联调（新电脑上的登录用途）

入口：`scripts/run_lp5_gpt_episode.py` + `configs/gpt_access.example.json`
（route A app-server / route B HTTP 网关；凭据只写环境变量）。
**待定拓扑决策**（联调时二选一）：
a) episode 循环跑在实验机 gap-sim 容器（环境就位），route B（HTTP）从
   新电脑的网关取模型——推荐，凭证只在新电脑；
b) route A（stdio app-server）要求 codex 二进制与登录态跟 episode 循环
   同机——若用此路线需把会话层也搬到新电脑或容器内注入登录态。
当前代码两条路线的传输层均已实现并有测试（`tests/test_agent_client_transport.py`）。

## 5. 排除项重建（仅在需要 W11 RoboDojo 资产时）

`organize_objects.tar.gz` / `Rigid/` / `Geometry/` 从实验服务器
`/workspace/data/`（objsrc、organize_objects）可复原；LIBERO 单线不需要。
