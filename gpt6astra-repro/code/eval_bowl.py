"""Phase 3 正式评测脚本：GPT-6 Astra / GLM-4.6V on pick-place 任务。

参数对齐 RoboCurve agent policy：
  medium thinking effort / 20 次 LLM 调用预算 / 25% 速度上限 / 默认安全护栏。

用法示例：
  # mock 世界（Mac，零成本验证）：
  .venv/bin/python eval_bowl.py --embodiment cubepick --trials 3 \
      --model glm-4.6v --base-url https://open.bigmodel.cn/api/paas/v4 \
      --key-env ZHIPUAI_API_KEY --log-dir logs-glm-mock

  # Isaac LiftCube（Kaggle T4）：
  /kaggle/tmp/venv/bin/python eval_bowl.py --embodiment isaacsim-liftcube \
      --trials 20 --model gpt-6-astra --key-env OPENAI_API_KEY --log-dir logs-astra
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embodiment", default="cubepick",
                    choices=["cubepick", "isaacsim-liftcube", "mujoco-bowl",
                             "robosuite-bowl"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None, help="OpenAI 兼容端点（GLM 等自定义接入）")
    ap.add_argument("--key-env", default=None, help="API key 的环境变量名")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-llm-calls", type=int, default=20)
    ap.add_argument("--max-speed-frac", type=float, default=0.25)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--instruction", default=None)
    args = ap.parse_args()

    from inspect_robots import eval
    from inspect_robots.scene import Scene
    from inspect_robots.scorer import episode_length, success_at_end
    from inspect_robots.task import Epochs, Task
    from inspect_robots_agent import LLMAgentPolicy

    if args.embodiment == "cubepick":
        from inspect_robots.mock import CubePickEmbodiment
        embodiment = CubePickEmbodiment()
        instruction = args.instruction or "reach the cube"
    elif args.embodiment == "mujoco-bowl":
        from mujoco_bowl import MuJoCoBowlEmbodiment
        embodiment = MuJoCoBowlEmbodiment()
        instruction = args.instruction or "pick up the red cube and place it in the bowl"
        if args.max_steps == 200:
            args.max_steps = 350  # 3D 抓放含重试，比 reach 需要更多步
    elif args.embodiment == "robosuite-bowl":
        # 新口径主路线：robosuite Panda + OSC + 漏斗碗（在 .venv-robosuite 运行）
        from robosuite_bowl import RobosuiteBowlEmbodiment
        embodiment = RobosuiteBowlEmbodiment()
        instruction = args.instruction or (
            "pick up the red cube and place it in the light-blue bowl")
        if args.max_steps == 200:
            args.max_steps = 350  # 20Hz 控制，~350 步 ≈ 17.5s，够一次完整抓放
    else:
        from inspect_robots_isaacsim import IsaacSimEmbodiment
        embodiment = IsaacSimEmbodiment(
            task_id="Isaac-Lift-Cube-Franka-v0",
            cameras=[("base_rgb", 224, 224)],
            headless=True,
            device="cuda:0",
        )
        instruction = args.instruction or "pick up the cube and lift it"

    task = Task(
        name=f"{args.embodiment}-{args.model.replace('/', '_')}",
        scenes=[
            Scene(id=f"seed-{i}", instruction=instruction,
                  init_seed=args.seed_base + i)
            for i in range(args.trials)
        ],
        scorer=[success_at_end(), episode_length()],
        max_steps=args.max_steps,
        epochs=Epochs(count=1, reducer="mean"),
    )
    policy = LLMAgentPolicy(
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.key_env,
        max_llm_calls=args.max_llm_calls,
        max_speed_frac=args.max_speed_frac,
        effort=args.effort,
    )
    (log,) = eval(task, policy, embodiment, log_dir=args.log_dir)

    print(f"\nstatus: {log.status}")
    print(f"model:  {args.model}   embodiment: {args.embodiment}   trials: {log.results.total_trials}")
    for name, value in sorted(log.results.metrics.items()):
        print(f"  {name}: {value:.4g}")
    print(f"\nlog: {log.location if hasattr(log, 'location') else args.log_dir}")


if __name__ == "__main__":
    sys.exit(main())
