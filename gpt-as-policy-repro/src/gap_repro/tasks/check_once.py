"""原 reward_manager.check_once 的逐行移植（E1，reward_manager.py L247-279）。

判据节点表示：
- 叶子 = ("call", func_name, args)：求值时经 call(func_name, args) -> 0.0/1.0；
- 结构 = list：与原文相同的 OR/AND 交替递归（子层取反）。

entry 的消费语义（E1，step() L412-422）：对 entry 的每个元素调用
check_once(element, op="or")，全部为真才 pop 该 entry（元素间 AND；
元素内部按 OR/AND 交替递归）。这修正了第二轮审查红1的复刻分歧：
"整条 entry 一次喂入 check_once" 不是 step() 的原文行为。
"""

from __future__ import annotations


def check_once(check, call, env_idx=0, op="or") -> bool:
    if isinstance(check, tuple):
        name, args = check[1], check[2]
        reward = call(name, {**args, "env_idx": env_idx})
        if reward < 1:
            return False
        return True
    elif isinstance(check, list):
        if op == "or":
            sub_success = False
            for sub_check in check:
                if isinstance(sub_check, list):
                    reward = 1 if check_once(sub_check, call, env_idx, op="and") else 0
                else:
                    name, args = sub_check[1], sub_check[2]
                    reward = call(name, {**args, "env_idx": env_idx})
                if reward is not None and reward > 0:
                    sub_success = True
                    break
            if not sub_success:
                return False
            return True
        elif op == "and":
            sub_success = True
            for sub_check in check:
                if isinstance(sub_check, list):
                    reward = 1 if check_once(sub_check, call, env_idx, op="or") else 0
                else:
                    name, args = sub_check[1], sub_check[2]
                    reward = call(name, {**args, "env_idx": env_idx})
                if reward is not None and reward < 1:
                    sub_success = False
                    break
            return sub_success
    raise ValueError(f"Invalid check type: {type(check)}")


def entry_passed(entry: list, call, env_idx=0) -> bool:
    """step() 语义：entry 的每个元素 check_once(element, op="or")，全真 → pop。"""
    return all(check_once(check, call, env_idx) for check in entry)
