#!/usr/bin/env python3
"""
explicit_fifo_unit_test_r2.py — R4 P0-2: unit test of the ACTUAL executed
`P3InstrumentedDelayedImplicitActuator` explicit per-physics-step position FIFO.

The real class file is imported (with minimal stubs for isaaclab/era bases that
need Omniverse Kit). This tests the shipped implementation, NOT the native
Isaac Lab DelayBuffer (that native test is kept separately as engineering
supplement fifo_unit_test_r1.json).

Verified semantics (must match MuJoCo PerStepDelayFifo repeat-first prefill):
  delay=4:   effective[k] == requested[max(0, k-4)]  (first 4 outputs == first input)
  delay=0:   effective[k] == requested[k]
  reset():   prefill restarts with next first requested (no leak across episodes)

Usage: python explicit_fifo_unit_test_r2.py --root .
"""
import argparse, pathlib, sys, types, json, hashlib, importlib.util
import numpy as np
import torch

# ---------------- stubs for isaaclab / era base (module import only) ----------------
def install_stubs():
    if "carb" in sys.modules:
        return
    isaaclab = types.ModuleType("isaaclab")
    actuators = types.ModuleType("isaaclab.actuators")
    act_pd = types.ModuleType("isaaclab.actuators.actuator_pd")
    utils = types.ModuleType("isaaclab.utils")
    types_mod = types.ModuleType("isaaclab.utils.types")

    class ArticulationActions:  # minimal stand-in
        def __init__(self, joint_positions=None, joint_velocities=None, joint_efforts=None):
            self.joint_positions = joint_positions
            self.joint_velocities = joint_velocities
            self.joint_efforts = joint_efforts

    class ImplicitActuator:  # mimics only what P3.compute calls via super()
        def __init__(self, cfg, joint_names, joint_ids, num_envs, device, **kw):
            self.cfg = cfg
            self.joint_names = list(joint_names)
            self.joint_ids = joint_ids
            self._num_envs = num_envs
            self._device = device
            self.stiffness = kw.get("stiffness")
            self.damping = kw.get("damping")
            if isinstance(self.stiffness, (int, float)):
                self.stiffness = torch.full((num_envs, len(self.joint_names)), float(self.stiffness), device=device)
            if isinstance(self.damping, (int, float)):
                self.damping = torch.full((num_envs, len(self.joint_names)), float(self.damping), device=device)
            self.computed_effort = torch.zeros(num_envs, len(self.joint_names), device=device)
            self.applied_effort = self.computed_effort.clone()
            self.num_joints = len(self.joint_names)

        def _clip_effort(self, effort):
            return effort

        def compute(self, control_action, joint_pos, joint_vel):
            error_pos = control_action.joint_positions - joint_pos
            self.computed_effort = self.stiffness * error_pos + self.damping * (0.0 if joint_vel is None else -joint_vel)
            self.applied_effort = self._clip_effort(self.computed_effort)
            return control_action

    act_pd.ImplicitActuator = ImplicitActuator
    actuators.ImplicitActuator = ImplicitActuator
    types_mod.ArticulationActions = ArticulationActions

    era_pkg = types.ModuleType("era_okcc_humanoid_lab")
    robots_pkg = types.ModuleType("era_okcc_humanoid_lab.robots")
    act_mod = types.ModuleType("era_okcc_humanoid_lab.robots.actuator")

    class DelayedImplicitActuatorCfg:
        def __init__(self, min_delay=0, max_delay=0):
            self.min_delay = min_delay
            self.max_delay = max_delay

    class _FakeDelayBuffer:
        def __init__(self, history_length, batch_size, device):
            self._hl = history_length
            self.time_lags = torch.zeros(batch_size, dtype=torch.int, device=device)
        @property
        def history_length(self):
            return self._hl
        def set_time_lag(self, lags, ids=None):
            pass
        def reset(self, ids=None):
            pass
        def compute(self, data):
            return data.clone()

    class DelayedImplicitActuator(ImplicitActuator):
        def __init__(self, cfg, joint_names, joint_ids, num_envs, device, **kw):
            super().__init__(cfg, joint_names, joint_ids, num_envs, device, **kw)
            self.positions_delay_buffer = _FakeDelayBuffer(cfg.max_delay, num_envs, device)
            self.velocities_delay_buffer = _FakeDelayBuffer(cfg.max_delay, num_envs, device)
            self.efforts_delay_buffer = _FakeDelayBuffer(cfg.max_delay, num_envs, device)
        def reset(self, env_ids=None):
            pass

    act_mod.DelayedImplicitActuator = DelayedImplicitActuator
    act_mod.DelayedImplicitActuatorCfg = DelayedImplicitActuatorCfg

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.actuators"] = actuators
    sys.modules["isaaclab.actuators.actuator_pd"] = act_pd
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.types"] = types_mod
    sys.modules["era_okcc_humanoid_lab"] = era_pkg
    sys.modules["era_okcc_humanoid_lab.robots"] = robots_pkg
    sys.modules["era_okcc_humanoid_lab.robots.actuator"] = act_mod


def load_class(act_path):
    install_stubs()
    spec = importlib.util.spec_from_file_location("p3_instr_r2_under_test", str(act_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.P3InstrumentedDelayedImplicitActuator


def make_actuator(cls, delay):
    cfg = types.SimpleNamespace(min_delay=delay, max_delay=delay)
    n = 3
    act = cls(cfg, ["j0", "j1", "j2"], [0, 1, 2], 1, "cpu",
              stiffness=torch.full((1, n), 10.0), damping=torch.full((1, n), 1.0))
    act.reset(None)
    return act


def step_actuator(act, value):
    n = len(act.joint_names)
    ca = types.SimpleNamespace(
        joint_positions=torch.full((1, n), float(value), device="cpu"),
        joint_velocities=None,
        joint_efforts=None)
    jp = torch.zeros(1, n, device="cpu")
    jv = torch.zeros(1, n, device="cpu")
    out = act.compute(ca, jp, jv)
    return out.joint_positions[0, 0].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root).resolve()
    act_path = root / "00_inputs/executed_sources/p3_instrumented_delayed_actuator_r2.py"
    if not act_path.is_file():
        print(f"actuator file missing {act_path}", file=sys.stderr)
        sys.exit(2)
    act_sha = hashlib.sha256(act_path.read_bytes()).hexdigest()
    cls = load_class(act_path)

    # 1) delay=4 (Isaac 20ms = 4 x 5ms physics steps): effective[k] == requested[max(0,k-4)]
    act = make_actuator(cls, 4)
    reqs = [0.0, 0.0, 1.0] + [0.0] * 25
    effs = [step_actuator(act, r) for r in reqs]
    expected = [reqs[max(0, k - 4)] for k in range(len(reqs))]
    diff4 = max(abs(a - b) for a, b in zip(effs, expected))
    t1 = diff4 < 1e-9
    first4 = [effs[i] for i in range(4)]
    t_prefill = all(abs(v - reqs[0]) < 1e-9 for v in first4)
    pulse_pos = int(np.argmax(effs))
    t_pulse = (pulse_pos == 6)  # pulse 1.0 at input idx2 -> output idx6
    phys_count = act._p3_physics_step_counter == len(reqs)

    # 2) delay=0 identity
    act0 = make_actuator(cls, 0)
    reqs0 = [float(i) for i in range(10)]
    effs0 = [step_actuator(act0, r) for r in reqs0]
    t_zero = max(abs(a - b) for a, b in zip(effs0, reqs0)) < 1e-9

    # 3) reset leak check: reset mid-stream then new episode must restart prefill
    #    general relation with per-step varying inputs: eff[k] == req[max(0,k-4)]
    #    (first primed copies + first appended input both equal req[0])
    act.reset(None)
    reqs3 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    effs3 = [step_actuator(act, r) for r in reqs3]
    expected3 = [reqs3[max(0, k - 4)] for k in range(len(reqs3))]
    t_reset = abs(effs3[0] - 7.0) < 1e-9  # no leak of old stream (7.0, not e.g. 1.0)
    t_reset_prefill = max(abs(a - b) for a, b in zip(effs3, expected3)) < 1e-9

    overall = bool(t1 and t_zero and t_reset and t_reset_prefill and t_prefill and t_pulse and phys_count)
    result = {
        "implementation_under_test": "00_inputs/executed_sources/p3_instrumented_delayed_actuator_r2.py",
        "actuator_source_sha256": act_sha,
        "expected_sha256": "85a06b932e91dff057a7f2f4ddcb2697b7bc321ba43deb5c4236680245da5582",
        "sha_match": act_sha == "85a06b932e91dff057a7f2f4ddcb2697b7bc321ba43deb5c4236680245da5582",
        "position_delay_backend": "explicit_deque",
        "tests": {
            "delay4_effective_equals_requested_shift4": {"pass": t1, "max_diff": diff4},
            "delay4_first4_repeat_first_prefill": {"pass": t_prefill, "first4": first4},
            "delay4_pulse_landing_idx6": {"pass": t_pulse, "idx": pulse_pos},
            "delay0_identity": {"pass": t_zero},
            "reset_no_leak_first_output_equals_new_input": {"pass": t_reset, "value": effs3[0]},
            "reset_after_shift4_relation_holds": {"pass": t_reset_prefill, "effs": effs3, "expected": expected3},
            "physics_step_counter_matches": {"pass": bool(phys_count)},
            "constant_per_control_semantics_note": "with inputs constant within a control cycle (real runner behavior), first control yields exactly 4 identical effective outputs then eff[k]==req[k-4]",
        },
        "pass": overall,
        "note": "P0-2: unit test of the actual executed explicit-deque FIFO (repeat-first-q-des); native DelayBuffer test kept separately as engineering supplement"
    }
    out = root / "01_delay_gate_r2/explicit_fifo_unit_test.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
