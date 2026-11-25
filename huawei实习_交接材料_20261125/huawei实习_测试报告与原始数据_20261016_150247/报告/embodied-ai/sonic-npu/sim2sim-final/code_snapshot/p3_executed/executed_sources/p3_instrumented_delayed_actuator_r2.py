#!/usr/bin/env python3
"""
p3_instrumented_delayed_actuator.py — P3 R1 可观测延迟执行器

Purpose:
  为 Isaac Lab 的 DelayedImplicitActuator 增加 P3 所需的 requested / effective 子步日志
  与显式的 first-q_des prehistory 证明，不改变 0 ms 时的原轨迹。

Design:
  - 继承 DelayedImplicitActuator，仅在 position target 上记录；
    velocity / effort 仍走原链路，仅记录 position。
  - reset() 后 primed=false；首次收到有效 position target 时，用该 target 填充
    完整 delay history（通过公开 DelayBuffer.compute 接口，不直接写私有 _buffer）；
    若无法通过公开接口完成，则在 manifest 中记录所用私有字段。
  - 每次 compute() 保存 requested 与 effective（DelayBuffer 输出后）；
  - 记录 episode physics-substep 计数；
  - 支持 runner 在一次 env.step() 前清空本周期 debug history，step 后读取 4 子步；
  - 仅 position target 为 P3 研究对象。

Notes on Isaac Lab DelayBuffer semantics:
  DelayBuffer(history_length= max_delay) with CircularBuffer(max_len = history+1).
  After reset, CircularBuffer.num_pushes==0 and buffer cleared to 0.
  First append fills entire buffer with first data (is_first_push logic), so first
  effective == first requested (repeat-first). Subsequent steps return data[max(0,i-lag)].
  This gives N+1 repeats of first data for lag N (e.g., lag 4 -> 5 repeats). Our
  validator accepts both N and N+1 as valid prefill as long as first 4 equal first
  requested and shift holds for k>=4. To provide provable first-q_des prefill,
  this instrumented actuator explicitly ensures that behaviour via public compute
  priming and via logged trace.

Manifest fields expected:
  actuator_source_sha256, isaac_lab commit, delay config, private field if used.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.actuators import ImplicitActuator
from isaaclab.actuators.actuator_pd import ImplicitActuator as _ImplAct
from isaaclab.utils.types import ArticulationActions

# Import parent
from era_okcc_humanoid_lab.robots.actuator import DelayedImplicitActuator, DelayedImplicitActuatorCfg


class P3InstrumentedDelayedImplicitActuator(DelayedImplicitActuator):
    """DelayedImplicitActuator with per-substep requested/effective logging."""

    def __init__(self, cfg: DelayedImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # Episode-level logs (list of np arrays, one per physics step)
        self._p3_requested_ep: list[torch.Tensor] = []
        self._p3_effective_ep: list[torch.Tensor] = []
        # Per-control debug (cleared by runner before each env.step)
        self._p3_req_per_control: list[torch.Tensor] = []
        self._p3_eff_per_control: list[torch.Tensor] = []
        self._p3_physics_step_counter: int = 0
        self._p3_primed: bool = False
        # flag to indicate whether private buffer was accessed (should be false if we use public API)
        self._p3_used_private_buffer: bool = False
        self._p3_private_field_name: str | None = None
        # P3 explicit FIFO for position (to guarantee repeat-first-q-des semantics)
        # We use deque to avoid native DelayBuffer's zero-prefill issue.
        import collections
        self._p3_fifo_pos: collections.deque | None = None
        self._p3_fifo_len: int = int(cfg.max_delay)  # for fixed delay, max==min
        # Physics step counter for P0-2 (1-indexed, increments per compute)
        self._p3_phys_indices_per_control: list[int] = []
        self._p3_phys_indices_ep: list[int] = []

    # Runner API ----------------------------------------------------------
    def clear_per_control_debug(self):
        """Clear per-control history before next env.step (called by runner)."""
        self._p3_req_per_control.clear()
        self._p3_eff_per_control.clear()
        self._p3_phys_indices_per_control.clear()

    def get_per_control_debug(self):
        """Return last control's 4 substeps (requested, effective) as cloned tensors."""
        req = [t.clone() if isinstance(t, torch.Tensor) else t for t in self._p3_req_per_control]
        eff = [t.clone() if isinstance(t, torch.Tensor) else t for t in self._p3_eff_per_control]
        return req, eff

    def get_per_control_physics_indices(self):
        """Return per-control physics step indices (1-indexed)."""
        return list(self._p3_phys_indices_per_control)

    def get_episode_physics_indices(self):
        return list(self._p3_phys_indices_ep)

    def get_episode_logs(self):
        """Return episode-level logs (list per physics step)."""
        return self._p3_requested_ep, self._p3_effective_ep

    def get_physics_step_count(self) -> int:
        return self._p3_physics_step_counter

    def is_primed(self) -> bool:
        return self._p3_primed

    # Overrides -----------------------------------------------------------
    def reset(self, env_ids: Sequence[int] | None = None):
        # Call parent which sets random delays and resets DelayBuffers (to zero)
        super().reset(env_ids)
        # Clear our logs. For simplicity, clear all even if env_ids subset (num_envs=1)
        self._p3_requested_ep.clear()
        self._p3_effective_ep.clear()
        self._p3_req_per_control.clear()
        self._p3_eff_per_control.clear()
        self._p3_physics_step_counter = 0
        self._p3_primed = False
        # Reset explicit FIFO and phys indices
        import collections
        self._p3_fifo_pos = None  # will be initialized on first valid compute with first target
        self._p3_fifo_len = int(self.positions_delay_buffer.history_length)  # should equal max_delay
        self._p3_phys_indices_per_control.clear()
        self._p3_phys_indices_ep.clear()
        # Note: native DelayBuffer.reset cleared buffer to 0, but we now use explicit FIFO for position
        # to guarantee repeat-first-q-des. Native buffers for vel/eff still used.

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # --- capture requested before delay (position only) ---
        req_pos = None
        if control_action.joint_positions is not None:
            # clone to avoid in-place mutation by DelayBuffer
            req_pos = control_action.joint_positions.clone()

        # --- NaN guard (same as parent) ---
        if req_pos is not None and torch.isnan(req_pos).any():
            print("WARNING: NaN detected in actions, applying correction!")
            control_action.joint_positions = torch.nan_to_num(control_action.joint_positions, nan=0.0)
            req_pos = control_action.joint_positions.clone()

        if req_pos is not None:
            assert not torch.isnan(req_pos).any(), "P3Instrumented NaN correction failed!"
            assert not torch.isinf(req_pos).any(), "P3Instrumented Inf detected!"

        # --- explicit first-q-des priming via explicit FIFO (public API) ---
        # For P3 we guarantee repeat-first-q-des: on first valid target after reset,
        # fill FIFO with that target repeated delay times. This avoids native DelayBuffer's
        # zero-prefill (which gives 0 for first 4 steps) and makes prehistory provable.
        # We use explicit deque for position; velocities/efforts still use native buffers.
        first_valid = (not self._p3_primed) and (req_pos is not None)
        eff_pos = None
        if req_pos is not None:
            import collections
            delay_len = int(self._p3_fifo_len)  # 4 for 20ms, 0 for 0ms
            if delay_len == 0:
                eff_pos = req_pos.clone()
                control_action.joint_positions = eff_pos
            else:
                if not self._p3_primed:
                    # Prime FIFO with first target repeated, then pop first as effective
                    self._p3_fifo_pos = collections.deque([req_pos.clone() for _ in range(delay_len)], maxlen=delay_len)
                    eff_pos = self._p3_fifo_pos.popleft()
                    self._p3_fifo_pos.append(req_pos.clone())
                    control_action.joint_positions = eff_pos.clone()
                else:
                    # Normal: pop front as effective, push current req
                    if self._p3_fifo_pos is None:
                        # Should not happen, but initialize if None
                        self._p3_fifo_pos = collections.deque([req_pos.clone() for _ in range(delay_len)], maxlen=delay_len)
                        eff_pos = self._p3_fifo_pos.popleft()
                        self._p3_fifo_pos.append(req_pos.clone())
                    else:
                        eff_pos = self._p3_fifo_pos.popleft()
                        self._p3_fifo_pos.append(req_pos.clone())
                    control_action.joint_positions = eff_pos.clone()
            # For logging, eff_pos is the delayed target that will be used for PD
        # Note: if req_pos is None, eff_pos stays None and control_action unchanged
        # velocities / efforts (keep original chain)
        if control_action.joint_velocities is not None:
            control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        if control_action.joint_efforts is not None:
            control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        # --- logging ---
        if req_pos is not None:
            # Detach and clone for logging (keep on same device for per_control, and cpu for ep)
            try:
                req_cpu = req_pos.detach().cpu().clone()
                eff_cpu = eff_pos.detach().cpu().clone() if eff_pos is not None else None
            except Exception:
                req_cpu = req_pos.clone()
                eff_cpu = eff_pos.clone() if eff_pos is not None else None

            self._p3_requested_ep.append(req_cpu)
            self._p3_effective_ep.append(eff_cpu)
            self._p3_req_per_control.append(req_cpu)
            self._p3_eff_per_control.append(eff_cpu)
            self._p3_physics_step_counter += 1
            # P0-2: physics step index is 1-indexed counter
            phys_idx = int(self._p3_physics_step_counter)  # 1 .. N*4
            self._p3_phys_indices_ep.append(phys_idx)
            self._p3_phys_indices_per_control.append(phys_idx)
            if first_valid:
                self._p3_primed = True

        # --- delegate to ImplicitActuator for torque computation (skip DelayedImplicitActuator's double delay) ---
        # DelayedImplicitActuator.compute would delay again, so we bypass it.
        # Use ImplicitActuator.compute directly.
        return super(DelayedImplicitActuator, self).compute(control_action, joint_pos, joint_vel)

    # Helper for manifest -------------------------------------------------
    def get_delay_info(self):
        return {
            "min_delay": int(self.cfg.min_delay),
            "max_delay": int(self.cfg.max_delay),
            "history_length_positions": int(self.positions_delay_buffer.history_length),
            "time_lags": self.positions_delay_buffer.time_lags.detach().cpu().tolist() if isinstance(self.positions_delay_buffer.time_lags, torch.Tensor) else str(self.positions_delay_buffer.time_lags),
            "primed": bool(self._p3_primed),
            "physics_steps": int(self._p3_physics_step_counter),
            "used_private_buffer": bool(self._p3_used_private_buffer),
            "private_field": self._p3_private_field_name,
            "position_delay_backend": "explicit_deque",
            "native_position_delay_buffer_bypassed": True,
            "native_velocity_effort_delay_buffer_used": True,
            "implementation": "collections.deque",
            "prehistory": "repeat-first-q-des",
        }
