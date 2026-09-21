"""Isolated recording and correctness fixes for the first pilot episode."""
import json
import time
from pathlib import Path

import numpy as np

from gap_repro.libero.gpt_bridge import LiberoRollout, InputError, _observation_packet
from gap_repro.libero.policy import build_state
from gap_repro.libero.agent_contract import validate_mode_steps


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Journal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "events.jsonl"

    def __call__(self, event):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"wall_time": time.time(), **event},
                                    default=json_default, ensure_ascii=False) + "\n")
            stream.flush()


class RecordedSession:
    """Delegates simulation, archiving every post-control observation."""
    def __init__(self, session, journal):
        self.session = session
        self.journal = journal
        self.step_id = 0
        (journal.directory / "frames").mkdir(exist_ok=True)

    def __getattr__(self, name):
        return getattr(self.session, name)

    def _save_observation(self, obs, action=None, **extra):
        import imageio.v2 as imageio
        frames = {}
        for name, key in (("agentview", "agentview_image"),
                          ("wrist", "robot0_eye_in_hand_image")):
            relative = f"frames/{self.step_id:04d}_{name}.png"
            imageio.imwrite(self.journal.directory / relative,
                            np.ascontiguousarray(obs[key][::-1, ::-1]))
            frames[name] = relative
        self.journal({"event": "observation", "step": self.step_id,
                      "action": action, "state": build_state(obs),
                      "eef_quaternion_xyzw": obs["robot0_eef_quat"],
                      "frames": frames, **extra})

    def reset_to(self, *args, **kwargs):
        obs = self.session.reset_to(*args, **kwargs)
        self.step_id = 0
        self._save_observation(obs, fingerprint_full=self.session.fingerprint_full(obs),
                               fixture_poses=self.session.fixture_poses())
        return obs

    def step(self, action):
        result = self.session.step(action)
        self.step_id += 1
        self._save_observation(result[0], action=np.asarray(action),
                               native_done=bool(result[2]))
        return result


class RecordedTransport:
    """Wrap FileNpzTransport before policy slicing to preserve full proposals."""
    def __init__(self, transport, journal):
        self.transport = transport
        self.journal = journal

    def infer(self, element, request_id):
        started = time.monotonic()
        actions, meta = self.transport.infer(element, request_id=request_id)
        self.journal({"event": "pi_proposal", "request_id": request_id,
                      "actions": actions, "meta": meta,
                      "inference_seconds": time.monotonic() - started,
                      "input_state": element["state"], "prompt": element["prompt"]})
        return actions, meta


class RecordedRollout(LiberoRollout):
    def __init__(self, *args, journal, **kwargs):
        super().__init__(*args, **kwargs)
        self.journal = journal

    def libero_start(self, task, output_dir=None):
        if hasattr(self, "_obs"):
            raise InputError("episode already started; continue from current observation")
        return super().libero_start(task, output_dir)

    def pi05_infer(self, **kwargs):
        proposal = super().pi05_infer(**kwargs)
        proposal["candidate_actions"] = np.asarray(self._pending_proposal).tolist()
        proposal["action_semantics"] = (
            "Normalized 7-D [dx,dy,dz,rx,ry,rz,gripper]; world-frame deltas; "
            "positive gripper closes, negative opens. These are the next "
            "5 execution steps, not object coordinates or future observations.")
        proposal["request_id"] = self._request_id()
        return proposal

    def _request_id(self):
        return (f"{self.case.campaign_id}|{self.case.episode_id}|"
                f"{getattr(self, '_rid_prefix', 'h')}{self.step_id}")

    def _execute_chunk(self, chunk, origin, info=None):
        if self.done:
            raise InputError("episode already terminated")
        remaining = self.case.max_steps - self.step_id
        if origin == "student" and getattr(self, "_requested_steps", None) is not None:
            remaining = min(remaining, self._requested_steps)
        chunk = np.asarray(chunk)[:remaining]
        if len(chunk) == 0:
            return self._terminal_packet(done=False)
        before = self.session.fingerprint_state(self._obs)
        start = self.step_id
        done = False
        executed = []
        for row in chunk:
            self._obs, _, done, _ = self.session.step(np.asarray(row, dtype=np.float64))
            executed.append(np.asarray(row).tolist())
            self.step_id += 1
            if done:
                break
        self.executed_gripper_closed = bool(executed[-1][6] > 0)
        entry = {"origin": origin, "t_start": start, "t_end": self.step_id,
                 "fingerprint_state_before": before,
                 "student_identity": self._active_identity,
                 "executed_info": info, "actual_actions": executed}
        self.record["chunks"].append(entry)
        self.journal({"event": "executed_chunk", **entry})
        # Every proposal is single use; another student action needs fresh inference.
        if hasattr(self, "_pending_proposal"):
            del self._pending_proposal
        if done or self.step_id >= self.case.max_steps:
            return self._terminal_packet(done=done)
        return _observation_packet(self._obs, self.case, self.step_id,
                                   self.executed_gripper_closed,
                                   request_id=self._request_id())

    def libero_execute(self, **kwargs):
        self.journal({"event": "gpt_decision", "step": self.step_id,
                      "arguments": kwargs})
        response = kwargs.get("response")
        if isinstance(response, dict):
            try:
                validate_mode_steps(response.get("mode"), response.get("steps"))
            except ValueError as error:
                raise InputError(str(error)) from error
        self._requested_steps = response.get("steps") if isinstance(response, dict) else None
        try:
            return super().libero_execute(**kwargs)
        finally:
            self._requested_steps = None

    def libero_act(self, **kwargs):
        self.journal({"event": "gpt_decision", "step": self.step_id,
                      "arguments": kwargs})
        return super().libero_act(**kwargs)
