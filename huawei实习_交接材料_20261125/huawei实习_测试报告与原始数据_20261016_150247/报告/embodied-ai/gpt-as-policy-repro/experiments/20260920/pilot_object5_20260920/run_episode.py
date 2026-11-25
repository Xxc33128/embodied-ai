"""Unified isolated pilot runner; waits for a shared GO after initial reset."""
import argparse
import dataclasses
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/repo/src")
from recorded_runtime import (Journal, RecordedSession, RecordedTransport,
                              RecordedRollout, json_default)
from gap_repro.libero.policy import CaseSpec, FileNpzTransport, LiberoStudentPolicy
from gap_repro.libero.runner import run_student_episode
from gap_repro.libero.gpt_bridge import run_rollout
from gap_repro.agent.client import AppServerTransport
from gap_repro.sim.libero_session import LiberoSession


class PreparedSession(RecordedSession):
    def prepare(self, init):
        self.prepared = super().reset_to(init, wait_steps=10)

    def reset_to(self, *args, **kwargs):
        if hasattr(self, "prepared"):
            obs = self.prepared
            del self.prepared
            return obs
        return super().reset_to(*args, **kwargs)


class AuditedAppServer(AppServerTransport):
    def __init__(self, config, journal):
        super().__init__(config)
        self.journal = journal

    def request(self, method, params, timeout):
        if method == "thread/start":
            params = dict(params)
            params["sandbox"] = "read-only"
            params["config"] = {**params.get("config", {}),
                                "features.shell_tool": False}
            for spec in params.get("dynamicTools", []):
                if spec["name"] == "pi05_infer":
                    spec["description"] = (
                        "Infer pi05 once on current observation; return the next "
                        "5 normalized 7-D robot actions and candidate identity. "
                        "No forward-kinematics trajectory or future images are "
                        "provided. Review current RGB/proprio and candidate intent "
                        "before libero_execute. Positive gripper closes; negative opens.")
        result = super().request(method, params, timeout)
        if method == "turn/start" and isinstance(result, dict):
            return result["turn"]["id"]
        return result

    def next_message(self, timeout):
        event = super().next_message(timeout)
        self.journal({"event": "model_event", "message": event})
        return event


def timeout_handler(signum, frame):
    raise TimeoutError("pilot episode exceeded 1800 second wall budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["student", "direct", "hybrid"], required=True)
    ap.add_argument("--root", default="/workspace/data/pilot_object5_20260920")
    ap.add_argument("--config")
    ap.add_argument("--inventory", default="/workspace/data/l1_case_inventory.json")
    args = ap.parse_args()
    root = Path(args.root)
    output = root / args.method
    journal = Journal(output)
    rows = json.loads(Path(args.inventory).read_text())["cases"]
    row = next(r for r in rows if r["benchmark"] == "libero_object"
               and r["condition"] == "Ori" and r["task_index"] == 5)
    method = {"student": "student_only", "direct": "gpt_only",
              "hybrid": "pi05_plus_gpt"}[args.method]
    case = CaseSpec.from_inventory(row, campaign_id=root.name,
                                  episode_id=f"object5-init0-{args.method}",
                                  init_index=0, method=method)
    metadata = {"case": dataclasses.asdict(case), "environment_seed": 0,
                "pilot": True, "model_config": None}
    transport = None
    record = {}
    status = "infrastructure_incomplete"
    started = time.monotonic()
    try:
        session = PreparedSession(LiberoSession(suite=case.benchmark,
                                               task_index=5, seed=0), journal)
        session.prepare(0)
        policy = None
        if args.method != "direct":
            exchange = root / ("pi_student" if args.method == "student" else "pi_hybrid")
            policy = LiberoStudentPolicy(RecordedTransport(
                FileNpzTransport(exchange / "req", exchange / "resp"), journal))
        if args.method != "student":
            config = json.loads(Path(args.config).read_text())
            config["workspace"] = str(output / "appserver")
            metadata["model_config"] = {k: config.get(k) for k in ("model", "provider", "effort")}
            transport = AuditedAppServer(config, journal)
            transport.connect()
        (output / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (output / "READY").write_text(str(time.time()), encoding="utf-8")
        waiting = time.monotonic()
        while not (root / "GO").exists():
            if time.monotonic() - waiting > 1800:
                raise TimeoutError("timed out waiting for synchronized GO")
            time.sleep(0.2)
        started = time.monotonic()
        journal({"event": "started", "method": args.method})
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)
        if args.method == "student":
            result, record = run_student_episode(case, policy=policy, session=session)
            status = result.status
        else:
            rollout = RecordedRollout(case, policy=policy, session=session, journal=journal)
            try:
                status, record = run_rollout(rollout, transport, method=method,
                                             turn_timeout=300, audit_sink=journal)
            finally:
                record = rollout.record
    except Exception as error:
        record["infra_error"] = f"{type(error).__name__}: {error}"
        status = "infrastructure_incomplete"
    finally:
        signal.alarm(0)
        if transport is not None:
            try:
                transport.close()
            except Exception as error:
                record["close_error"] = f"{type(error).__name__}: {error}"
        final = {"status": status, "record": record,
                 "wall_seconds": time.monotonic() - started}
        journal({"event": "finished", **final})
        (output / "result.json").write_text(json.dumps(final, default=json_default,
                                                       indent=2), encoding="utf-8")
        print(json.dumps({"method": args.method, "status": status,
                          "success": record.get("success")}), flush=True)


if __name__ == "__main__":
    main()
