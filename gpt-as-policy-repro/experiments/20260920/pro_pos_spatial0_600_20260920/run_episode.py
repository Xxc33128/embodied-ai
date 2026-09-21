"""Unified isolated pilot runner; waits for a shared GO after initial reset."""
import argparse
import dataclasses
import hashlib
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
    ap.add_argument("--root", required=True)
    ap.add_argument("--config")
    ap.add_argument("--benchmark", default="libero_spatial_swap")
    ap.add_argument("--condition", default="Pos")
    ap.add_argument("--inventory", default="/workspace/data/l1_case_inventory.json")
    args = ap.parse_args()
    root = Path(args.root)
    output = root / args.method
    if (output / "events.jsonl").exists() or (output / "result.json").exists():
        raise ValueError("Refusing to overwrite an existing episode")
    journal = Journal(output)
    rows = json.loads(Path(args.inventory).read_text())["cases"]
    if args.benchmark != "libero_spatial_swap" or args.condition != "Pos":
        raise ValueError("This run requires LIBERO-PRO Pos: libero_spatial_swap")
    row = next(r for r in rows if r["benchmark"] == args.benchmark
               and r["condition"] == args.condition and r["task_index"] == 0)
    method = {"student": "student_only", "direct": "gpt_only",
              "hybrid": "pi05_plus_gpt"}[args.method]
    case = CaseSpec.from_inventory(row, campaign_id=root.name,
                                  episode_id=f"spatial0-pos-init0-600-{args.method}",
                                  init_index=0, method=method)
    case = dataclasses.replace(case, max_steps=600)
    metadata = {"case": dataclasses.asdict(case), "environment_seed": 0,
                "pilot": True, "model_config": None,
                "control_budget": 600, "settling_steps": 10,
                "wall_timeout_seconds": 1800, "max_tool_calls": 1400}
    transport = None
    record = {}
    status = "infrastructure_incomplete"
    started = time.monotonic()
    try:
        session = PreparedSession(LiberoSession(suite=case.benchmark,
                                               task_index=case.task_index, seed=0), journal)
        bddl_path = Path(session.bddl_file)
        actual_hash = hashlib.sha256(bddl_path.read_bytes()).hexdigest()
        if bddl_path.parent.name != args.benchmark or actual_hash != row["bddl_sha256"]:
            raise ValueError("PRO BDDL path/hash does not match selected inventory")
        ori = next(r for r in rows if r["benchmark"] == "libero_spatial"
                   and r["condition"] == "Ori" and r["task_index"] == case.task_index)
        if actual_hash == ori["bddl_sha256"]:
            raise ValueError("Pos asset unexpectedly equals Ori")
        metadata["asset_verification"] = {"benchmark": args.benchmark,
            "condition": args.condition, "bddl_path": str(bddl_path),
            "bddl_sha256": actual_hash, "ori_bddl_sha256": ori["bddl_sha256"]}
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
                                             turn_timeout=300, max_tool_calls=1400,
                                             audit_sink=journal)
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
