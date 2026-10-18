#!/usr/bin/env python3
"""Run live CRIUgpu preempt→resume demo; write criu-live-demo.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRATCH = Path(
    os.environ.get("XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer")
)
DEMO_SCRIPT = ROOT / "scripts" / "criu_process_demo.sh"


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    from criu_hosts import probe_checkpoint_stack

    probe = probe_checkpoint_stack()
    out: dict = {
        "ok": False,
        "output_unchanged": False,
        "checkpoint_class": probe.get("checkpoint_class") or "",
        "criu_available": probe.get("criu_available"),
        "docker_experimental": probe.get("docker_experimental"),
        "nvidia_driver": probe.get("nvidia_driver"),
        "probe": probe,
        "reason": "CRIU stack incomplete",
    }

    if not probe.get("criu_available"):
        out["reason"] = "criu binary not available on PATH"
    elif not probe.get("checkpoint_class"):
        out["reason"] = "checkpoint_class empty (need criu + docker experimental + driver ≥570)"
    else:
        workdir = SCRATCH / "criu-live-workdir"
        try:
            proc = subprocess.run(
                ["bash", str(DEMO_SCRIPT), str(workdir)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=180,
            )
            result_path = workdir / "result.json"
            if result_path.is_file():
                demo = json.loads(result_path.read_text(encoding="utf-8"))
                out.update(demo)
                out["probe"] = probe
                out["demo_stdout"] = (proc.stdout or "")[-2000:]
                out["demo_stderr"] = (proc.stderr or "")[-2000:]
                out["reason"] = (
                    "live CRIU preempt→resume demonstrated (same-host + simulated migrate)"
                    if demo.get("ok") and demo.get("output_unchanged")
                    else "criu demo finished without output_unchanged"
                )
            else:
                out["reason"] = f"demo script exit {proc.returncode}: {(proc.stderr or proc.stdout)[-500:]}"
        except subprocess.TimeoutExpired:
            out["reason"] = "criu demo timed out after 180s"
        except OSError as exc:
            out["reason"] = f"criu demo failed: {exc}"

    path = SCRATCH / "criu-live-demo.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)
    return 0 if out.get("ok") and out.get("output_unchanged") else 1


if __name__ == "__main__":
    raise SystemExit(main())