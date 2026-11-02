"""Switching the local worker to the test stack and back restores today's prod env.

The prod backup used to be written once and reused: a June config came back on
2026-11-02, without the agent gateway URL, client certificate or v2 paths, and
the host answered every heartbeat with 410 until it was rebuilt by hand.
"""

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "api/scripts/switch_worker_env.sh"


def switch(tmp_path, mode):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    for name in ("docker", "systemctl", "sudo"):
        (fake / name).write_text("#!/bin/sh\nexit 1\n")
        (fake / name).chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{fake}:{os.environ['PATH']}",
        XCELSIOR_WORKER_ENV=str(tmp_path / "worker.env"),
        XCELSIOR_WORKER_MODE_FILE=str(tmp_path / "worker.mode"),
    )
    return subprocess.run(
        ["bash", str(SCRIPT), mode], env=env, capture_output=True, text=True, timeout=30
    )


def test_round_trip_restores_the_current_prod_env_not_the_first_one(tmp_path):
    worker_env = tmp_path / "worker.env"
    worker_env.write_text("XCELSIOR_SCHEDULER_URL=https://old.example\n")
    assert switch(tmp_path, "test").returncode == 0
    assert switch(tmp_path, "prod").returncode == 0

    current = "XCELSIOR_SCHEDULER_URL=https://agent.example\nXCELSIOR_AGENT_V2_PATHS=1\n"
    worker_env.write_text(current)
    assert switch(tmp_path, "test").returncode == 0
    assert "localhost:9501" in worker_env.read_text()
    assert switch(tmp_path, "prod").returncode == 0
    assert worker_env.read_text() == current


def test_a_second_switch_to_test_keeps_the_prod_backup(tmp_path):
    worker_env = tmp_path / "worker.env"
    prod = "XCELSIOR_SCHEDULER_URL=https://agent.example\n"
    worker_env.write_text(prod)
    assert switch(tmp_path, "test").returncode == 0
    assert switch(tmp_path, "test").returncode == 0
    assert switch(tmp_path, "prod").returncode == 0
    assert worker_env.read_text() == prod
