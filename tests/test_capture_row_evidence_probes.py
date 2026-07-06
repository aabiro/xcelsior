"""Row-evidence probes must not verify fleet-gate rows when live gates fail."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_CLOSURE_POLICY", "engineering_partial")

SCRATCH = Path(
    os.environ.get("XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer")
)
ROOT = Path(__file__).resolve().parent.parent

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "capture_row_evidence",
    ROOT / "scripts" / "capture_row_evidence.py",
)
cre = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cre)


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    scratch = SCRATCH if SCRATCH.is_dir() else tmp_path
    monkeypatch.setattr(cre, "SCRATCH", scratch)
    monkeypatch.setenv("XCELSIOR_GOAL_SCRATCH", str(scratch))
    return scratch


def _write(scratch: Path, name: str, data: dict) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / name).write_text(json.dumps(data), encoding="utf-8")


class TestRow2ProbeHonesty:
    def test_row2_blocked_when_accept_below_threshold(self, _scratch):
        _write(
            _scratch,
            "live-vllm-e2e.json",
            {
                "ok": True,
                "upstream_mode": "live_vllm",
                "model": "Qwen/Qwen3-4B-AWQ",
                "acceptance_rate": 0.5125,
                "eagle3_enabled": True,
                "mesh_host_count": 2,
            },
        )
        _write(
            _scratch,
            "live-speculative-proxy-evidence.json",
            {
                "upstream_mode": "live_vllm",
                "model": "Qwen/Qwen3-4B-AWQ",
                "proxy_requests": 3,
                "preset_startup_command": "--speculative-algorithm EAGLE3",
            },
        )
        _write(_scratch, "row-2-env-limit.json", {"unblock_path": "GPU > 6GB"})

        row = cre._probe_row_2()
        assert row["status"] == "blocked"
        assert row.get("gate_met") is False
        assert row.get("eagle_kept_on") is False
        assert float(row.get("mean_acceptance_rate") or 0) < 0.75

    def test_engineering_partial_keeps_row2_unchecked(self, _scratch):
        rows = {
            "2": cre._row(
                "blocked",
                "live EAGLE accept=0.52 < 0.75",
                mean_acceptance_rate=0.52,
                eagle_kept_on=False,
                gate_met=False,
                evidence_tier="live_eagle_validated_gated_off",
            ),
        }
        rows, verified, blocked, _ = cre._apply_engineering_partial_closure(rows)
        assert 2 in blocked
        assert rows["2"]["status"] == "blocked"
        assert 2 not in verified