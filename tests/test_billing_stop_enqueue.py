"""C7 — verify billing.py enqueues correct agent commands.

Per the Phase 3 hardening plan (P3.2), all four sites in billing.py
that previously used SSH to stop/start containers were converted to
push commands through the agent queue. This test asserts those
contracts are still in place — protecting against accidental
regressions back to the old SSH path.

We use a static-source check rather than full integration because:
  1. The full `BillingService.stop_instance` path requires a live
     Postgres + 8 mock layers; the value tested (the call shape) is
     entirely determined by source.
  2. A semantic break (wrong command type, wrong field name) shows up
     immediately as a missing match in the source, which is exactly
     what we want to detect.

Sites covered:
- BillingService.stop_instance        → "pause_container"  (state-machine: STOPPED→RESTARTING needs container)
- suspended-wallet sweep / grace expired → "stop_container"
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

BILLING_PY = Path(__file__).resolve().parent.parent / "billing.py"


@pytest.fixture(scope="module")
def billing_src() -> str:
    return BILLING_PY.read_text(encoding="utf-8")


def _enqueue_calls(src: str) -> list[tuple[str, str]]:
    """Return list of (created_by, command_type) for every enqueue_agent_command.

    Matches both ``"created_by="billing_xxx"`` and the command-type
    string literal that appears as the second positional arg.
    """
    pat = re.compile(
        r"enqueue_agent_command\s*\(\s*"
        r"[^,]+,\s*"  # host_id
        r'"(?P<cmd>[a-z_]+)"\s*,\s*'  # command type
        r"\{[^}]+\}\s*,\s*"  # args dict
        r'created_by\s*=\s*"(?P<by>[a-z_]+)"',
        re.MULTILINE | re.DOTALL,
    )
    return [(m.group("by"), m.group("cmd")) for m in pat.finditer(src)]


def test_stop_instance_preserves_container(billing_src: str) -> None:
    pairs = _enqueue_calls(billing_src)
    assert ("billing_stop", "pause_container") in pairs, (
        "stop_instance must enqueue pause_container (NOT stop_container) — "
        "the STOPPED→RESTARTING transition requires the container to still "
        "exist for `docker start`. Using stop_container leaves restart broken. "
        "Found: " + repr(pairs)
    )


def test_terminal_paths_use_stop_container(billing_src: str) -> None:
    """Wallet-suspension / grace-expired truly destroy the container."""
    pairs = _enqueue_calls(billing_src)
    cmds_for_terminal = [
        c for by, c in pairs
        if by in {"billing_suspended", "billing_grace_expired", "billing_topup_failure"}
    ]
    assert any(c == "stop_container" for c in cmds_for_terminal), (
        "Terminal billing sweepers must enqueue stop_container (docker stop + rm). "
        "Found created_by labels: "
        + repr([by for by, _ in pairs])
    )


def test_no_ssh_exec_for_lifecycle(billing_src: str) -> None:
    """Regression guard: no `ssh_exec(... docker (start|stop|kill|rm) ...)` in billing.

    The whole point of P3.2 was to remove direct SSH-from-VPS-to-host
    lifecycle ops. Catching this in CI prevents reverts.
    """
    bad = re.findall(
        r"ssh_exec\([^)]*docker\s+(?:start|stop|kill|rm)\b", billing_src
    )
    assert not bad, f"billing.py reintroduced SSH-based lifecycle ops: {bad}"


def test_all_enqueues_pass_container_name_and_job_id(billing_src: str) -> None:
    """Args dict must always contain container_name AND job_id.

    Worker handlers branch on container_name; job_id is needed for the
    callback report path. Missing either is a silent breakage.
    """
    blocks = re.findall(
        r"enqueue_agent_command\s*\([^)]*\)", billing_src, re.DOTALL
    )
    assert blocks, "no enqueue_agent_command calls found in billing.py — "
    "P3.2 may have been reverted entirely"
    for blk in blocks:
        assert "container_name" in blk, (
            f"enqueue call missing container_name arg: {blk[:200]}"
        )
        assert "job_id" in blk, (
            f"enqueue call missing job_id arg: {blk[:200]}"
        )
