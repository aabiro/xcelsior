"""Source guard against reintroducing unfenced billing lifecycle writers."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

BILLING_PY = Path(__file__).resolve().parent.parent / "billing" / "billing.py"


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


def test_lifecycle_has_no_unfenced_enqueues(billing_src: str) -> None:
    assert _enqueue_calls(billing_src) == []
    for obsolete in ("pause_container", "start_container", "update_job_status"):
        assert obsolete not in billing_src
    assert "request_fenced_stop" in billing_src
    assert "request_fenced_stop_remove" in billing_src
    assert "request_fresh_attempt_resume" in billing_src


def test_billing_sweepers_delegate_to_stop_instance(billing_src: str) -> None:
    """Automated stops must use the same fenced lifecycle service as users."""
    delegated = re.findall(
        r"self\.stop_instance\(\s*job_id,\s*reason=\"billing_suspended\"\s*\)",
        billing_src,
    )
    assert len(delegated) >= 2, (
        "Grace-expired and suspended-wallet sweepers must delegate to "
        "stop_instance so v2 jobs receive attempt-bound commands."
    )


def test_no_ssh_exec_for_lifecycle(billing_src: str) -> None:
    """Regression guard: no `ssh_exec(... docker (start|stop|kill|rm) ...)` in billing.

    The whole point of P3.2 was to remove direct SSH-from-VPS-to-host
    lifecycle ops. Catching this in CI prevents reverts.
    """
    bad = re.findall(r"ssh_exec\([^)]*docker\s+(?:start|stop|kill|rm)\b", billing_src)
    assert not bad, f"billing.py reintroduced SSH-based lifecycle ops: {bad}"


def test_billing_lifecycle_uses_native_identity(billing_src: str) -> None:
    import ast

    tree = ast.parse(billing_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "stop_instance",
            "start_instance",
            "restart_instance",
            "terminate_instance",
        }:
            source = ast.get_source_segment(billing_src, node)
            assert "native_admission_problem" in source
            assert "spec->>'owner_id'" in source
            assert "payload->>'owner'" not in source
