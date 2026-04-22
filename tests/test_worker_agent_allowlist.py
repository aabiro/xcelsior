"""P3/C7 — worker_agent drain-side allowlist (defence-in-depth)."""
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / "worker_agent.py"


def test_c7_allowlist_constant_exists():
    src = SRC.read_text()
    assert "_AGENT_COMMAND_ALLOWED = frozenset({" in src


def test_c7_allowlist_covers_known_commands():
    import worker_agent  # noqa

    # Must match routes/agent.py _AGENT_COMMAND_ALLOWED exactly.
    expected = {
        "reinject_shell",
        "upgrade_agent",
        "stop_container",
        "pause_container",
        "start_container",
        "snapshot_container",
    }
    assert worker_agent._AGENT_COMMAND_ALLOWED == expected


def test_c7_drain_rejects_unknown_at_top():
    """The rejection block must appear BEFORE any if/elif dispatch."""
    src = SRC.read_text()
    idx = src.find("def drain_agent_commands(")
    body_end = src.find("\ndef ", idx + 1)
    body = src[idx:body_end]
    reject_idx = body.find("if name not in _AGENT_COMMAND_ALLOWED")
    first_dispatch_idx = body.find('if name == "reinject_shell"')
    assert reject_idx > 0
    assert first_dispatch_idx > 0
    assert reject_idx < first_dispatch_idx, (
        "allowlist check must precede the first dispatch branch"
    )


def test_c7_no_else_soft_warning():
    """The old `else: log.warning('Unknown agent command ...')` fallback
    must be gone — the allowlist at the top replaces it."""
    src = SRC.read_text()
    assert "Unknown agent command cmd=" not in src, (
        "legacy soft-warning else branch still present"
    )


def test_c7_rejection_counter_labeled():
    src = SRC.read_text()
    assert "_agent_commands_rejected_total" in src
    assert ".labels(command=" in src


def test_c7_allowlist_matches_api_side():
    """Worker and API allowlists must agree to avoid drift."""
    import worker_agent  # noqa
    from routes import agent as agent_route

    assert worker_agent._AGENT_COMMAND_ALLOWED == agent_route._AGENT_COMMAND_ALLOWED, (
        "worker and API allowlists diverged"
    )
