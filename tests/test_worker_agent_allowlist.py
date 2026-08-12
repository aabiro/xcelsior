"""P3/C7 — worker_agent drain-side allowlist (defence-in-depth)."""

from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "worker_agent.py"


def test_c7_allowlist_constant_exists():
    src = SRC.read_text()
    assert "_AGENT_COMMAND_ALLOWED = frozenset(" in src


def _server_side_allowlist() -> set[str]:
    """`routes/agent.py::_AGENT_COMMAND_ALLOWED`, read from its own source.

    Parsed rather than imported so this comparison does not depend on the
    module importing cleanly, and derived rather than restated because the
    third copy is the one that rots: adding a command meant editing the worker,
    the route **and a literal in this file**, and the literal is the one nobody
    remembers.
    """
    import ast

    tree = ast.parse((Path(__file__).resolve().parent.parent / "routes" / "agent.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "_AGENT_COMMAND_ALLOWED":
                    return {
                        elt.value
                        for elt in getattr(node.value, "elts", [])
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    }
    raise AssertionError("routes/agent.py no longer declares _AGENT_COMMAND_ALLOWED")


def test_the_derivation_finds_both_sides():
    """A guard over two empty sets passes; this is what stops that reading green."""
    import worker_agent

    assert len(_server_side_allowlist()) > 5
    assert len(worker_agent._AGENT_COMMAND_ALLOWED) > 5


def test_the_two_allowlists_agree():
    """Defence-in-depth only works if both sides list the same commands.

    A command the server will enqueue and the worker will not run is a
    directive that vanishes; one the worker will run and the server will not
    enqueue is dead code that reads as a capability. Both sides derived, so
    neither can drift from the other and there is no third copy to forget.
    """
    import worker_agent

    server = _server_side_allowlist()
    worker = set(worker_agent._AGENT_COMMAND_ALLOWED)
    assert server == worker, (
        "the agent command allowlists disagree — "
        f"server-only {sorted(server - worker)}, worker-only {sorted(worker - server)}"
    )


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
    assert reject_idx < first_dispatch_idx, "allowlist check must precede the first dispatch branch"


def test_c7_no_else_soft_warning():
    """The old `else: log.warning('Unknown agent command ...')` fallback
    must be gone — the allowlist at the top replaces it."""
    src = SRC.read_text()
    assert "Unknown agent command cmd=" not in src, "legacy soft-warning else branch still present"


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
