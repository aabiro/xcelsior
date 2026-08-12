"""`.env.worker.example` must document every variable the agent reads.

A worker env template is the only description of the agent's interface that a
host operator ever sees. When it drifts, the failure is silent in the worst
direction: the agent reads an unset variable, takes a default nobody chose, and
reports healthy.

**Derived, not hand-maintained.** The required set comes from `worker_agent.py`
itself, so a new `os.environ.get` cannot ship without appearing in the template.
Two lists in two files, with a test on only one of them, is the drift vector
this exists to close.

**AST, not a source grep.** A textual search matches the variable name in a
comment or a docstring and reports coverage that isn't there — the same defect
that made the A0 boundary test match the word "report" in its own prose. This
walks actual attribute calls and subscripts.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AGENT = REPO / "worker_agent.py"
EXAMPLE = REPO / ".env.worker.example"


#: Modules the agent **ships to the host and imports**, which read env vars of
#: their own. Walking `worker_agent.py` alone understates the interface: it
#: misses `XCELSIOR_IMAGE_CACHE_MAX_GB` and `XCELSIOR_IMAGE_CACHE_EVICT_LOW_GB`,
#: which are unambiguously worker config — an operator sizing a host's image
#: cache sets exactly those — and which this test would therefore have called
#: undocumented in one direction and stale in the other.
#:
#: Derived from the agent's own shipped-file list rather than typed here, so a
#: module split out of `worker_agent.py` tomorrow is covered without an edit.
def _shipped_modules() -> list[Path]:
    """Every `.py` the agent lists as part of itself, that exists in the repo."""
    import re as _re

    source = AGENT.read_text(encoding="utf-8")
    names = set(_re.findall(r'"([a-z_][a-z0-9_]*\.py)"', source))
    return sorted(
        {AGENT} | {REPO / n for n in names if (REPO / n).is_file()},
        key=lambda p: p.name,
    )


#: Read by the agent but deliberately absent from the template.
#:
#: `XCELSIOR_AGENT_PUBLIC_INGRESS` is control-plane configuration. It decides
#: whether the *server* refuses public worker ingress; a worker setting it
#: changes nothing. It appears in a real `.env.worker` on this machine because
#: server config was pasted in, which is exactly what the template exists to
#: stop — a worker env file is copied onto rented machines the operator does
#: not control.
NOT_WORKER_CONFIG = frozenset({"XCELSIOR_AGENT_PUBLIC_INGRESS"})


def _env_names_read_by(path: Path) -> set[str]:
    """Every literal env var name the module reads, via AST."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    def _literal(node: ast.AST) -> str | None:
        return (
            node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None
        )

    for node in ast.walk(tree):
        # os.environ.get("X") / os.getenv("X")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ("get", "getenv") and node.args:
                name = _literal(node.args[0])
                # `.get` is common on plain dicts; keep only SHOUT_CASE names,
                # which is what an env var looks like and a dict key rarely is.
                if name and name.isupper() and "_" in name:
                    found.add(name)
        # os.environ["X"]
        if isinstance(node, ast.Subscript):
            name = _literal(node.slice)
            if name and name.isupper() and "_" in name:
                found.add(name)

    return found


def _names_declared_in(path: Path) -> set[str]:
    """Keys declared in an env template, ignoring comments and blanks."""
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def _required_names() -> set[str]:
    """Union across everything the agent ships, not just its entry point."""
    found: set[str] = set()
    for module in _shipped_modules():
        found |= _env_names_read_by(module)
    return found


def test_the_derivation_spans_more_than_the_entry_point():
    """The gap this closed, asserted so it cannot quietly reopen.

    `worker_image_cache.py` is split out of the agent and shipped with it, and
    it reads two cache-sizing variables an operator must set. Walking only
    `worker_agent.py` reported them as both undocumented and stale — the same
    variable failing two tests in opposite directions, which is the signature
    of a derivation that is looking in too few places.
    """
    modules = {p.name for p in _shipped_modules()}
    assert "worker_agent.py" in modules
    assert len(modules) > 1, (
        "the derivation is back to a single file; a variable read by a shipped "
        "helper would be reported as undocumented and as stale at once"
    )
    names = _required_names()
    assert "XCELSIOR_IMAGE_CACHE_MAX_GB" in names, (
        "the image-cache sizing variables are no longer being found; they are "
        "worker configuration read by a module the agent ships"
    )


def test_example_documents_every_variable_the_agent_reads():
    required = _required_names() - NOT_WORKER_CONFIG
    documented = _names_declared_in(EXAMPLE)

    missing = sorted(required - documented)
    assert not missing, (
        "`.env.worker.example` does not document variables `worker_agent.py` "
        f"reads: {missing}. An operator setting up a host cannot know to set "
        "them, so the agent will take an unchosen default and report healthy. "
        "Add them to the template, or to NOT_WORKER_CONFIG with the reason."
    )


def test_example_declares_nothing_the_agent_does_not_read():
    """The other direction: a stale entry is a documented lie."""
    required = _required_names()
    documented = _names_declared_in(EXAMPLE)

    extra = sorted(documented - required)
    assert not extra, (
        f"`.env.worker.example` documents variables `worker_agent.py` never "
        f"reads: {extra}. Setting one has no effect, which is worse than it "
        "being absent — the operator believes they configured something."
    )


def test_the_template_carries_no_values():
    """A template with a value in it is a secret waiting to be committed."""
    offenders = []
    for lineno, line in enumerate(EXAMPLE.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.split("=", 1)[1].strip():
            offenders.append(f"{lineno}: {stripped.split('=', 1)[0]}")
    assert not offenders, (
        f"`.env.worker.example` has values, not placeholders: {offenders}. "
        "This file is tracked; the real one is not."
    )


@pytest.mark.parametrize("path", [AGENT, EXAMPLE])
def test_the_inputs_exist(path: Path):
    """Both halves must be present, or the tests above pass vacuously."""
    assert path.is_file(), f"{path} is missing, so the comparison proves nothing"
