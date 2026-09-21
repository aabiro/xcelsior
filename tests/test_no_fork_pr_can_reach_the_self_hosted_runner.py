"""No workflow on the sandboxed runner may be triggered from a fork.

This repository is public. A `pull_request` trigger plus `runs-on:
[self-hosted, sandboxed]` lets anyone who opens a PR from a fork run code on
that machine. `gates-sandboxed.yml` documents the threat model and avoids the
trigger deliberately.

`scripts/ci-runner/run-runner.sh` already refuses to start while any such
workflow exists, and that check is correct — but its failure mode is silent in
the worst way. `mcp.yml` moved its `test` job onto the runner when hosted
Actions stayed blocked and brought its `pull_request` trigger along. The runner
then exited on startup **1,865 times**, and nothing anywhere said so: a runner
that refuses to start looks exactly like a runner that is offline, and a job
waiting for one reads as "queued" rather than "never going to run". Every
static gate in the repository was dark for weeks. The pyright gate drifted from
zero to nine findings in that window, and a forged-signature bug and six
corrupted characters shipped underneath it.

So the same rule is checked twice, on purpose. The runner enforces it where it
matters, and this test reports it where somebody is looking — before the push,
not after the runner quietly gives up.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"

#: Matches the runner's own discovery in `run-runner.sh`.
SANDBOXED = re.compile(r"runs-on:\s*\[\s*self-hosted\s*,\s*sandboxed\s*\]")

#: `on:` triggers only, anchored at the two-space indent a trigger sits at —
#: a `paths:` entry naming pull_request, or a job named for one, is not a
#: trigger. Same expression the runner uses.
FORK_TRIGGER = re.compile(r"^\s{0,2}pull_request(_target)?\s*:", re.MULTILINE)


def _sandboxed_workflows() -> list[tuple[Path, str]]:
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        if SANDBOXED.search(text):
            out.append((path, text))
    return out


def test_the_runner_still_serves_something() -> None:
    """A rule about an empty set passes for the wrong reason.

    If every workflow moved off the runner, the check below would be vacuously
    true forever and would never notice one moving back.
    """
    assert _sandboxed_workflows(), (
        "no workflow targets [self-hosted, sandboxed]; either they were renamed "
        "— in which case this guard and run-runner.sh are both looking in the "
        "wrong place — or the runner is serving nothing"
    )


def test_no_sandboxed_workflow_can_be_triggered_from_a_fork() -> None:
    offenders: list[str] = []
    for path, text in _sandboxed_workflows():
        match = FORK_TRIGGER.search(text)
        if match:
            line = text[: match.start()].count("\n") + 1
            offenders.append(f".github/workflows/{path.name}:{line}: {match.group().strip()}")

    assert not offenders, (
        "these run on the self-hosted runner AND can be triggered from a fork, "
        "so an outside contributor's PR would execute on that machine. "
        "scripts/ci-runner/run-runner.sh will refuse to start until this is "
        "fixed, which takes every other gate down with it and looks like the "
        "runner merely being offline:\n  " + "\n  ".join(offenders)
    )


def test_the_runner_preflight_still_enforces_this() -> None:
    """Belt and braces, both halves.

    This test is the visible half; the runner is the half that actually stops
    the code running. If the preflight were removed, this test would carry the
    rule alone — and a test can be skipped, deselected, or simply not run by
    whoever pushes.
    """
    script = (WORKFLOWS.parents[1] / "scripts" / "ci-runner" / "run-runner.sh").read_text(
        encoding="utf-8"
    )
    assert "REFUSING TO START" in script, (
        "run-runner.sh no longer refuses to start on a fork-triggered workflow; "
        "the enforcement that actually protects the machine is gone"
    )
    assert "self-hosted" in script and "sandboxed" in script, (
        "run-runner.sh no longer discovers the workflows that target this runner"
    )
