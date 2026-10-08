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

import os
import re
import subprocess
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
RUN_RUNNER = WORKFLOWS.parents[1] / "scripts" / "ci-runner" / "run-runner.sh"

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


# ── The runner serving another repository ───────────────────────────────────────
#
# `XCELSIOR_CI_REPO` points the runner at another repository; aabiro.github.io
# deploys from it. The preflight used to read this checkout's workflows whatever
# that variable said, so it validated xcelsior and then registered for a
# repository it had never looked at. These run the real script against a foreign
# repository, with `gh` serving that repository's workflow files and `docker`
# recording what it was asked to do.

_STUB_GH = r"""#!/usr/bin/env bash
for arg in "$@"; do
    case "$arg" in
        */actions/runners/registration-token) echo stub-token; exit 0 ;;
        */contents/.github/workflows)
            [[ -d "$STUB_WORKFLOWS" ]] || exit 1
            for f in "$STUB_WORKFLOWS"/*; do echo ".github/workflows/${f##*/}"; done
            exit 0 ;;
        */contents/.github/workflows/*) cat "$STUB_WORKFLOWS/${arg##*/}"; exit ;;
    esac
done
exit 1
"""

_STUB_DOCKER = """#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
"""

_SAFE = """\
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: [self-hosted, sandboxed]
"""

_FORKABLE = """\
on:
  push:
  pull_request:
jobs:
  test:
    runs-on: [self-hosted, sandboxed]
"""

_HOSTED = """\
on:
  pull_request:
jobs:
  lint:
    runs-on: ubuntu-latest
"""


def _run_runner_for_foreign_repo(
    tmp_path: Path, workflows: dict[str, str] | None
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("gh", _STUB_GH), ("docker", _STUB_DOCKER)):
        stub = bin_dir / name
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)
    wf_dir = tmp_path / "workflows"
    if workflows is not None:
        wf_dir.mkdir()
        for name, text in workflows.items():
            (wf_dir / name).write_text(text, encoding="utf-8")
    log = tmp_path / "docker.log"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "XCELSIOR_CI_REPO": "example/portfolio",
        "STUB_WORKFLOWS": str(wf_dir),
        "STUB_LOG": str(log),
    }
    result = subprocess.run(
        ["bash", str(RUN_RUNNER)], env=env, capture_output=True, text=True, timeout=60
    )
    return result, log.read_text(encoding="utf-8") if log.exists() else ""


def test_a_foreign_repos_fork_triggered_workflow_stops_the_runner(tmp_path: Path) -> None:
    result, docker = _run_runner_for_foreign_repo(
        tmp_path, {"deploy.yml": _FORKABLE, "lint.yml": _HOSTED}
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "REFUSING TO START" in result.stderr
    assert "deploy.yml in example/portfolio" in result.stderr
    assert docker == "", "the runner was built or started despite the refusal"


def test_a_foreign_repo_is_checked_rather_than_this_checkout(tmp_path: Path) -> None:
    """A pull_request trigger on a hosted job is not this runner's concern."""
    result, docker = _run_runner_for_foreign_repo(
        tmp_path, {"deploy.yml": _SAFE, "lint.yml": _HOSTED}
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 workflow(s) in example/portfolio" in result.stdout
    assert "RUNNER_REPO_URL=https://github.com/example/portfolio" in docker


def test_unreadable_foreign_workflows_stop_the_runner(tmp_path: Path) -> None:
    """No workflows to read is no evidence that none of them is fork-triggerable."""
    result, docker = _run_runner_for_foreign_repo(tmp_path, None)

    assert result.returncode == 1, result.stdout + result.stderr
    assert "could not list the workflows" in result.stderr
    assert docker == ""
