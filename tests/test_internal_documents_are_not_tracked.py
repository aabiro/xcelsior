"""Internal documents and generated billing exports must never be tracked again.

On 2026-11-11 this repository's history was rewritten in two git filter-repo
passes and force-pushed, to remove internal documents that should never have
been public. They came from two places:

* The platform's agent workflow writes review, evidence and handoff documents as
  it works. They were written into `api/docs/review/`, and `.gitignore` un-ignored
  that directory on purpose, so every record was committed and published.
* Billing tooling exported live Stripe configuration as JSON into the tree.

The scrub removed the files. It did not stop either source: the workflow regrows
the directory, and one `git add -A` (or `git add -f`) publishes it again. This
guard is what stops it.

## What it checks

1. No tracked file is at one of the paths below. This is the check that catches a
   `git add -f`, which `.gitignore` cannot prevent.
2. `.gitignore` really ignores each path. The old exemption was a `!` negation
   with a reasoned comment. A new one would let `git add -A` publish the directory
   again while every rule in the block still read as present, so this asks git
   rather than reading the file.

The list is held here as well as in `.gitignore` on purpose. If this test read its
patterns from `.gitignore`, deleting a rule there would delete the check with it.

## Where these documents go instead

`~/xcelsior-internal-docs`, outside the repository. See CONTRIBUTING.md.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: gitignore-syntax patterns, matching the block at the end of `.gitignore` that
#: starts "Internal documents and generated exports - never commit".
NEVER_TRACKED = (
    # Internal review, evidence and handoff documents.
    "/docs/review/",
    "/api/docs/review/",
    "/api/docs/billing/",
    "/api/docs/billing-money-path.md",
    "/docs/billing-money-path.md",
    "/SPOT_RUNBOOK.md",
    "/SPOT_ROLLOUT.md",
    "/api/docs/SPOT_RUNBOOK.md",
    "/archive/SPOT_RUNBOOK.md",
    "/archive/SPOT_ROLLOUT.md",
    "/google_support.md",
    "/archive/google_support.md",
    # Generated exports of live Stripe configuration. The code that reads them
    # (`billing/stripe_catalog.py`) is ordinary code and stays tracked.
    "/api/config/stripe_dashboard.json",
    "/api/config/stripe_catalog.json",
    "/config/stripe_dashboard.json",
    "/config/stripe_catalog.json",
    "/billing/stripe_dashboard.json",
    "/billing/stripe_catalog.json",
    # Removed in the scrub, not to be re-added. Unanchored: any directory.
    "polish_stripe_dashboard.py",
    # Local research and evidence, CVP briefs and results among them. Ignored
    # before the scrub only by the old checkout's .git/info/exclude.
    "/reports/",
    "/research_notes/",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def _is_checkout() -> bool:
    return _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


pytestmark = pytest.mark.skipif(
    not _is_checkout(), reason="needs a git checkout: these checks read the index and .gitignore"
)


def _probes(pattern: str) -> list[str]:
    """Concrete paths `pattern` has to cover, for `git check-ignore`."""
    rel = pattern.lstrip("/")
    if rel.endswith("/"):
        # A file the workflow would write, and one a level deeper.
        return [f"{rel}probe.md", f"{rel}evidence/probe.json"]
    if not pattern.startswith("/"):
        return [rel, f"scripts/{rel}", f"api/scripts/{rel}"]
    return [rel]


def test_no_internal_document_or_export_is_tracked() -> None:
    args = ["ls-files", "--cached", "--ignored"]
    for pattern in NEVER_TRACKED:
        args += ["--exclude", pattern]
    result = _git(*args)
    assert result.returncode == 0, result.stderr

    tracked = result.stdout.splitlines()
    assert not tracked, (
        "These files are internal documents or generated billing exports, which "
        "were removed from this repository's history on 2026-11-11 and must not "
        "come back. Untrack them with `git rm --cached <path>` (the working copy "
        "stays on disk) and keep them in ~/xcelsior-internal-docs:\n  " + "\n  ".join(tracked)
    )


@pytest.mark.parametrize("path", [p for pattern in NEVER_TRACKED for p in _probes(pattern)])
def test_gitignore_keeps_it_out(path: str) -> None:
    # `--no-index` asks about the rules alone; the test above covers the index.
    if _git("check-ignore", "--no-index", "--quiet", path).returncode == 0:
        return
    why = _git("check-ignore", "--no-index", "--verbose", "--non-matching", path).stdout.strip()
    pytest.fail(
        f"{path} is not ignored by .gitignore, so `git add -A` would publish it. "
        f"Matching rule (empty if none): {why!r}. The block at the end of "
        ".gitignore must cover it and stay last, after every `!` negation."
    )
