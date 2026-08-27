"""A document that shipped code points at must be in the repository.

`scheduler.py` tells its reader that changing the verified-host fallback is "a
P5 C2 decision — `docs/placement-preference-plan.md` §5.4 — not a drive-by
edit". `control_plane/scheduler/preference.py`, `migration_gate.py` and
**migration 105** cite the same document. It was not in the repository: `docs/*`
is gitignored with an exemption list, and that file had never been added to it.

So the pointer worked for whoever wrote it and for nobody who cloned the repo. A
migration's comment is permanent history — it will still be pointing at a missing
file years from now.

`.gitignore` predicts this failure in its own words: *"An incident write-up that
silently is not in the repository is worth less than no write-up, because it
reads as done."* It reached the tree anyway, through code references rather than
through someone forgetting to add an exemption. That is why this check reads the
**references** rather than the exemption list.

## What this does not do

It does not require the document to be *good*, or current, or to contain the
section number cited. It requires it to exist for the next reader. That is the
difference between a citation and a dead link.

Paths into other repositories are excluded by shape — `scripts/audit_ai_ml_docs.py`
greps `~/Projects/pxl-registry/docs/...`, and treating that as a broken local
reference is how a guard starts crying wolf about a file it was never that
project's job to hold.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

from tests._source_tree import iter_source_files, read_source

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: `docs/foo.md`, but only when it is not part of a longer path belonging to
#: some other checkout.
CITATION = re.compile(r"(?<![\w/~.-])(docs/[A-Za-z0-9_./-]+\.md)")

#: Referenced and deliberately not in the repository, each with the reason.
ALLOWED_ABSENT = {
    # A business document. It is cited, and publishing it is a decision for its
    # owner rather than a side effect of tidying references.
    "docs/BUSINESS_STRATEGY.md",
    # Committed twice historically and no longer in the tree. Two tests still
    # cite it for their scoping rationale. Recorded here rather than silently
    # tolerated: it is a real dead link, and restoring or re-pointing it is a
    # decision, not a cleanup.
    "docs/review/workaround-elimination-plan.md",
}


def _tracked() -> set[str]:
    out = subprocess.run(
        ["git", "ls-files", "docs"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(out.stdout.split())


def _citations() -> dict[str, list[str]]:
    """`docs/path.md -> [source files citing it]`, from shipped Python.

    `include_tests=True` because two test modules cite documents for their
    scoping rationale, and a citation from a test is a pointer a reader follows
    exactly like any other.
    """
    found: dict[str, list[str]] = {}
    for path, rel in iter_source_files("*.py", include_tests=True):
        # Not this file: its docstring and allowlist quote paths as examples,
        # and a guard that reports its own prose is noise the next reader has to
        # learn to ignore — which is how a real entry gets skimmed past.
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue
        for cited in CITATION.findall(read_source(path)):
            found.setdefault(cited, []).append(rel)
    return found


def test_the_scan_finds_citations_at_all():
    """Calibration — an empty result would satisfy the assertion below."""
    citations = _citations()
    assert len(citations) > 5, f"only {len(citations)} doc citations found; the pattern is wrong"
    assert "docs/placement-preference-plan.md" in citations


def test_every_cited_document_is_in_the_repository():
    tracked = _tracked()
    missing = {
        doc: sources
        for doc, sources in _citations().items()
        if doc not in tracked and doc not in ALLOWED_ABSENT
    }
    assert not missing, (
        "these documents are cited by shipped code and are not in the "
        "repository, so a clone follows the pointer to nothing: "
        + "; ".join(f"{d} (from {', '.join(sorted(set(s)))})" for d, s in sorted(missing.items()))
    )
