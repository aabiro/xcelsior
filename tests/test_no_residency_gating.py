"""Xcelsior is a global marketplace. This test keeps it that way.

Capacity comes from independent hosts worldwide and is selected on price,
availability, GPU model, and host reputation — never on geography. The platform
does not gate placement on location, does not market itself as Canada-first, and
does not assert data-residency guarantees it cannot keep for a workload running
on a host in another country.

**Why this file is written the way it is.** The first version of this guard
scanned only git-tracked files, only ten filename suffixes, and skipped a long
exclusion list. It reported zero while whole blog posts, a published API docs
page titled "Compliance & Data Residency", the agent-facing `llms.txt`, a served
dashboard template, and an entire generated SDK resource were still full of it —
because `.mdx`, `.html`, and `.txt` were simply not in the suffix list. A guard
narrowed until it passes is worse than no guard: it turns an open problem into a
false assurance.

So this version scans **every file** under the repository. Exclusions are only
for things genuinely not ours to edit (dependencies, build output, vendored
copies) or historical records whose subject *is* the removal. Each is named and
justified, so adding one is a change a reviewer can see.

**To fix a failure: delete the reference.** There is no budget to raise.
"""

from __future__ import annotations

import pathlib
import re
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The vocabulary of the removed placement model. Deliberately excludes bare
# "CAD" (a currency, not a location) and "province" where it means tax
# collection, which is a real obligation rather than a placement constraint.
FORBIDDEN = re.compile(
    r"pipeda"
    r"|lprpde"
    r"|law\s*25|loi\s*25"
    r"|canada.first|canadian.first"
    r"|data.sovereignty|sovereignty|sovereign\b"
    r"|quebec.?pia"
    r"|residency"
    r"|jurisdiction"
    r"|require_residency"
    r"|is_canadian"
    r"|canada.only|canada_only"
    r"|x-data-residency",
    re.I,
)

# Directories not ours to edit, or regenerated wholesale.
EXCLUDED_DIRS = (
    "node_modules",
    ".git",
    ".next",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "htmlcov",
    "target",          # Rust/Tauri build output, incl. vendored licence files
    "site-assets",     # vendored third-party marketing bundle
    "coverage",
    ".hypothesis",     # property-test corpus: machine-generated string soup
    # Applied migration history. A migration that removed a column must name it;
    # rewriting one would falsify the record of what the schema used to be.
    "migrations/versions",
)

# Individual files that record the removal itself, or are third-party verbatim.
EXCLUDED_FILES = (
    "tests/test_no_residency_gating.py",
    "docs/mcp_tools_chatgpt.md",   # verbatim third-party research output
    "docs/mcp_tools_gemini.md",
    "docs/mcp_tools_grok.md",
    "stripe_embed.md",            # verbatim Stripe documentation dump
    # Submitted as written; JSON carries no comment syntax for the marker.
    "docs/compliance/scip_partner_loi.json",
)

# Binary and lockfile extensions: scanning them yields noise, not signal.
SKIP_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".pdf",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".webm", ".mov", ".zip",
    ".gz", ".tar", ".so", ".dylib", ".dll", ".bin", ".wasm", ".lock",
    ".appimage", ".deb", ".rpm", ".node",
    # Local, untracked backups of a developer's own env file. Editing someone's
    # backup to satisfy a test would be rewriting their record of a past state.
    ".bak",
    # Captured runtime logs: a record of what the system said at the time.
    ".log",
}

# A line asserting the concept is *absent* is the guard, not the thing coming
# back. Without this, a test proving `is_canadian_compute` no longer exists
# would fail this check, and the only way to satisfy both would be to delete
# the proof.
ABSENCE_ASSERTION = re.compile(
    r"assert not\b|not hasattr\b|not in \b|no longer|was removed|is gone"
    r"|must not\b|never\b|removed in\b|dropped in\b|# noqa: residency-guard",
    re.I,
)

# "Jurisdiction" in a governing-law clause means *which courts hear a dispute*,
# not where data lives. Every contract has one, and Xcelsior Compute Inc. is an
# Ontario company, so this is both accurate and legally necessary. Recognised
# narrowly — the line must be about courts, venue, or governing law.
LEGAL_VENUE = re.compile(
    r"governed by the laws|governing law|courts? of|venue|arbitrat|dispute",
    re.I,
)

# A file may opt out entirely by carrying this marker, for the narrow case of
# prose whose whole subject is the removal — a migration's rationale, or a test
# explaining what it used to assert and why that was wrong. The marker is
# explicit and greppable, so every exemption is auditable rather than silent.
DOCUMENTS_REMOVAL = "residency-guard: documents-removal"


def _is_excluded(rel: str) -> bool:
    if set(rel.split("/")) & set(EXCLUDED_DIRS):
        return True
    if rel.startswith("migrations/versions/"):
        return True
    # Generated crawl/link reports: a snapshot of URLs that existed at the time.
    if rel.startswith("link-checker-results"):
        return True
    # `.env.bak.<timestamp>` — a local snapshot, not configuration in use.
    if ".bak." in rel or rel.endswith(".bak"):
        return True
    if rel.startswith("docs/archive/"):
        return True
    return rel in EXCLUDED_FILES


def scan(root: pathlib.Path | None = None) -> dict[str, list[str]]:
    """Every offending line under *root* (the repository by default), by file.

    `root` exists so the self-check below can point the *same* scanner at a
    temporary directory. Walking the whole repository nine times to prove the
    scanner reads nine file types took longer than the suite's per-test timeout
    and killed the run — a guard nobody can afford to execute is not a guard.
    """
    base = root or ROOT
    hits: dict[str, list[str]] = {}
    for path in base.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        rel = path.relative_to(base).as_posix()
        if root is None and _is_excluded(rel):
            continue
        try:
            body = path.read_text(encoding="utf-8", errors="strict")
        except (OSError, UnicodeDecodeError):
            continue  # binary or unreadable: nothing a human wrote as prose
        if DOCUMENTS_REMOVAL in body:
            continue
        offending = [
            f"{n}: {line.strip()[:120]}"
            for n, line in enumerate(body.splitlines(), 1)
            if FORBIDDEN.search(line)
            and not ABSENCE_ASSERTION.search(line)
            and not LEGAL_VENUE.search(line)
        ]
        if offending:
            hits[rel] = offending
    return hits


def test_no_residency_or_sovereignty_anywhere():
    """Zero. Not a budget, not a ratchet — zero."""
    hits = scan()
    assert not hits, (
        "Residency / sovereignty / jurisdiction vocabulary is present. Xcelsior "
        "is a global marketplace: delete the reference. If a file's entire "
        f"subject is the removal, mark it `{DOCUMENTS_REMOVAL}`.\n"
        + "\n".join(
            f"  {f}\n" + "\n".join(f"      {line}" for line in lines[:6])
            for f, lines in sorted(hits.items())
        )
    )


def test_the_guard_reads_the_file_types_that_hid_it():
    """The first version of this guard missed `.mdx`, `.html` and `.txt`.

    Those are exactly where the worst of it lived: blog posts, a published docs
    page, a served dashboard template, and the agent-facing `llms.txt`. If the
    scanner ever stops reading them, this fails rather than silently reporting
    a clean repository.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        suffixes = (".mdx", ".html", ".txt", ".py", ".ts", ".tsx", ".yml", ".json", ".csv", ".md")
        for suffix in suffixes:
            (root / f"probe{suffix}").write_text(
                "this line mentions data residency\n", encoding="utf-8"
            )
        # One walk, all suffixes: cheap enough to actually run in CI.
        found = set(scan(root))
        missing = sorted(s for s in suffixes if f"probe{s}" not in found)
        assert not missing, f"scanner does not read these file types: {missing}"

        # And it must skip binaries rather than choke on them.
        (root / "probe.png").write_bytes(b"\x89PNG\r\n\x1a\n residency")
        assert "probe.png" not in scan(root)


def test_absence_assertions_are_not_counted():
    """A test proving the thing is gone must not itself trip the guard."""
    assert ABSENCE_ASSERTION.search('assert not hasattr(meter, "is_canadian_compute")')
    assert ABSENCE_ASSERTION.search("residency is no longer a placement input")
    assert not ABSENCE_ASSERTION.search("province-level residency controls apply")


@pytest.mark.parametrize(
    "path",
    [
        "llms.txt",
        "templates/dashboard.html",
        "fern/pages/compliance.mdx",
        "fern/pages/security.mdx",
        "fern/pages/introduction.mdx",
        "frontend/src/lib/i18n/en.ts",
        "frontend/src/lib/i18n/fr.ts",
        "README.md",
        "mcp/src/auth/scopes.ts",
        "routes/_deps.py",
        "scheduler.py",
        "billing.py",
    ],
)
def test_high_traffic_surfaces_are_clean(path):
    """Named explicitly, because each of these shipped the old positioning.

    A wildcard scan can be narrowed; a parametrised case cannot be dropped
    without it showing up in review.
    """
    target = ROOT / path
    if not target.exists():
        pytest.skip(f"{path} does not exist")
    body = target.read_text(encoding="utf-8", errors="ignore")
    if DOCUMENTS_REMOVAL in body:
        pytest.skip(f"{path} documents the removal")
    offending = [
        line.strip()
        for line in body.splitlines()
        if FORBIDDEN.search(line)
        and not ABSENCE_ASSERTION.search(line)
        and not LEGAL_VENUE.search(line)
    ]
    assert not offending, f"{path} still carries it: {offending[:5]}"


def test_deleted_modules_stay_deleted():
    for path in (
        "jurisdiction.py",
        "routes/jurisdiction.py",
        "frontend/sdk/api/resources/jurisdiction",
        "frontend/content/blog/pipeda-compliant-gpu-cloud.mdx",
        "frontend/content/blog/law-25-gpu-compute.mdx",
        "frontend/content/blog/rent-gpu-canada.mdx",
        "frontend/content/blog/canadian-ai-compute.mdx",
    ):
        assert not (ROOT / path).exists(), f"{path} came back"


def test_no_placement_gating_symbols_survive():
    """The functions and flags that actually gated placement."""
    banned = (
        "CANADA_ONLY",
        "set_canada_only",
        "allocate_jurisdiction_aware",
        "process_queue_sovereign",
        "filter_hosts_by_jurisdiction",
        "compute_fund_eligible_amount",
        "requires_quebec_pia",
        "generate_residency_trace",
        "SOVEREIGNTY_PREMIUM_PCT",
    )
    offenders: dict[str, list[str]] = {}
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if _is_excluded(rel) or rel == "tests/test_no_residency_gating.py":
            continue
        body = path.read_text(encoding="utf-8", errors="ignore")
        found = [s for s in banned if s in body]
        if found:
            offenders[rel] = found
    assert not offenders, f"placement-gating symbols are back: {offenders}"
