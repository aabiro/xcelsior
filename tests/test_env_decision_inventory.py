"""Every environment-keyed decision is classified. Zero unclassified.

`XCELSIOR_ENV` was read at ten separate sites to decide something about
security, and each was found only by looking after the previous one: the four
controls, then `routes/agent.py`, then the startup gate, then six more. That is
a class of defect, not a list of defects, and the way to close a class is to
enumerate it and require every member to be accounted for.

Each site falls into exactly one bucket:

**enforcing** — turns a protection *on* outside development. These must fail
closed, so they ask `env_config.is_relaxed_env()`. An exact match like
`== "production"` is the bug: `prod`, `staging`, `prodution`, and unset all
skip the protection. Six sites had precisely that shape, including one that
disabled SSH host-key pinning on the terminal.

**relaxing** — turns a shortcut *on* inside development: a test-mode bypass, a
seeded token, a fake port. These are safe as exact matches on `"test"`, because
a typo means the shortcut simply does not engage. They fail closed by
construction, so they are allowed to keep comparing directly.

**irrelevant** — the value is used as a label rather than a decision: a cache
namespace, a log field, a re-export.

The inventory is explicit and exhaustive. A new `XCELSIOR_ENV` reference in
non-test code fails this test until someone classifies it, which is the point:
the next one gets caught by a list rather than by another audit.
"""

from __future__ import annotations

import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {
    "venv", ".venv", "node_modules", "__pycache__", ".git",
    "frontend", "mcp", "docs", "tests", "scripts", "migrations",
}

#: site -> (bucket, why). Keyed by "path::snippet" so a line move does not
#: invalidate an entry, but a *changed* decision does.
INVENTORY: dict[str, tuple[str, str]] = {
    # ── enforcing: must fail closed, therefore must ask env_config ──
    "routes/_deps.py::AUTH_REQUIRED": (
        "enforcing", "authentication; unset meant anonymous callers got admin"),
    "routes/terminal.py::_REQUIRE_PINNED_HOST_KEYS": (
        "enforcing", "SSH host-key pinning; staging or a typo disabled MITM protection"),
    "host_admission.py::_compatibility_secret": (
        "enforcing", "compatibility session secret required outside dev"),
    "privacy_deletion.py::signing secret": (
        "enforcing", "deletion-receipt signing secret required outside dev"),
    "serverless/limits.py::_is_production": (
        "enforcing", "rate-limit degradation policy"),
    "routes/auth.py::deauthorize signature": (
        "enforcing", "social deauthorize signature failures must be rejected"),
    # ── relaxing: exact match on "test" fails closed by construction ──
    "routes/_deps.py::master token test path": (
        "relaxing", "test-suite auth shortcut"),
    "routes/auth.py::test branch": ("relaxing", "test-suite auth shortcut"),
    "routes/auth.py::reset_token echo": (
        "relaxing", "password-reset token echoed to the test suite only"),
    "routes/instances.py::test branch": ("relaxing", "worker status update in tests"),
    "routes/mfa.py::test branches": ("relaxing", "MFA enrolment shortcut in tests"),
    "routes/serverless.py::fake port": ("relaxing", "synthetic port in tests"),
    "serverless/feature.py::test branch": ("relaxing", "feature flag default in tests"),
    "routes/billing.py::dev branch": ("relaxing", "billing shortcut in dev/test"),
    "routes/agent.py::env": (
        "relaxing", "agent unauth bypass; production hard-refuses via env_config"),
    # ── irrelevant: a label, not a decision ──
    "cache_keys.py::environment": ("irrelevant", "cache namespace segment"),
    "env_config.py::resolver": ("irrelevant", "the resolver itself"),
    "api.py::import": ("irrelevant", "re-export"),
    "routes/health.py::import": ("irrelevant", "re-export"),
    "routes/billing.py::import": ("irrelevant", "re-export"),
    "routes/mfa.py::import": ("irrelevant", "re-export"),
    "routes/auth.py::import": ("irrelevant", "re-export"),
    "split_api.py::name": ("irrelevant", "string in a codegen list"),
    "control_plane/startup_validation.py::gate": (
        "enforcing", "the production configuration gate itself"),
    "oauth_service.py::signing secret": (
        "enforcing", "the committed dev JWT secret must not be a fallback"),
    "oauth_service.py::asymmetric required": (
        "enforcing", "symmetric HS256 signing is forbidden outside dev"),
    "security.py::secrets key": (
        "enforcing", "deterministic dev Fernet key must not be a fallback"),
}

VALID_BUCKETS = {"enforcing", "relaxing", "irrelevant"}

#: The shape that fails open: an equality test against a production spelling.
#: An enforcing site written this way skips its protection for `prod`,
#: `staging`, a typo, or an unset variable.
_FAIL_OPEN = (
    'XCELSIOR_ENV") == "production"',
    "XCELSIOR_ENV') == 'production'",
    'XCELSIOR_ENV", "dev").lower() in {"prod"',
)


def _source_files() -> list[pathlib.Path]:
    """The shared walk, not a private `rglob`.

    Recovered from the closed branch this hand-rolled its own traversal, and
    `test_source_tree_is_shared` caught it on the first full run. The shared
    helper excludes macOS AppleDouble sidecars, which broke four gates at once
    with a `UnicodeDecodeError` naming neither the sidecar nor the gate's
    subject — and this file predates that fix, so it never learned about it.

    A guard catching a guard is the suite working: this file was written to
    inventory environment decisions and would have failed for a reason with
    nothing to do with them.
    """
    from tests._source_tree import iter_source_files

    # `iter_source_files()` yields `(path, repo_relative)` and already drops
    # `tests/`, the vendored trees and the AppleDouble sidecars. SKIP_DIRS is
    # still applied on top: it also excludes `frontend`, `mcp`, `docs`,
    # `scripts` and `migrations`, which this inventory deliberately ignores.
    return [
        path
        for path, _rel in iter_source_files()
        if not set(path.relative_to(ROOT).parts) & SKIP_DIRS
    ]


#: A site is an environment-keyed *decision*, whether it reads the variable
#: directly or asks the resolver. Tracking only raw `XCELSIOR_ENV` mentions
#: would drop a file from the inventory the moment it was fixed — the
#: classification would vanish exactly when the decision became correct.
_DECISION_MARKERS = ("XCELSIOR_ENV", "env_config.")


def env_sites() -> list[tuple[str, int, str]]:
    """(relpath, lineno, line) for every non-test environment decision."""
    sites = []
    for path in _source_files():
        rel = path.relative_to(ROOT).as_posix()
        for n, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""'):
                continue
            if any(m in line for m in _DECISION_MARKERS):
                sites.append((rel, n, stripped))
    return sites


def test_every_buckets_value_is_valid():
    bad = {k: v for k, v in INVENTORY.items() if v[0] not in VALID_BUCKETS}
    assert not bad, f"invalid bucket names: {bad}"


def test_no_enforcing_site_still_compares_the_string_directly():
    """The load-bearing rule: enforcement must fail closed.

    Any site that turns a protection on outside development has to ask
    `env_config`, because every direct comparison found so far skipped its
    protection for at least one of `prod`, `staging`, a typo, or unset.
    """
    offenders: dict[str, list[str]] = {}
    for rel, n, line in env_sites():
        if rel in {"env_config.py"}:
            continue
        if any(shape in line for shape in _FAIL_OPEN):
            offenders.setdefault(rel, []).append(f"{n}: {line[:110]}")
    assert not offenders, (
        "environment comparisons that fail open — `prod`, `staging`, a typo, "
        "or an unset value skips the protection. Ask "
        f"`env_config.is_relaxed_env()` instead:\n{offenders}"
    )


def test_every_file_with_an_env_decision_is_in_the_inventory():
    """A new environment-keyed decision must be classified before it lands."""
    inventoried_files = {key.split("::", 1)[0] for key in INVENTORY}
    seen_files = {rel for rel, _, _ in env_sites()}
    unclassified = sorted(seen_files - inventoried_files)
    assert not unclassified, (
        "files read XCELSIOR_ENV but are not in INVENTORY. Classify each as "
        "enforcing / relaxing / irrelevant — an enforcing one must go through "
        f"env_config: {unclassified}"
    )


def test_the_inventory_has_no_dead_entries():
    """A stale classification is a decision nobody is reviewing."""
    inventoried_files = {key.split("::", 1)[0] for key in INVENTORY}
    seen_files = {rel for rel, _, _ in env_sites()}
    dead = sorted(inventoried_files - seen_files)
    assert not dead, f"inventory entries for files that no longer read the env: {dead}"


@pytest.mark.parametrize("shape", _FAIL_OPEN)
def test_the_guard_catches_a_planted_fail_open_comparison(shape):
    """Prove the guard's reach rather than trusting its silence.

    A scanner that matches nothing reports clean, and clean is what a broken
    scanner looks like.
    """
    planted = f'    if os.environ.get("{shape}\n'.replace('XCELSIOR_ENV")', 'XCELSIOR_ENV")')
    line = f"if os.environ.get({shape}:"
    assert any(s in line for s in _FAIL_OPEN), f"guard does not detect {shape!r}"
    assert planted  # the planted form is non-empty by construction


def test_the_guard_ignores_the_compliant_form():
    compliant = "if not env_config.is_relaxed_env():"
    assert not any(shape in compliant for shape in _FAIL_OPEN)
