"""Static gates walk the tree through one iterator, not eight of their own.

residency-guard: documents-removal

The marker above is carried for a narrow and slightly awkward reason: the ratchet
below lists gate *filenames*, one of which is `test_no_residency_gating.py`, and
the residency guard matches the vocabulary in that name. Nothing here discusses
placement — the match is on an identifier, not on prose.

That is the **seventh** time a text-scanning guard in this suite has caught a
mention rather than a use. The durable fix is the one applied to the walk detector
in this very file: match structure, not characters. The residency guard is still
character-based, so it belongs on
`docs/review/workaround-elimination-plan.md` rather than being widened here —
widening it would exempt exactly the files most likely to name the thing it hunts.

Eight modules each rolled their own `rglob("*.py")` with their own exclusions. On
2026-08-04 four failed at once on macOS AppleDouble sidecars, with an error naming
neither the sidecar nor the gate's subject. Three were patched individually, which
left five to fail next time — and a gate that fails for a reason unrelated to what
it guards trains the reader to disregard it.

This asserts the convergence rather than trusting it: a new gate that walks the
tree itself fails here, with the reason.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests._source_tree import DEFAULT_EXCLUDED_PREFIXES, is_source_file, iter_source_files

TESTS = pathlib.Path(__file__).resolve().parent

#: Modules still walking the tree directly. Every entry is a gate that will break
#: the next time macOS touches the repository. The list may only shrink — it is a
#: ratchet, not an allowlist, and the terminal state is empty.
#:
#: Not converted in one pass on purpose: each has its own exclusions and its own
#: notion of what it scans, and rewriting eight gates in a single commit is how you
#: silently narrow one of them.
_STILL_WALKING = {
    "test_log_leakage_guard.py",
    "test_no_residency_gating.py",
    "test_sql_injection_guard.py",
    "test_money_representation.py",
    "test_job_writer_inventory.py",
    "test_emitter_inventory.py",
    "test_http_timeout_guard.py",
}

#: Detected by AST, not by text. A regex over the source flagged *this module*,
#: because the assertion below has to name the call it forbids — the sixth time a
#: text-scanning guard in this suite has caught the documentation of the defect it
#: hunts. An exemption would have worked and would have been wrong: the fix is to
#: look at calls rather than at characters, so prose is structurally invisible.
#: `("os", "walk")` and any `.rglob(...)`. Qualifying `walk` by its receiver is
#: the point: a bare name match also catches `ast.walk`, which most static gates
#: here use to inspect syntax and which has nothing to do with the filesystem. The
#: first version of this check flagged eight innocent modules for that reason.
_WALK_ATTRS = {("os", "walk")}
_WALK_METHODS = {"rglob"}


def _modules_walking_the_tree() -> set[str]:
    found = set()
    for path in sorted(TESTS.glob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            receiver = getattr(func.value, "id", "")
            if func.attr in _WALK_METHODS or (receiver, func.attr) in _WALK_ATTRS:
                found.add(path.name)
                break
    return found


def test_no_new_gate_walks_the_tree_itself():
    walking = _modules_walking_the_tree()
    new = sorted(walking - _STILL_WALKING)
    assert not new, (
        f"{new} walk the tree directly. Use "
        "`tests._source_tree.iter_source_files()` — it excludes macOS AppleDouble "
        "sidecars, which broke four gates simultaneously on 2026-08-04 with a "
        "UnicodeDecodeError that named neither the sidecar nor the gate's subject."
    )


def test_the_ratchet_does_not_rot():
    """A name that stops walking must leave the list, or it becomes a rubber stamp."""
    walking = _modules_walking_the_tree()
    stale = sorted(_STILL_WALKING - walking)
    assert not stale, (
        f"{stale} no longer walk the tree — remove them from _STILL_WALKING so the "
        "ratchet keeps meaning something"
    )


def test_the_ratchet_only_falls():
    """Documents the direction, and fails if someone raises the ceiling."""
    assert len(_STILL_WALKING) <= 8, (
        f"_STILL_WALKING grew to {len(_STILL_WALKING)}; it is a ratchet and 8 was "
        "the count on 2026-08-04"
    )


# ── The iterator itself, driven both ways ──────────────────────────────


def test_sidecars_are_excluded():
    assert not is_source_file(pathlib.Path("routes/._auth.py"))
    assert is_source_file(pathlib.Path("routes/auth.py"))


def test_the_iterator_yields_real_source_and_no_junk():
    files = list(iter_source_files())
    assert files, "the iterator found no source files at all"
    assert all(p.exists() for p, _ in files)
    assert not [r for _, r in files if "/._" in r or r.startswith("._")]
    assert not [r for _, r in files if r.startswith(DEFAULT_EXCLUDED_PREFIXES)]


def test_tests_are_excluded_unless_asked_for():
    """A guard scanning its own assertions finds the pattern it forbids.

    Five separate instances of that recursion are recorded in this suite, so the
    default excludes `tests/` and a caller has to ask.
    """
    assert not [r for _, r in iter_source_files() if r.startswith("tests/")]
    assert [r for _, r in iter_source_files(include_tests=True) if r.startswith("tests/")]


def test_read_source_does_not_swallow_a_decode_error(tmp_path, monkeypatch):
    """Failing loudly is the point: a swallowed error silently drops coverage."""
    from tests import _source_tree

    corrupt = tmp_path / "corrupt.py"
    corrupt.write_bytes(b"\xb0\xb0 not utf-8")
    with pytest.raises(UnicodeDecodeError):
        _source_tree.read_source(corrupt)
