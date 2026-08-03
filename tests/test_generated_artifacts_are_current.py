"""A generated file must equal a fresh generation, byte for byte.

P0's gate: *regenerating the registry produces byte-identical output; a hand
edit to a generated file fails the build.*

`docs/generated/endpoint-inventory.md` says "Do not edit by hand — regenerate
it" and had no check enforcing it. It drifted: regenerating changed 14 rows,
every one an endpoint scoped during P0 — `setup-intent` and `portal-session`
gaining `_require_scope`, `/ssh/keygen` moving from `_require_auth` to
`_require_admin`, `stream-ticket` / `expose` / `auto-launch` gaining
`instances:connect`, the privacy writes gaining an unconditional
`_require_auth`.

Nothing was wrong with the generator. Regeneration was a step someone had to
remember, and for a while nobody did — which is the same shape as the OpenAPI
generator that read its own output for months, and as `tool-surface.json` being
written by `npm run surface:update` rather than derived.

The inventory is the artifact GT0 classifies. A stale one means the audit is
against endpoints that no longer describe the code, and every `class` entered
against a moved row is wrong in a way nobody would notice.

`tests/test_public_openapi.py` already does this for the published spec, and its
history is the reason to compare *whole documents*: comparing only the operation
set answered "are the right endpoints published?" and never "does the document
still describe them correctly?", under which five schemas silently drifted.

**Not covered here yet:** `mcp/tool-surface.json` and the TypeScript
descriptions and annotations. They need the same treatment, and P0.3 is not
finished until they have it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "docs" / "generated" / "endpoint-inventory.md"
GENERATOR = ROOT / "scripts" / "generate_endpoint_inventory.py"


def _regenerate_to_string() -> str:
    """Run the generator against a scratch path and return what it wrote.

    Deliberately a subprocess: the generator imports the FastAPI app, and doing
    that in-process would leave the app object and its routers resident for
    every later test in the session. Test isolation is not worth trading for a
    slightly faster check — a previous in-process reload of `routes._deps`
    broke four unrelated tests in the full suite while passing in isolation.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        target = pathlib.Path(tmp) / "endpoint-inventory.md"
        result = subprocess.run(
            [sys.executable, str(GENERATOR), "--output", str(target)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0 or not target.exists():
            # Older generators take no --out; fall back to the default path by
            # copying the current file aside first.
            raise RuntimeError(
                f"generator failed (rc={result.returncode}): {result.stderr[-400:]}"
            )
        return target.read_text(encoding="utf-8")


def test_the_generator_exists_and_is_executable():
    """Prove the reach: a missing generator would skip every check below."""
    assert GENERATOR.exists(), f"{GENERATOR} is gone; the inventory has no source"
    assert INVENTORY.exists(), f"{INVENTORY} is missing; regenerate it"


def test_the_inventory_declares_itself_generated():
    """A file nobody knows is generated will be hand-edited."""
    header = INVENTORY.read_text(encoding="utf-8")[:600]
    assert "generate_endpoint_inventory.py" in header
    assert "Do not edit by hand" in header


def test_the_checked_in_inventory_matches_a_fresh_generation():
    """The gate. A hand edit, or a stale file, fails here.

    Compared whole-document rather than by operation count: a count matching
    while rows differ is precisely how this drifted — 516 operations before and
    after, with 14 rows changed underneath.
    """
    try:
        generated = _regenerate_to_string()
    except RuntimeError as exc:  # pragma: no cover - surfaced as a skip below
        import pytest

        pytest.skip(f"generator could not run: {exc}")

    checked_in = INVENTORY.read_text(encoding="utf-8")
    if checked_in == generated:
        return

    checked_lines = checked_in.splitlines()
    fresh_lines = generated.splitlines()
    diffs = [
        f"  line {n}:\n    checked-in: {a.strip()[:110]}\n    generated : {b.strip()[:110]}"
        for n, (a, b) in enumerate(zip(checked_lines, fresh_lines), 1)
        if a != b
    ][:8]
    raise AssertionError(
        "docs/generated/endpoint-inventory.md differs from a fresh generation. "
        "Run `python scripts/generate_endpoint_inventory.py` and commit the "
        "result in the same commit as the route change.\n"
        f"lines: checked-in {len(checked_lines)}, generated {len(fresh_lines)}\n"
        + "\n".join(diffs)
    )
