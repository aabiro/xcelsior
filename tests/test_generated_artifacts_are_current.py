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

**`mcp/tool-surface.json` now has the same treatment**, in
`mcp/tests/unit/surface.test.ts` — "is byte-identical to a fresh generation".
It lives in the TypeScript suite because that is where its generator lives.

It was worth chasing: the checks already there did not catch a hand edit.
`changes.every(c => !c.breaking)` is *true* for any non-breaking drift, so it
passed while printing "run `npm run surface:update` to record these changes" —
a guard that could not fail for the reason its own message gave. The only other
currency check compared tool *counts*. Editing a description in the committed
snapshot left all 13 tests green.

Now covered on both sides: `mcp/tests/unit/surface.test.ts` gained
*"is byte-identical to a fresh generation"*, verified by editing a description
in the committed snapshot.

Keep this paragraph honest. The previous version still said the TypeScript
descriptions were uncovered and that "P0.3 is not finished", long after both
had stopped being true — a document about drift, carrying drift. A stale
docstring on a gate is worse than none: it describes a hole someone may go and
re-plug, or discourages relying on coverage that exists.
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
            raise RuntimeError(f"generator failed (rc={result.returncode}): {result.stderr[-400:]}")
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
    # A generator that cannot run is a FAILURE of this gate, never a skip.
    #
    # This used to catch the error and `pytest.skip`, so any import error,
    # missing environment variable or timeout turned the drift gate into a
    # green skip — with drift sitting underneath it, unreported. That is
    # precisely the failure shape this file was written to replace, reproduced
    # inside the thing doing the replacing: it gated on the *absence of a
    # failure* rather than on a positive result.
    #
    # Verified by pointing the generator at a stub that exits non-zero: the old
    # code reported "2 passed, 1 skipped"; this reports a failure.
    try:
        generated = _regenerate_to_string()
    except RuntimeError as exc:
        raise AssertionError(
            f"the drift gate could not run its generator, so nothing is checking "
            f"{INVENTORY.relative_to(ROOT)} for staleness: {exc}"
        ) from exc

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
        f"lines: checked-in {len(checked_lines)}, generated {len(fresh_lines)}\n" + "\n".join(diffs)
    )
