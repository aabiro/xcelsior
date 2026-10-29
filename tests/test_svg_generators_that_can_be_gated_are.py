"""The SVG generators whose output still matches their assets are held there.

`tests/test_every_generator_has_a_currency_gate.py` lists the SVG generators as
debt with the reason *"writes checked-in SVGs under frontend/public"*. That is a
description, not a verdict, and it was wrong for half of them.

Measured 2026-09-19 by running each against a clean tree:

    generate_features_svgs      0 modified, 0 untracked  -> matches
    generate_gpu_page_svgs      0 modified, 0 untracked  -> matches
    generate_ai_onboarding_svgs 1 modified               -> diverged
    generate_mcp_page_svgs      1 modified, 1 untracked  -> diverged
    generate_dashboard_svgs     2 modified, 4 untracked  -> diverged

The two that match are ordinary generated artifacts and are gated here. The
three that diverged are **not** stale output to be regenerated: `/gpu.svg` is
live in the marketplace page, carries no "generated" header, and was last
touched in the same feature commit as its generator before the two parted ways.
Regenerating those would overwrite live artwork with a superseded design, which
is the damage a currency gate is supposed to prevent — so they stay listed as
what they are, design scaffolds whose output has been hand-evolved.

The generators write to a fixed `OUT` with no `--output`, so each is run with
its assets copied aside and restored in a `finally`. Editing the tree as a side
effect of *checking* it would be its own kind of drift.
"""

from __future__ import annotations

import filecmp
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: generator -> the directory it writes. Only generators whose output currently
#: matches their committed assets belong here; adding a diverged one would
#: demand regeneration that destroys hand-evolved artwork.
GATED = {
    "generate_features_svgs.py": ROOT / "frontend" / "public" / "features",
    "generate_gpu_page_svgs.py": ROOT / "frontend" / "public" / "gpu-fleet",
}


def _regenerate_and_compare(generator: str, out_dir: Path) -> list[str]:
    """Run the generator over a saved copy of its outputs; return differing names."""
    with tempfile.TemporaryDirectory() as tmp:
        backup = Path(tmp) / "before"
        shutil.copytree(out_dir, backup)
        try:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / generator)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                # A generator that cannot run is a FAILURE of this gate, never a
                # skip — the hole closed in tests/test_generated_artifacts_are_current.py.
                raise AssertionError(
                    f"{generator} could not run, so nothing is checking "
                    f"{out_dir.name}: rc={result.returncode} {result.stderr[-300:]}"
                )
            names = sorted({p.name for p in backup.iterdir()} | {p.name for p in out_dir.iterdir()})
            return [
                n
                for n in names
                if not (backup / n).is_file()
                or not (out_dir / n).is_file()
                or not filecmp.cmp(backup / n, out_dir / n, shallow=False)
            ]
        finally:
            shutil.rmtree(out_dir)
            shutil.copytree(backup, out_dir)


@pytest.mark.parametrize("generator,out_dir", sorted(GATED.items()))
def test_the_committed_svgs_match_a_fresh_generation(generator: str, out_dir: Path) -> None:
    assert out_dir.is_dir(), f"{out_dir} is gone; {generator} has nowhere to write"
    differing = _regenerate_and_compare(generator, out_dir)
    assert not differing, (
        f"{out_dir.name}/ no longer matches what {generator} produces: {differing}. "
        f"Either run `python scripts/{generator}` and commit, or — if these assets "
        f"have been hand-evolved on purpose — move this generator to "
        f"UNGATED_BACKLOG in tests/test_every_generator_has_a_currency_gate.py with "
        f"that reason, as was done for the dashboard, onboarding and mcp-page sets."
    )


def test_the_gated_set_is_not_empty() -> None:
    """An emptied map turns the parametrised test above into no test at all."""
    assert GATED, "no SVG generator is gated; this file asserts nothing"


def test_every_gated_generator_and_output_directory_exists() -> None:
    """A renamed generator would silently stop being checked."""
    for generator, out_dir in GATED.items():
        assert (ROOT / "scripts" / generator).is_file(), f"{generator} no longer exists"
        assert out_dir.is_dir(), f"{out_dir} no longer exists"
