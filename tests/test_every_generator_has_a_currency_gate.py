"""Every generator must have something checking its output is current.

The endpoint inventory drifted because it was generated, checked in, and
ungated — 535 rows against a 557-row reality, for as long as nobody regenerated
by hand. That was fixed by adding a gate. This file exists so the *fifth* gate
does not get forgotten the same way the regeneration did: a new
`scripts/generate_*.py` arriving without a currency check fails here, at the
moment it is added, rather than years later when its output is silently wrong.

The backlog below is not an opinion that those generators are fine. Measured
2026-09-19 by running `scripts/generate_dashboard_svgs.py` against a clean
tree: `frontend/public/gpu.svg` and `gpu-light.svg` came back **modified** —
committed assets that no longer match their generator — and four further
outputs (`rocket.svg`, `rocket-light.svg`, and two
`xcelsior-hosts-setup-transparent*`) are not committed at all. The generators
are deterministic, so this is real drift, not noise, and every one of these is
gateable. They are listed as debt, with the count pinned so it can only shrink.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._source_tree import iter_source_files

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SELF = Path(__file__).name


def _test_sources() -> list[tuple[Path, str]]:
    """Python and TypeScript test files, via the shared walker.

    Not `rglob`: `tests/_source_tree.py` excludes macOS AppleDouble sidecars,
    which broke four gates at once on 2026-08-04 with a UnicodeDecodeError
    naming neither the sidecar nor the gate's subject. `mcp/` is re-admitted
    because the TypeScript surface gate lives there and is exactly what this
    file needs to see.
    """
    out: list[tuple[Path, str]] = []
    for suffix in ("*.py", "*.ts"):
        out.extend(
            iter_source_files(
                suffix,
                include_prefixes=("mcp/",),
                include_tests=True,
            )
        )
    # Match on the test DIRECTORY, not the substring "test": `"test" in rel`
    # admitted `scripts/regenerate_untested_endpoints.py` — "un-TEST-ed" — and
    # the gate then reported a nonexistent generator referenced by a "test".
    return [
        (p, rel)
        for p, rel in out
        if (rel.startswith("tests/") or "/tests/" in rel) and p.name != SELF
    ]

#: Generators whose output has no currency check yet. Each entry is debt.
#: Adding to this list is not allowed — see the count assertion below.
UNGATED_BACKLOG = {
    "generate_ai_onboarding_svgs.py": "writes checked-in SVGs under frontend/public",
    "generate_dashboard_svgs.py": (
        "writes checked-in SVGs under frontend/public; VERIFIED STALE 2026-09-19 "
        "(gpu.svg and gpu-light.svg differ from a fresh run; four more outputs "
        "are not committed at all)"
    ),
    "generate_features_svgs.py": "writes checked-in SVGs under frontend/public",
    "generate_gpu_page_svgs.py": "writes checked-in SVGs under frontend/public",
    "generate_mcp_page_svgs.py": "writes checked-in SVGs under frontend/public",
    "regenerate_untested_endpoints.py": (
        "writes the checked-in UNTESTED_ENDPOINTS.md; found only after fixing "
        "this file's own glob, which looked for generate_* and so could not see "
        "a generator named regenerate_*"
    ),
    "gen_alaska_path.py": (
        "one-shot SVG path derivation that fetches over the network; its output "
        "is pasted by hand rather than written to a tracked path, so a currency "
        "check would need a different shape from the others"
    ),
}

#: Pinned to the measured count — 7, not the 5 first counted: the original
#: glob missed two generators entirely.
#:
#: Pinned to the measured count. It may go DOWN as gates are added; it must
#: never go up. A ratchet with slack is not a ratchet — an earlier version of
#: this pattern in this repo was written with headroom and passed its own
#: control.
MAX_UNGATED = 7


#: Generator name shapes. `generate_*` alone missed
#: `regenerate_untested_endpoints.py` and `gen_alaska_path.py` — the gate meant
#: to catch an unlisted generator could not see two of them.
GENERATOR_GLOBS = ("generate_*.py", "regenerate_*.py", "gen_*.py")


def _generators() -> list[str]:
    names = {p.name for g in GENERATOR_GLOBS for p in SCRIPTS.glob(g)}
    return sorted(names)


def _is_gated(generator: str) -> list[str]:
    """Test files that name this generator — the proxy for 'something checks it'."""
    hits: list[str] = []
    for path, rel in _test_sources():
        try:
            if generator in path.read_text(encoding="utf-8", errors="ignore"):
                hits.append(rel)
        except OSError:  # pragma: no cover
            continue
    return hits


def test_every_generator_is_gated_or_explicitly_listed_as_debt() -> None:
    unaccounted = [
        g for g in _generators() if not _is_gated(g) and g not in UNGATED_BACKLOG
    ]
    assert not unaccounted, (
        f"these generators write output that nothing checks for staleness: "
        f"{unaccounted}. Add a currency gate (see "
        "tests/test_generated_artifacts_are_current.py for the shape), or add it "
        "to UNGATED_BACKLOG with a reason and raise MAX_UNGATED deliberately."
    )


def test_the_backlog_does_not_grow() -> None:
    assert len(UNGATED_BACKLOG) <= MAX_UNGATED, (
        f"{len(UNGATED_BACKLOG)} ungated generators, up from {MAX_UNGATED}. A new "
        "generator is gated in the commit that adds it; this number never rises."
    )


def test_the_discovery_finds_every_generator_the_backlog_names() -> None:
    """The gate must be able to see its own blind spot.

    `GENERATOR_GLOBS` was `("generate_*.py",)`, which silently missed
    `regenerate_untested_endpoints.py` and `gen_alaska_path.py`. Nothing failed:
    a generator the discovery cannot see is simply absent from every check, so
    the gate reported a clean backlog of 5 while 7 generators were ungated.

    Narrowing the globs now fails here, because the backlog names files the
    discovery would no longer return. Verified: with the old single glob all
    other tests stay green and only this one goes red.
    """
    discovered = set(_generators())
    invisible = sorted(g for g in UNGATED_BACKLOG if g not in discovered)
    assert not invisible, (
        f"GENERATOR_GLOBS does not discover {invisible}, which the backlog names. "
        "The discovery is narrower than reality, so generators outside it are "
        "checked by nothing and this gate cannot tell."
    )


def test_the_backlog_has_no_stale_entries() -> None:
    """An exemption for a generator that is gone, or has since been gated, lies."""
    gone = [g for g in UNGATED_BACKLOG if not (SCRIPTS / g).exists()]
    assert not gone, f"UNGATED_BACKLOG names generators that no longer exist: {gone}"

    now_gated = {g: _is_gated(g) for g in UNGATED_BACKLOG if _is_gated(g)}
    assert not now_gated, (
        f"these are listed as ungated but something now checks them: {now_gated}. "
        "Remove them from UNGATED_BACKLOG and lower MAX_UNGATED in the same commit."
    )


def test_the_gated_ones_really_are_gated() -> None:
    """The positive control: this file's own notion of 'gated' must work."""
    for generator in ("generate_endpoint_inventory.py", "generate_public_openapi.py"):
        assert (SCRIPTS / generator).exists(), f"{generator} disappeared"
        assert _is_gated(generator), (
            f"{generator} has a gate in this repo, but _is_gated() cannot see it — "
            "the detection is broken, so every result above is meaningless"
        )


def test_every_gate_names_a_generator_that_exists() -> None:
    """A gate pointed at a renamed generator silently stops checking anything."""
    referenced: set[str] = set()
    for path, _rel in _test_sources():
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover
            continue
        # `\b` is load-bearing: without it this matched `generate_untested…`
        # inside `REgenerate_untested_endpoints.py` and reported a generator
        # that never existed.
        referenced.update(re.findall(r"\b(?:re)?generate_[a-z0-9_]+\.py", body))
    missing = sorted(g for g in referenced if not (SCRIPTS / g).exists())
    assert not missing, (
        f"tests reference generators that do not exist: {missing} — those gates "
        "are checking nothing"
    )
