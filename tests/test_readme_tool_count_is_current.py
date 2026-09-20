"""The tool count in the README headline must match the published surface.

The headline claims *"through 86 MCP tools"*. That number was **84** in August
and 75 before that — it moves whenever the surface does, and a headline nobody
regenerates goes quietly false. This repository already gates every other
generated number for exactly that reason; a marketing claim deserves the same
treatment, because it is the first thing a reader believes and the last thing
anyone re-checks.

`mcp/tool-surface.json` is the source of truth — the same file
`tests/test_generated_artifacts_are_current.py` treats as authoritative.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
SURFACE = ROOT / "mcp" / "tool-surface.json"


def _published_count() -> int:
    data = json.loads(SURFACE.read_text(encoding="utf-8"))
    tools = data.get("tools") if isinstance(data, dict) else data
    assert tools, f"{SURFACE} lists no tools; the count below would be meaningless"
    return len(tools)


def _counts_claimed_in_readme() -> list[int]:
    """Every '<N> MCP tools' claim in the README."""
    return [int(n) for n in re.findall(r"(\d+)\s+MCP tools", README.read_text(encoding="utf-8"))]


def test_the_readme_claims_a_tool_count_at_all() -> None:
    """If the phrasing changes, this gate must fail rather than silently pass.

    A regex that matches nothing is a guard that cannot fail — the failure mode
    this file exists to prevent, reproduced inside it.
    """
    claims = _counts_claimed_in_readme()
    assert claims, (
        "no '<N> MCP tools' claim found in README.md. Either the headline was "
        "reworded — in which case update this pattern — or the count was dropped, "
        "and this gate is now asserting nothing."
    )


def test_every_tool_count_in_the_readme_matches_the_published_surface() -> None:
    published = _published_count()
    for claimed in _counts_claimed_in_readme():
        assert claimed == published, (
            f"README.md advertises {claimed} MCP tools; {SURFACE.name} publishes "
            f"{published}. The headline is the first thing a reader believes and "
            f"the last thing anyone re-checks — update it, or update the surface."
        )
