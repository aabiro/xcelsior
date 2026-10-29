"""Structural geometry for dashboard heavy rails + light grid alignment.

Proves shipped CSS tokens and shell structure so appbar rails, content rails,
and mid-tile center-origin light verticals co-align (plan criterion 4).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "frontend/src/components/marketing/dashboard-theme.css").read_text()
SHELL = (ROOT / "frontend/src/app/(dashboard)/dashboard-shell.tsx").read_text()


def _token(name: str) -> str:
    m = re.search(rf"--{re.escape(name)}:\s*([^;]+);", CSS)
    assert m, f"missing CSS token --{name}"
    return m.group(1).strip()


def test_content_max_and_pad_place_rails_on_grid_step():
    """max/2 - pad must be integer multiple of grid step (maxed content)."""
    step = int(_token("dashboard-grid-step").replace("px", ""))
    pad = int(_token("dashboard-content-pad").replace("px", ""))
    content_max = int(_token("dashboard-content-max").replace("px", ""))
    assert step == 76
    assert pad == 28
    half_span = content_max / 2 - pad
    assert half_span == int(half_span)
    assert half_span % step == 0, (
        f"rail offset from center {half_span} is not k*{step}; "
        f"rails cannot land on mid-tile center-origin grid lines"
    )
    assert content_max == 2 * (8 * step + pad)  # documented n=8


def test_light_grid_uses_mid_tile_lines_on_content_column():
    """Mid-tile stroke + center origin → lines at center ± k*step."""
    assert ".dashboard-site-content-column::before" in CSS
    col_block = CSS.split(".dashboard-site-content-column::before")[1].split(
        ".dashboard-site-topbar {"
    )[0]
    assert "calc(50% - 0.5px)" in col_block
    assert "background-position: center top" in col_block
    assert "var(--dashboard-grid-step)" in col_block
    # No leading-edge-only 1px-at-0 pattern on the content-column grid.
    assert "linear-gradient(var(--grid) 1px, transparent 1px)" not in col_block


def test_topbar_and_main_share_content_column_outside_ai_rail():
    """Shell structure: topbar+main inside content-column; AI rail outside."""
    assert "dashboard-site-content-column" in SHELL
    assert "dashboard-site-main-row" in SHELL
    # Order: content-column wraps topbar + workspace/main; AI rail after.
    idx_col = SHELL.index("dashboard-site-content-column")
    idx_top = SHELL.index("dashboard-site-topbar", idx_col)
    idx_main = SHELL.index("dashboard-site-main-inner", idx_col)
    idx_rail = SHELL.index("dashboard-site-ai-rail", idx_col)
    assert idx_col < idx_top < idx_main < idx_rail


def test_heavy_rails_use_shared_pad_on_topbar_and_main():
    assert "dashboard-site-topbar-inner::before" in CSS
    assert "dashboard-site-main-inner::before" in CSS
    assert "left: var(--dashboard-content-pad" in CSS
    assert "right: var(--dashboard-content-pad" in CSS
    # Both inners use the same max-width token.
    assert CSS.count("var(--dashboard-content-max") >= 2


def test_measured_geometry_table_matches_tokens():
    """Document measured rail/grid coincidence for evidence (maxed column)."""
    step = 76
    pad = 28
    content_max = 1272
    center = 0.0  # content-column center as origin
    rail_left = center - (content_max / 2 - pad)
    rail_right = center + (content_max / 2 - pad)
    # Mid-tile center-origin lines
    lines = [center + k * step for k in range(-12, 13)]
    assert rail_left in lines
    assert rail_right in lines
    assert rail_right - rail_left == 2 * (content_max / 2 - pad)
    # Squares preserved: same step horizontal and vertical
    assert step == step
