"""The /mcp link preview the site serves is the card in the brand kit.

The kit keeps its images under `public/brand-system/assets/social/`, but
`robots.txt` disallows `/brand-system/`, and X's crawler honours robots.txt: an
`og:image` or `twitter:image` pointing there would not render. So the page's
metadata points at a copy under `site-assets/`. Two copies of one image drift
silently (someone re-shoots the kit, the live card stays old), so this pins them
together.
"""

from __future__ import annotations

import filecmp
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "api" / "frontend" / "public"

KIT_CARD = (
    PUBLIC
    / "brand-system"
    / "assets"
    / "social"
    / "mcp-launch"
    / "link-preview"
    / "og-1200x630.png"
)
SERVED_CARD = PUBLIC / "site-assets" / "assets" / "png" / "og-mcp-1200x630.png"


def test_the_served_mcp_card_is_the_kit_card() -> None:
    assert KIT_CARD.is_file(), f"{KIT_CARD.relative_to(ROOT)} is missing"
    assert SERVED_CARD.is_file(), f"{SERVED_CARD.relative_to(ROOT)} is missing"
    assert filecmp.cmp(KIT_CARD, SERVED_CARD, shallow=False), (
        "The /mcp link preview differs from the brand kit's card. Copy "
        f"{KIT_CARD.relative_to(ROOT)} over {SERVED_CARD.relative_to(ROOT)}."
    )


def test_the_mcp_page_points_at_the_served_card() -> None:
    page = (
        ROOT / "api" / "frontend" / "src" / "app" / "(marketing)" / "mcp" / "page.tsx"
    ).read_text(encoding="utf-8")
    assets = (ROOT / "api" / "frontend" / "src" / "lib" / "brand-assets.ts").read_text(
        encoding="utf-8"
    )
    assert "SITE_ASSETS.ogMcp1200x630" in page
    assert "/og-mcp-1200x630.png" in assets
