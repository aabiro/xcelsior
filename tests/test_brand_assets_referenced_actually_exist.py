"""Every brand asset we send to a user must exist on disk.

`routes/teams.py` mailed a team invitation with
`https://xcelsior.ca/xcelsior-logo-rounded.svg`, and `routes/health.py` rendered
the status page with `xcelsior-logo-wordmark-iconbg.svg`. **Neither file exists.**
They belong to a brand generation that was replaced by `site-assets/` — a
different mark entirely, red and blue with a maple leaf, rather than the cyan →
purple gradient the product ships now.

So the invite email had a broken image in it, and had for as long as the assets
had been gone. Nothing failed: an email is fire-and-forget, and a missing `<img>`
renders as a grey box in someone else's client. `health.py` even has an
`onerror` handler that hides the logo and shows a text title, so the page
degraded silently and looked deliberate.

## What this checks

Absolute `https://xcelsior.ca/...` asset URLs in server-rendered HTML resolve to
a file under `frontend/public/`. That is the whole property: a reference we mail
out must be a file we actually serve.

It cannot check that the asset is the *right* one — `lockup-light.png` on a
white background is invisible and this would pass. The filenames are a trap
worth knowing: **"light" means light-coloured text**, for dark backgrounds, so
`lockup-light.png` belongs on the dark status page and `lockup-dark.png` on
Stripe's white one.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "frontend" / "public"

#: Absolute asset URLs on the production origin, as embedded in server HTML.
ASSET_URL = re.compile(r'https://xcelsior\.ca/([A-Za-z0-9._/-]+\.(?:svg|png|ico|jpg|jpeg))')


def _referenced() -> dict[str, list[str]]:
    """`asset path -> [source files referencing it]`."""
    found: dict[str, list[str]] = {}
    for path in sorted(ROOT.glob("routes/*.py")) + [ROOT / "api.py"]:
        if not path.is_file():
            continue
        for asset in ASSET_URL.findall(path.read_text(encoding="utf-8")):
            found.setdefault(asset, []).append(path.name)
    return found


def test_the_scan_finds_the_references_at_all():
    """Calibration. An empty result satisfies the assertion below for free, and
    these URLs live in HTML string literals that are easy to reformat."""
    referenced = _referenced()
    assert referenced, "no absolute brand asset URLs found; the extractor is broken"


def test_every_referenced_asset_exists():
    missing = {
        asset: sources
        for asset, sources in _referenced().items()
        if not (PUBLIC / asset).is_file()
    }
    assert not missing, (
        "these assets are referenced in server-rendered HTML and do not exist "
        "under frontend/public/, so users receive a broken image: "
        + "; ".join(f"{a} (from {', '.join(s)})" for a, s in sorted(missing.items()))
    )


def test_the_retired_brand_generation_is_not_referenced():
    """The stale marks specifically.

    `frontend/public/xcelsior-*.png` is the previous identity and several of its
    files are already deleted. Naming it keeps a copy-paste from an old template
    from quietly reintroducing the wrong logo — which would render, because some
    of those files do still exist.
    """
    offenders = []
    for asset, sources in _referenced().items():
        if pathlib.PurePath(asset).name.startswith("xcelsior-logo") or pathlib.PurePath(
            asset
        ).name.startswith("xcelsior-icon"):
            offenders.append(f"{asset} (from {', '.join(sources)})")
    assert not offenders, (
        "these reference the retired brand generation rather than site-assets/: "
        + "; ".join(sorted(offenders))
    )
