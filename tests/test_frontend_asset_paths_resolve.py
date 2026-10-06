"""Every static asset path the frontend names must exist under `public/`.

Commit 034a4f1 ("archive old assets") moved files from `frontend/public/` into
`frontend/public/_archive/`. Four of them were still in use:

* `canada-map-arc.svg` and `canada-map-arc-light.svg`, the map in the hero on
  `/dashboard`, the first page every signed-in user sees. Production served a
  404 for both, and the hero rendered as an empty box.
* `desktop-dashboard-screenshot.svg` and `desktop-control-center-screenshot.svg`,
  the PWA manifest's screenshots, which install prompts then failed to load.

Nothing failed at build time. `<img src>` and manifest entries are strings, so
the only signal was a browser's network log, which is how the signed-in
Playwright run against production found them.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
PUBLIC = FRONTEND / "public"

SOURCE_SUFFIXES = {".ts", ".tsx", ".js", ".jsx", ".css", ".mdx", ".md"}
ASSET = re.compile(
    r"""["'`(]"""
    r"""(/[A-Za-z0-9_\-./]+\."""
    r"""(?:svg|png|jpe?g|webp|gif|ico|avif|mp4|webm|woff2?|ttf|otf|pdf|json|txt|xml))"""
    r"""(?:[?#][^"'`)]*)?["'`)]"""
)


def _served_by_app_router(path: str) -> bool:
    """`/sitemap.xml` from `app/sitemap.ts`, or a `route.ts` at that path."""
    rel = path.lstrip("/")
    stem = rel.rsplit(".", 1)[0]
    app = SRC / "app"
    return any((app / f"{stem}{ext}").exists() for ext in (".ts", ".tsx")) or any(
        (app / rel / f"route{ext}").exists() for ext in (".ts", ".tsx")
    )


def _references() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for f in SRC.rglob("*"):
        if f.suffix not in SOURCE_SUFFIXES or "__tests__" in f.parts:
            continue
        for match in ASSET.finditer(f.read_text(errors="ignore")):
            path = match.group(1)
            if path.startswith("//"):
                continue
            found.setdefault(path, []).append(str(f.relative_to(ROOT)))
    return found


def test_the_scan_sees_the_assets_it_is_meant_to_check():
    """A pattern that matched nothing would pass every tree."""
    refs = _references()
    assert "/canada-map-arc.svg" in refs, "the dashboard hero's map is no longer seen by the scan"
    assert len(refs) >= 10, f"only {len(refs)} asset paths found; the pattern has gone blind"


def test_every_referenced_asset_exists():
    missing = {
        path: sorted(set(files))
        for path, files in _references().items()
        if not (PUBLIC / path.lstrip("/")).is_file() and not _served_by_app_router(path)
    }
    archived = {p for p in missing if (PUBLIC / "_archive" / p.lstrip("/")).is_file()}
    assert not missing, (
        "frontend source names assets that `public/` does not have; each is a 404 "
        f"in production: {missing}"
        + (f"\nThese were moved to public/_archive while still in use: {sorted(archived)}" if archived else "")
    )
