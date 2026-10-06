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

import functools
import pathlib
import re

from tests._source_tree import iter_source_files, read_source

ROOT = pathlib.Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
SRC = FRONTEND / "src"
PUBLIC = FRONTEND / "public"

#: What `frontend/src` is written in. Each suffix is one walk of the repository,
#: `node_modules` included, so this is the list that exists rather than every
#: one that could.
SOURCE_SUFFIXES = ("*.ts", "*.tsx", "*.css")
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


@functools.cache
def _references() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for suffix in SOURCE_SUFFIXES:
        # `include_prefixes` beats every exclude, so tests are dropped here.
        for f, rel in iter_source_files(suffix, include_prefixes=("frontend/src/",)):
            if not rel.startswith("frontend/src/") or rel.startswith("frontend/src/__tests__/"):
                continue
            for match in ASSET.finditer(read_source(f)):
                path = match.group(1)
                if not path.startswith("//"):
                    found.setdefault(path, []).append(rel)
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
