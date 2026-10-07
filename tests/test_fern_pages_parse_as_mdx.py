"""Fern pages are MDX, and MDX has no HTML comments.

`fern/pages/compliance.mdx` carried `<!-- ... -->`. `fern check` passed it and
the publish then refused the whole site ("Unexpected character `!`"), so docs
could not be republished when docs.xcelsior.ca went dark on 2026-10-06. A
comment in MDX is `{/* ... */}`.
"""

from __future__ import annotations

from tests._source_tree import iter_source_files, read_source


def _pages():
    return [(f, rel) for f, rel in iter_source_files("*.mdx", include_prefixes=("fern/",)) if rel.startswith("fern/")]


def test_there_are_pages_to_check():
    assert len(_pages()) >= 5, "found almost no Fern pages; the scan is pointed at the wrong place"


def test_no_page_uses_an_html_comment():
    offenders = []
    for f, rel in _pages():
        for n, line in enumerate(read_source(f).splitlines(), 1):
            if "<!--" in line:
                offenders.append(f"{rel}:{n}")
    assert not offenders, f"HTML comments break the Fern publish; use {{/* */}}: {offenders}"
