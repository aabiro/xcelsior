"""One iterator over the repository's source files, for every static gate.

Eight test modules walk the tree looking for `*.py`, each with its own exclusion
list. On 2026-08-04 four of them failed simultaneously with

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 27

because macOS had written AppleDouble sidecars (`._foo.py`, binary resource forks)
into the tree over a network share. Three were then fixed with a per-file
`._` skip, which left five to fail the next time — and the failure names neither
the sidecar nor the gate's actual subject, so it reads as the gate being broken.

The exclusions belong in one place. A gate that has to remember to skip junk is a
gate that will forget.

Deliberately not clever: no caching, no globbing config, no plugin. Callers pass
their own extra prefixes if they need them, and everything else is shared.
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterator

REPO = pathlib.Path(__file__).resolve().parent.parent

#: Directories that are never repository source. `mcp/` is TypeScript with its own
#: vitest suite; `tests/` is excluded by most callers because a guard scanning its
#: own assertions finds the pattern it forbids — the recursion that has flagged
#: prose about a defect five times in this suite.
DEFAULT_EXCLUDED_PREFIXES = (
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    "mcp/",
    "desktop/",
    "frontend/",
    "__pycache__/",
    "build/",
    "dist/",
)


def is_source_file(path: pathlib.Path) -> bool:
    """False for anything that is not repository source, whatever its suffix.

    The AppleDouble check is by *basename*, not by content: a `._` file is junk
    regardless of whether it happens to decode. Sniffing for valid UTF-8 instead
    would also swallow a genuinely corrupt source file, which is a defect a gate
    should surface rather than skip.
    """
    return not path.name.startswith("._")


def iter_source_files(
    suffix: str = "*.py",
    *,
    exclude_prefixes: tuple[str, ...] = (),
    include_tests: bool = False,
) -> Iterator[tuple[pathlib.Path, str]]:
    """Yield `(path, repo_relative_posix)` for every source file to be scanned."""
    excluded = DEFAULT_EXCLUDED_PREFIXES + tuple(exclude_prefixes)
    if not include_tests:
        excluded += ("tests/",)
    for path in REPO.rglob(suffix):
        rel = path.relative_to(REPO).as_posix()
        if rel.startswith(excluded):
            continue
        if not is_source_file(path):
            continue
        yield path, rel


def read_source(path: pathlib.Path) -> str:
    """Read a source file, failing loudly rather than silently skipping.

    A gate that swallows a decode error stops covering whatever it could not read.
    `iter_source_files` has already removed the known-binary cases, so anything
    that fails here is a finding.
    """
    return path.read_text(encoding="utf-8")
