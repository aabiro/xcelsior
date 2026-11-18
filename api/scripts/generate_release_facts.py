#!/usr/bin/env python3
"""Write the copies of release facts that cannot read them at run time.

``core/release_facts.py`` reads each value from the file that decides it. Two
consumers cannot call it, so they get a written copy:

- the frontend: the Next.js build context is ``api/frontend`` alone, so it reads
  ``api/frontend/src/lib/release-facts.json``;
- ``api/mcp/server.json``: the MCP registry reads it as published, so the
  server's version and the npm package's version are literals in it.

Run it after changing a source (a package version, install.sh's wizard pin, or
``npm run surface:update``)::

    python3 api/scripts/generate_release_facts.py

``--check`` writes nothing and exits 1 when a copy is stale.
``tests/test_release_facts.py`` runs the check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))

import release_facts as facts  # noqa: E402

FRONTEND_FACTS = ROOT / "api" / "frontend" / "src" / "lib" / "release-facts.json"
REGISTRY_ENTRY = ROOT / "api" / "mcp" / "server.json"


def _json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def frontend_facts() -> str:
    return _json(facts.frontend_facts())


def registry_entry(current: str) -> str:
    entry = json.loads(current)
    entry["version"] = facts.package_version(facts.MCP_SERVER_PACKAGE)
    cli = json.loads(facts.MCP_CLI_PACKAGE.read_text(encoding="utf-8"))
    matched = [p for p in entry.get("packages", []) if p.get("identifier") == cli["name"]]
    if len(matched) != 1:
        raise SystemExit(
            f"{REGISTRY_ENTRY}: expected one {cli['name']} package, found {len(matched)}"
        )
    matched[0]["version"] = cli["version"]
    return _json(entry)


def outputs() -> dict[Path, str]:
    """Every written copy, as this script would write it now."""
    return {
        FRONTEND_FACTS: frontend_facts(),
        REGISTRY_ENTRY: registry_entry(REGISTRY_ENTRY.read_text(encoding="utf-8")),
    }


def stale() -> list[Path]:
    return [
        path
        for path, text in outputs().items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]


def main(argv: list[str]) -> int:
    if argv == ["--check"]:
        behind = stale()
        for path in behind:
            print(f"stale: {path.relative_to(ROOT)}", file=sys.stderr)
        return 1 if behind else 0
    if argv:
        print("usage: generate_release_facts.py [--check]", file=sys.stderr)
        return 2
    for path, text in outputs().items():
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
