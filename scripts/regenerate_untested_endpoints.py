#!/usr/bin/env python3
"""Regenerate UNTESTED_ENDPOINTS.md from routes/*.py and tests/ heuristic."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
ROUTES_DIR = PROJECT / "routes"
TESTS_DIR = PROJECT / "tests"
OUT = PROJECT / "UNTESTED_ENDPOINTS.md"

ROUTE_RE = re.compile(
    r'@router\.(get|post|put|patch|delete)\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
DEF_RE = re.compile(r"^async def (\w+)|^def (\w+)", re.MULTILINE)
CLI_CMD_RE = re.compile(r"^def (cmd_\w+)\(", re.MULTILINE)


def _load_test_corpus() -> str:
    parts: list[str] = []
    for path in sorted(TESTS_DIR.rglob("*.py")):
        try:
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return "\n".join(parts)


def _coverage_file_for(path: str, handler: str, corpus_by_file: dict[str, str]) -> str | None:
    path_base = path.split("{", 1)[0]
    for fname, text in corpus_by_file.items():
        if not fname.startswith("test_") or not fname.endswith(".py"):
            continue
        if handler in text or path in text or (path_base and path_base in text):
            if "coverage" in fname:
                return fname
    return None


def _is_tested(path: str, handler: str, corpus: str, corpus_by_file: dict[str, str]) -> tuple[bool, str]:
    generic = re.sub(r"\{[^}/]+\}", "{*}", path)
    path_base = path.split("{", 1)[0] if "{" in path else path
    if (
        path in corpus
        or generic in corpus
        or handler in corpus
        or (path_base and path_base in corpus)
        or path.replace("{", "").replace("}", "") in corpus
    ):
        cov = _coverage_file_for(path, handler, corpus_by_file)
        if cov:
            return True, cov
        return True, ""
    return False, ""


def _parse_routes() -> dict[str, list[tuple[str, str, str, bool, str]]]:
    corpus = _load_test_corpus()
    corpus_by_file = {
        p.name: p.read_text(encoding="utf-8", errors="replace")
        for p in TESTS_DIR.rglob("*.py")
    }
    by_file: dict[str, list[tuple[str, str, str, bool, str]]] = defaultdict(list)

    for route_file in sorted(ROUTES_DIR.glob("*.py")):
        if route_file.name.startswith("_"):
            continue
        text = route_file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        pending: list[tuple[str, str]] = []
        for i, line in enumerate(lines):
            m = ROUTE_RE.search(line)
            if m:
                pending.append((m.group(1).upper(), m.group(2)))

            dm = DEF_RE.match(line.strip())
            if dm and pending:
                handler = dm.group(1) or dm.group(2)
                method, path = pending.pop(0)
                tested, cov_file = _is_tested(path, handler, corpus, corpus_by_file)
                by_file[route_file.name].append((method, path, handler, tested, cov_file))

    return by_file


def _parse_cli_commands(corpus: str) -> list[tuple[str, str, bool]]:
    cli = PROJECT / "cli.py"
    if not cli.exists():
        return []
    text = cli.read_text(encoding="utf-8", errors="replace")
    commands: list[tuple[str, str, bool]] = []
    for m in CLI_CMD_RE.finditer(text):
        handler = m.group(1)
        name = handler[4:].replace("_", "-")  # cmd_host_add -> host-add
        tested = handler in corpus or name in corpus
        commands.append((name, handler, tested))
    return sorted(commands, key=lambda x: x[0])


def _render(by_file: dict[str, list], cli_commands: list[tuple[str, str, bool]]) -> str:
    total = sum(len(v) for v in by_file.values())
    untested_routes = sum(1 for entries in by_file.values() for e in entries if not e[3])
    untested_cli = sum(1 for c in cli_commands if not c[2])

    lines = [
        "# Untested endpoints — coverage-gap worklist",
        "",
        f"_Regenerated {date.today().isoformat()} by `scripts/regenerate_untested_endpoints.py`: "
        "a route/CLI command counts as covered when **either** its path (prefix before `{…}`) "
        "**or** its handler function name appears anywhere under `tests/`._",
        "",
        f"**{untested_routes} of {total} routes ({100 * untested_routes // max(total, 1)}%)** "
        f"and **{untested_cli} of {len(cli_commands)} CLI commands** "
        f"({100 * untested_cli // max(len(cli_commands), 1)}%) have no test signal.",
        "",
        "Workflow per item: write a `TestClient` (or CLI) test → if it works, tick the box; "
        "if it 500s/throws, fix-or-delete then tick. Caveat: a few may be exercised transitively; "
        "confirm with the test.",
        "",
        f"## Routes ({untested_routes} untested)",
        "",
    ]

    for fname in sorted(by_file.keys()):
        entries = by_file[fname]
        open_count = sum(1 for e in entries if not e[3])
        lines.append(f"### `routes/{fname}` ({open_count} untested)")
        for method, path, handler, tested, cov_file in sorted(entries, key=lambda x: x[1]):
            mark = "x" if tested else " "
            suffix = f"  ✓ {cov_file}" if tested and cov_file else ""
            lines.append(f"- [{mark}] `{method} {path}` — `{handler}`{suffix}")
        lines.append("")

    lines.append(f"## CLI commands ({untested_cli} untested)")
    lines.append("")
    for name, handler, tested in cli_commands:
        mark = "x" if tested else " "
        lines.append(f"- [{mark}] `{name}` — `{handler}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    corpus = _load_test_corpus()
    by_file = _parse_routes()
    cli_commands = _parse_cli_commands(corpus)
    OUT.write_text(_render(by_file, cli_commands), encoding="utf-8")
    total = sum(len(v) for v in by_file.values())
    untested = sum(1 for entries in by_file.values() for e in entries if not e[3])
    print(f"Wrote {OUT} — {untested}/{total} routes untested, {sum(1 for c in cli_commands if not c[2])}/{len(cli_commands)} CLI")


if __name__ == "__main__":
    main()