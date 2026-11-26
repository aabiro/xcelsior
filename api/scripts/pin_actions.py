#!/usr/bin/env python3
"""Pin every GitHub Action the workflows use to a full commit SHA.

A tag such as ``actions/checkout@v4`` is a pointer its owner can move. Whoever
controls the action's repository (or steals a maintainer's token, as happened to
``tj-actions/changed-files`` in March 2025) can repoint it, and every workflow
that names the tag runs the new code on its next run. Two of this repository's
workflows run on a self-hosted runner on our own machine, so that code would run
there, not on a throwaway GitHub VM.

A commit SHA cannot be repointed. This script rewrites

    uses: actions/checkout@v4
to
    uses: actions/checkout@<40-hex sha> # v4

keeping the tag as a comment so a person can still read the version, and so a
later run can refresh the pin to wherever that tag points now.

    python3 api/scripts/pin_actions.py            # pin anything not yet pinned
    python3 api/scripts/pin_actions.py --refresh  # re-resolve every comment tag
    python3 api/scripts/pin_actions.py --check    # exit 1 if anything floats

Resolving a tag reads the GitHub API through the ``gh`` CLI; ``--check`` needs no
network and is what ``tests/test_workflow_actions_are_pinned.py`` runs.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"

#: ``uses: owner/repo[/path]@ref  # comment``
USES = re.compile(
    r"^(?P<lead>\s*(?:-\s+)?uses:\s*)(?P<action>[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)@(?P<ref>[^\s#]+)"
    r"(?:\s+#\s*(?P<comment>\S+))?\s*$"
)
SHA = re.compile(r"^[0-9a-f]{40}$")

#: Actions that read their own ref as an input. ``dtolnay/rust-toolchain@stable``
#: installs the stable toolchain because the ref is called ``stable``; pinned to
#: a SHA it needs ``toolchain: stable`` spelled out, or it fails to start.
REF_IS_INPUT = {"dtolnay/rust-toolchain": "toolchain"}


def workflow_files() -> list[Path]:
    return sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])


def unpinned(paths: list[Path] | None = None) -> list[str]:
    """Every remote action reference that is not a commit SHA with its tag beside it."""
    problems = []
    for path in paths or workflow_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = USES.match(line)
            if match is None:
                if re.match(r"^\s*(?:-\s+)?uses:\s*[^.\s]", line) and "docker://" not in line:
                    problems.append(
                        f"{path.name}:{number}: cannot read this uses: line: {line.strip()}"
                    )
                continue
            if not SHA.match(match.group("ref")):
                problems.append(
                    f"{path.name}:{number}: {match.group('action')}@{match.group('ref')} is not pinned"
                )
            elif not match.group("comment"):
                problems.append(
                    f"{path.name}:{number}: {match.group('action')} is pinned but has no '# <tag>' comment"
                )
    return problems


def resolve(action: str, ref: str, cache: dict[tuple[str, str], str]) -> str:
    repo = "/".join(action.split("/")[:2])
    key = (repo, ref)
    if key not in cache:
        result = subprocess.run(
            ["gh", "api", f"repos/{repo}/commits/{ref}", "--jq", ".sha"],
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        if result.returncode != 0 or not SHA.match(sha):
            raise SystemExit(f"could not resolve {repo}@{ref}: {result.stderr.strip() or sha}")
        cache[key] = sha
    return cache[key]


def _needs_input(lines: list[str], index: int, action: str) -> str | None:
    """The ``with:`` input a REF_IS_INPUT action needs once its ref is a SHA, if missing."""
    name = REF_IS_INPUT.get(action)
    if name is None:
        return None
    indent = len(lines[index]) - len(lines[index].lstrip(" -"))
    for following in lines[index + 1 :]:
        stripped = following.strip()
        if not stripped:
            continue
        if len(following) - len(following.lstrip(" ")) < indent or stripped.startswith("- "):
            break
        if stripped.startswith(f"{name}:"):
            return None
    return name


def pin(path: Path, refresh: bool, cache: dict[tuple[str, str], str]) -> int:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        match = USES.match(line.rstrip("\n"))
        if match is None:
            index += 1
            continue
        action, ref, comment = match.group("action"), match.group("ref"), match.group("comment")
        if SHA.match(ref) and not refresh:
            index += 1
            continue
        tag = comment if SHA.match(ref) else ref
        if not tag:
            raise SystemExit(
                f"{path.name}:{index + 1}: {action} is pinned with no tag comment to refresh from"
            )
        sha = resolve(action, tag, cache)
        new = f"{match.group('lead')}{action}@{sha} # {tag}\n"
        missing = _needs_input([text.rstrip("\n") for text in lines], index, action)
        if new != line:
            lines[index] = new
            changed += 1
        if missing:
            body_indent = " " * (len(line) - len(line.lstrip(" -")) + 2)
            has_with = index + 1 < len(lines) and lines[index + 1].strip() == "with:"
            insert = (
                [f"{body_indent}{missing}: {tag}\n"]
                if has_with
                else [
                    f"{body_indent[:-2]}with:\n",
                    f"{body_indent}{missing}: {tag}\n",
                ]
            )
            lines[index + 1 + (1 if has_with else 0) : index + 1 + (1 if has_with else 0)] = insert
            changed += 1
        index += 1
    if changed:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="report unpinned actions; change nothing"
    )
    mode.add_argument(
        "--refresh", action="store_true", help="re-resolve pinned actions from their tag comments"
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = unpinned()
        for problem in problems:
            print(problem)
        return 1 if problems else 0
    cache: dict[tuple[str, str], str] = {}
    total = 0
    for path in workflow_files():
        count = pin(path, args.refresh, cache)
        if count:
            print(f"{path.relative_to(ROOT)}: {count} line(s) changed")
        total += count
    print(
        f"{total} line(s) changed across {len(workflow_files())} workflows; {len(cache)} action refs resolved"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
