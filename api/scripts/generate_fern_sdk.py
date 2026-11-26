#!/usr/bin/env python3
"""Regenerate the TypeScript SDK in `frontend/sdk` from the published spec.

`fern/generators.yml` points its `local` group at `../frontend/sdk`, but
nothing ran it. The checked-in SDK had gone untouched since it was copied in
during the October package extraction, and it had been built from the **full**
internal API: 210 operations, 131 of them on no public surface (MFA removal,
account deletion, host deletion, team administration), and missing three
operations the published spec does list. No check compared the two, so nobody
noticed.

This script is the one way to write that directory. It runs the generator in a
scratch copy of the Fern workspace, replaces `frontend/sdk` with the output,
and writes `frontend/sdk/FERN_GENERATION.json`: a digest of every input the
generator reads and of the tree it produced. `tests/test_fern_sdk_is_current.py`
recomputes both without Docker, so a spec change without a regeneration, or a
hand edit to the SDK, fails the suite.

    python api/scripts/generate_fern_sdk.py           # regenerate and stamp
    python api/scripts/generate_fern_sdk.py --check   # regenerate to a scratch
                                                      # dir; fail on any difference

Needs the `fern` CLI and Docker. The CLI follows the version in
`fern/fern.config.json`, and the generator image is pinned in
`fern/generators.yml`; two runs produce byte-identical trees, which is what
makes the digest meaningful. Regenerate `fern/openapi.json` first
(`scripts/generate_public_openapi.py`): the SDK is a projection of that file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FERN_DIR = ROOT / "fern"
SDK_DIR = ROOT / "frontend" / "sdk"
STAMP_NAME = "FERN_GENERATION.json"
STAMP_PATH = SDK_DIR / STAMP_NAME
GROUP = "local"

#: Every file the `local` generator group reads. `docs.yml` and the pages feed
#: the docs site, not the SDK, so editing them must not demand a regeneration.
INPUTS = (
    "fern/fern.config.json",
    "fern/generators.yml",
    "fern/openapi.json",
    "fern/openapi-overrides.yml",
)

#: Editor and OS litter that is not part of the generated tree. AppleDouble
#: sidecars (`._x`) broke four gates at once on macOS; see
#: `tests/test_source_tree_is_shared.py`.
_IGNORED_NAMES = {".DS_Store"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def input_digests() -> dict[str, str]:
    """`{"fern/openapi.json": "<sha256>", …}` for the files the generator reads."""
    return {rel: _sha256((ROOT / rel).read_bytes()) for rel in INPUTS}


def tree_files(root: Path) -> list[str]:
    """Relative POSIX paths of the generated files, sorted, stamp excluded."""
    out = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == STAMP_NAME or path.name in _IGNORED_NAMES or path.name.startswith("._"):
            continue
        out.append(rel)
    return sorted(out)


def tree_digest(root: Path) -> str:
    """One digest over every file's path and bytes, so a rename counts too."""
    lines = [f"{rel}\0{_sha256((root / rel).read_bytes())}" for rel in tree_files(root)]
    return _sha256("\n".join(lines).encode("utf-8"))


def pinned_versions() -> dict[str, str]:
    """The fern CLI version and the generator image the `local` group pins."""
    config = json.loads((FERN_DIR / "fern.config.json").read_text(encoding="utf-8"))
    generators = yaml.safe_load((FERN_DIR / "generators.yml").read_text(encoding="utf-8"))
    (generator,) = generators["groups"][GROUP]["generators"]
    return {
        "fern_cli": str(config["version"]),
        "generator": f"fernapi/{generator['name']}:{generator['version']}",
    }


def build_stamp(sdk_root: Path) -> dict:
    return {
        "note": (
            "Written by api/scripts/generate_fern_sdk.py. Do not edit by hand; "
            "tests/test_fern_sdk_is_current.py checks it."
        ),
        **pinned_versions(),
        "inputs": input_digests(),
        "files": len(tree_files(sdk_root)),
        "output_sha256": tree_digest(sdk_root),
    }


def _require(tool: str) -> str:
    found = shutil.which(tool)
    if not found:
        sys.exit(f"`{tool}` is not on PATH; the Fern SDK generator needs it.")
    return found


def generate_into(workdir: Path) -> Path:
    """Run the `local` group against a scratch copy of the workspace.

    A copy rather than the real `fern/` because the output path is rewritten to
    land inside `workdir`: `--check` must never touch the checked-in SDK, and
    the write path only replaces it once generation has succeeded.
    """
    fern = _require("fern")
    _require("docker")
    workspace = workdir / "fern"
    shutil.copytree(FERN_DIR, workspace)
    generators_path = workspace / "generators.yml"
    generators = yaml.safe_load(generators_path.read_text(encoding="utf-8"))
    (generator,) = generators["groups"][GROUP]["generators"]
    generator["output"] = {"location": "local-file-system", "path": "../sdk"}
    generators_path.write_text(yaml.safe_dump(generators, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [fern, "generate", "--local", "--group", GROUP, "--no-prompt"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = workdir / "sdk"
    if result.returncode != 0 or not (output / "index.ts").exists():
        sys.exit(
            f"fern generate failed (rc={result.returncode}):\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return output


def _write(generated: Path) -> None:
    if SDK_DIR.exists():
        shutil.rmtree(SDK_DIR)
    shutil.copytree(generated, SDK_DIR)
    stamp = build_stamp(SDK_DIR)
    STAMP_PATH.write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SDK_DIR.relative_to(ROOT)}: {stamp['files']} files, {stamp['generator']}")


def _check(generated: Path) -> int:
    fresh, checked_in = set(tree_files(generated)), set(tree_files(SDK_DIR))
    changed = sorted(
        rel
        for rel in fresh & checked_in
        if (generated / rel).read_bytes() != (SDK_DIR / rel).read_bytes()
    )
    problems = [f"only in a fresh generation: {rel}" for rel in sorted(fresh - checked_in)]
    problems += [f"only in the checked-in SDK: {rel}" for rel in sorted(checked_in - fresh)]
    problems += [f"differs: {rel}" for rel in changed]
    if not problems:
        recorded = (
            json.loads(STAMP_PATH.read_text(encoding="utf-8")) if STAMP_PATH.exists() else None
        )
        if recorded != build_stamp(SDK_DIR):
            problems.append(f"{STAMP_NAME} does not match the inputs and tree")
    if problems:
        print(f"{SDK_DIR.relative_to(ROOT)} is stale; run api/scripts/generate_fern_sdk.py")
        for line in problems[:50]:
            print(f"  {line}")
        if len(problems) > 50:
            print(f"  … and {len(problems) - 50} more")
        return 1
    print(f"{SDK_DIR.relative_to(ROOT)} matches a fresh generation.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="generate to a scratch directory and compare; change nothing",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="fern-sdk-") as tmp:
        generated = generate_into(Path(tmp))
        if args.check:
            return _check(generated)
        _write(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())
