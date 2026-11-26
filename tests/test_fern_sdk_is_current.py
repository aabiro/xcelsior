"""The checked-in TypeScript SDK must be a generation of the published spec.

`api/frontend/sdk` is what `fern/generators.yml` writes, and nothing checked it.
It sat untouched for months while the spec moved: by 2026-11-25 it was a build
of the *full internal* API, 210 operations with 131 on no public surface, and
it lacked three operations the published spec lists. The public-spec gate
(`test_public_openapi.py`) stops at `fern/openapi.json`; this carries the same
guarantee one step further, to the code built from it.

Regenerating takes Docker and the fern CLI, so these checks do not regenerate.
`api/scripts/generate_fern_sdk.py` records a digest of the generator's inputs
and of the tree it wrote; here both are recomputed, which catches the two ways
the SDK goes stale: the spec changes and nobody regenerates, or someone edits
the generated code by hand. `generate_fern_sdk.py --check` is the full
regeneration comparison, for when Docker is available.

The operation check reads the SDK itself, not the stamp, so a stamp copied
forward by hand still cannot hide an SDK that calls a different surface.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml
from scripts.generate_fern_sdk import (
    INPUTS,
    SDK_DIR,
    STAMP_NAME,
    STAMP_PATH,
    input_digests,
    pinned_versions,
    tree_digest,
    tree_files,
)

from tests._source_tree import iter_source_files, read_source

ROOT = Path(__file__).resolve().parent.parent
FERN_DIR = ROOT / "api" / "fern"
RESOURCES = "api/frontend/sdk/api/resources/"
REGENERATE = "run api/scripts/generate_fern_sdk.py"

#: Every generated endpoint method ends in the same error helper, which is
#: handed the verb and the path template exactly as the spec spells them.
SDK_OPERATION = re.compile(
    r"handleNonStatusCodeError\(\s*_response\.error,\s*_response\.rawResponse,\s*"
    r'"([A-Z]+)",\s*"([^"]+)"',
    re.S,
)
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _stamp() -> dict:
    assert STAMP_PATH.exists(), f"{STAMP_PATH} is missing; {REGENERATE}"
    return json.loads(STAMP_PATH.read_text(encoding="utf-8"))


def _sdk_operations() -> set[tuple[str, str]]:
    # Through the shared iterator, which skips macOS AppleDouble sidecars; see
    # `tests/test_source_tree_is_shared.py`.
    found: set[tuple[str, str]] = set()
    for client, rel in iter_source_files("Client.ts", include_prefixes=(RESOURCES,)):
        if rel.startswith(RESOURCES):
            found |= set(SDK_OPERATION.findall(read_source(client)))
    return found


def _spec_operations() -> set[tuple[str, str]]:
    spec = json.loads((FERN_DIR / "openapi.json").read_text(encoding="utf-8"))
    return {
        (method.upper(), path)
        for path, methods in spec["paths"].items()
        for method in methods
        if method in HTTP_METHODS
    }


def test_the_stamp_names_every_input_the_generator_reads():
    """Prove the reach: a stamp over no inputs would match every spec."""
    stamp = _stamp()
    assert set(stamp["inputs"]) == set(INPUTS)
    assert stamp["files"] > 50, f"only {stamp['files']} files; the SDK tree is mostly gone"
    assert "fern/openapi.json" in INPUTS


def test_the_sdk_was_generated_from_the_current_spec():
    """The gate for "the spec changed and nobody regenerated"."""
    recorded = _stamp()["inputs"]
    stale = sorted(rel for rel, digest in input_digests().items() if recorded.get(rel) != digest)
    assert (
        not stale
    ), f"{', '.join(stale)} changed since api/frontend/sdk was generated; {REGENERATE}"


def test_the_sdk_matches_what_the_generator_wrote():
    """The gate for a hand edit, or a partial copy, of generated code."""
    stamp = _stamp()
    assert tree_digest(SDK_DIR) == stamp["output_sha256"], (
        "api/frontend/sdk differs from the tree recorded in "
        f"{STAMP_NAME}: generated code was edited by hand or copied in part; {REGENERATE}"
    )
    # Inputs are the test above. Comparing the whole stamp here would report a
    # spec change as a hand edit to the SDK, which sends the reader the wrong way.
    assert stamp["files"] == len(tree_files(SDK_DIR)), f"{STAMP_NAME} miscounts the tree"


def test_the_stamp_records_the_pinned_generator():
    stamp = _stamp()
    assert {key: stamp[key] for key in ("fern_cli", "generator")} == pinned_versions()


def test_the_sdk_calls_exactly_the_published_operations():
    """Read from the SDK's own code, so it cannot be satisfied by the stamp alone."""
    sdk, spec = _sdk_operations(), _spec_operations()
    assert len(sdk) > 50, f"only {len(sdk)} SDK operations parsed; SDK_OPERATION is wrong"
    assert not sdk - spec, f"the SDK calls operations the public spec omits: {sorted(sdk - spec)}"
    assert not spec - sdk, f"the SDK lacks published operations: {sorted(spec - sdk)}"


def test_the_published_package_uses_the_same_generator_as_the_checked_in_sdk():
    """`publish` ships `@xcelsior-gpu/sdk` to npm; `local` writes the copy here.

    Two pins of one version: if they part, the package on npm stops being the
    code these checks cover.
    """
    groups = yaml.safe_load((FERN_DIR / "generators.yml").read_text(encoding="utf-8"))["groups"]
    pins = {
        name: {(g["name"], g["version"]) for g in group["generators"]}
        for name, group in groups.items()
    }
    assert pins["publish"] == pins["local"], pins
