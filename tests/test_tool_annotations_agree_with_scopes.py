"""A published annotation must not contradict the scope the tool requires.

`mcp-tool-surface-synthesis.md` §5.2 names five files that must agree and are
maintained by hand: contracts, scopes, annotations, descriptions and
`tool-surface.json`. Only one pair of them is asserted today —
`tests/test_tool_scope_registry_completeness.py` compares `TOOL_SCOPES` against
the published manifest. The `READ_ONLY` and `DESTRUCTIVE` sets in
`mcp/src/tools/contracts.ts` and every entry in `descriptions.ts` are checked by
nothing.

## Why the read-only pair is the one that matters

`readOnlyHint` is not documentation. A model reads it to decide whether a call
needs care, and a directory reviewer reads it to decide whether to trust the
rest of what we publish. A tool annotated read-only whose `TOOL_SCOPES` entry
demands `instances:write` is telling the model the call is safe **and** telling
the authorization layer it mutates. One of those is wrong, and the model is the
one that acts on it.

Nothing prevented that combination before this file.

## Why this is the precursor to S1 rather than S1 itself

§5.2's fix is to invert the direction — make the registry the source and
generate the annotations, so drift becomes unrepresentable rather than merely
detectable. That is the right end state and it is a cross-cutting change to the
MCP package. **Asserting the agreement first is what makes that inversion safe
to attempt**: without these tests, the refactor's own correctness would rest on
reading five files carefully, which is the thing that failed in the first place.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCOPES_TS = ROOT / "mcp" / "src" / "auth" / "scopes.ts"
CONTRACTS_TS = ROOT / "mcp" / "src" / "tools" / "contracts.ts"
DESCRIPTIONS_TS = ROOT / "mcp" / "src" / "tools" / "descriptions.ts"

#: Scope suffixes that mutate. `read` is the only one that does not, and
#: `connect` is deliberately here: issuing connection material is a write in
#: every sense that matters, even though nothing is "modified".
MUTATING_SUFFIXES = ("write", "operate", "evict", "connect", "approve")


def _tool_scopes() -> dict[str, str]:
    """tool name -> raw requirement text, from `TOOL_SCOPES`."""
    text = SCOPES_TS.read_text(encoding="utf-8")
    block = text.split("export const TOOL_SCOPES", 1)[1].split("\n};", 1)[0]
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^\s{2}(\w+):\s*(\{[^}]*\})", block, re.MULTILINE)
    }


def _named_set(name: str) -> set[str]:
    """The tool names inside a `const NAME = new Set([...])` in contracts.ts."""
    text = CONTRACTS_TS.read_text(encoding="utf-8")
    match = re.search(rf"const {name} = new Set\(\[(.*?)\]\)", text, re.S)
    assert match, f"contracts.ts no longer declares {name}"
    # Strip comments before harvesting names, or a tool mentioned in the
    # reasoning above the set reads as a member of it — the match-a-mention
    # defect this suite has caught repeatedly.
    body = re.sub(r"//[^\n]*", "", match.group(1))
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    return set(re.findall(r'"([a-z_][a-z0-9_]*)"', body))


def _described() -> set[str]:
    text = DESCRIPTIONS_TS.read_text(encoding="utf-8")
    block = text.split("=", 1)[1]
    return set(re.findall(r"^\s{2}([a-z_][a-z0-9_]*):", block, re.MULTILINE))


def _is_mutating(requirement: str) -> bool:
    scopes = re.findall(r'"([a-z_]+):([a-z_]+)"', requirement)
    return any(suffix in MUTATING_SUFFIXES for _domain, suffix in scopes)


# ── Calibration ───────────────────────────────────────────────────────


def test_all_four_sources_parse():
    """Four empty sets agree with each other perfectly. This stops that."""
    assert len(_tool_scopes()) > 40, "TOOL_SCOPES did not parse"
    assert len(_named_set("READ_ONLY")) > 20, "READ_ONLY did not parse"
    assert len(_named_set("DESTRUCTIVE")) >= 3, "DESTRUCTIVE did not parse"
    assert len(_described()) > 40, "descriptions did not parse"


# ── The contradiction that matters ────────────────────────────────────


def test_no_read_only_tool_requires_a_mutating_scope():
    """`readOnlyHint` is what a model reads to decide a call is safe.

    A tool annotated read-only whose scope demands `instances:write` tells the
    model one thing and the authorization layer another. The model is the one
    that acts on it.
    """
    scopes = _tool_scopes()
    offenders = []
    for tool in sorted(_named_set("READ_ONLY")):
        requirement = scopes.get(tool)
        if requirement is None:
            continue  # covered by the registration test below
        if _is_mutating(requirement):
            offenders.append(f"{tool} (requires {requirement.strip()})")
    assert not offenders, (
        "these tools are published as read-only but require a mutating scope: "
        + "; ".join(offenders)
        + ". The annotation tells a model the call is safe; the scope says it "
        "is not."
    )


def test_no_tool_is_both_read_only_and_destructive():
    """A direct contradiction, and nothing forbade it."""
    both = sorted(_named_set("READ_ONLY") & _named_set("DESTRUCTIVE"))
    assert not both, f"annotated both read-only and destructive: {both}"


def test_every_destructive_tool_requires_a_mutating_scope():
    """The other direction. A destructive tool behind a read scope is either a
    mis-annotation or an authorization hole, and both are worth failing on."""
    scopes = _tool_scopes()
    offenders = [
        f"{tool} (requires {scopes[tool].strip()})"
        for tool in sorted(_named_set("DESTRUCTIVE"))
        if tool in scopes and not _is_mutating(scopes[tool])
    ]
    assert not offenders, (
        "these tools are published as destructive but require only a read "
        "scope: " + "; ".join(offenders)
    )


# ── Names that refer to nothing ───────────────────────────────────────


@pytest.mark.parametrize("set_name", ["READ_ONLY", "DESTRUCTIVE"])
def test_every_annotated_name_is_a_registered_tool(set_name: str):
    """A typo here annotates nothing and fails silently.

    `terminate_instnace` in `DESTRUCTIVE` leaves `terminate_instance`
    unflagged — published as neither destructive nor read-only, with no error
    anywhere.
    """
    scopes = _tool_scopes()
    unknown = sorted(_named_set(set_name) - set(scopes))
    assert not unknown, (
        f"{set_name} names tools that are not in TOOL_SCOPES: {unknown}. A "
        "misspelled entry annotates nothing and reports no error."
    )


# ── Descriptions ──────────────────────────────────────────────────────


def test_every_registered_tool_has_a_description():
    """A directory reviewer calls every tool and compares behaviour to the
    description. A tool with none is one they cannot review."""
    missing = sorted(set(_tool_scopes()) - _described())
    assert not missing, f"registered tools with no description: {missing}"


def test_no_description_names_a_tool_that_does_not_exist():
    """A stale description is a promise about a tool nobody can call."""
    orphans = sorted(_described() - set(_tool_scopes()))
    assert not orphans, (
        f"descriptions exist for tools not in TOOL_SCOPES: {orphans}. Either "
        "the tool was removed and its description was not, or the registry "
        "entry was lost."
    )
