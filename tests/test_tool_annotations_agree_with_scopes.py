"""A published annotation must not contradict the scope the tool requires.

`mcp-tool-surface-synthesis.md` §5.2 named five files that must agree and were
maintained by hand: contracts, scopes, annotations, descriptions and
`tool-surface.json`.

## What S1 changed, and what it did not

The inversion landed. `TOOL_SCOPE_REGISTRY` in `scopes.ts` is the source, its
key set is exported as `ToolName`, and every other per-tool table is declared
`Record<ToolName, …>`. A tool registered with no policy row, no description, or
a row for a tool that does not exist, is now a **compile error** — verified by
injecting each of the three and watching `tsc` reject it. Completeness is no
longer something a test has to notice.

**Coherence still is.** The compiler counts rows; it cannot read them. Nothing
in the type system knows that `instances:write` mutates, so
`readOnly: true` beside a write scope type-checks perfectly. That contradiction
is what this file exists for, and S1 did not touch it.

## Why the read-only pair is the one that matters

`readOnlyHint` is not documentation. A model reads it to decide whether a call
needs care, and a directory reviewer reads it to decide whether to trust the
rest of what we publish. A tool annotated read-only whose scope entry demands
`instances:write` is telling the model the call is safe **and** telling the
authorization layer it mutates. One of those is wrong, and the model is the one
that acts on it.

## The guard that protects the guard

The parsers here assert on the *declaration text* — `Record<ToolName,
ToolPolicy>`, `Record<ToolName, string>`. That is deliberate. Widening either
back to `Record<string, …>` would silently switch the compiler off, leaving
these tests as the only thing standing, and a parse that finds nothing passes
vacuously. Failing loudly on the widened declaration is what stops the
inversion being undone by a change that looks like a type cleanup.
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
    block = text.split("const TOOL_SCOPE_REGISTRY = {", 1)[1].split("\n} satisfies", 1)[0]
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^\s{2}(\w+):\s*(\{[^}]*\})", block, re.MULTILINE)
    }


def _tool_policy() -> dict[str, str]:
    """tool name -> raw policy body, from `TOOL_POLICY` in contracts.ts.

    Comments are stripped before parsing. `drain_host` is named in the
    reasoning above its own entry *and* discussed in `evict_host_workloads`'s;
    without stripping, a tool mentioned in prose reads as an entry — the
    match-a-mention defect this suite has caught repeatedly.
    """
    text = CONTRACTS_TS.read_text(encoding="utf-8")
    marker = "const TOOL_POLICY: Record<ToolName, ToolPolicy> = {"
    assert marker in text, (
        "contracts.ts no longer declares TOOL_POLICY as Record<ToolName, "
        "ToolPolicy>. If the type was widened, the compiler has stopped "
        "checking completeness and this file is all that remains."
    )
    block = text.split(marker, 1)[1].split("\n};", 1)[0]
    block = re.sub(r"//[^\n]*", "", block)
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^\s{2}(\w+):\s*\{([^}]*)\}", block, re.MULTILINE)
    }


def _flagged(field: str) -> set[str]:
    """Tools whose policy sets `field: true`."""
    return {n for n, body in _tool_policy().items() if f"{field}: true" in body}


def _named_set(name: str) -> set[str]:
    """Back-compat shim for the two set names this file was written against."""
    return _flagged({"READ_ONLY": "readOnly", "DESTRUCTIVE": "destructive"}[name])


def _described() -> set[str]:
    text = DESCRIPTIONS_TS.read_text(encoding="utf-8")
    marker = "const DESCRIPTIONS: Record<ToolName, string> = {"
    assert marker in text, (
        "descriptions.ts no longer declares DESCRIPTIONS as "
        "Record<ToolName, string>; the compiler has stopped checking that "
        "every registered tool has one."
    )
    block = text.split(marker, 1)[1]
    return set(re.findall(r"^\s{2}([a-z_][a-z0-9_]*):", block, re.MULTILINE))


def _is_mutating(requirement: str) -> bool:
    scopes = re.findall(r'"([a-z_]+):([a-z_]+)"', requirement)
    return any(suffix in MUTATING_SUFFIXES for _domain, suffix in scopes)


# ── Calibration ───────────────────────────────────────────────────────


def test_all_four_sources_parse():
    """Four empty sets agree with each other perfectly. This stops that."""
    assert len(_tool_scopes()) > 40, "TOOL_SCOPES did not parse"
    assert len(_tool_policy()) > 40, "TOOL_POLICY did not parse"
    assert len(_flagged("readOnly")) > 20, "no read-only tools parsed"
    assert len(_flagged("destructive")) >= 3, "no destructive tools parsed"
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


# ── The inversion must stay inverted ──────────────────────────────────


def test_the_scope_registry_still_types_the_other_tables():
    """`ToolName` is the whole mechanism. Without it nothing is checked.

    `satisfies` rather than a `Record<string, …>` annotation is load-bearing:
    annotating the literal would widen its key type to `string`, `ToolName`
    would become `string`, and every `Record<ToolName, …>` elsewhere would
    accept anything. The inversion would still *look* present in every file
    while checking nothing — which is the failure mode worth a named test.
    """
    text = SCOPES_TS.read_text(encoding="utf-8")
    assert "const TOOL_SCOPE_REGISTRY = {" in text, (
        "the scope registry is no longer declared as a bare const; ToolName "
        "cannot be derived from it"
    )
    assert "} satisfies Record<string, ScopeRequirement>;" in text, (
        "the registry lost its `satisfies` clause. Annotating it instead "
        "widens the key type to string and silently disables every "
        "completeness check in this package."
    )
    assert "export type ToolName = keyof typeof TOOL_SCOPE_REGISTRY;" in text, (
        "ToolName is no longer derived from the registry's key set"
    )


def test_the_policy_and_description_tables_are_keyed_by_tool_name():
    """Both tables must stay narrow, or the compiler stops counting rows."""
    assert "Record<ToolName, ToolPolicy>" in CONTRACTS_TS.read_text(encoding="utf-8"), (
        "TOOL_POLICY is no longer keyed by ToolName; a tool with no policy row "
        "would compile again"
    )
    assert "Record<ToolName, string>" in DESCRIPTIONS_TS.read_text(encoding="utf-8"), (
        "the description table is no longer keyed by ToolName; a tool with no "
        "description would compile again"
    )


def test_every_tool_declares_the_three_required_policy_fields():
    """Optional-with-a-default is what the five hand-kept Sets already were.

    A tool that omits `destructive` would publish as reversible by default,
    which is the drift S1 removed. The type requires all three; this asserts
    the requirement was not quietly relaxed to optional.
    """
    incomplete = sorted(
        name
        for name, body in _tool_policy().items()
        if not all(field in body for field in ("readOnly:", "destructive:", "audience:"))
    )
    assert not incomplete, (
        f"policy rows missing a required field: {incomplete}. If these compile, "
        "ToolPolicy has been relaxed to make them optional."
    )
