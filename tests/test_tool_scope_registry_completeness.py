"""A tool cannot exist without a scope requirement in the tool layer.

§0.1 of the plan is explicit: *a scope enforced in one layer only is the exact
defect the `api` wildcard was.* That defect was real — `userHasScope` short-
circuited on `api` in TypeScript while the Python layer had already removed it,
and every agent key reached every scoped endpoint.

The same gap reopens the moment a tool ships without a `TOOL_SCOPES` entry: the
route checks a scope, the tool advertises none, and the mismatch is invisible
until someone reads both files. "We will add the entry when the tool lands" is a
promise, and this codebase's history is a list of promises that were kept right
up until they weren't.

So the registry is required to be complete rather than intended to be. Every
tool the server registers must declare what it requires, and every scope it
declares must be one the authorization server can actually issue — a tool
requiring a scope no client can hold is unreachable, which is the sealed-endpoint
failure in a different costume.

This is a structural check, not a behavioural one: it reads the sources rather
than starting a server, so it runs everywhere and cannot be skipped for being
slow.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCOPES_TS = ROOT / "mcp" / "src" / "auth" / "scopes.ts"


def _tool_scope_entries() -> dict[str, str]:
    """tool name -> raw requirement text, from TOOL_SCOPES."""
    text = SCOPES_TS.read_text(encoding="utf-8")
    block = text.split("export const TOOL_SCOPES", 1)[1]
    block = block.split("\n};", 1)[0]
    return {
        m.group(1): m.group(2)
        for m in re.finditer(r"^\s{2}(\w+):\s*(\{[^}]*\})", block, re.MULTILINE)
    }


def _declared_scopes() -> set[str]:
    return set(re.findall(r'"([a-z_]+:[a-z_]+)"', SCOPES_TS.read_text(encoding="utf-8")))


SURFACE_JSON = ROOT / "mcp" / "tool-surface.json"


def _published_surface() -> list[dict]:
    """The generated snapshot of what the server actually publishes.

    This is the registry, not a re-derivation of it: `npm run surface:update`
    writes it, and `tests/test_mcp_*` already assert it matches the running
    server. Reading it here means this guard checks the same artifact the
    server ships rather than a hand-maintained parallel list — which is the
    whole point of P0's "generate, never hand-maintain".
    """
    import json

    return json.loads(SURFACE_JSON.read_text(encoding="utf-8"))["tools"]


def _registered_tool_names() -> set[str]:
    return {t["name"] for t in _published_surface()}


#: The plan's eval baseline. Every later phase's delta is measured against it,
#: so a wrong number silently invalidates every future gate comparison.
#:
#: Was 39 when this file was written on `feat/mcp-p0-scopes`; the surface has
#: since grown by sixteen. P2 added `register_ssh_key` and
#: `open_instance_access`; P3 added the eight durable-state tools; and the
#: two serverless exits closed half of what GT0 found — entrances everywhere,
#: exits missing. The
#: plan's Gate P0 line is restated in the same commit, which is what this
#: test's failure message demands and the reason it is worth failing over.
#:
#: **A baseline now exists.** Captured 2026-08-08 at 46 published tools:
#: `expected_tool_accuracy` 0.9111 against the unmoved 0.90 threshold, abstention
#: 1.0, unsafe-write rate 0.0, in `eval-baseline.json`. It was taken against a
#: *local* surface built from the working tree — the JSON records
#: `base: http://127.0.0.1:…` — because production runs an older commit, so it
#: grades what the repo publishes rather than what is deployed. Check that field
#: before comparing two baselines.
#:
#: The comment this replaces said no baseline had ever been captured, which was
#: true when written and stopped being true the moment one was. A stale claim
#: about a gate is worse than none, because it argues against re-checking.
#:
#: This constant pins the *count*. 55 → 57 with `promote_artifact_to_volume`
#: and `get_promotion_status` (P3 A4); 57 → 59 with `run_pipeline` and
#: `get_pipeline_status` (P4 B4). The two new
#: tools do not change the eval's blast radius — both are satisfiable by a Quick
#: Connect token, asserted in `test_connector_tokens_are_scope_restricted.py` —
#: but the surface is larger than the baseline was measured on, so the next
#: capture is owed and will not be directly comparable.
EXPECTED_TOOL_TOTAL = 59

#: The customer profile is what `mcp.xcelsior.ca/mcp` serves and what a
#: directory lists. It is the total minus two exclusions, and the decomposition
#: is asserted rather than assumed — checking only the published snapshot means
#: checking 30 of 39, leaving the operator tools (`evict_host_workloads`,
#: `drain_host`) unverified, which are the ones carrying the sharpest scopes.
OPERATOR_TOOLS = {
    "drain_host", "undrain_host", "evict_host_workloads", "get_host_capacity",
    "get_scheduler_health", "list_reconciliation_findings", "retry_agent_command",
}
COMPANY_KNOWLEDGE_TOOLS = {"search", "fetch"}


def test_the_registry_holds_the_baseline_number_of_tools():
    """The plan states the baseline; if the count moved, the plan must say so."""
    entries = _tool_scope_entries()
    assert len(entries) == EXPECTED_TOOL_TOTAL, (
        f"TOOL_SCOPES holds {len(entries)} tools, not the {EXPECTED_TOOL_TOTAL} "
        "the plan's eval baseline is measured against. Restate the baseline in "
        "docs/mcp-agent-native-implementation-plan.md with the reason, in the "
        "same commit."
    )


def test_the_customer_snapshot_is_the_total_minus_its_stated_exclusions():
    """Explains the 30-vs-39 difference instead of leaving it to be rediscovered.

    Checking the published snapshot alone verifies the customer profile and
    silently ignores nine tools. This pins *why* the numbers differ, so a tool
    vanishing from the snapshot for any other reason fails here.
    """
    all_tools = set(_tool_scope_entries())
    expected_customer = all_tools - OPERATOR_TOOLS - COMPANY_KNOWLEDGE_TOOLS
    published = _registered_tool_names()
    assert published == expected_customer, (
        "the customer snapshot is not the total minus operator and "
        "company-knowledge tools:\n"
        f"  published but not expected: {sorted(published - expected_customer)}\n"
        f"  expected but not published: {sorted(expected_customer - published)}"
    )


def test_every_tool_in_the_registry_declares_a_scope_requirement():
    """All 39, not the 30 that happen to be published.

    The gap §0.1 warns about, made impossible instead of promised — and covering
    the operator tools specifically, since those hold `hosts:evict`.
    """
    empty = sorted(
        name
        for name, raw in _tool_scope_entries().items()
        if not re.search(r'"[a-z_]+:[a-z_]+"', raw)
    )
    assert not empty, f"tools declaring no scope at all: {empty}"


def test_operator_tools_require_operator_scopes():
    """An operator tool reachable with tenant scopes is a profile leak."""
    entries = _tool_scope_entries()
    leaks = {}
    for name in sorted(OPERATOR_TOOLS):
        raw = entries.get(name, "")
        scopes = set(re.findall(r'"([a-z_]+:[a-z_]+)"', raw))
        if not any(s.startswith(("hosts:", "control_plane:")) for s in scopes):
            leaks[name] = sorted(scopes)
    assert not leaks, (
        f"operator tools requiring no operator scope: {leaks}"
    )


def test_every_registered_tool_declares_its_scope_requirement():
    """The published surface is a subset of the registry, never beyond it."""
    missing = sorted(_registered_tool_names() - set(_tool_scope_entries()))
    assert not missing, (
        "these tools are published but have no TOOL_SCOPES entry, so the tool "
        "layer advertises no requirement while the route enforces one — a scope "
        f"enforced in one layer only: {missing}"
    )


def test_no_tool_requires_a_scope_that_cannot_be_issued():
    """A requirement no client can satisfy is an unreachable tool."""
    import oauth_service

    grantable = {s for s in oauth_service.SCOPE_DESCRIPTIONS if ":" in s}
    required: set[str] = set()
    for raw in _tool_scope_entries().values():
        required |= set(re.findall(r'"([a-z_]+:[a-z_]+)"', raw))
    for tool in _published_surface():
        required |= set(tool.get("requiredScopes") or ())
    unissuable = sorted(required - grantable)
    assert not unissuable, (
        "tools require scopes the authorization server will never issue, so "
        f"they are unreachable by any credential: {unissuable}"
    )


def test_every_tool_scope_entry_is_non_empty():
    """`{}` would pass a presence check while requiring nothing."""
    empty = sorted(
        name
        for name, raw in _tool_scope_entries().items()
        if not re.search(r'"[a-z_]+:[a-z_]+"', raw)
    )
    assert not empty, (
        f"these tools declare a requirement that names no scope: {empty}"
    )


def test_the_discovery_actually_finds_tools():
    """Prove the scanner's reach rather than trusting its silence.

    If the tool directory moved, every assertion above would pass on an empty
    set and report clean — the shape of the vocabulary guard that scanned ten
    file types and found nothing.
    """
    assert len(_registered_tool_names()) >= 20, (
        "tool discovery found almost nothing; the checks above would pass "
        "vacuously"
    )
    assert all(t.get("requiredScopes") for t in _published_surface()), (
        "a published tool declares no required scope at all"
    )
    assert len(_tool_scope_entries()) >= 20, "TOOL_SCOPES parsing found almost nothing"
