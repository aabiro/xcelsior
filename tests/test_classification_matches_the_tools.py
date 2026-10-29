"""`covered` must mean a tool calls it, and a tool's calls must be covered.

`docs/endpoint-classification.json` is the input to every "what is left to
build" decision, and it was wrong in **both** directions:

* six endpoints labelled `covered` that no tool called — the read half of
  auto-top-up among them, so an agent had to *write* the setting to learn it;
* twelve labelled `gap` that tools have called all along — almost the whole
  volume surface, plus `open_instance_access` and
  `evaluate_placement_preference`.

Together that overstated the work remaining by twelve and understated the
surface by six, on the file used to choose what to build next.

## Why this one is machine-checkable when the earlier attempt was not

An earlier pass concluded `covered` could not be verified mechanically, after
three scans gave three different wrong answers. That conclusion was about a
*fuzzy* matcher: it joined path segments with `[^"`]*`, so `/api/billing/invoice`
matched `/api/billing/invoices`, and `/api/events` matched
`/api/v1/instances/{id}/events`. Segment-joining cannot distinguish
`/api/v2/volumes` from `/api/v2/volumes/{id}` either, since removing the
parameter leaves the same parts.

Exact comparison has none of those failure modes. The path literal is taken
whole, `${…}` becomes `{}`, the classification's `{name}` becomes `{}`, and the
two strings must be equal. What made the earlier approach unreliable was the
fuzziness, not the idea.

**The calibration is what earns the trust**: every one of the ~69 distinct paths
the tools call resolves to a classified endpoint. A matcher that silently failed
would leave unmatched calls, so `test_every_tool_call_maps_to_a_known_endpoint`
failing is the signal that this file has stopped measuring anything.

## What it still cannot check

That the tool calling an endpoint is a *sensible* one for it, or that `internal`
and `redundant` are correctly applied — those stay human judgements. This checks
the one relationship that is mechanical: a label claiming tool coverage, and a
tool call, must agree.
"""

from __future__ import annotations

import json
import pathlib
import re

from tests._source_tree import iter_source_files

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLASSIFICATION = ROOT / "docs" / "endpoint-classification.json"

#: `client.get<T>("…")` / `client.post(`…`)` — the verb, then the first string
#: literal argument. `[^(]*` spans generic type parameters, including nested
#: ones: `client.get<Record<string, unknown>>(` defeated a `[^>]*` version.
CALL = re.compile(r"client\.(get|post|put|patch|delete)[^(]*\(\s*([\"'`])([^\"'`]+)\2", re.S)
TEMPLATE = re.compile(r"\$\{[^}]*\}")
PARAM = re.compile(r"\{[^}]+\}")


def _tool_calls() -> dict[str, str]:
    """`"GET /api/v2/volumes/{}" -> "volumes.ts:68"`, one entry per path."""
    # Through the shared iterator, not `rglob` — eight gates each rolled their
    # own walk and four broke at once on macOS AppleDouble sidecars, with an
    # error naming neither the sidecar nor the gate's subject. See
    # `tests/test_source_tree_is_shared.py`.
    found: dict[str, str] = {}
    for path, rel in sorted(
        iter_source_files("*.ts", include_prefixes=("mcp/src/",)), key=lambda pair: pair[1]
    ):
        if not rel.startswith("mcp/src/"):
            continue
        text = path.read_text(encoding="utf-8")
        for match in CALL.finditer(text):
            verb, literal = match.group(1).upper(), match.group(3)
            if not literal.startswith("/"):
                continue
            key = f"{verb} {TEMPLATE.sub('{}', literal)}"
            found.setdefault(key, f"{path.name}:{text[: match.start()].count(chr(10)) + 1}")
    return found


def _classified() -> dict[str, list[tuple[str, str]]]:
    """Normalised `"GET /a/{}"` -> [(original entry, class), …]."""
    raw = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[str, str]]] = {}
    for entry, value in raw.items():
        klass = value["class"] if isinstance(value, dict) else value
        method, path = entry.split(" ", 1)
        out.setdefault(f"{method} {PARAM.sub('{}', path)}", []).append((entry, klass))
    return out


# ── Calibration ───────────────────────────────────────────────────────


def test_both_sides_parse():
    """Two empty sets agree perfectly, and this file would report green."""
    calls = _tool_calls()
    assert len(calls) > 50, f"only {len(calls)} tool call paths found; CALL is wrong"
    classified = _classified()
    assert len(classified) > 400, f"only {len(classified)} endpoints parsed"


def test_every_tool_call_maps_to_a_known_endpoint():
    """The assertion that makes the rest of this file trustworthy.

    If the extractor or the normalisation breaks, calls stop matching and land
    here — rather than silently matching nothing and reporting that every label
    is correct.
    """
    classified = _classified()
    orphans = sorted(
        f"{where} calls {call}" for call, where in _tool_calls().items() if call not in classified
    )
    assert not orphans, (
        "these tool calls match no classified endpoint. Either a route was "
        "removed under a tool, or the path matching has broken and every other "
        "assertion here is now vacuous: " + "; ".join(orphans)
    )


# ── The two directions the file was wrong in ──────────────────────────


def test_every_endpoint_a_tool_calls_is_labelled_covered():
    """Twelve were labelled `gap` while a tool called them — most of `volumes`.

    A `gap` is work to do. Listing something already built inflates the backlog
    and sends the next person to build a tool that exists.
    """
    classified = _classified()
    wrong = []
    for call, where in sorted(_tool_calls().items()):
        for entry, klass in classified.get(call, []):
            if klass != "covered":
                wrong.append(f"{entry} is `{klass}` but {where} calls it")
    assert not wrong, "\n  ".join(["endpoints a tool calls are not marked covered:"] + wrong)


#: Paths the tools reach through a **lambda in a table** rather than a literal
#: beside `client.post`, so the extractor cannot pair them with a verb:
#:
#:     ["drain_host", "hosts:operate", schema, (a) => `/api/v1/hosts/${…}/drain`]
#:
#: `mutate()` supplies the POST later. Each was confirmed by reading
#: `src/tools/operator.ts`. Named rather than pattern-matched, because a
#: pattern that tolerated indirection would tolerate a genuinely uncalled
#: endpoint too — which is the defect this direction exists to catch.
CALLED_INDIRECTLY = {
    "POST /api/v1/hosts/{host_id}/drain",
    "POST /api/v1/hosts/{host_id}/undrain",
    "POST /api/v1/instances/{job_id}/retry",
    "POST /api/v1/instances/{job_id}/reconcile",
    "POST /api/v1/control-plane/commands/{command_id}/retry",
}


def test_every_covered_endpoint_is_called_by_a_tool():
    """The other direction. Six claimed coverage that did not exist.

    `GET /api/v2/billing/auto-topup` was the costly one: labelled covered with
    no read tool, so the only way to learn the auto-top-up settings was to POST
    a change and read the previous values back — writing in order to read, on
    the surface that authorises unattended charges.
    """
    calls = set(_tool_calls())
    raw = json.loads(CLASSIFICATION.read_text(encoding="utf-8"))
    uncalled = sorted(
        entry
        for entry, value in raw.items()
        if (value["class"] if isinstance(value, dict) else value) == "covered"
        and entry not in CALLED_INDIRECTLY
        and f"{entry.split(' ', 1)[0]} {PARAM.sub('{}', entry.split(' ', 1)[1])}" not in calls
    )
    assert not uncalled, (
        "these endpoints claim tool coverage and no tool calls them: "
        f"{uncalled}. Either the tool was never built, or it was removed and "
        "the label outlived it."
    )
