#!/usr/bin/env python3
"""GX1 evaluation set — does a model pick the right tool from our descriptions?

Adoption plan X1.14 / gate GX1. The cases cover direct requests, indirect
phrasings, follow-ups that reuse an id from an earlier turn, write actions that
must route through approval, and — the ones that matter most — requests that
should call **no** tool at all.

The tool definitions are read from the **live server**, not from the source
tree. That is the whole point: a reviewer's model sees exactly what
`tools/list` publishes, so the eval has to grade the published surface. A
regression in a description shows up here even though no test file changed.

    export XCELSIOR_MCP_TOKEN=...          # MCP-audience bearer
    export ANTHROPIC_API_KEY=...           # or `ant auth login`
    python3 scripts/mcp_tool_eval.py --base https://mcp.xcelsior.ca/mcp

Without an Anthropic credential the run reports BLOCKED(env) and exits 0 — a
gate that cannot run is never green, but it is also not a failure of the code
under test. With one, exit status is non-zero unless the pass rate meets
`--threshold`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any

EVAL_FILE = pathlib.Path(__file__).resolve().parent.parent / "mcp" / "evals" / "tool-selection.jsonl"

SYSTEM = (
    "You are an assistant connected to the Xcelsior GPU cloud over MCP. "
    "Use the available tools when they are the right way to answer, and answer "
    "directly when they are not. Do not call a tool merely because one exists."
)


@dataclass
class Case:
    id: str
    category: str
    prompt: str
    why: str
    expect_any_of: list[str] = field(default_factory=list)
    expect_none: bool = False
    context: list[dict[str, Any]] = field(default_factory=list)


def load_cases(path: pathlib.Path) -> list[Case]:
    cases: list[Case] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        cases.append(
            Case(
                id=raw["id"],
                category=raw["category"],
                prompt=raw["prompt"],
                why=raw.get("why", ""),
                expect_any_of=raw.get("expect_any_of", []),
                expect_none=bool(raw.get("expect_none")),
                context=raw.get("context", []),
            )
        )
    return cases


# ── The published tool surface ────────────────────────────────────────────


def fetch_tools(base: str, token: str, timeout: float) -> list[dict[str, Any]]:
    """`initialize` + `tools/list` against the real endpoint."""
    import httpx

    headers = {
        "authorization": f"Bearer {token}",
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    with httpx.Client(timeout=timeout) as http:
        init = http.post(
            base,
            headers=headers,
            content=json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "xcelsior-tool-eval", "version": "1.0.0"},
                },
            }),
        )
        init.raise_for_status()
        listed = http.post(
            base,
            headers=headers,
            content=json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        )
        listed.raise_for_status()
    return _decode(listed.text)["result"]["tools"]


def _decode(body: str) -> dict[str, Any]:
    """Streamable HTTP may answer as JSON or as a one-shot SSE frame."""
    if body.lstrip().startswith("event:"):
        body = "".join(line[5:].strip() for line in body.splitlines() if line.startswith("data:"))
    return json.loads(body)


def to_anthropic_tools(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
        }
        for tool in mcp_tools
    ]


# ── Running one case ──────────────────────────────────────────────────────


def selected_tools(client, model: str, tools: list[dict[str, Any]], case: Case) -> tuple[list[str], str]:
    messages = [*case.context, {"role": "user", "content": case.prompt}]
    response = client.beta.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        tools=tools,
        messages=messages,
        # Claude Opus 5's safety classifiers can decline a request outright.
        # Routing the decline to the recommended fallback keeps one unlucky
        # case from being scored as a tool-selection failure.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return [], "refusal"
    names = [block.name for block in response.content if getattr(block, "type", "") == "tool_use"]
    text = " ".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    return names, text[:200]


def grade(case: Case, chosen: list[str]) -> tuple[bool, str]:
    if case.expect_none:
        if chosen:
            return False, f"expected no tool call, got {', '.join(chosen)}"
        return True, "abstained"
    if not chosen:
        return False, f"expected one of {', '.join(case.expect_any_of)}, called nothing"
    if any(name in case.expect_any_of for name in chosen):
        return True, ", ".join(chosen)
    return False, f"called {', '.join(chosen)}, expected one of {', '.join(case.expect_any_of)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="https://mcp.xcelsior.ca/mcp")
    parser.add_argument("--token", default=os.environ.get("XCELSIOR_MCP_TOKEN", ""))
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--threshold", type=float, default=0.90, help="required overall pass rate")
    parser.add_argument(
        "--abstention-threshold",
        type=float,
        default=1.0,
        help="required pass rate on the no-tool cases; a connector that calls a "
        "tool at a greeting is worse than one that misroutes a real request",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--only", default="", help="run one category: direct|indirect|followup|approval|no_tool")
    args = parser.parse_args()

    cases = load_cases(EVAL_FILE)
    if args.only:
        cases = [case for case in cases if case.category == args.only]
    print(f"GX1 tool-selection eval — {len(cases)} cases against {args.base}\n")

    if not args.token:
        print("BLOCKED(env): set XCELSIOR_MCP_TOKEN to an MCP-audience bearer token.")
        return 0
    try:
        import anthropic
    except ImportError:
        print("BLOCKED(env): pip install anthropic")
        return 0
    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        print(f"BLOCKED(env): no Anthropic credential ({exc}). Run `ant auth login` or set ANTHROPIC_API_KEY.")
        return 0

    try:
        tools = to_anthropic_tools(fetch_tools(args.base, args.token, args.timeout))
    except Exception as exc:
        print(f"FAIL: could not read the published tool surface: {exc}")
        return 1
    print(f"Published surface: {len(tools)} tools\n")

    failures: list[tuple[Case, str]] = []
    by_category: dict[str, list[bool]] = {}
    for case in cases:
        try:
            chosen, detail = selected_tools(client, args.model, tools, case)
        except Exception as exc:  # network/API problem is a harness failure, not a miss
            print(f"[ ERR  ] {case.id}: {exc}")
            failures.append((case, str(exc)))
            by_category.setdefault(case.category, []).append(False)
            continue
        ok, note = grade(case, chosen)
        by_category.setdefault(case.category, []).append(ok)
        print(f"[{'  ok  ' if ok else ' FAIL '}] {case.id:<28} {note}")
        if not ok:
            failures.append((case, note))
            if detail:
                print(f"           said: {detail}")

    total = sum(len(v) for v in by_category.values())
    passed = sum(sum(v) for v in by_category.values())
    rate = passed / total if total else 0.0
    print("\nBy category:")
    for category, results in sorted(by_category.items()):
        print(f"  {category:<10} {sum(results)}/{len(results)}")
    print(f"\nOverall: {passed}/{total} ({rate:.0%}), threshold {args.threshold:.0%}")

    abstention = by_category.get("no_tool", [])
    abstention_rate = (sum(abstention) / len(abstention)) if abstention else 1.0
    if abstention:
        print(f"Abstention: {sum(abstention)}/{len(abstention)} ({abstention_rate:.0%}), "
              f"threshold {args.abstention_threshold:.0%}")

    if failures:
        print("\nFailures — each one is a description that does not say what it should:")
        for case, note in failures:
            print(f"  {case.id}: {note}\n    why this case exists: {case.why}")

    ok = rate >= args.threshold and abstention_rate >= args.abstention_threshold
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
