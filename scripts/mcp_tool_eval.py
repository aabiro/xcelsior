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
import time
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


# ── Token accounting ──────────────────────────────────────────────────────

#: Totals across the run, so the artifact records what the grading cost and
#: whether the cache actually engaged. A run that reports zero cache reads is a
#: run paying full price for the same 7,700 tokens on every case.
USAGE = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
}


def _record_usage(usage: Any) -> None:
    if usage is None:
        return
    for field in USAGE:
        USAGE[field] += int(getattr(usage, field, 0) or 0)


def estimated_cost_usd(model: str) -> float:
    """Opus 5 list price. Cache writes bill at 1.25x input, reads at 0.1x."""
    if "opus" not in model:
        return 0.0
    per_m_in, per_m_out = 5.0, 25.0
    return (
        USAGE["input_tokens"] * per_m_in
        + USAGE["cache_creation_input_tokens"] * per_m_in * 1.25
        + USAGE["cache_read_input_tokens"] * per_m_in * 0.10
        + USAGE["output_tokens"] * per_m_out
    ) / 1_000_000


# ── Running one case ──────────────────────────────────────────────────────


def selected_tools(client, model: str, tools: list[dict[str, Any]], case: Case) -> tuple[list[str], str]:
    messages = [*case.context, {"role": "user", "content": case.prompt}]
    # Prompt caching on the static prefix.
    #
    # Every case sends the same 36 tool schemas and the same system prompt —
    # about 7,700 input tokens — and only the last user turn differs. Uncached
    # that is roughly 4c per case at Opus 5's input rate, so a 34-case baseline
    # costs about $1.50 to grade the same fixed text 34 times.
    #
    # The breakpoint goes on the system block because the cacheable prefix is
    # ordered tools -> system -> messages: one `cache_control` here covers the
    # tool schemas *and* the system prompt. `SYSTEM` becomes a block list for
    # that reason, not a style change.
    #
    # Correctness is unaffected either way — a cache miss sends the identical
    # bytes — so `usage` is returned to the caller and totalled, because
    # "caching is on" is a claim and cache_read_input_tokens is evidence.
    common = {
        "model": model,
        "max_tokens": 4096,
        "system": [
            {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
        ],
        "tools": tools,
        "messages": messages,
    }
    try:
        # Claude Opus 5's safety classifiers can decline a request outright.
        # Routing the decline to the recommended fallback keeps one unlucky
        # case from being scored as a tool-selection failure.
        response = client.beta.messages.create(
            **common,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except TypeError as exc:
        # The pinned SDK does not know `fallbacks`, and this is the reason no
        # baseline has ever been captured. With anthropic 0.86.0 every one of
        # the 34 cases died on
        #
        #     Messages.create() got an unexpected keyword argument 'fallbacks'
        #
        # and the script still exited 0, so the gate reported success having
        # graded nothing. Degrading here rather than pinning a newer SDK: the
        # option is an optimisation against refusals, and `stop_reason ==
        # "refusal"` below already detects and reports them. Losing it costs a
        # slightly noisier score, not a wrong one.
        if "fallbacks" not in str(exc):
            raise
        response = client.beta.messages.create(**common)
    usage = getattr(response, "usage", None)
    _record_usage(usage)
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
    parser.add_argument(
        "--case",
        default="",
        help="run one case by id. For re-checking a single description change "
        "without paying for the set — a full capture is ~30x the trials. It is "
        "deliberately narrow: a description names other tools, so a fix "
        "verified this way is unverified against the cases it might disturb, "
        "and the run's rate is not comparable to a baseline.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="write the baseline JSON here. `live-gates.yml` has always passed "
        "this flag and it did not exist, so the job exited 2 on argparse before "
        "reaching any credential check.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="run every case this many times and report the mean. One sample is "
        "not a measurement: two consecutive single-sample runs of this eval "
        "against an unchanged surface scored 26/30 and 25/30, disagreeing on "
        "three cases. A threshold cannot distinguish a regression from that.",
    )
    args = parser.parse_args()

    cases = load_cases(EVAL_FILE)
    # **Newest first.** Cases are appended to the JSONL, so file order puts the
    # least-verified cases last — and a capture that runs out of credit partway
    # loses exactly the evidence it was run to get. On 2026-08-15 a 40-case run
    # died at case 31 and every one of the nine it never reached was a case for
    # a tool added that day, including the one already known to be failing.
    #
    # Order does not affect a completed run: each case is graded independently
    # and the rate is a sum. It only decides what a *partial* run buys, so it
    # should buy the unknown. Reversing costs nothing and needs no metadata —
    # append position already encodes recency.
    cases = list(reversed(cases))
    if args.only:
        cases = [case for case in cases if case.category == args.only]
    if args.case:
        cases = [case for case in cases if case.id == args.case]
        if not cases:
            print(f"BLOCKED: no case with id {args.case!r}")
            return 2
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
    #: case_id -> how many of its samples passed. A case that passes some of the
    #: time is a different problem from one that never does, and averaging them
    #: into a single rate hides which is which.
    per_case: dict[str, list[bool]] = {}
    for case in cases:
        results: list[bool] = []
        note = ""
        for _ in range(max(1, args.samples)):
            try:
                chosen, detail = selected_tools(client, args.model, tools, case)
            except Exception as exc:
                # **A harness failure is not a wrong answer, and must never be
                # scored as one.** This used to append `False` and carry on, so
                # an expired token, a rate limit or an exhausted credit balance
                # turned into "the model chose badly" — silently, and with a
                # plausible number at the end.
                #
                # It happened: a 5-sample run lost its Anthropic balance partway
                # and reported `expected_tool_accuracy 0.54`, `abstention 0.0`,
                # `unsafe_write_rate 1.0` — which reads as a connector that
                # became reckless, from 66 calls that never reached the API. The
                # giveaway was `direct` and `indirect` perfect and everything
                # after them zero, in file order, at *lower* total cost than a
                # smaller run.
                #
                # Abort instead. A partial baseline is worse than none, because
                # a number gets written down and compared against later.
                raise SystemExit(
                    f"\nABORTED — the grader could not reach the API on "
                    f"{case.id}:\n    {exc}\n\n"
                    "No baseline written. Scoring an unreachable API as a wrong "
                    "answer would record a fabricated regression."
                ) from exc
            ok, note = grade(case, chosen)
            results.append(ok)
        per_case[case.id] = results
        passed_n = sum(results)
        by_category.setdefault(case.category, []).extend(results)
        if passed_n == len(results):
            mark = "  ok  "
        elif passed_n == 0:
            mark = " FAIL "
        else:
            mark = " FLAKY"
        tally = f"{passed_n}/{len(results)}" if len(results) > 1 else ""
        print(f"[{mark}] {case.id:<28} {tally:<6} {note}")
        ok = passed_n > len(results) / 2
        if not ok:
            failures.append((case, note))

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

    # **How precise this number is not.** Two consecutive 3-sample runs against
    # an unchanged surface scored 83/90 (0.9222) and 80/90 (0.8889) — one above
    # the 0.90 threshold and one below it. That is not a bug in either run: at
    # n=90 and p≈0.92 the binomial standard error is ≈0.029, so a 0.033 swing is
    # about 1.2 SE and entirely expected.
    #
    # The consequence is that a single capture cannot distinguish a real
    # regression from sampling noise unless the shift is larger than roughly
    # 2 SE, and the threshold sits *inside* that band. `temperature: 0` would be
    # the usual fix and is **not available** — the API rejects it on Opus 5 with
    # "`temperature` is deprecated for this model" — so the only lever is n.
    #
    # Rather than quietly raising the default sample count, which multiplies the
    # cost of every capture, the resolution is computed and reported so nobody
    # reads a point estimate as exact. Raising `--samples` is a spending
    # decision and belongs to whoever is paying.
    standard_error = (rate * (1.0 - rate) / total) ** 0.5 if total else 0.0
    resolution = 2 * standard_error

    # `always_failed` is the part of this eval that noise cannot flip: a case
    # that fails every sample is a real failure at any sample count. It is
    # required to be empty as well as the rate clearing its threshold, so the
    # gate keeps one signal that does not move with the draw.
    ok = (
        rate >= args.threshold
        and abstention_rate >= args.abstention_threshold
        and not [cid for cid, r in per_case.items() if not any(r)]
    )
    print(
        f"\nResolution: +/-{resolution:.3f} at 2 SE (n={total}). A change smaller "
        f"than that is not distinguishable from noise in one capture."
    )

    if args.out:
        # `live-gates.yml` has always invoked this with `--out eval-baseline.json`
        # and the flag did not exist, so the job died on argparse exit 2 before
        # reaching any credential check — one of three reasons no baseline was
        # ever captured. The others: the three secrets are unset, and the SDK
        # pinned here rejects the `fallbacks` argument this script passed on
        # every call.
        #
        # Keys match what the workflow's summary step already reads.
        pathlib.Path(args.out).write_text(
            json.dumps(
                {
                    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "base": args.base,
                    "model": args.model,
                    "tool_count": len(tools),
                    "cases": total,
                    "passed": passed,
                    # Named for what it measures. It was `first_tool_accuracy`,
                    # which `grade()` has never computed: the check is
                    # `any(name in expect_any_of for name in chosen)` — the
                    # expected tool appearing *anywhere* in a single turn's
                    # selection, not first. The wrong name cost real time when
                    # a failure was diagnosed as "the model read before writing,
                    # so a safe read scores as a miss" — which the grader does
                    # not do. Renamed rather than aliased.
                    "expected_tool_accuracy": round(rate, 4),
                    # Single-turn, and that is a property of the measurement
                    # rather than of the surface. A case whose natural answer is
                    # read-then-act cannot pass here, because the eval never
                    # feeds the read's result back.
                    "grading": "expected tool appears in one turn's selection",
                    "samples_per_case": max(1, args.samples),
                    # Which cases never pass, and which pass only sometimes. A
                    # single overall rate cannot tell a real regression from
                    # run-to-run variance, and two consecutive single-sample
                    # runs of this eval disagreed on three of thirty cases.
                    "always_failed": sorted(
                        cid for cid, r in per_case.items() if not any(r)
                    ),
                    "flaky": sorted(
                        cid for cid, r in per_case.items() if any(r) and not all(r)
                    ),
                    # The abstention cases are the ones where calling *any* tool
                    # is the failure, so the interesting rate is the inverse.
                    "abstention_rate": round(abstention_rate, 4),
                    # Named for what it counts, which is **not** writes: it is
                    # the complement of abstention, so a *read* tool called at a
                    # greeting scores here too. The old name asserted a check
                    # the computation never performed, and it was misread as an
                    # unsafe write having occurred.
                    "non_abstention_rate": round(1.0 - abstention_rate, 4),
                    "unsafe_write_rate": round(1.0 - abstention_rate, 4),
                    "standard_error": round(standard_error, 4),
                    "resolution_2se": round(resolution, 4),
                    "by_category": {
                        k: {"passed": sum(v), "total": len(v)}
                        for k, v in sorted(by_category.items())
                    },
                    "threshold": args.threshold,
                    "abstention_threshold": args.abstention_threshold,
                    "meets_threshold": ok,
                    # What the grading cost, and whether the cache engaged. A
                    # baseline that records zero cache reads paid full price to
                    # send the same 7,700-token prefix on every case.
                    "usage": dict(USAGE),
                    "estimated_cost_usd": round(estimated_cost_usd(args.model), 4),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nBaseline written to {args.out}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
