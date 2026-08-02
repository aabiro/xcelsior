#!/usr/bin/env python3
"""Soak the MCP edge at a chosen rate and check it degrades rather than collapses.

Gate GX6 (adoption plan X6.31): "soak test at 2× published limits with graceful
degradation". Graceful means a specific, checkable thing:

  * excess calls are **refused** with 429 and a `Retry-After`, not dropped and
    not served slowly;
  * accepted calls stay within the published p95 latency;
  * **no request returns 5xx** — a 5xx under load is collapse, and the
    difference between "we are shedding load" and "we fell over" is exactly
    this line.

    python3 scripts/mcp_soak.py --base https://mcp.xcelsior.ca/mcp \\
        --token "$XCELSIOR_MCP_TOKEN" --rate 240 --duration 300

Exits non-zero when any pass condition fails. Read-only: it calls `tools/list`
and one read tool, so a soak never launches, cancels, or bills anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

try:
    import httpx
except ImportError:  # pragma: no cover - operator-facing script
    print("mcp_soak requires httpx (pip install httpx)", file=sys.stderr)
    raise SystemExit(2) from None


@dataclass
class Results:
    lock: threading.Lock = field(default_factory=threading.Lock)
    latencies_ms: list[float] = field(default_factory=list)
    statuses: Counter = field(default_factory=Counter)
    throttled_without_retry_after: int = 0
    transport_errors: int = 0

    def record(self, status: int, elapsed_ms: float, retry_after: str | None) -> None:
        with self.lock:
            self.statuses[status] += 1
            if status == 429 and not retry_after:
                self.throttled_without_retry_after += 1
            # Only accepted calls count toward the latency objective; a fast
            # rejection is not evidence the service is fast.
            if 200 <= status < 300:
                self.latencies_ms.append(elapsed_ms)

    def error(self) -> None:
        with self.lock:
            self.transport_errors += 1


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def one_call(http: httpx.Client, base: str, token: str, body: dict, results: Results) -> None:
    started = time.perf_counter()
    try:
        response = http.post(
            base,
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            },
            content=json.dumps(body),
        )
    except Exception:
        results.error()
        return
    results.record(
        response.status_code,
        (time.perf_counter() - started) * 1000,
        response.headers.get("retry-after"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="https://mcp.xcelsior.ca/mcp")
    parser.add_argument("--token", required=True, help="MCP-audience bearer token")
    parser.add_argument("--rate", type=int, default=240, help="requests per minute to offer")
    parser.add_argument("--duration", type=int, default=300, help="seconds")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--p95-ms", type=float, default=2000.0, help="read-tool p95 objective")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    interval = 60.0 / max(1, args.rate)
    deadline = time.monotonic() + args.duration
    results = Results()
    # A read that touches the upstream API, so the soak exercises the real path
    # rather than a handler that answers from memory.
    body = {
        "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": {"name": "get_pricing_reference", "arguments": {}},
    }

    print(
        f"Soaking {args.base} at {args.rate}/min for {args.duration}s "
        f"({args.concurrency} workers)\n"
    )

    stop = threading.Event()
    pending = threading.Semaphore(args.concurrency)

    def worker(payload: dict) -> None:
        try:
            one_call(http, args.base, args.token, payload, results)
        finally:
            pending.release()

    threads: list[threading.Thread] = []
    with httpx.Client(timeout=args.timeout) as http:
        next_at = time.monotonic()
        while time.monotonic() < deadline and not stop.is_set():
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(next_at - now, 0.05))
                continue
            next_at += interval
            # Offered load is capped by concurrency: if the server slows down we
            # stop adding load rather than queueing an unbounded backlog, which
            # would measure our own client instead of the server.
            if not pending.acquire(blocking=False):
                results.record(0, 0.0, None)  # counted as "offered, not sent"
                continue
            thread = threading.Thread(target=worker, args=(body,), daemon=True)
            thread.start()
            threads.append(thread)
            threads = [t for t in threads if t.is_alive()]
        for thread in threads:
            thread.join(timeout=args.timeout)

    accepted = sum(count for status, count in results.statuses.items() if 200 <= status < 300)
    throttled = results.statuses.get(429, 0)
    server_errors = sum(count for status, count in results.statuses.items() if status >= 500)
    skipped = results.statuses.get(0, 0)
    total = accepted + throttled + server_errors + results.transport_errors

    print("Status distribution:")
    for status, count in sorted(results.statuses.items()):
        label = "offered-but-not-sent (client saturated)" if status == 0 else str(status)
        print(f"  {label:<40} {count}")
    if results.transport_errors:
        print(f"  transport errors                         {results.transport_errors}")

    p50 = percentile(results.latencies_ms, 0.50)
    p95 = percentile(results.latencies_ms, 0.95)
    p99 = percentile(results.latencies_ms, 0.99)
    if results.latencies_ms:
        print(
            f"\nAccepted latency: p50 {p50:.0f}ms  p95 {p95:.0f}ms  p99 {p99:.0f}ms  "
            f"(mean {statistics.fmean(results.latencies_ms):.0f}ms, n={len(results.latencies_ms)})"
        )

    failures: list[str] = []
    if server_errors:
        failures.append(
            f"{server_errors} request(s) returned 5xx — that is collapse, not degradation"
        )
    if results.transport_errors:
        failures.append(
            f"{results.transport_errors} transport error(s) — connections were dropped rather than refused"
        )
    if throttled and results.throttled_without_retry_after:
        failures.append(
            f"{results.throttled_without_retry_after} of {throttled} throttled responses "
            f"carried no Retry-After — a client cannot back off correctly"
        )
    if results.latencies_ms and p95 > args.p95_ms:
        failures.append(f"accepted p95 {p95:.0f}ms exceeds the {args.p95_ms:.0f}ms objective")
    if not accepted:
        failures.append("no request was accepted — this measured an outage, not a limit")
    if skipped:
        print(
            f"\nNote: {skipped} calls were never sent because the client hit its own "
            f"concurrency cap. Raise --concurrency if that is a large share of {total}."
        )

    print()
    if failures:
        print("SOAK FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print(
        f"SOAK PASSED: {accepted} accepted, {throttled} shed with Retry-After, "
        f"0 server errors, p95 {p95:.0f}ms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
