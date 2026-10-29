#!/usr/bin/env python3
"""Build kv-served-traffic-kpi.json from plan step-3 usage-sample (openai_proxy path)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(
    os.environ.get("XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer")
)


def _usage_kv(sample: dict) -> tuple[float | None, dict]:
    run1 = (sample.get("usage_run1") or {}).get("usage") or {}
    last24 = run1.get("last_24h") or {}
    rate = last24.get("kv_cache_hit_rate")
    return (float(rate) if rate is not None else None, last24)


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    usage_path = SCRATCH / "usage-sample.json"
    if not usage_path.is_file():
        print(f"WARN: missing {usage_path}", file=sys.stderr)
        return 1
    sample = json.loads(usage_path.read_text(encoding="utf-8"))
    kv_rate, last24 = _usage_kv(sample)
    ledger = sample.get("ledger_rows") or []
    proxy_rows = [
        r for r in ledger if "usage-proxy" in str(r.get("idempotency_key", ""))
    ]
    out = {
        "source": "openai_proxy_served_traffic",
        "evidence_tier": "launch_served_traffic",
        "note": (
            "KV hit rate from /usage API after HTTP chat/completions through "
            "openai_proxy (plan step 3). Not a separate 30-day production soak."
        ),
        "kv_cache_hit_rate": kv_rate,
        "target": 0.30,
        "kpi_met": kv_rate is not None and kv_rate >= 0.30,
        "proxy_ledger_rows": len(proxy_rows),
        "total_input_tokens": last24.get("total_input_tokens"),
        "total_cached_tokens": last24.get("total_cached_tokens"),
        "slo_status": last24.get("slo_status"),
        "usage_endpoint_id": (
            ((sample.get("usage_run1") or {}).get("usage") or {}).get("endpoint_id")
        ),
    }
    path = SCRATCH / "kv-served-traffic-kpi.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(path)
    return 0 if out["kpi_met"] and proxy_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())