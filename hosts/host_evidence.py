"""One read model of a host's evidence, computed from the source tables at read time.

Listings, the host page, the trust page and the MCP tools all show what this
returns, so a renter and an agent see the same facts:

- verification: state, when it last passed, when the next check is due, and
  whether it is overdue (``verification_freshness``, the one rule);
- the latest xcelsior-bench run (``benchmarks``, 167) and when it ran;
- reputation tier and score (``reputation_scores``);
- the job success rate over a window: the share of runs that started on the
  host and ended (``job_attempts``) without a host-caused failure
  (``job_failure_log``, which only host-fault failures reach). A renter's own
  crash or cancel is not counted against the host.

Nothing here is copied into another table. ``gpu_offers.verified_level``,
``reliability_score`` and ``dlperf_score`` were never written after an offer
was created (prod showed a verified host's offer as ``unverified``); they are
deprecated, left out of every response (``DEPRECATED_OFFER_FIELDS``), and this
is what replaces them.

Public by design, so it leaves out what ``/api/verified-hosts`` keeps behind a
sign-in: the de-verification reason, holds, the GPU fingerprint and UUID.
"""

from __future__ import annotations

import time
from typing import Any

#: The window the job success rate covers.
SUCCESS_WINDOW_SEC = 30 * 86400

#: gpu_offers columns nothing maintains; responses drop them (see module doc).
DEPRECATED_OFFER_FIELDS = ("verified_level", "reliability_score", "dlperf_score")

#: Attempt statuses that mean a run that started on the host has ended, whoever
#: ended it (job_attempts_status_check). `fenced` is a superseded placement.
_ENDED = ("succeeded", "failed", "lost", "preempted", "cancelled")


def _f(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def bench_summary(details: dict | None, row: dict) -> dict:
    """The fields a listing shows from one xcelsior-bench run."""
    details = details if isinstance(details, dict) else {}
    raw_sustained = details.get("sustained")
    sustained: dict = raw_sustained if isinstance(raw_sustained, dict) else {}
    raw_verdict = details.get("verdict")
    verdict: dict = raw_verdict if isinstance(raw_verdict, dict) else {}
    return {
        "ran_at": _f(row.get("run_at")),
        "bench_version": row.get("bench_version") or details.get("bench_version") or "",
        "verdict": row.get("verdict") or verdict.get("result") or "",
        "faults": list(verdict.get("faults") or []),
        "gpu_model": row.get("gpu_model") or details.get("gpu_model") or "",
        "fp16_tflops": _f(details.get("fp16_tflops")),
        "bf16_tflops": _f(details.get("bf16_tflops")),
        "memory_bandwidth_gbps": _f(details.get("memory_bandwidth_gbps")),
        "allocatable_vram_gb": _f(details.get("allocatable_vram_gb")),
        "pcie_h2d_gbps": _f(details.get("pcie_h2d_gbps")),
        "pcie_d2h_gbps": _f(details.get("pcie_d2h_gbps")),
        "sustained": {
            "duration_s": _f(sustained.get("duration_s")),
            "sm_clock_p50_mhz": _f(sustained.get("sm_clock_p50_mhz")),
            "sm_clock_min_mhz": _f(sustained.get("sm_clock_min_mhz")),
            "temp_peak_c": _f(sustained.get("temp_peak_c")),
            "thermal_throttle_share": _f(sustained.get("thermal_throttle_share")),
            "power_fault_share": _f(sustained.get("power_fault_share")),
            "power_cap_share": _f(sustained.get("power_cap_share")),
        },
        "driver_version": details.get("driver_version") or "",
        "cuda_version": details.get("cuda_version") or "",
    }


def host_evidence(
    cur, host_ids: list[str], *, now: float | None = None, history: int = 0
) -> dict[str, dict]:
    """Evidence for each host id; a host with no rows anywhere reads as unverified.

    ``cur`` is a dict-row cursor. ``history`` > 0 adds the dates (and verdicts)
    of up to that many earlier benchmark runs.
    """
    from reputation import score_to_tier
    from verification import verification_freshness

    ids = sorted({str(h) for h in host_ids if h})
    if not ids:
        return {}
    current = now if now is not None else time.time()
    since = current - SUCCESS_WINDOW_SEC

    verifications = {
        r["host_id"]: r
        for r in cur.execute(
            """SELECT host_id, state, verified_at, last_check_at, next_check_at
                 FROM host_verifications WHERE host_id = ANY(%s)""",
            (ids,),
        ).fetchall()
    }
    benches: dict[str, list[dict]] = {}
    for r in cur.execute(
        """SELECT * FROM (
               SELECT host_id, gpu_model, run_at, details, bench_version, verdict,
                      row_number() OVER (PARTITION BY host_id ORDER BY run_at DESC) AS n
                 FROM benchmarks
                WHERE host_id = ANY(%s) AND benchmark_type = 'xcelsior-bench') b
            WHERE n <= %s ORDER BY host_id, run_at DESC""",
        (ids, 1 + max(0, history)),
    ).fetchall():
        benches.setdefault(r["host_id"], []).append(r)
    reputation = {
        r["entity_id"]: r
        for r in cur.execute(
            """SELECT entity_id, tier, final_score FROM reputation_scores
                WHERE entity_id = ANY(%s)""",
            (ids,),
        ).fetchall()
    }
    ended = {
        r["host_id"]: int(r["n"])
        for r in cur.execute(
            """SELECT host_id, count(*) AS n FROM job_attempts
                WHERE host_id = ANY(%s) AND started_at IS NOT NULL
                  AND status = ANY(%s) AND ended_at >= to_timestamp(%s)
                GROUP BY host_id""",
            (ids, list(_ENDED), since),
        ).fetchall()
    }
    host_faults = {
        r["host_id"]: int(r["n"])
        for r in cur.execute(
            """SELECT host_id, count(DISTINCT job_id) AS n FROM job_failure_log
                WHERE host_id = ANY(%s) AND failed_at >= %s GROUP BY host_id""",
            (ids, since),
        ).fetchall()
    }

    out: dict[str, dict] = {}
    for host_id in ids:
        v = verifications.get(host_id) or {}
        state = str(v.get("state") or "unverified")
        freshness = verification_freshness(
            state, v.get("last_check_at"), v.get("next_check_at"), now=current
        )
        runs = benches.get(host_id) or []
        latest = runs[0] if runs else None
        n_ended = ended.get(host_id, 0)
        n_faults = min(host_faults.get(host_id, 0), n_ended) if n_ended else 0
        rep = reputation.get(host_id) or {}
        score = _f(rep.get("final_score"))
        evidence = {
            "host_id": host_id,
            "verification": {
                "state": state,
                "freshness": freshness,
                "overdue": freshness == "overdue",
                "last_verified_at": _f(v.get("verified_at")),
                "last_check_at": _f(v.get("last_check_at")),
                "next_check_at": _f(v.get("next_check_at")),
            },
            "benchmark": bench_summary(latest["details"], latest) if latest else None,
            # The tier is read from the score with the one ladder, so the two
            # cannot disagree; no row means no reputation yet, not a zero.
            "reputation": {
                "tier": score_to_tier(score).value if score is not None else None,
                "score": score,
            },
            "jobs": {
                "window_days": SUCCESS_WINDOW_SEC // 86400,
                "runs_ended": n_ended,
                "host_fault_failures": n_faults,
                "success_rate": round(1 - n_faults / n_ended, 4) if n_ended else None,
            },
        }
        if history:
            evidence["benchmark_history"] = [
                {
                    "ran_at": _f(r["run_at"]),
                    "bench_version": r["bench_version"],
                    "verdict": r["verdict"],
                }
                for r in runs[1:]
            ]
        out[host_id] = evidence
    return out


def evidence_badge(evidence: dict | None) -> dict:
    """The short form a listing carries: verified/overdue, last check, benchmark date."""
    if not evidence:
        return {
            "verification": "unverified",
            "last_check_at": None,
            "next_check_at": None,
            "benchmark": None,
            "reputation_tier": None,
            "reputation_score": None,
            "job_success_rate": None,
        }
    bench = evidence.get("benchmark") or None
    return {
        "verification": evidence["verification"]["freshness"],
        "last_check_at": evidence["verification"]["last_check_at"],
        "next_check_at": evidence["verification"]["next_check_at"],
        "benchmark": (
            {
                "ran_at": bench["ran_at"],
                "bench_version": bench["bench_version"],
                "verdict": bench["verdict"],
                "fp16_tflops": bench["fp16_tflops"],
                "memory_bandwidth_gbps": bench["memory_bandwidth_gbps"],
                "allocatable_vram_gb": bench["allocatable_vram_gb"],
                "temp_peak_c": bench["sustained"]["temp_peak_c"],
                "thermal_throttle_share": bench["sustained"]["thermal_throttle_share"],
            }
            if bench
            else None
        ),
        "reputation_tier": evidence["reputation"]["tier"],
        "reputation_score": evidence["reputation"]["score"],
        "job_success_rate": evidence["jobs"]["success_rate"],
    }


def evidence_summary(evidences: list[dict | None]) -> dict:
    """For a row that stands for several hosts: how many are verified, the newest benchmark."""
    present = [e for e in evidences if e]
    freshness = [e["verification"]["freshness"] for e in present]
    runs = [e["benchmark"]["ran_at"] for e in present if e.get("benchmark")]
    return {
        "hosts": len(evidences),
        "verified": freshness.count("verified"),
        "overdue": freshness.count("overdue"),
        "latest_benchmark_at": max((r for r in runs if r is not None), default=None),
    }


def attach_evidence(rows: list[dict]) -> list[dict]:
    """Give each listing row its host's evidence badge, and drop the deprecated fields.

    One read for the whole page. A database failure leaves every row badged
    ``unverified``, never verified.
    """
    evidence = evidence_for_hosts([str(r.get("host_id") or "") for r in rows])
    for row in rows:
        for name in DEPRECATED_OFFER_FIELDS:
            row.pop(name, None)
        row["evidence"] = evidence_badge(evidence.get(str(row.get("host_id") or "")))
    return rows


def reputation_score(row: dict) -> float | None:
    """The evidence reputation score on a badged row (None when the host has none)."""
    return (row.get("evidence") or {}).get("reputation_score")


def evidence_for_hosts(host_ids: list[str], *, history: int = 0) -> dict[str, dict]:
    """``host_evidence`` on a pooled connection; empty when the database is unreachable."""
    try:
        from psycopg.rows import dict_row

        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            return host_evidence(cur, host_ids, history=history)
    except Exception:
        import logging

        logging.getLogger("xcelsior").warning("host evidence read failed", exc_info=True)
        return {}
