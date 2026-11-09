"""Per-run billing for Slurm jobs: the ledger, its rules, and the drift report.

Billing time comes from the cluster's accounting (slurmdbd), never from the
reconciler's polling clock (plan D3). ``sacct -D -X`` returns one allocation
record per *run* of a job: a requeued or preempted job has several under one
Slurm id, and without ``-D`` only the latest shows, so the earlier runs were
never billed. When a mapping settles, every run of its job becomes one row of
``slurm_billing_segments`` (migration 158): the raw fields, the seconds it
bills under the written rules, and the rule's name, so a bill can be explained
record by record.

The rules (plan P2.4, §5a and decision D8), applied per record:

- **never_started**: no Start in accounting (cancelled or failed while
  pending) bills nothing.
- **ran_to_end**: COMPLETED, FAILED, OUT_OF_MEMORY, DEADLINE, and a CANCELLED
  job that had started, bill ElapsedRaw.
- **timeout_capped_to_limit**: a TIMEOUT run bills ``min(ElapsedRaw, time
  limit)``. Enforcement granularity plus ``KillWait`` ran a 60 s job for 78 s
  on the lab cluster; that lag is the cluster's, not the tenant's. The cap
  applies to ElapsedRaw only. Held suspension is added after it, so a timeout
  can bill more than its walltime by the seconds it still held its GPUs.
- **preempted_to_preemption**: a PREEMPTED run bills ElapsedRaw, which ends at
  the preemption.
- **requeued_run**: a REQUEUED, NODE_FAIL, BOOT_FAIL or similar earlier run
  bills its own ElapsedRaw; the run that follows is its own record.
- **suspended_gres_held / suspended_gres_released** (D8): ElapsedRaw excludes
  suspended time. A suspended job keeps its allocation unless the site's
  suspend policy releases GRES, so where ``suspend_holds_gres`` is true the
  suspended seconds are added; where false they are free. The rule name
  carries which applied.

``shadow_cost_micros`` is what the cluster's rate card would charge.
``mark_charged`` is the later settlement step and nothing calls it: every
row is stored with ``charged`` false. The ``slurm_placement`` switch does
not turn charging on. Charging is Phase 2.4 and is not built.

The drift report compares the ledger's raw GPU seconds for a day with what the
cluster's own accounting says for the same day, per cluster, so a week of
shadow mode can show they agree (blueprint 2.3: under 1 %).
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

import slurm_adapter as sa

log = logging.getLogger("xcelsior-slurm-billing")

RULE_NEVER_STARTED = "never_started"
RULE_RAN_TO_END = "ran_to_end"
RULE_TIMEOUT_CAPPED = "timeout_capped_to_limit"
RULE_PREEMPTED = "preempted_to_preemption"
RULE_REQUEUED_RUN = "requeued_run"
RULE_SUSPENDED_HELD = "suspended_gres_held"
RULE_SUSPENDED_FREE = "suspended_gres_released"

#: Records that end a run Slurm then starts again under the same id.
_REQUEUE_STATES = frozenset(
    {"REQUEUED", "NODE_FAIL", "BOOT_FAIL", "LAUNCH_FAILED", "RECONFIG_FAIL"}
)

#: Drift over this is an alert (blueprint 2.3).
DRIFT_ALERT_PCT = 1.0

_BILLING_TRES_RE = re.compile(r"(?:^|,)billing=(\d+)")


def segment_charge(record: dict, *, suspend_holds_gres: bool = True) -> tuple[int, str]:
    """(billable seconds, rule) for one ``sacct -D -X`` record.

    ``record`` is one entry of ``slurm_adapter.accounting_window``: ``state``
    (normalised), ``started_at``, ``elapsed_sec``, ``suspended_sec`` and
    ``time_limit_sec``.
    """
    state = str(record.get("state") or "").upper()
    elapsed = max(0, int(record.get("elapsed_sec") or 0))
    suspended = max(0, int(record.get("suspended_sec") or 0))
    limit = record.get("time_limit_sec")
    if record.get("started_at") is None:
        return 0, RULE_NEVER_STARTED
    if state == "TIMEOUT":
        billable = min(elapsed, int(limit)) if limit else elapsed
        rule = RULE_TIMEOUT_CAPPED
    elif state == "PREEMPTED":
        billable, rule = elapsed, RULE_PREEMPTED
    elif state in _REQUEUE_STATES:
        billable, rule = elapsed, RULE_REQUEUED_RUN
    else:
        billable, rule = elapsed, RULE_RAN_TO_END
    if suspended:
        if suspend_holds_gres:
            billable += suspended
            rule += "+" + RULE_SUSPENDED_HELD
        else:
            rule += "+" + RULE_SUSPENDED_FREE
    return billable, rule


def rate_card(cluster: dict | None) -> dict:
    """The cluster's rate card as a dict of ints: ``gpu_hour_micros``,
    ``cpu_hour_micros`` and the ``gpu_model`` it bills as (a string)."""
    raw = (cluster or {}).get("rate_card") or {}
    if not isinstance(raw, dict):
        return {}
    card: dict[str, Any] = {}
    for key in ("gpu_hour_micros", "cpu_hour_micros"):
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        card[key] = value
    if raw.get("gpu_model"):
        card["gpu_model"] = str(raw["gpu_model"])
    return card


def segment_cost_micros(card: dict, *, billable_sec: int, gpu_count: int, cpu_count: int):
    """What the rate card would charge for one segment, or None without a card."""
    if not card:
        return None
    cost = 0
    if gpu_count and card.get("gpu_hour_micros") is not None:
        cost += round(billable_sec * gpu_count * card["gpu_hour_micros"] / 3600)
    elif not gpu_count and card.get("cpu_hour_micros") is not None:
        cost += round(billable_sec * max(1, cpu_count) * card["cpu_hour_micros"] / 3600)
    elif gpu_count:
        return None  # a GPU run on a cluster that prices no GPU hour
    return cost


def _epoch_ts(value) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.fromtimestamp(float(value), tz=dt.timezone.utc)


def build_segments(mapping: dict, runs: list[dict], cluster: dict | None) -> list[dict]:
    """The ledger rows for a settled mapping, from its job's accounting runs."""
    card = rate_card(cluster)
    holds = bool((cluster or {}).get("suspend_holds_gres", True))
    rows = []
    for index, run in enumerate(runs):
        tres = sa.parse_alloc_tres(run.get("alloc_tres") or "")
        billable, rule = segment_charge(run, suspend_holds_gres=holds)
        gpu_seconds = billable * tres["gpu"]
        billing_match = _BILLING_TRES_RE.search(run.get("alloc_tres") or "")
        rows.append(
            {
                "mapping_id": mapping["mapping_id"],
                "run_index": index,
                "cluster_id": mapping["cluster_id"],
                "slurm_job_id": str(mapping["slurm_job_id"]),
                "xcelsior_job_id": mapping["xcelsior_job_id"],
                "attempt_id": mapping.get("xcelsior_attempt_id"),
                "tenant_id": mapping.get("tenant_id"),
                "slurm_state": run["state"],
                "submitted_at": _epoch_ts(run.get("submitted_at")),
                "started_at": _epoch_ts(run.get("started_at")),
                "ended_at": _epoch_ts(run.get("ended_at")),
                "elapsed_sec": max(0, int(run.get("elapsed_sec") or 0)),
                "suspended_sec": max(0, int(run.get("suspended_sec") or 0)),
                "time_limit_sec": run.get("time_limit_sec"),
                "gpu_count": tres["gpu"],
                "cpu_count": tres["cpu"],
                "node_count": tres["node"],
                "billing_tres": int(billing_match.group(1)) if billing_match else None,
                "alloc_tres": run.get("alloc_tres"),
                "billable_sec": billable,
                "billing_rule": rule,
                "gpu_seconds": gpu_seconds,
                "rate_micros_per_gpu_hour": card.get("gpu_hour_micros"),
                "shadow_cost_micros": segment_cost_micros(
                    card, billable_sec=billable, gpu_count=tres["gpu"], cpu_count=tres["cpu"]
                ),
                "raw": {
                    k: run.get(k)
                    for k in (
                        "name",
                        "raw_state",
                        "submitted_at",
                        "started_at",
                        "ended_at",
                        "elapsed_sec",
                        "suspended_sec",
                        "time_limit_sec",
                        "exit_code",
                        "derived_exit_code",
                        "alloc_tres",
                        "node_list",
                    )
                },
            }
        )
    return rows


def write_segments(conn, mapping: dict, runs: list[dict], cluster: dict | None) -> int:
    """Insert the ledger rows for a settled mapping; existing runs are kept.

    Called in the transaction that settles the mapping, so the evidence a
    settlement bills from is durable with the settlement itself.
    """
    from psycopg.types.json import Jsonb

    written = 0
    for row in build_segments(mapping, runs, cluster):
        cur = conn.execute(
            """
            INSERT INTO slurm_billing_segments
                (mapping_id, run_index, cluster_id, slurm_job_id, xcelsior_job_id,
                 attempt_id, tenant_id, slurm_state, submitted_at, started_at, ended_at,
                 elapsed_sec, suspended_sec, time_limit_sec, gpu_count, cpu_count,
                 node_count, billing_tres, alloc_tres, billable_sec, billing_rule,
                 gpu_seconds, rate_micros_per_gpu_hour, shadow_cost_micros, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (mapping_id, run_index) DO NOTHING
            """,
            (
                row["mapping_id"],
                row["run_index"],
                row["cluster_id"],
                row["slurm_job_id"],
                row["xcelsior_job_id"],
                row["attempt_id"],
                row["tenant_id"],
                row["slurm_state"],
                row["submitted_at"],
                row["started_at"],
                row["ended_at"],
                row["elapsed_sec"],
                row["suspended_sec"],
                row["time_limit_sec"],
                row["gpu_count"],
                row["cpu_count"],
                row["node_count"],
                row["billing_tres"],
                row["alloc_tres"],
                row["billable_sec"],
                row["billing_rule"],
                row["gpu_seconds"],
                row["rate_micros_per_gpu_hour"],
                row["shadow_cost_micros"],
                Jsonb(row["raw"]),
            ),
        )
        written += cur.rowcount
    return written


def attempt_totals(conn, attempt_id: str) -> dict | None:
    """The ledger's totals for one attempt: billable and GPU seconds, cost, and
    the run count. None when the ledger has nothing for it yet."""
    row = conn.execute(
        """
        SELECT count(*), COALESCE(sum(billable_sec), 0), COALESCE(sum(gpu_seconds), 0),
               sum(shadow_cost_micros), max(gpu_count),
               EXTRACT(EPOCH FROM min(started_at)), EXTRACT(EPOCH FROM max(ended_at))
          FROM slurm_billing_segments WHERE attempt_id = %s
        """,
        (attempt_id,),
    ).fetchone()
    if row is None or not row[0]:
        return None
    return {
        "runs": int(row[0]),
        "billable_sec": int(row[1]),
        "gpu_seconds": int(row[2]),
        "shadow_cost_micros": None if row[3] is None else int(row[3]),
        "gpu_count": int(row[4] or 0),
        "started_at": None if row[5] is None else float(row[5]),
        "ended_at": None if row[6] is None else float(row[6]),
    }


def mark_charged(conn, attempt_id: str, meter_id: str) -> int:
    cur = conn.execute(
        "UPDATE slurm_billing_segments SET charged = TRUE, meter_id = %s "
        "WHERE attempt_id = %s AND NOT charged",
        (meter_id, attempt_id),
    )
    return cur.rowcount


# ── Drift report ─────────────────────────────────────────────────────


def day_window(day: dt.date, tz_name: str) -> tuple[float, float]:
    """[start, end) of a calendar day in the cluster's timezone, as epochs."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(tz_name or "UTC")
    start = dt.datetime(day.year, day.month, day.day, tzinfo=tz)
    return start.timestamp(), (start + dt.timedelta(days=1)).timestamp()


def compare_day(records: dict, segments: list[tuple], *, start: float, end: float) -> dict:
    """Accounting's records against the ledger's rows for one window.

    ``records`` is ``accounting_between``'s ``records`` (every run of every
    job in the window, keyed by Slurm id); ``segments`` are the ledger rows
    ``(slurm_job_id, submitted_at epoch, elapsed_sec, gpu_count)`` whose
    ``ended_at`` falls in the window. Both sides are keyed by (Slurm id,
    Submit), the key ``sacct -D`` itself distinguishes runs by. Only runs that
    ended in the window and carry the adapter's name prefix count.
    """
    acct: dict[tuple[str, int], int] = {}
    for sid, runs in records.items():
        for run in runs:
            ended = run.get("ended_at")
            if ended is None or not start <= float(ended) < end:
                continue
            if not str(run.get("name") or "").startswith("xcl-"):
                continue
            gpus = sa.parse_alloc_tres(run.get("alloc_tres") or "")["gpu"]
            key = (str(sid), int(run.get("submitted_at") or 0))
            acct[key] = acct.get(key, 0) + int(run.get("elapsed_sec") or 0) * gpus
    ledger: dict[tuple[str, int], int] = {}
    for sid, submitted, elapsed, gpus in segments:
        key = (str(sid), int(submitted or 0))
        ledger[key] = ledger.get(key, 0) + int(elapsed) * int(gpus)
    acct_total = sum(acct.values())
    ledger_total = sum(ledger.values())
    if acct_total == 0:
        drift = 0.0 if ledger_total == 0 else 100.0
    else:
        drift = abs(ledger_total - acct_total) / acct_total * 100.0
    return {
        "sacct_gpu_seconds": acct_total,
        "ledger_gpu_seconds": ledger_total,
        "sacct_records": len(acct),
        "ledger_records": len(ledger),
        "drift_pct": round(drift, 4),
        "unmatched": {
            "only_in_sacct": sorted(f"{k[0]}@{k[1]}" for k in acct.keys() - ledger.keys()),
            "only_in_ledger": sorted(f"{k[0]}@{k[1]}" for k in ledger.keys() - acct.keys()),
            "differing": sorted(
                f"{k[0]}@{k[1]}" for k in acct.keys() & ledger.keys() if acct[k] != ledger[k]
            ),
        },
    }


def _ledger_rows(conn, cluster_id: str, start: float, end: float) -> list[tuple]:
    return conn.execute(
        """
        SELECT slurm_job_id, EXTRACT(EPOCH FROM submitted_at), elapsed_sec, gpu_count
          FROM slurm_billing_segments
         WHERE cluster_id = %s AND ended_at >= to_timestamp(%s) AND ended_at < to_timestamp(%s)
        """,
        (cluster_id, start, end),
    ).fetchall()


def report_day(cluster: dict, day: dt.date, *, runner=None) -> dict:
    """Compare one day of the ledger with the cluster's accounting and record it.

    One ``sacct -D -X -S <day start - 1 d> -E <day end + 1 h>`` through the
    gate: the window is a superset, and membership is decided by each run's
    End. Returns the comparison, or an error dict when accounting didn't
    answer (nothing is recorded then).
    """
    from psycopg.types.json import Jsonb

    from db import pg_connection, pg_transaction

    profile = sa._get_profile(cluster["profile_key"])
    tz_name = str(profile.get("timezone") or "UTC")
    start, end = day_window(day, tz_name)
    acct = sa.accounting_between(
        start - 86400, end + 3600, profile_name=cluster["profile_key"], runner=runner
    )
    if "error" in acct:
        return acct
    with pg_connection() as conn:
        segments = _ledger_rows(conn, cluster["cluster_id"], start, end)
    result = compare_day(acct["records"], segments, start=start, end=end)
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO slurm_billing_drift
                (cluster_id, day, sacct_gpu_seconds, ledger_gpu_seconds, sacct_records,
                 ledger_records, drift_pct, unmatched, checked_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, clock_timestamp())
            ON CONFLICT (cluster_id, day) DO UPDATE
               SET sacct_gpu_seconds = EXCLUDED.sacct_gpu_seconds,
                   ledger_gpu_seconds = EXCLUDED.ledger_gpu_seconds,
                   sacct_records = EXCLUDED.sacct_records,
                   ledger_records = EXCLUDED.ledger_records,
                   drift_pct = EXCLUDED.drift_pct,
                   unmatched = EXCLUDED.unmatched,
                   checked_at = clock_timestamp()
            """,
            (
                cluster["cluster_id"],
                day,
                result["sacct_gpu_seconds"],
                result["ledger_gpu_seconds"],
                result["sacct_records"],
                result["ledger_records"],
                result["drift_pct"],
                Jsonb(result["unmatched"]),
            ),
        )
    if result["drift_pct"] > DRIFT_ALERT_PCT:
        log.warning(
            "slurm billing drift %s %s: ledger %d vs sacct %d GPU-seconds (%.2f%%)",
            cluster["cluster_id"],
            day,
            result["ledger_gpu_seconds"],
            result["sacct_gpu_seconds"],
            result["drift_pct"],
        )
    return {"cluster_id": cluster["cluster_id"], "day": day.isoformat(), **result}


def report_due_clusters(today: dt.date | None = None) -> list[dict]:
    """The scheduled task: yesterday's report for every enabled cluster that
    has none yet (so a day is compared once, after accounting has had the
    night to catch up)."""
    from psycopg.rows import dict_row

    import slurm_reconciler as rc
    from db import pg_connection

    with pg_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        clusters = cur.execute("""
            SELECT c.* FROM slurm_clusters c
             WHERE c.enabled
               AND (c.breaker_open_until IS NULL OR c.breaker_open_until <= clock_timestamp())
             ORDER BY c.cluster_id
            """).fetchall()
    results = []
    for cluster in clusters:
        profile = sa._get_profile(cluster["profile_key"])
        tz_name = str(profile.get("timezone") or "UTC")
        from zoneinfo import ZoneInfo

        day = (today or dt.datetime.now(ZoneInfo(tz_name or "UTC")).date()) - dt.timedelta(days=1)
        with pg_connection() as conn:
            done = conn.execute(
                "SELECT 1 FROM slurm_billing_drift WHERE cluster_id = %s AND day = %s",
                (cluster["cluster_id"], day),
            ).fetchone()
        if done:
            continue
        try:
            with rc._runner_for(cluster) as runner:
                results.append(report_day(cluster, day, runner=runner))
        except sa.SlurmAdapterError as e:
            results.append(
                {"cluster_id": cluster["cluster_id"], "error_code": e.code, "error": str(e)}
            )
    return results
