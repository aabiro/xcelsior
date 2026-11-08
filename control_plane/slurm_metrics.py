"""Slurm reconciler signals for Prometheus, from durable database state.

The reconciler runs in the background worker, so a counter in its process
would never reach the API's /metrics endpoint. Everything here is derived
from what it records instead: cluster health and ``failure_counts`` on
``slurm_clusters``, and the open rows of ``slurm_job_mappings``. One series
set per enabled cluster, so cardinality is the number of clusters, plus one
counter per stage and error code that cluster has actually hit.

The alerts in ``prometheus/native-control-plane-rules.yml`` read these:
reconcile lag past three poll intervals, a job waiting on accounting for
15 minutes (``ending_awaiting_accounting``), and any refused host key.
"""

from __future__ import annotations

import json
import time

_CLUSTERS = """
    SELECT c.cluster_id,
           c.poll_interval_seconds,
           EXTRACT(EPOCH FROM clock_timestamp() - COALESCE(c.controller_ok_at, c.created_at)),
           c.consecutive_failures,
           (c.breaker_open_until > clock_timestamp()),
           c.failure_counts,
           COALESCE(jsonb_array_length(c.facts->'orphans'), 0),
           count(m.xcelsior_job_id),
           count(m.xcelsior_job_id) FILTER (
               WHERE m.last_observed_at IS NULL
                  OR m.last_observed_at < clock_timestamp()
                     - make_interval(secs => 3 * c.poll_interval_seconds)),
           count(m.xcelsior_job_id) FILTER (WHERE m.left_controller_at IS NOT NULL),
           COALESCE(EXTRACT(EPOCH FROM clock_timestamp() - min(m.left_controller_at)), 0)
      FROM slurm_clusters c
      LEFT JOIN slurm_job_mappings m
        ON m.cluster_id = c.cluster_id AND m.terminal_at IS NULL
     WHERE c.enabled
     GROUP BY c.cluster_id
     ORDER BY c.cluster_id
"""


def collect(conn) -> list[dict]:
    clusters = []
    for row in conn.execute(_CLUSTERS).fetchall():
        cid, interval, lag, failures, breaker, counts, orphans, open_, stale, waiting, oldest = row
        if isinstance(counts, str):
            counts = json.loads(counts)
        clusters.append(
            {
                "cluster": cid,
                "poll_interval_seconds": int(interval),
                "reconcile_lag_seconds": round(float(lag or 0), 1),
                "consecutive_failures": int(failures),
                "breaker_open": int(bool(breaker)),
                "failures": {k: int(v) for k, v in (counts or {}).items()},
                "orphan_jobs": int(orphans),
                "open_jobs": int(open_),
                "stale_jobs": int(stale),
                "awaiting_accounting_jobs": int(waiting),
                "awaiting_accounting_oldest_seconds": round(float(oldest or 0), 1),
            }
        )
    return clusters


_GAUGES = (
    ("poll_interval_seconds", "Seconds between the reconciler's passes over the cluster"),
    (
        "reconcile_lag_seconds",
        "Seconds since a reconcile pass last got an answer from the cluster's controller",
    ),
    ("consecutive_failures", "Failed passes in a row"),
    ("breaker_open", "1 while the reconciler is backing off the cluster"),
    ("orphan_jobs", "Adapter-named jobs on the cluster that no open mapping accounts for"),
    ("open_jobs", "Mappings not yet settled"),
    ("stale_jobs", "Open mappings not confirmed for three poll intervals"),
    (
        "awaiting_accounting_jobs",
        "Open mappings whose job has left the controller without a finished accounting record",
    ),
    (
        "awaiting_accounting_oldest_seconds",
        "How long the oldest of those has waited on accounting (0 when none)",
    ),
)


def _label(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=True)


def render(clusters: list[dict] | None) -> list[str]:
    lines = [
        "# HELP xcelsior_slurm_metrics_available 1 when the Slurm reconciler's state was read",
        "# TYPE xcelsior_slurm_metrics_available gauge",
        f"xcelsior_slurm_metrics_available {int(clusters is not None)}",
    ]
    if clusters is None:
        return lines
    lines += [
        "# HELP xcelsior_slurm_metrics_last_success_timestamp_seconds Unix time of the last read",
        "# TYPE xcelsior_slurm_metrics_last_success_timestamp_seconds gauge",
        f"xcelsior_slurm_metrics_last_success_timestamp_seconds {time.time():.3f}",
    ]
    for key, help_text in _GAUGES:
        name = f"xcelsior_slurm_{key}"
        lines += [f"# HELP {name} {help_text}", f"# TYPE {name} gauge"]
        lines += [f"{name}{{cluster={_label(c['cluster'])}}} {c[key]}" for c in clusters]
    name = "xcelsior_slurm_failures_total"
    lines += [
        f"# HELP {name} Failed passes, accounting calls and output fetches, by stage and error code",
        f"# TYPE {name} counter",
    ]
    for c in clusters:
        for key, count in sorted(c["failures"].items()):
            stage, _, code = key.partition(":")
            lines.append(
                f"{name}{{cluster={_label(c['cluster'])},stage={_label(stage)},"
                f"error_code={_label(code)}}} {count}"
            )
    return lines
