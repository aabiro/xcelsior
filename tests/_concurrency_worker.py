"""Worker-process body for the §26.2 scheduler concurrency stress test.

Runs in a separate OS process (spawn): opens its own raw psycopg
connection (no shared pool across fork/spawn), then races the other
replicas claiming and reserving until the queue is exhausted. Domain
conflicts release the claim and move on; transient serialization errors
retry bounded — exactly the production replica loop shape.
"""

from __future__ import annotations

import random
import time

import psycopg

from control_plane.db import is_transient_error
from control_plane.scheduler.claim import claim_next_job, release_claim
from control_plane.scheduler.reservation import (
    ReservationConflict,
    reserve_and_bind,
)


def run_replica(dsn: str, replica_id: str, host_ids: list[str], marker: str) -> dict:
    """Claim+reserve until no work remains. Returns per-replica stats."""
    stats = {"reserved": 0, "conflicts": 0, "released": 0, "transient_retries": 0}
    rng = random.Random(replica_id)
    empty_polls = 0
    with psycopg.connect(dsn) as conn:
        while empty_polls < 3:
            claimed = None
            try:
                # Scoped to this test's jobs (gpu_model carries the marker)
                # so replicas never churn on shared-DB residue.
                claimed = claim_next_job(
                    conn, replica_id=replica_id,
                    scope_gpu_models=[f"stress-{marker}"],
                )
                conn.commit()
            except Exception as exc:
                conn.rollback()
                if is_transient_error(exc):
                    stats["transient_retries"] += 1
                    continue
                raise
            if claimed is None:
                empty_polls += 1
                time.sleep(0.02)
                continue
            empty_polls = 0
            # Only touch this test's own jobs; anything else in the shared
            # test DB gets its claim released untouched.
            if not claimed.job_id.startswith(f"job-{marker}"):
                release_claim(
                    conn, claimed.job_id, claimed.claim_token,
                    reason_code="foreign_test_job", requeue_delay_sec=30.0,
                )
                conn.commit()
                continue
            placed = False
            for host_id in rng.sample(host_ids, len(host_ids)):
                try:
                    reserve_and_bind(
                        conn,
                        job_id=claimed.job_id,
                        claim_token=claimed.claim_token,
                        replica_id=replica_id,
                        host_id=host_id,
                    )
                    conn.commit()
                    stats["reserved"] += 1
                    placed = True
                    break
                except ReservationConflict:
                    conn.rollback()
                    stats["conflicts"] += 1
                except Exception as exc:
                    conn.rollback()
                    if is_transient_error(exc):
                        stats["transient_retries"] += 1
                        continue
                    raise
            if not placed:
                # All candidates conflicted: durable reason + long backoff
                # so this replica set stops re-claiming a hopeless job.
                release_claim(
                    conn, claimed.job_id, claimed.claim_token,
                    reason_code="no_capacity", requeue_delay_sec=3600.0,
                )
                conn.commit()
                stats["released"] += 1
    return stats
