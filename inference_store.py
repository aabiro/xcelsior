# Xcelsior — Inference Job Persistence (PostgreSQL)
# Replaces in-memory dicts so inference jobs survive API restarts.

import json
import os
import time
from contextlib import contextmanager
from typing import Optional

INFERENCE_JOB_TTL_SEC = int(os.environ.get("XCELSIOR_INFERENCE_TTL", "86400"))  # 24h


@contextmanager
def _inference_db():
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def store_inference_job(
    job_id: str,
    customer_id: str,
    model: str,
    inputs: list,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
):
    """Persist an inference job submission."""
    from psycopg.types.json import Jsonb
    with _inference_db() as conn:
        conn.execute(
            "INSERT INTO inference_jobs "
            "(job_id, customer_id, model, inputs, max_tokens, temperature, timeout_sec, status, submitted_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', %s) "
            "ON CONFLICT (job_id) DO UPDATE SET "
            "customer_id=EXCLUDED.customer_id, model=EXCLUDED.model, inputs=EXCLUDED.inputs, "
            "max_tokens=EXCLUDED.max_tokens, temperature=EXCLUDED.temperature, "
            "timeout_sec=EXCLUDED.timeout_sec, status='queued', submitted_at=EXCLUDED.submitted_at",
            (job_id, customer_id, model, Jsonb(inputs), max_tokens, temperature, timeout_sec, time.time()),
        )


def get_inference_job(job_id: str) -> Optional[dict]:
    """Return inference job metadata, or None if not found."""
    with _inference_db() as conn:
        row = conn.execute(
            "SELECT * FROM inference_jobs WHERE job_id = %s", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d["inputs"], str):
            d["inputs"] = json.loads(d["inputs"])
        return d


def store_inference_result(job_id: str, outputs: list, model: str, latency_ms: float):
    """Persist inference results from a worker callback."""
    from psycopg.types.json import Jsonb
    now = time.time()
    with _inference_db() as conn:
        conn.execute(
            "INSERT INTO inference_results "
            "(job_id, outputs, model, latency_ms, completed_at) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (job_id) DO UPDATE SET "
            "outputs=EXCLUDED.outputs, model=EXCLUDED.model, latency_ms=EXCLUDED.latency_ms, completed_at=EXCLUDED.completed_at",
            (job_id, Jsonb(outputs), model, latency_ms, now),
        )
        conn.execute(
            "UPDATE inference_jobs SET status = 'completed', completed_at = %s WHERE job_id = %s",
            (now, job_id),
        )


def get_inference_result(job_id: str) -> Optional[dict]:
    """Return inference results, or None if not yet completed."""
    with _inference_db() as conn:
        row = conn.execute(
            "SELECT * FROM inference_results WHERE job_id = %s", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        if isinstance(d["outputs"], str):
            d["outputs"] = json.loads(d["outputs"])
        return d


def delete_inference_job(job_id: str):
    """Remove an inference job and its results."""
    with _inference_db() as conn:
        conn.execute("DELETE FROM inference_results WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM inference_jobs WHERE job_id = %s", (job_id,))


def purge_expired_jobs(ttl_sec: int = INFERENCE_JOB_TTL_SEC):
    """Delete inference jobs older than ttl_sec."""
    cutoff = time.time() - ttl_sec
    with _inference_db() as conn:
        expired = conn.execute(
            "SELECT job_id FROM inference_jobs WHERE submitted_at < %s", (cutoff,)
        ).fetchall()
        for row in expired:
            conn.execute("DELETE FROM inference_results WHERE job_id = %s", (row["job_id"],))
            conn.execute("DELETE FROM inference_jobs WHERE job_id = %s", (row["job_id"],))
        return len(expired)
