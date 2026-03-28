# Xcelsior — Inference Job Persistence (SQLite)
# Replaces in-memory dicts so inference jobs survive API restarts.

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from typing import Optional

_INFERENCE_DB_DIR = os.path.join(os.path.dirname(__file__), "data")
_INFERENCE_DB_PATH = os.path.join(_INFERENCE_DB_DIR, "inference.db")
INFERENCE_JOB_TTL_SEC = int(os.environ.get("XCELSIOR_INFERENCE_TTL", "86400"))  # 24h


def _ensure_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inference_jobs (
            job_id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            model TEXT NOT NULL,
            inputs TEXT NOT NULL,
            max_tokens INTEGER NOT NULL,
            temperature REAL NOT NULL,
            timeout_sec INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            submitted_at REAL NOT NULL,
            completed_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inference_results (
            job_id TEXT PRIMARY KEY,
            outputs TEXT NOT NULL,
            model TEXT NOT NULL,
            latency_ms REAL NOT NULL DEFAULT 0,
            completed_at REAL NOT NULL,
            FOREIGN KEY (job_id) REFERENCES inference_jobs(job_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inf_jobs_status "
        "ON inference_jobs(status, submitted_at ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_inf_jobs_submitted "
        "ON inference_jobs(submitted_at)"
    )


@contextmanager
def _inference_db():
    os.makedirs(_INFERENCE_DB_DIR, exist_ok=True)
    conn = sqlite3.connect(_INFERENCE_DB_PATH, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)
    try:
        yield conn
    finally:
        conn.close()


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
    with _inference_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO inference_jobs "
            "(job_id, customer_id, model, inputs, max_tokens, temperature, timeout_sec, status, submitted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
            (job_id, customer_id, model, json.dumps(inputs), max_tokens, temperature, timeout_sec, time.time()),
        )


def get_inference_job(job_id: str) -> Optional[dict]:
    """Return inference job metadata, or None if not found."""
    with _inference_db() as conn:
        row = conn.execute(
            "SELECT * FROM inference_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["inputs"] = json.loads(d["inputs"])
        return d


def store_inference_result(job_id: str, outputs: list, model: str, latency_ms: float):
    """Persist inference results from a worker callback."""
    now = time.time()
    with _inference_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO inference_results "
            "(job_id, outputs, model, latency_ms, completed_at) VALUES (?, ?, ?, ?, ?)",
            (job_id, json.dumps(outputs), model, latency_ms, now),
        )
        conn.execute(
            "UPDATE inference_jobs SET status = 'completed', completed_at = ? WHERE job_id = ?",
            (now, job_id),
        )


def get_inference_result(job_id: str) -> Optional[dict]:
    """Return inference results, or None if not yet completed."""
    with _inference_db() as conn:
        row = conn.execute(
            "SELECT * FROM inference_results WHERE job_id = ?", (job_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["outputs"] = json.loads(d["outputs"])
        return d


def delete_inference_job(job_id: str):
    """Remove an inference job and its results."""
    with _inference_db() as conn:
        conn.execute("DELETE FROM inference_results WHERE job_id = ?", (job_id,))
        conn.execute("DELETE FROM inference_jobs WHERE job_id = ?", (job_id,))


def purge_expired_jobs(ttl_sec: int = INFERENCE_JOB_TTL_SEC):
    """Delete inference jobs older than ttl_sec."""
    cutoff = time.time() - ttl_sec
    with _inference_db() as conn:
        expired = conn.execute(
            "SELECT job_id FROM inference_jobs WHERE submitted_at < ?", (cutoff,)
        ).fetchall()
        for row in expired:
            conn.execute("DELETE FROM inference_results WHERE job_id = ?", (row["job_id"],))
            conn.execute("DELETE FROM inference_jobs WHERE job_id = ?", (row["job_id"],))
        return len(expired)
