# Xcelsior Database Abstraction Layer
# Supports SQLite (dev/default) and PostgreSQL (production).
# Implements expand/contract dual-write migration pattern.
#
# Architecture follows REPORT_EXCELSIOR_TECHNICAL_FINAL.md:
#   - psycopg3 for Postgres (native SQLAlchemy 2.0+ integration)
#   - JSONB + GIN indexing for flexible payloads
#   - Dual-write for zero-downtime migration
#   - Feature-flagged read/write switching

# Auto-load .env file (must be before any os.environ reads)
from dotenv import load_dotenv

load_dotenv()

import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("xcelsior")

# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_DB_FILE = os.path.join(os.path.dirname(__file__), "xcelsior.db")

# Database backend: "sqlite", "postgres", or "dual" (dual-write migration mode)
DB_BACKEND = os.environ.get("XCELSIOR_DB_BACKEND", "postgres").lower()

# PostgreSQL connection string (used when backend is "postgres" or "dual")
POSTGRES_DSN = os.environ.get(
    "XCELSIOR_POSTGRES_DSN",
    os.environ.get(
        "DATABASE_URL",
        "postgresql://xcelsior:xcelsior@localhost:5432/xcelsior",
    ),
)

# During dual-write, which DB to read from: "sqlite" or "postgres"
DUAL_READ_FROM = os.environ.get("XCELSIOR_DUAL_READ_FROM", "sqlite").lower()

# Connection pool settings (Postgres)
PG_POOL_SIZE = int(os.environ.get("XCELSIOR_PG_POOL_SIZE", "20"))
PG_MAX_OVERFLOW = int(os.environ.get("XCELSIOR_PG_MAX_OVERFLOW", "10"))


def _db_path():
    return os.environ.get("XCELSIOR_DB_PATH", DEFAULT_DB_FILE)


# ── SQLite Backend ────────────────────────────────────────────────────


def _ensure_sqlite_tables(conn):
    """Ensure all scheduler persistence tables and indexes exist in SQLite."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS state " "(namespace TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL,
            submitted_at REAL NOT NULL,
            host_id TEXT,
            payload TEXT NOT NULL
        )
        """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS hosts (
            host_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            registered_at REAL NOT NULL,
            payload TEXT NOT NULL
        )
        """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_queue "
        "ON jobs(status, priority DESC, submitted_at ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_status " "ON hosts(status, registered_at ASC)"
    )


@contextmanager
def sqlite_connection():
    """SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(_db_path(), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_sqlite_tables(conn)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def sqlite_transaction():
    """Execute a mutation in a single SQLite write transaction."""
    with sqlite_connection() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise


# ── PostgreSQL Backend ────────────────────────────────────────────────

_pg_pool = None
_pg_pool_lock = threading.Lock()


def _get_pg_pool():
    """Lazy-initialize PostgreSQL connection pool using psycopg3.

    Retries with exponential backoff if PG is temporarily unavailable
    (up to 5 attempts: 1s, 2s, 4s, 8s, 16s).
    """
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    with _pg_pool_lock:
        if _pg_pool is not None:
            return _pg_pool

        try:
            from psycopg_pool import ConnectionPool
        except ImportError:
            log.error(
                "psycopg or psycopg_pool not installed. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            )
            raise

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            try:
                pool = ConnectionPool(
                    POSTGRES_DSN,
                    min_size=2,
                    max_size=PG_POOL_SIZE,
                    max_idle=PG_MAX_OVERFLOW,
                    kwargs={"autocommit": False},
                )
                # Verify the pool can actually connect
                with pool.connection() as conn:
                    _ensure_pg_tables(conn)
                    conn.commit()

                _pg_pool = pool
                log.info(
                    "PostgreSQL connection pool initialized (size=%d, max_idle=%d)",
                    PG_POOL_SIZE,
                    PG_MAX_OVERFLOW,
                )
                return _pg_pool
            except Exception as e:
                if attempt == max_attempts:
                    log.error(
                        "Failed to connect to PostgreSQL after %d attempts: %s",
                        max_attempts,
                        e,
                    )
                    raise
                delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s
                log.warning(
                    "PostgreSQL connection attempt %d/%d failed: %s — retrying in %ds",
                    attempt,
                    max_attempts,
                    e,
                    delay,
                )
                time.sleep(delay)


# ── GPU Pricing seed data ─────────────────────────────────────────────
# Platform-controlled GPU rates in CAD. Volume costs are computed at
# runtime in billing.py and are NOT stored here.

_GPU_PRICING_SEED = [
    # (gpu_model, vram_gb, tier, pricing_mode, base_rate_cad, priority_mult, sovereignty_prem, spot_disc, mg4_disc, mg8_disc)
    # ── RTX 3090 24GB ──
    ("RTX 3090",  24, "standard",  "on_demand",    0.30, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "standard",  "spot",         0.12, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "standard",  "reserved_1mo", 0.24, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "standard",  "reserved_1yr", 0.17, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "premium",   "on_demand",    0.39, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "premium",   "spot",         0.156,1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "premium",   "reserved_1mo", 0.312,1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "premium",   "reserved_1yr", 0.221,1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 3090",  24, "sovereign", "on_demand",    0.429,1.0, 0.10, 0.0,  0.05, 0.10),
    ("RTX 3090",  24, "sovereign", "spot",         0.172,1.0, 0.10, 0.0,  0.05, 0.10),
    ("RTX 3090",  24, "sovereign", "reserved_1mo", 0.343,1.0, 0.10, 0.0,  0.05, 0.10),
    ("RTX 3090",  24, "sovereign", "reserved_1yr", 0.243,1.0, 0.10, 0.0,  0.05, 0.10),
    # ── RTX 4090 24GB ──
    ("RTX 4090",  24, "standard",  "on_demand",    0.55, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "standard",  "spot",         0.22, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "standard",  "reserved_1mo", 0.44, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "standard",  "reserved_1yr", 0.30, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "premium",   "on_demand",    0.715,1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "premium",   "spot",         0.286,1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "premium",   "reserved_1mo", 0.572,1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "premium",   "reserved_1yr", 0.39, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("RTX 4090",  24, "sovereign", "on_demand",    0.787,1.0, 0.10, 0.0,  0.05, 0.10),
    ("RTX 4090",  24, "sovereign", "spot",         0.315,1.0, 0.10, 0.0,  0.05, 0.10),
    ("RTX 4090",  24, "sovereign", "reserved_1mo", 0.629,1.0, 0.10, 0.0,  0.05, 0.10),
    ("RTX 4090",  24, "sovereign", "reserved_1yr", 0.429,1.0, 0.10, 0.0,  0.05, 0.10),
    # ── A100 40GB ──
    ("A100 40GB", 40, "standard",  "on_demand",    1.50, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "standard",  "spot",         0.60, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "standard",  "reserved_1mo", 1.20, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "standard",  "reserved_1yr", 0.83, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "premium",   "on_demand",    1.95, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "premium",   "spot",         0.78, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "premium",   "reserved_1mo", 1.56, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "premium",   "reserved_1yr", 1.079,1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 40GB", 40, "sovereign", "on_demand",    2.145,1.0, 0.10, 0.0,  0.05, 0.10),
    ("A100 40GB", 40, "sovereign", "spot",         0.858,1.0, 0.10, 0.0,  0.05, 0.10),
    ("A100 40GB", 40, "sovereign", "reserved_1mo", 1.716,1.0, 0.10, 0.0,  0.05, 0.10),
    ("A100 40GB", 40, "sovereign", "reserved_1yr", 1.187,1.0, 0.10, 0.0,  0.05, 0.10),
    # ── A100 80GB ──
    ("A100 80GB", 80, "standard",  "on_demand",    2.20, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "standard",  "spot",         0.88, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "standard",  "reserved_1mo", 1.76, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "standard",  "reserved_1yr", 1.21, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "premium",   "on_demand",    2.86, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "premium",   "spot",         1.144,1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "premium",   "reserved_1mo", 2.288,1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "premium",   "reserved_1yr", 1.573,1.0, 0.0,  0.0,  0.05, 0.10),
    ("A100 80GB", 80, "sovereign", "on_demand",    3.146,1.0, 0.10, 0.0,  0.05, 0.10),
    ("A100 80GB", 80, "sovereign", "spot",         1.258,1.0, 0.10, 0.0,  0.05, 0.10),
    ("A100 80GB", 80, "sovereign", "reserved_1mo", 2.517,1.0, 0.10, 0.0,  0.05, 0.10),
    ("A100 80GB", 80, "sovereign", "reserved_1yr", 1.730,1.0, 0.10, 0.0,  0.05, 0.10),
    # ── H100 80GB ──
    ("H100 80GB", 80, "standard",  "on_demand",    3.50, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "standard",  "spot",         1.40, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "standard",  "reserved_1mo", 2.80, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "standard",  "reserved_1yr", 1.93, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "premium",   "on_demand",    4.55, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "premium",   "spot",         1.82, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "premium",   "reserved_1mo", 3.64, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "premium",   "reserved_1yr", 2.509,1.0, 0.0,  0.0,  0.05, 0.10),
    ("H100 80GB", 80, "sovereign", "on_demand",    5.005,1.0, 0.10, 0.0,  0.05, 0.10),
    ("H100 80GB", 80, "sovereign", "spot",         2.002,1.0, 0.10, 0.0,  0.05, 0.10),
    ("H100 80GB", 80, "sovereign", "reserved_1mo", 4.004,1.0, 0.10, 0.0,  0.05, 0.10),
    ("H100 80GB", 80, "sovereign", "reserved_1yr", 2.760,1.0, 0.10, 0.0,  0.05, 0.10),
    # ── L40S 48GB ──
    ("L40S 48GB", 48, "standard",  "on_demand",    1.80, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "standard",  "spot",         0.72, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "standard",  "reserved_1mo", 1.44, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "standard",  "reserved_1yr", 0.99, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "premium",   "on_demand",    2.34, 1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "premium",   "spot",         0.936,1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "premium",   "reserved_1mo", 1.872,1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "premium",   "reserved_1yr", 1.287,1.0, 0.0,  0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "sovereign", "on_demand",    2.574,1.0, 0.10, 0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "sovereign", "spot",         1.030,1.0, 0.10, 0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "sovereign", "reserved_1mo", 2.059,1.0, 0.10, 0.0,  0.05, 0.10),
    ("L40S 48GB", 48, "sovereign", "reserved_1yr", 1.416,1.0, 0.10, 0.0,  0.05, 0.10),
]

# Priority multipliers applied at query time, not stored per-row
GPU_PRIORITY_MULTIPLIERS = {
    "low":      0.8,
    "normal":   1.0,
    "high":     1.3,
    "critical": 1.6,
}


def _seed_gpu_pricing(cur):
    """Insert seed pricing rows if table is empty."""
    cur.execute("SELECT COUNT(*) FROM gpu_pricing")
    count = cur.fetchone()[0]
    if count > 0:
        return
    for row in _GPU_PRICING_SEED:
        cur.execute(
            """INSERT INTO gpu_pricing
               (gpu_model, vram_gb, tier, pricing_mode, base_rate_cad,
                priority_multiplier, sovereignty_premium, spot_discount,
                multi_gpu_discount_4, multi_gpu_discount_8)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (gpu_model, tier, pricing_mode) DO NOTHING""",
            row,
        )
    log.info("Seeded gpu_pricing table with %d rows", len(_GPU_PRICING_SEED))


def _ensure_pg_tables(conn):
    """Ensure all scheduler persistence tables and indexes exist in PostgreSQL.

    Uses JSONB for payload columns with GIN indexes for query-critical fields.
    """
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS state (
            namespace TEXT PRIMARY KEY,
            payload JSONB NOT NULL
        )
        """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL,
            submitted_at DOUBLE PRECISION NOT NULL,
            host_id TEXT,
            payload JSONB NOT NULL
        )
        """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hosts (
            host_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            registered_at DOUBLE PRECISION NOT NULL,
            payload JSONB NOT NULL
        )
        """)

    # Indexed columns for scheduler queries
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_queue "
        "ON jobs(status, priority DESC, submitted_at ASC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_status " "ON hosts(status, registered_at ASC)"
    )

    # GIN indexes on JSONB payloads for GPU capability queries
    # Allows: SELECT * FROM hosts WHERE payload->'gpu_specs' @> '{"vram_gb": 24}'
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hosts_payload_gin " "ON hosts USING GIN (payload)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_payload_gin " "ON jobs USING GIN (payload)")

    # Expression indexes for hot-path queries
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_gpu_model " "ON hosts ((payload->>'gpu_model'))"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_tier " "ON jobs ((payload->>'tier'))")

    # ── Job logs (persistent container output) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS job_logs (
            id BIGSERIAL PRIMARY KEY,
            job_id TEXT NOT NULL,
            ts DOUBLE PRECISION NOT NULL,
            level TEXT NOT NULL DEFAULT 'info',
            line TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL DEFAULT EXTRACT(EPOCH FROM NOW())
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_logs_job_ts "
        "ON job_logs (job_id, ts)"
    )

    # ── Billing cycles (charge records for running instances) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS billing_cycles (
            cycle_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            customer_id TEXT NOT NULL,
            host_id TEXT DEFAULT '',
            resource_type TEXT NOT NULL DEFAULT 'gpu',
            period_start DOUBLE PRECISION NOT NULL,
            period_end DOUBLE PRECISION NOT NULL,
            duration_seconds DOUBLE PRECISION NOT NULL,
            rate_per_hour DOUBLE PRECISION NOT NULL,
            gpu_model TEXT DEFAULT '',
            tier TEXT DEFAULT 'free',
            tier_multiplier DOUBLE PRECISION DEFAULT 1.0,
            amount_cad DOUBLE PRECISION NOT NULL,
            status TEXT DEFAULT 'charged',
            created_at DOUBLE PRECISION NOT NULL
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_cycles_job "
        "ON billing_cycles (job_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_cycles_customer "
        "ON billing_cycles (customer_id)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_billing_cycles_created_at "
        "ON billing_cycles (created_at)"
    )

    # ── Persistent volumes ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS volumes (
            volume_id TEXT PRIMARY KEY,
            owner_id TEXT NOT NULL,
            name TEXT NOT NULL,
            storage_type TEXT DEFAULT 'nfs',
            size_gb INTEGER NOT NULL DEFAULT 50,
            region TEXT DEFAULT '',
            province TEXT DEFAULT '',
            encrypted BOOLEAN DEFAULT TRUE,
            status TEXT NOT NULL DEFAULT 'provisioning',
            encryption_key_id TEXT DEFAULT '',
            key_ciphertext TEXT DEFAULT '',
            mount_path_host TEXT DEFAULT '',
            created_at DOUBLE PRECISION NOT NULL,
            deleted_at DOUBLE PRECISION DEFAULT 0
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_volumes_owner "
        "ON volumes (owner_id, status)"
    )

    # ── Volume attachments ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS volume_attachments (
            attachment_id TEXT PRIMARY KEY,
            volume_id TEXT NOT NULL REFERENCES volumes(volume_id),
            instance_id TEXT NOT NULL,
            mount_path TEXT DEFAULT '/workspace',
            mode TEXT DEFAULT 'rw',
            attached_at DOUBLE PRECISION NOT NULL,
            detached_at DOUBLE PRECISION DEFAULT 0
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_volume_attachments_volume "
        "ON volume_attachments (volume_id, detached_at)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_volume_attachments_instance "
        "ON volume_attachments (instance_id, detached_at)"
    )

    # ── GPU Pricing (platform-controlled rates) ──
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gpu_pricing (
            id SERIAL PRIMARY KEY,
            gpu_model TEXT NOT NULL,
            vram_gb INTEGER NOT NULL,
            tier TEXT NOT NULL DEFAULT 'standard',
            pricing_mode TEXT NOT NULL DEFAULT 'on_demand',
            base_rate_cad DOUBLE PRECISION NOT NULL,
            priority_multiplier DOUBLE PRECISION DEFAULT 1.0,
            sovereignty_premium DOUBLE PRECISION DEFAULT 0.0,
            spot_discount DOUBLE PRECISION DEFAULT 0.0,
            multi_gpu_discount_4 DOUBLE PRECISION DEFAULT 0.0,
            multi_gpu_discount_8 DOUBLE PRECISION DEFAULT 0.0,
            active BOOLEAN DEFAULT TRUE,
            updated_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
            UNIQUE (gpu_model, tier, pricing_mode)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_gpu_pricing_model "
        "ON gpu_pricing (gpu_model, tier, pricing_mode) WHERE active = TRUE"
    )
    _seed_gpu_pricing(cur)


@contextmanager
def pg_connection():
    """PostgreSQL connection from pool."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        yield conn


@contextmanager
def pg_transaction():
    """PostgreSQL transactional block."""
    pool = _get_pg_pool()
    with pool.connection() as conn:
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


# ── Unified Interface ─────────────────────────────────────────────────


class DatabaseOps:
    """Unified database operations that work across SQLite and PostgreSQL."""

    @staticmethod
    def decode_payload(payload):
        """Decode a JSON payload from either backend."""
        if payload is None:
            return None
        if isinstance(payload, dict):
            return payload  # Already decoded (psycopg3 JSONB auto-decodes)
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return None

    @staticmethod
    def encode_payload(data):
        """Encode data for storage. Returns JSON string for SQLite, dict for Postgres."""
        if isinstance(data, str):
            return data
        return json.dumps(data)

    @staticmethod
    def upsert_job(conn, job, backend="sqlite"):
        """Upsert a job record."""
        job_id = str(job.get("job_id", "")).strip()
        if not job_id:
            return

        status = str(job.get("status") or "queued")
        priority = int(job.get("priority", 0) or 0)
        submitted_at = float(job.get("submitted_at", time.time()) or time.time())
        host_id = job.get("host_id")

        if backend == "postgres":
            from psycopg.types.json import Jsonb

            conn.execute(
                """
                INSERT INTO jobs(job_id, status, priority, submitted_at, host_id, payload)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    priority = EXCLUDED.priority,
                    submitted_at = EXCLUDED.submitted_at,
                    host_id = EXCLUDED.host_id,
                    payload = EXCLUDED.payload
                """,
                (job_id, status, priority, submitted_at, host_id, Jsonb(job)),
            )
        else:
            conn.execute(
                """
                INSERT INTO jobs(job_id, status, priority, submitted_at, host_id, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    priority = excluded.priority,
                    submitted_at = excluded.submitted_at,
                    host_id = excluded.host_id,
                    payload = excluded.payload
                """,
                (job_id, status, priority, submitted_at, host_id, json.dumps(job)),
            )

    @staticmethod
    def upsert_host(conn, host, backend="sqlite"):
        """Upsert a host record."""
        host_id = str(host.get("host_id", "")).strip()
        if not host_id:
            return

        status = str(host.get("status") or "active")
        registered_at = float(host.get("registered_at", time.time()) or time.time())

        if backend == "postgres":
            from psycopg.types.json import Jsonb

            conn.execute(
                """
                INSERT INTO hosts(host_id, status, registered_at, payload)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(host_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    registered_at = EXCLUDED.registered_at,
                    payload = EXCLUDED.payload
                """,
                (host_id, status, registered_at, Jsonb(host)),
            )
        else:
            conn.execute(
                """
                INSERT INTO hosts(host_id, status, registered_at, payload)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(host_id) DO UPDATE SET
                    status = excluded.status,
                    registered_at = excluded.registered_at,
                    payload = excluded.payload
                """,
                (host_id, status, registered_at, json.dumps(host)),
            )

    @staticmethod
    def get_job(conn, job_id, backend="sqlite"):
        """Fetch a single job by ID."""
        ph = "%s" if backend == "postgres" else "?"
        row = conn.execute(f"SELECT payload FROM jobs WHERE job_id = {ph}", (job_id,)).fetchone()
        if not row:
            return None
        payload = row["payload"] if isinstance(row, dict) else row[0]
        return DatabaseOps.decode_payload(payload)

    @staticmethod
    def get_host(conn, host_id, backend="sqlite"):
        """Fetch a single host by ID."""
        ph = "%s" if backend == "postgres" else "?"
        row = conn.execute(f"SELECT payload FROM hosts WHERE host_id = {ph}", (host_id,)).fetchone()
        if not row:
            return None
        payload = row["payload"] if isinstance(row, dict) else row[0]
        return DatabaseOps.decode_payload(payload)

    @staticmethod
    def load_jobs(conn, status=None, backend="sqlite"):
        """Load jobs from DB, optionally filtered by status."""
        ph = "%s" if backend == "postgres" else "?"
        if status:
            rows = conn.execute(
                f"SELECT payload FROM jobs WHERE status = {ph} "
                "ORDER BY submitted_at ASC, job_id ASC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload FROM jobs ORDER BY submitted_at ASC, job_id ASC"
            ).fetchall()

        jobs = []
        for row in rows:
            payload = row["payload"] if isinstance(row, dict) else row[0]
            item = DatabaseOps.decode_payload(payload)
            if isinstance(item, dict):
                jobs.append(item)
        return jobs

    @staticmethod
    def load_hosts(conn, active_only=False, backend="sqlite"):
        """Load hosts from DB, optionally filtered to active only."""
        if active_only:
            rows = conn.execute(
                "SELECT payload FROM hosts WHERE status = 'active' "
                "ORDER BY registered_at ASC, host_id ASC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload FROM hosts ORDER BY registered_at ASC, host_id ASC"
            ).fetchall()

        hosts = []
        for row in rows:
            payload = row["payload"] if isinstance(row, dict) else row[0]
            item = DatabaseOps.decode_payload(payload)
            if isinstance(item, dict):
                hosts.append(item)
        return hosts

    @staticmethod
    def delete_job(conn, job_id, backend="sqlite"):
        """Delete a job by ID."""
        ph = "%s" if backend == "postgres" else "?"
        conn.execute(f"DELETE FROM jobs WHERE job_id = {ph}", (job_id,))

    @staticmethod
    def delete_host(conn, host_id, backend="sqlite"):
        """Delete a host by ID."""
        ph = "%s" if backend == "postgres" else "?"
        conn.execute(f"DELETE FROM hosts WHERE host_id = {ph}", (host_id,))

    @staticmethod
    def delete_all_jobs(conn, backend="sqlite"):
        conn.execute("DELETE FROM jobs")

    @staticmethod
    def delete_all_hosts(conn, backend="sqlite"):
        conn.execute("DELETE FROM hosts")

    @staticmethod
    def upsert_state(conn, namespace, data, backend="sqlite"):
        """Upsert a state namespace record."""
        if backend == "postgres":
            from psycopg.types.json import Jsonb

            conn.execute(
                """
                INSERT INTO state(namespace, payload) VALUES (%s, %s)
                ON CONFLICT(namespace) DO UPDATE SET payload = EXCLUDED.payload
                """,
                (namespace, Jsonb(data) if isinstance(data, (dict, list)) else data),
            )
        else:
            payload = json.dumps(data) if not isinstance(data, str) else data
            conn.execute(
                "INSERT INTO state(namespace, payload) VALUES (?, ?) "
                "ON CONFLICT(namespace) DO UPDATE SET payload = excluded.payload",
                (namespace, payload),
            )

    @staticmethod
    def get_state(conn, namespace, backend="sqlite"):
        """Load a state namespace."""
        ph = "%s" if backend == "postgres" else "?"
        row = conn.execute(
            f"SELECT payload FROM state WHERE namespace = {ph}", (namespace,)
        ).fetchone()
        if not row:
            return None
        payload = row["payload"] if isinstance(row, dict) else row[0]
        return DatabaseOps.decode_payload(payload)

    @staticmethod
    def query_hosts_by_gpu(conn, gpu_model=None, min_vram_gb=None, backend="sqlite"):
        """Query hosts using JSONB operators (Postgres) or payload scan (SQLite).

        This demonstrates the GIN index advantage described in the reports.
        """
        if backend == "postgres":
            conditions = ["status = 'active'"]
            params = []
            if gpu_model:
                conditions.append("payload->>'gpu_model' = %s")
                params.append(gpu_model)
            if min_vram_gb is not None:
                conditions.append("(payload->>'free_vram_gb')::float >= %s")
                params.append(min_vram_gb)

            query = (
                "SELECT payload FROM hosts WHERE "
                + " AND ".join(conditions)
                + " ORDER BY (payload->>'free_vram_gb')::float DESC"
            )
            rows = conn.execute(query, params).fetchall()
            return [DatabaseOps.decode_payload(row["payload"] if isinstance(row, dict) else row[0]) for row in rows if row]
        else:
            # SQLite fallback: load all and filter in Python
            hosts = DatabaseOps.load_hosts(conn, active_only=True, backend="sqlite")
            result = []
            for h in hosts:
                if gpu_model and h.get("gpu_model") != gpu_model:
                    continue
                if min_vram_gb is not None and h.get("free_vram_gb", 0) < min_vram_gb:
                    continue
                result.append(h)
            return sorted(result, key=lambda h: -h.get("free_vram_gb", 0))


# ── Dual-Write Engine ─────────────────────────────────────────────────


class DualWriteEngine:
    """Implements the expand/contract dual-write migration pattern.

    Phase 1 (Dual-Write / Shadow):
        - Primary (SQLite): All reads and writes go here
        - Secondary (Postgres): Writes mirrored, failures logged but don't block

    Phase 2 (Read Switch):
        - Set XCELSIOR_DUAL_READ_FROM=postgres
        - Reads switch to Postgres, writes still go to both

    Phase 3 (Cutover):
        - Set XCELSIOR_DB_BACKEND=postgres
        - Full cutover, SQLite stops being written

    Phase 4 (Cleanup):
        - Remove dual-write code, archive SQLite
    """

    def __init__(self):
        self.backend = DB_BACKEND
        self.read_from = DUAL_READ_FROM
        self.ops = DatabaseOps

    @contextmanager
    def connection(self):
        """Get a connection for the current read backend."""
        if self.backend == "postgres" or (self.backend == "dual" and self.read_from == "postgres"):
            with pg_connection() as conn:
                yield conn, "postgres"
        else:
            with sqlite_connection() as conn:
                yield conn, "sqlite"

    @contextmanager
    def transaction(self):
        """Get a transactional connection.

        In dual mode, yields the primary (SQLite) connection.
        Secondary writes happen in the commit hook.
        """
        if self.backend == "postgres":
            with pg_transaction() as conn:
                yield conn, "postgres"
        elif self.backend == "dual":
            with sqlite_transaction() as conn:
                yield conn, "sqlite"
        else:
            with sqlite_transaction() as conn:
                yield conn, "sqlite"

    def mirror_to_secondary(self, operation, *args, **kwargs):
        """Mirror a write operation to the secondary database.

        Called after successful primary write in dual mode.
        Failures are logged but never propagate.
        """
        if self.backend != "dual":
            return

        secondary = "postgres" if self.read_from == "sqlite" else "sqlite"

        def _do_mirror():
            try:
                if secondary == "postgres":
                    with pg_transaction() as conn:
                        operation(conn, *args, backend="postgres", **kwargs)
                else:
                    with sqlite_transaction() as conn:
                        operation(conn, *args, backend="sqlite", **kwargs)
            except Exception as e:
                log.warning(
                    "Dual-write mirror failed (secondary=%s, op=%s): %s",
                    secondary,
                    operation.__name__,
                    e,
                )

        # Run mirror in background thread to avoid latency impact
        threading.Thread(target=_do_mirror, daemon=True).start()


# ── Singleton ─────────────────────────────────────────────────────────

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Get the global DualWriteEngine instance."""
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine
        _engine = DualWriteEngine()
        log.info(
            "Database engine initialized: backend=%s, read_from=%s",
            _engine.backend,
            _engine.read_from if _engine.backend == "dual" else _engine.backend,
        )
        return _engine


# ── PostgreSQL LISTEN/NOTIFY for Event Bus ────────────────────────────


class PgEventBus:
    """Lightweight event bus using PostgreSQL LISTEN/NOTIFY.

    Used for SSE dashboard streaming. Payloads are kept < 8000 bytes
    (Postgres NOTIFY limit). Publish IDs and let clients fetch details.
    """

    def __init__(self, channel="xcelsior_events"):
        self.channel = channel
        self._listeners = []
        self._running = False

    def notify(self, event_type, data):
        """Publish an event via NOTIFY. Non-blocking."""
        if DB_BACKEND not in ("postgres", "dual"):
            # Fallback: dispatch to in-memory listeners
            self._dispatch_inmemory(event_type, data)
            return

        def _do_notify():
            try:
                with pg_connection() as conn:
                    payload = json.dumps({"type": event_type, "data": data, "ts": time.time()})
                    # Truncate to stay under 8000 bytes
                    if len(payload) > 7900:
                        payload = json.dumps(
                            {
                                "type": event_type,
                                "data": {"id": data.get("id", "?")},
                                "ts": time.time(),
                            }
                        )
                    conn.execute(f"NOTIFY {self.channel}, %s", (payload,))
                    conn.commit()
            except Exception as e:
                log.debug("PgEventBus notify failed: %s", e)

        threading.Thread(target=_do_notify, daemon=True).start()

    def _dispatch_inmemory(self, event_type, data):
        """Dispatch events via in-memory callbacks (SQLite fallback)."""
        event = {"type": event_type, "data": data, "ts": time.time()}
        for listener in self._listeners:
            try:
                listener(event)
            except Exception:
                pass

    def add_listener(self, callback):
        """Register an in-memory event listener."""
        self._listeners.append(callback)

    def remove_listener(self, callback):
        """Remove an in-memory event listener."""
        try:
            self._listeners.remove(callback)
        except ValueError:
            pass


# ── Persistent Auth Storage ────────────────────────────────────────────

# Legacy compat — kept for tests that monkeypatch this value
AUTH_DB_FILE = os.environ.get("XCELSIOR_AUTH_DB_PATH", "data/auth.db")
_auth_schema_lock = threading.Lock()
_auth_schema_ensured = False


def _ensure_oauth_auth_tables(conn) -> None:
    """Ensure incremental OAuth auth tables/columns exist."""
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id TEXT PRIMARY KEY,
            client_name TEXT NOT NULL,
            client_type TEXT NOT NULL,
            redirect_uris JSONB NOT NULL DEFAULT '[]'::jsonb,
            grant_types JSONB NOT NULL DEFAULT '[]'::jsonb,
            scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
            client_secret_hash TEXT,
            client_secret_salt TEXT,
            created_by_email TEXT,
            is_first_party INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            last_used DOUBLE PRECISION,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    # Add columns if missing (for migrations)
    cur.execute("ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active'")
    cur.execute("ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS last_used DOUBLE PRECISION")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
            token_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            family_id TEXT NOT NULL,
            parent_token_id TEXT,
            session_token TEXT UNIQUE,
            client_id TEXT NOT NULL,
            email TEXT,
            user_id TEXT,
            session_type TEXT NOT NULL DEFAULT 'browser',
            scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            consumed_at DOUBLE PRECISION,
            revoked_at DOUBLE PRECISION,
            replaced_by_token_id TEXT,
            reuse_detected_at DOUBLE PRECISION
        )
        """
    )
    cur.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_type TEXT NOT NULL DEFAULT 'legacy'"
    )
    cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS client_id TEXT")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_oauth_clients_owner ON oauth_clients (created_by_email)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_family ON oauth_refresh_tokens (family_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_email ON oauth_refresh_tokens (email)")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_session ON oauth_refresh_tokens (session_token)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_expires ON oauth_refresh_tokens (expires_at)"
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_session_type ON sessions (session_type)")

    # Team invites
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS team_invites (
            token TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            invited_by TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_team_invites_email ON team_invites (email)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_team_invites_team ON team_invites (team_id)")


def _ensure_auth_schema(conn) -> None:
    global _auth_schema_ensured
    if _auth_schema_ensured:
        return
    with _auth_schema_lock:
        if _auth_schema_ensured:
            return
        _ensure_oauth_auth_tables(conn)
        _auth_schema_ensured = True


@contextmanager
def auth_connection():
    """PostgreSQL connection for auth tables (users, sessions, API keys, teams, notifications, SSH keys)."""
    from psycopg.rows import dict_row
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row
        try:
            _ensure_auth_schema(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


class UserStore:
    """Persistent user/session/API key storage backed by PostgreSQL."""

    # ── Users ──

    @staticmethod
    def get_user(email: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = %s", (email,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_user_by_id(user_id: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = %s", (user_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def create_user(user: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (email, user_id, name, password_hash, salt, role, is_admin,
                    customer_id, provider_id, country, province, oauth_provider, team_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    user["email"],
                    user["user_id"],
                    user.get("name", ""),
                    user.get("password_hash", ""),
                    user.get("salt", ""),
                    user.get("role", "submitter"),
                    int(user.get("is_admin", 0)),
                    user.get("customer_id"),
                    user.get("provider_id"),
                    user.get("country", "CA"),
                    user.get("province", "ON"),
                    user.get("oauth_provider"),
                    user.get("team_id"),
                    user.get("created_at", time.time()),
                ),
            )

    @staticmethod
    def update_user(email: str, updates: dict) -> None:
        allowed = {
            "name",
            "role",
            "is_admin",
            "country",
            "province",
            "provider_id",
            "team_id",
            "password_hash",
            "salt",
            "reset_token",
            "reset_token_expires",
            "notifications_enabled",
            "canada_only_routing",
            "preferences",
            "mfa_enabled",
            "email_verified",
            "email_verification_token",
            "email_verification_expires",
        }
        fields = {k: v for k, v in updates.items() if k in allowed}
        if not fields:
            return
        # Wrap JSONB fields for psycopg3
        if "preferences" in fields and isinstance(fields["preferences"], dict):
            from psycopg.types.json import Jsonb
            fields["preferences"] = Jsonb(fields["preferences"])
        # Cast is_admin to int for PostgreSQL INTEGER column
        if "is_admin" in fields:
            fields["is_admin"] = int(fields["is_admin"])
        set_clause = ", ".join(f"{k} = %s" for k in fields)
        values = list(fields.values()) + [email]
        with auth_connection() as conn:
            conn.execute(f"UPDATE users SET {set_clause} WHERE email = %s", values)

    @staticmethod
    def set_admin(email: str, is_admin: int) -> None:
        """Set user admin flag. Only callable from admin endpoints."""
        with auth_connection() as conn:
            conn.execute("UPDATE users SET is_admin = %s WHERE email = %s", (int(is_admin), email))

    @staticmethod
    def delete_user(email: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM users WHERE email = %s", (email,))
            conn.execute("DELETE FROM sessions WHERE email = %s", (email,))
            conn.execute("DELETE FROM api_keys WHERE email = %s", (email,))
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE email = %s", (email,))
            conn.execute("DELETE FROM web_push_subscriptions WHERE user_email = %s", (email,))

    @staticmethod
    def user_exists(email: str) -> bool:
        with auth_connection() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE email = %s", (email,)).fetchone()
            return row is not None

    @staticmethod
    def list_users(team_id: str | None = None) -> list[dict]:
        with auth_connection() as conn:
            if team_id:
                rows = conn.execute("SELECT * FROM users WHERE team_id = %s", (team_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM users").fetchall()
            return [dict(r) for r in rows]

    # ── Sessions ──

    @staticmethod
    def create_session(session: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    token, email, user_id, role, is_admin, name, created_at, expires_at,
                    ip_address, user_agent, last_active, session_type, client_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    session["token"],
                    session["email"],
                    session["user_id"],
                    session.get("role", "submitter"),
                    int(session.get("is_admin", 0)),
                    session.get("name", ""),
                    session.get("created_at", time.time()),
                    session["expires_at"],
                    session.get("ip_address"),
                    session.get("user_agent"),
                    session.get("last_active", session.get("created_at", time.time())),
                    session.get("session_type", "legacy"),
                    session.get("client_id"),
                ),
            )

    @staticmethod
    def get_session(token: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE token = %s AND expires_at > %s",
                (token, time.time()),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def delete_session(token: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE token = %s", (token,))
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE session_token = %s", (token,))

    @staticmethod
    def delete_user_sessions(email: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE email = %s", (email,))
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE email = %s", (email,))

    @staticmethod
    def cleanup_expired_sessions() -> int:
        now = time.time()
        max_lifetime = 86400 * 30  # Must match MAX_SESSION_LIFETIME in routes/_deps.py
        with auth_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE expires_at < %s OR created_at < %s",
                (now, now - max_lifetime),
            )
            conn.execute("DELETE FROM oauth_refresh_tokens WHERE expires_at < %s", (now,))
            return cursor.rowcount

    @staticmethod
    def list_user_sessions(email: str) -> list[dict]:
        now = time.time()
        max_lifetime = 86400 * 30  # Must match MAX_SESSION_LIFETIME in routes/_deps.py
        with auth_connection() as conn:
            rows = conn.execute(
                "SELECT token, email, user_id, role, name, created_at, expires_at, ip_address, user_agent, last_active, session_type, client_id "
                "FROM sessions WHERE email = %s AND expires_at > %s AND created_at > %s ORDER BY last_active DESC",
                (email, now, now - max_lifetime),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def update_session_last_active(token: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                "UPDATE sessions SET last_active = %s WHERE token = %s",
                (time.time(), token),
            )

    @staticmethod
    def rotate_session_token(old_token: str, new_session: dict) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM sessions WHERE token = %s", (old_token,))
            conn.execute(
                """
                INSERT INTO sessions (
                    token, email, user_id, role, is_admin, name, created_at, expires_at,
                    ip_address, user_agent, last_active, session_type, client_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    new_session["token"],
                    new_session["email"],
                    new_session["user_id"],
                    new_session.get("role", "submitter"),
                    int(new_session.get("is_admin", 0)),
                    new_session.get("name", ""),
                    new_session.get("created_at", time.time()),
                    new_session["expires_at"],
                    new_session.get("ip_address"),
                    new_session.get("user_agent"),
                    new_session.get("last_active", time.time()),
                    new_session.get("session_type", "legacy"),
                    new_session.get("client_id"),
                ),
            )

    # ── API Keys ──

    @staticmethod
    def create_api_key(key_data: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (key, name, email, user_id, role, is_admin, scope, created_at, last_used)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
                (
                    key_data["key"],
                    key_data.get("name", "default"),
                    key_data["email"],
                    key_data["user_id"],
                    key_data.get("role", "submitter"),
                    int(key_data.get("is_admin", 0)),
                    key_data.get("scope", "full-access"),
                    key_data.get("created_at", time.time()),
                    key_data.get("last_used"),
                ),
            )

    @staticmethod
    def get_api_key(key: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE key = %s", (key,)).fetchone()
            if row:
                d = dict(row)
                conn.execute("UPDATE api_keys SET last_used = %s WHERE key = %s", (time.time(), key))
                return d
            return None

    @staticmethod
    def list_api_keys(email: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute("SELECT * FROM api_keys WHERE email = %s", (email,)).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def delete_api_key_by_preview(email: str, preview: str) -> bool:
        with auth_connection() as conn:
            rows = conn.execute("SELECT key FROM api_keys WHERE email = %s", (email,)).fetchall()
            for row in rows:
                k = row["key"]
                if (k[:12] + "..." + k[-4:]) == preview:
                    conn.execute("DELETE FROM api_keys WHERE key = %s", (k,))
                    return True
            return False

    @staticmethod
    def delete_user_api_keys(email: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM api_keys WHERE email = %s", (email,))

    # ── Teams ──

    @staticmethod
    def create_team(team: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO teams (team_id, name, owner_email, created_at, plan, max_members)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (
                    team["team_id"],
                    team["name"],
                    team["owner_email"],
                    team.get("created_at", time.time()),
                    team.get("plan", "free"),
                    team.get("max_members", 5),
                ),
            )
            # Add owner as admin member
            conn.execute(
                """
                INSERT INTO team_members (team_id, email, role, joined_at)
                VALUES (%s, %s, 'admin', %s)
            """,
                (team["team_id"], team["owner_email"], time.time()),
            )

    @staticmethod
    def get_team(team_id: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute("SELECT * FROM teams WHERE team_id = %s", (team_id,)).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_team_members(team_id: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute(
                """
                SELECT tm.email, tm.role, tm.joined_at, u.name, u.user_id
                FROM team_members tm LEFT JOIN users u ON tm.email = u.email
                WHERE tm.team_id = %s
            """,
                (team_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def add_team_member(team_id: str, email: str, role: str = "member") -> bool:
        with auth_connection() as conn:
            team = conn.execute(
                "SELECT max_members FROM teams WHERE team_id = %s", (team_id,)
            ).fetchone()
            if not team:
                return False
            count = conn.execute(
                "SELECT COUNT(*) as c FROM team_members WHERE team_id = %s", (team_id,)
            ).fetchone()["c"]
            if count >= team["max_members"]:
                return False
            conn.execute(
                """
                INSERT INTO team_members (team_id, email, role, joined_at)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (team_id, email) DO UPDATE SET role = EXCLUDED.role, joined_at = EXCLUDED.joined_at
            """,
                (team_id, email, role, time.time()),
            )
            conn.execute("UPDATE users SET team_id = %s WHERE email = %s", (team_id, email))
            return True

    @staticmethod
    def remove_team_member(team_id: str, email: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                "DELETE FROM team_members WHERE team_id = %s AND email = %s", (team_id, email)
            )
            conn.execute(
                "UPDATE users SET team_id = NULL WHERE email = %s AND team_id = %s", (email, team_id)
            )

    @staticmethod
    def update_team_member_role(team_id: str, email: str, role: str) -> bool:
        """Update a team member's role. Returns False if member not found."""
        with auth_connection() as conn:
            cur = conn.execute(
                "UPDATE team_members SET role = %s WHERE team_id = %s AND email = %s",
                (role, team_id, email),
            )
            return cur.rowcount > 0

    @staticmethod
    def delete_team(team_id: str) -> None:
        with auth_connection() as conn:
            conn.execute("UPDATE users SET team_id = NULL WHERE team_id = %s", (team_id,))
            conn.execute("DELETE FROM team_members WHERE team_id = %s", (team_id,))
            conn.execute("DELETE FROM teams WHERE team_id = %s", (team_id,))

    @staticmethod
    def get_user_teams(email: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute(
                """
                SELECT t.*, tm.role as member_role
                FROM teams t JOIN team_members tm ON t.team_id = tm.team_id
                WHERE tm.email = %s
            """,
                (email,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Team Invites ──

    @staticmethod
    def create_team_invite(invite: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO team_invites (token, team_id, email, role, invited_by, created_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (token) DO NOTHING
                """,
                (
                    invite["token"],
                    invite["team_id"],
                    invite["email"].lower(),
                    invite.get("role", "member"),
                    invite["invited_by"],
                    invite["created_at"],
                    invite["expires_at"],
                ),
            )

    @staticmethod
    def get_team_invite(token: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM team_invites WHERE token = %s", (token,)
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def delete_team_invite(token: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM team_invites WHERE token = %s", (token,))

    @staticmethod
    def get_pending_invites_for_email(email: str) -> list[dict]:
        with auth_connection() as conn:
            import time as _time
            rows = conn.execute(
                "SELECT * FROM team_invites WHERE email = %s AND expires_at > %s",
                (email.lower(), _time.time()),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── SSH Keys ──

    @staticmethod
    def add_ssh_key(key_data: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                "INSERT INTO user_ssh_keys (id, email, user_id, name, public_key, fingerprint, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (
                    key_data["id"],
                    key_data["email"],
                    key_data["user_id"],
                    key_data.get("name", "default"),
                    key_data["public_key"],
                    key_data["fingerprint"],
                    key_data.get("created_at", time.time()),
                ),
            )

    @staticmethod
    def list_ssh_keys(email: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM user_ssh_keys WHERE email = %s ORDER BY created_at DESC",
                (email,),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def delete_ssh_key(email: str, key_id: str) -> bool:
        with auth_connection() as conn:
            cur = conn.execute(
                "DELETE FROM user_ssh_keys WHERE id = %s AND email = %s",
                (key_id, email),
            )
            return cur.rowcount > 0

    @staticmethod
    def delete_user_ssh_keys(email: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM user_ssh_keys WHERE email = %s", (email,))


class NotificationStore:
    """Per-user in-app notification storage backed by PostgreSQL."""

    @staticmethod
    def create(
        user_email: str,
        notif_type: str,
        title: str,
        body: str = "",
        data: dict | None = None,
        *,
        action_url: str = "",
        entity_type: str = "",
        entity_id: str = "",
        priority: int = 0,
    ) -> str:
        from psycopg.types.json import Jsonb
        import uuid as _uuid

        nid = f"notif-{_uuid.uuid4().hex[:12]}"
        payload = dict(data or {})
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO notifications (
                    id,
                    user_email,
                    type,
                    title,
                    body,
                    data,
                    read,
                    created_at,
                    action_url,
                    entity_type,
                    entity_id,
                    priority
                )
                VALUES (%s, %s, %s, %s, %s, %s, 0, %s, %s, %s, %s, %s)
                """,
                (
                    nid,
                    user_email,
                    notif_type,
                    title,
                    body,
                    Jsonb(payload),
                    time.time(),
                    action_url,
                    entity_type,
                    entity_id,
                    priority,
                ),
            )

        try:
            from web_push import deliver_web_push_notification

            deliver_web_push_notification(
                user_email,
                {
                    "id": nid,
                    "type": notif_type,
                    "title": title,
                    "body": body,
                    "data": payload,
                    "action_url": action_url,
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "priority": priority,
                },
            )
        except Exception as exc:
            log.debug("Web push delivery error for %s: %s", user_email, exc)

        return nid

    @staticmethod
    def list_for_user(user_email: str, unread_only: bool = False,
                      limit: int = 50) -> list[dict]:
        with auth_connection() as conn:
            sql = "SELECT * FROM notifications WHERE user_email = %s"
            params: list = [user_email]
            if unread_only:
                sql += " AND read = 0"
            sql += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def unread_count(user_email: str) -> int:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM notifications WHERE user_email = %s AND read = 0",
                (user_email,),
            ).fetchone()
            return row["c"] if row else 0

    @staticmethod
    def total_count() -> int:
        with auth_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM notifications").fetchone()
            return int(row["c"]) if row else 0

    @staticmethod
    def mark_read(notification_id: str, user_email: str) -> bool:
        with auth_connection() as conn:
            cur = conn.execute(
                "UPDATE notifications SET read = 1 WHERE id = %s AND user_email = %s",
                (notification_id, user_email),
            )
            return cur.rowcount > 0

    @staticmethod
    def mark_all_read(user_email: str) -> int:
        with auth_connection() as conn:
            cur = conn.execute(
                "UPDATE notifications SET read = 1 WHERE user_email = %s AND read = 0",
                (user_email,),
            )
            return cur.rowcount

    @staticmethod
    def delete_old(days: int = 30) -> int:
        cutoff = time.time() - days * 86400
        with auth_connection() as conn:
            cur = conn.execute("DELETE FROM notifications WHERE created_at < %s", (cutoff,))
            return cur.rowcount

    @staticmethod
    def delete(notification_id: str, user_email: str) -> bool:
        with auth_connection() as conn:
            cur = conn.execute(
                "DELETE FROM notifications WHERE id = %s AND user_email = %s",
                (notification_id, user_email),
            )
            return cur.rowcount > 0


class WebPushSubscriptionStore:
    """Stores active web push subscriptions for authenticated users."""

    @staticmethod
    def upsert(
        user_email: str,
        endpoint: str,
        p256dh: str,
        auth: str,
        *,
        user_agent: str = "",
    ) -> str:
        import uuid as _uuid

        subscription_id = f"wps-{_uuid.uuid4().hex[:12]}"
        now = time.time()

        with auth_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO web_push_subscriptions (
                    id,
                    user_email,
                    endpoint,
                    p256dh,
                    auth,
                    user_agent,
                    created_at,
                    last_used_at,
                    revoked_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NULL)
                ON CONFLICT (endpoint) DO UPDATE
                SET user_email = EXCLUDED.user_email,
                    p256dh = EXCLUDED.p256dh,
                    auth = EXCLUDED.auth,
                    user_agent = EXCLUDED.user_agent,
                    last_used_at = EXCLUDED.last_used_at,
                    revoked_at = NULL
                RETURNING id
                """,
                (subscription_id, user_email, endpoint, p256dh, auth, user_agent, now, now),
            ).fetchone()
            return row["id"] if row else subscription_id

    @staticmethod
    def list_active_for_user(user_email: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM web_push_subscriptions
                WHERE user_email = %s AND revoked_at IS NULL
                ORDER BY last_used_at DESC, created_at DESC
                """,
                (user_email,),
            ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def count_active_for_user(user_email: str) -> int:
        with auth_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM web_push_subscriptions
                WHERE user_email = %s AND revoked_at IS NULL
                """,
                (user_email,),
            ).fetchone()
            return int(row["count"]) if row else 0

    @staticmethod
    def count_active() -> int:
        with auth_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM web_push_subscriptions
                WHERE revoked_at IS NULL
                """
            ).fetchone()
            return int(row["count"]) if row else 0

    @staticmethod
    def count_revoked() -> int:
        with auth_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM web_push_subscriptions
                WHERE revoked_at IS NOT NULL
                """
            ).fetchone()
            return int(row["count"]) if row else 0

    @staticmethod
    def count_stale_active(max_age_days: int = 30) -> int:
        cutoff = time.time() - max(max_age_days, 1) * 86400
        with auth_connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM web_push_subscriptions
                WHERE revoked_at IS NULL
                  AND last_used_at < %s
                """,
                (cutoff,),
            ).fetchone()
            return int(row["count"]) if row else 0

    @staticmethod
    def revoke(user_email: str, endpoint: str) -> bool:
        with auth_connection() as conn:
            cur = conn.execute(
                """
                UPDATE web_push_subscriptions
                SET revoked_at = %s
                WHERE user_email = %s AND endpoint = %s AND revoked_at IS NULL
                """,
                (time.time(), user_email, endpoint),
            )
            return cur.rowcount > 0

    @staticmethod
    def delete_revoked_older_than(days: int = 30) -> int:
        cutoff = time.time() - max(days, 1) * 86400
        with auth_connection() as conn:
            cur = conn.execute(
                """
                DELETE FROM web_push_subscriptions
                WHERE revoked_at IS NOT NULL
                  AND revoked_at < %s
                """,
                (cutoff,),
            )
            return cur.rowcount

    @staticmethod
    def revoke_endpoint(endpoint: str) -> bool:
        with auth_connection() as conn:
            cur = conn.execute(
                """
                UPDATE web_push_subscriptions
                SET revoked_at = %s
                WHERE endpoint = %s AND revoked_at IS NULL
                """,
                (time.time(), endpoint),
            )
            return cur.rowcount > 0

    @staticmethod
    def touch(endpoint: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                "UPDATE web_push_subscriptions SET last_used_at = %s WHERE endpoint = %s",
                (time.time(), endpoint),
            )


# Global event bus instance
event_bus = PgEventBus()


def emit_event(event_type, data):
    """Convenience: emit a scheduler event for dashboard SSE streaming."""
    event_bus.notify(event_type, data)


def start_pg_listen(callback, channel="xcelsior_events"):
    """Start a background thread that LISTENs on a Postgres channel.

    When a NOTIFY arrives, `callback(event_type, data)` is called.
    This bridges PgEventBus → SSE delivery in multi-process deployments.

    Only starts if DB_BACKEND is 'postgres' or 'dual'.
    Returns the thread (or None if not applicable).
    """
    if DB_BACKEND not in ("postgres", "dual"):
        # SQLite mode: register callback directly as in-memory listener
        def _adapter(event):
            callback(event.get("type", "message"), event.get("data", {}))

        event_bus.add_listener(_adapter)
        log.info("PgEventBus: registered in-memory listener (SQLite mode)")
        return None

    def _listen_loop():
        import select

        while True:
            try:
                # Dedicated connection for LISTEN (not from pool)
                import psycopg

                conn = psycopg.connect(
                    os.environ.get("XCELSIOR_PG_DSN") or POSTGRES_DSN,
                    autocommit=True,
                )
                conn.execute(f"LISTEN {channel}")
                log.info("PgEventBus: LISTEN started on channel '%s'", channel)

                while True:
                    # Use select() to wait for notifications without busy-polling
                    if select.select([conn.fileno()], [], [], 30.0) == ([], [], []):
                        # Timeout — send a keepalive query
                        conn.execute("SELECT 1")
                        continue

                    # Process all pending notifications
                    gen = conn.notifies()
                    for notify in gen:
                        try:
                            payload = json.loads(notify.payload)
                            callback(
                                payload.get("type", "message"),
                                payload.get("data", {}),
                            )
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        break  # One notification per select cycle; re-check
            except Exception as e:
                log.warning("PgEventBus LISTEN error: %s — reconnecting in 5s", e)
                time.sleep(5)

    t = threading.Thread(target=_listen_loop, daemon=True, name="pg-listen")
    t.start()
    return t


# ────────────────────────────────────────────────────────────────────
# MFA Store
# ────────────────────────────────────────────────────────────────────

class MfaStore:
    """MFA methods, backup codes, and challenges backed by PostgreSQL."""

    # ── Methods ──

    @staticmethod
    def list_methods(email: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mfa_methods WHERE email = %s ORDER BY created_at",
                (email,),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def get_method(method_id: int) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM mfa_methods WHERE id = %s", (method_id,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def get_method_by_type(email: str, method_type: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM mfa_methods WHERE email = %s AND method_type = %s AND enabled = 1",
                (email, method_type),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def create_method(data: dict) -> int:
        with auth_connection() as conn:
            row = conn.execute(
                """
                INSERT INTO mfa_methods (email, method_type, secret, phone_number,
                    credential_id, public_key, sign_count, device_name, enabled, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    data["email"],
                    data["method_type"],
                    data.get("secret"),
                    data.get("phone_number"),
                    data.get("credential_id"),
                    data.get("public_key"),
                    data.get("sign_count", 0),
                    data.get("device_name"),
                    data.get("enabled", 1),
                    data.get("created_at", time.time()),
                ),
            ).fetchone()
            return row["id"]

    @staticmethod
    def delete_method(method_id: int, email: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                "DELETE FROM mfa_methods WHERE id = %s AND email = %s",
                (method_id, email),
            )

    @staticmethod
    def delete_methods_by_type(email: str, method_type: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                "DELETE FROM mfa_methods WHERE email = %s AND method_type = %s",
                (email, method_type),
            )

    # ── Passkey helpers ──

    @staticmethod
    def get_passkey_by_credential(credential_id: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM mfa_methods WHERE credential_id = %s AND method_type = 'passkey' AND enabled = 1",
                (credential_id,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def update_passkey_sign_count(method_id: int, sign_count: int) -> None:
        with auth_connection() as conn:
            conn.execute(
                "UPDATE mfa_methods SET sign_count = %s WHERE id = %s",
                (sign_count, method_id),
            )

    # ── Backup Codes ──

    @staticmethod
    def list_backup_codes(email: str) -> list[dict]:
        with auth_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mfa_backup_codes WHERE email = %s ORDER BY id",
                (email,),
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def create_backup_codes(email: str, code_hashes: list[str]) -> None:
        now = time.time()
        with auth_connection() as conn:
            # Delete old codes
            conn.execute("DELETE FROM mfa_backup_codes WHERE email = %s", (email,))
            for h in code_hashes:
                conn.execute(
                    "INSERT INTO mfa_backup_codes (email, code_hash, used, created_at) VALUES (%s, %s, 0, %s)",
                    (email, h, now),
                )

    @staticmethod
    def use_backup_code(email: str, code_hash: str) -> bool:
        """Mark a backup code as used. Returns True if a valid unused code was found."""
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT id FROM mfa_backup_codes WHERE email = %s AND code_hash = %s AND used = 0",
                (email, code_hash),
            ).fetchone()
            if not row:
                return False
            conn.execute("UPDATE mfa_backup_codes SET used = 1 WHERE id = %s", (row["id"],))
            return True

    # ── Challenges ──

    @staticmethod
    def create_challenge(data: dict) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO mfa_challenges (challenge_id, email, session_token, challenge_data, created_at, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    data["challenge_id"],
                    data["email"],
                    data.get("session_token"),
                    data.get("challenge_data"),
                    data.get("created_at", time.time()),
                    data["expires_at"],
                ),
            )

    @staticmethod
    def get_challenge(challenge_id: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM mfa_challenges WHERE challenge_id = %s", (challenge_id,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def delete_challenge(challenge_id: str) -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM mfa_challenges WHERE challenge_id = %s", (challenge_id,))

    @staticmethod
    def cleanup_expired_challenges() -> None:
        with auth_connection() as conn:
            conn.execute("DELETE FROM mfa_challenges WHERE expires_at < %s", (time.time(),))


class OAuthStore:
    @staticmethod
    def update_client(client_id: str, updates: dict, created_by_email: str | None = None) -> bool:
            allowed = {"client_name", "redirect_uris", "grant_types", "scopes", "status"}
            fields = {k: v for k, v in updates.items() if k in allowed}
            if not fields:
                return False
            # Wrap JSONB fields for psycopg3
            from psycopg.types.json import Jsonb
            for k in ["redirect_uris", "grant_types", "scopes"]:
                if k in fields:
                    fields[k] = Jsonb(list(fields[k]))
            set_clause = ", ".join(f"{k} = %s" for k in fields)
            values = list(fields.values())
            values.append(time.time())
            set_clause += ", updated_at = %s"
            if created_by_email:
                values.extend([client_id, created_by_email])
                where = "client_id = %s AND created_by_email = %s AND is_first_party = 0"
            else:
                values.append(client_id)
                where = "client_id = %s"
            with auth_connection() as conn:
                cur = conn.execute(f"UPDATE oauth_clients SET {set_clause} WHERE {where}", values)
                return cur.rowcount > 0

    @staticmethod
    def rotate_client_secret(client_id: str, new_secret_hash: str, new_secret_salt: str, created_by_email: str | None = None) -> bool:
            set_clause = "client_secret_hash = %s, client_secret_salt = %s, updated_at = %s"
            values = [new_secret_hash, new_secret_salt, time.time()]
            if created_by_email:
                values.extend([client_id, created_by_email])
                where = "client_id = %s AND created_by_email = %s AND is_first_party = 0"
            else:
                values.append(client_id)
                where = "client_id = %s"
            with auth_connection() as conn:
                cur = conn.execute(f"UPDATE oauth_clients SET {set_clause} WHERE {where}", values)
                return cur.rowcount > 0

    @staticmethod
    def set_client_status(client_id: str, status: str, created_by_email: str | None = None) -> bool:
            values = [status, time.time()]
            if created_by_email:
                values.extend([client_id, created_by_email])
                where = "client_id = %s AND created_by_email = %s AND is_first_party = 0"
            else:
                values.append(client_id)
                where = "client_id = %s"
            with auth_connection() as conn:
                cur = conn.execute(f"UPDATE oauth_clients SET status = %s, updated_at = %s WHERE {where}", values)
                return cur.rowcount > 0


    @staticmethod
    def update_last_used(client_id: str):
        with auth_connection() as conn:
            conn.execute("UPDATE oauth_clients SET last_used = %s WHERE client_id = %s", (time.time(), client_id))

    @staticmethod
    def create_client(client: dict) -> None:
        from psycopg.types.json import Jsonb

        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO oauth_clients (
                    client_id,
                    client_name,
                    client_type,
                    redirect_uris,
                    grant_types,
                    scopes,
                    client_secret_hash,
                    client_secret_salt,
                    created_by_email,
                    is_first_party,
                    created_at,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (client_id) DO UPDATE SET
                    client_name = EXCLUDED.client_name,
                    client_type = EXCLUDED.client_type,
                    redirect_uris = EXCLUDED.redirect_uris,
                    grant_types = EXCLUDED.grant_types,
                    scopes = EXCLUDED.scopes,
                    client_secret_hash = EXCLUDED.client_secret_hash,
                    client_secret_salt = EXCLUDED.client_secret_salt,
                    created_by_email = EXCLUDED.created_by_email,
                    is_first_party = EXCLUDED.is_first_party,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    client["client_id"],
                    client["client_name"],
                    client["client_type"],
                    Jsonb(list(client.get("redirect_uris") or [])),
                    Jsonb(list(client.get("grant_types") or [])),
                    Jsonb(list(client.get("scopes") or [])),
                    client.get("client_secret_hash"),
                    client.get("client_secret_salt"),
                    client.get("created_by_email"),
                    int(client.get("is_first_party", 0)),
                    client.get("created_at", time.time()),
                    client.get("updated_at", time.time()),
                ),
            )

    @staticmethod
    def get_client(client_id: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM oauth_clients WHERE client_id = %s",
                (client_id,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def list_clients(created_by_email: str | None = None) -> list[dict]:
        with auth_connection() as conn:
            if created_by_email:
                rows = conn.execute(
                    "SELECT * FROM oauth_clients WHERE created_by_email = %s OR is_first_party = 1 ORDER BY created_at DESC",
                    (created_by_email,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM oauth_clients ORDER BY created_at DESC"
                ).fetchall()
            return [dict(row) for row in rows]

    @staticmethod
    def delete_client(client_id: str, created_by_email: str | None = None) -> bool:
        with auth_connection() as conn:
            if created_by_email:
                cur = conn.execute(
                    "DELETE FROM oauth_clients WHERE client_id = %s AND created_by_email = %s AND is_first_party = 0",
                    (client_id, created_by_email),
                )
            else:
                cur = conn.execute("DELETE FROM oauth_clients WHERE client_id = %s", (client_id,))
            return cur.rowcount > 0

    @staticmethod
    def create_refresh_token(token_data: dict) -> None:
        from psycopg.types.json import Jsonb

        with auth_connection() as conn:
            conn.execute(
                """
                INSERT INTO oauth_refresh_tokens (
                    token_id,
                    token_hash,
                    family_id,
                    parent_token_id,
                    session_token,
                    client_id,
                    email,
                    user_id,
                    session_type,
                    scopes,
                    created_at,
                    expires_at,
                    consumed_at,
                    revoked_at,
                    replaced_by_token_id,
                    reuse_detected_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    token_data["token_id"],
                    token_data["token_hash"],
                    token_data["family_id"],
                    token_data.get("parent_token_id"),
                    token_data.get("session_token"),
                    token_data["client_id"],
                    token_data.get("email"),
                    token_data.get("user_id"),
                    token_data.get("session_type", "browser"),
                    Jsonb(list(token_data.get("scopes") or [])),
                    token_data.get("created_at", time.time()),
                    token_data["expires_at"],
                    token_data.get("consumed_at"),
                    token_data.get("revoked_at"),
                    token_data.get("replaced_by_token_id"),
                    token_data.get("reuse_detected_at"),
                ),
            )

    @staticmethod
    def get_refresh_token(token_hash: str) -> dict | None:
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT * FROM oauth_refresh_tokens WHERE token_hash = %s",
                (token_hash,),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def mark_refresh_token_rotated(token_id: str, replaced_by_token_id: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                """
                UPDATE oauth_refresh_tokens
                SET consumed_at = %s, replaced_by_token_id = %s
                WHERE token_id = %s
                """,
                (time.time(), replaced_by_token_id, token_id),
            )

    @staticmethod
    def revoke_refresh_family(family_id: str, *, reuse_detected: bool = False) -> None:
        now = time.time()
        with auth_connection() as conn:
            conn.execute(
                """
                UPDATE oauth_refresh_tokens
                SET revoked_at = COALESCE(revoked_at, %s),
                    reuse_detected_at = CASE WHEN %s THEN COALESCE(reuse_detected_at, %s) ELSE reuse_detected_at END
                WHERE family_id = %s
                """,
                (now, reuse_detected, now, family_id),
            )

    @staticmethod
    def delete_refresh_token_by_session(session_token: str) -> None:
        with auth_connection() as conn:
            conn.execute(
                "DELETE FROM oauth_refresh_tokens WHERE session_token = %s",
                (session_token,),
            )
