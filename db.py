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
DB_BACKEND = os.environ.get("XCELSIOR_DB_BACKEND", "sqlite").lower()

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
        "CREATE TABLE IF NOT EXISTS state "
        "(namespace TEXT PRIMARY KEY, payload TEXT NOT NULL)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL,
            submitted_at REAL NOT NULL,
            host_id TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS hosts (
            host_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            registered_at REAL NOT NULL,
            payload TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_queue "
        "ON jobs(status, priority DESC, submitted_at ASC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_status "
        "ON hosts(status, registered_at ASC)"
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
    """Lazy-initialize PostgreSQL connection pool using psycopg3."""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool

    with _pg_pool_lock:
        if _pg_pool is not None:
            return _pg_pool

        try:
            from psycopg_pool import ConnectionPool

            _pg_pool = ConnectionPool(
                POSTGRES_DSN,
                min_size=2,
                max_size=PG_POOL_SIZE,
                kwargs={"autocommit": False},
            )
            log.info("PostgreSQL connection pool initialized (size=%d)", PG_POOL_SIZE)

            # Ensure tables exist
            with _pg_pool.connection() as conn:
                _ensure_pg_tables(conn)
                conn.commit()

            return _pg_pool
        except ImportError:
            log.error(
                "psycopg or psycopg_pool not installed. "
                "Install with: pip install 'psycopg[binary]' psycopg_pool"
            )
            raise
        except Exception as e:
            log.error("Failed to connect to PostgreSQL: %s", e)
            raise


def _ensure_pg_tables(conn):
    """Ensure all scheduler persistence tables and indexes exist in PostgreSQL.

    Uses JSONB for payload columns with GIN indexes for query-critical fields.
    """
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS state (
            namespace TEXT PRIMARY KEY,
            payload JSONB NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            priority INTEGER NOT NULL,
            submitted_at DOUBLE PRECISION NOT NULL,
            host_id TEXT,
            payload JSONB NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS hosts (
            host_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            registered_at DOUBLE PRECISION NOT NULL,
            payload JSONB NOT NULL
        )
        """
    )

    # Indexed columns for scheduler queries
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_queue "
        "ON jobs(status, priority DESC, submitted_at ASC)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_status "
        "ON hosts(status, registered_at ASC)"
    )

    # GIN indexes on JSONB payloads for GPU capability queries
    # Allows: SELECT * FROM hosts WHERE payload->'gpu_specs' @> '{"vram_gb": 24}'
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_payload_gin "
        "ON hosts USING GIN (payload)"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_payload_gin "
        "ON jobs USING GIN (payload)"
    )

    # Expression indexes for hot-path queries
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_hosts_gpu_model "
        "ON hosts ((payload->>'gpu_model'))"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_jobs_tier "
        "ON jobs ((payload->>'tier'))"
    )


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
        row = conn.execute(
            f"SELECT payload FROM jobs WHERE job_id = {ph}", (job_id,)
        ).fetchone()
        if not row:
            return None
        payload = row[0] if backend == "postgres" else row["payload"]
        return DatabaseOps.decode_payload(payload)

    @staticmethod
    def get_host(conn, host_id, backend="sqlite"):
        """Fetch a single host by ID."""
        ph = "%s" if backend == "postgres" else "?"
        row = conn.execute(
            f"SELECT payload FROM hosts WHERE host_id = {ph}", (host_id,)
        ).fetchone()
        if not row:
            return None
        payload = row[0] if backend == "postgres" else row["payload"]
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
            payload = row[0] if backend == "postgres" else row["payload"]
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
            payload = row[0] if backend == "postgres" else row["payload"]
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
        payload = row[0] if backend == "postgres" else row["payload"]
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
                conditions.append(
                    "(payload->>'free_vram_gb')::float >= %s"
                )
                params.append(min_vram_gb)

            query = (
                "SELECT payload FROM hosts WHERE "
                + " AND ".join(conditions)
                + " ORDER BY (payload->>'free_vram_gb')::float DESC"
            )
            rows = conn.execute(query, params).fetchall()
            return [row[0] for row in rows if row[0]]
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
        if self.backend == "postgres" or (
            self.backend == "dual" and self.read_from == "postgres"
        ):
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
                    payload = json.dumps(
                        {"type": event_type, "data": data, "ts": time.time()}
                    )
                    # Truncate to stay under 8000 bytes
                    if len(payload) > 7900:
                        payload = json.dumps(
                            {"type": event_type, "data": {"id": data.get("id", "?")}, "ts": time.time()}
                        )
                    conn.execute(
                        f"NOTIFY {self.channel}, %s", (payload,)
                    )
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


# Global event bus instance
event_bus = PgEventBus()


def emit_event(event_type, data):
    """Convenience: emit a scheduler event for dashboard SSE streaming."""
    event_bus.notify(event_type, data)
