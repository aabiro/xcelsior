"""Routes: admin."""

import time

from fastapi import APIRouter, HTTPException, Request

from routes._deps import (
    _USE_PERSISTENT_AUTH,
    _admin_flag,
    _is_platform_admin,
    _require_admin,
    _users_db,
    log,
)
from scheduler import (
    list_hosts,
    list_jobs,
    log,
)
from db import UserStore, emit_event
from billing import get_billing_engine
from verification import get_verification_engine
from events import get_event_store

router = APIRouter()


# ── Helper: _admin_flag ──

def _admin_flag(value) -> int:
    """Normalize truthy admin values from DB/session payloads."""
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "on"} else 0
    return 1 if value else 0

@router.get("/api/admin/stats", tags=["Admin"])
def api_admin_stats(request: Request):
    """Get admin dashboard statistics."""
    _require_admin(request)
    hosts = list_hosts(active_only=False)
    active_hosts = [h for h in hosts if h.get("status") == "active"]
    jobs = list_jobs()
    running = [j for j in jobs if j.get("status") in ("running", "assigned")]
    if _USE_PERSISTENT_AUTH:
        users = UserStore.list_users()
    else:
        users = list(_users_db.values())

    # Real revenue MTD from billing engine
    revenue_mtd = 0.0
    try:
        be = get_billing_engine()
        import datetime as _dt
        now = _dt.datetime.now(_dt.timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()
        with be._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_cad), 0) AS rev FROM usage_meters WHERE created_at >= %s",
                (month_start,),
            ).fetchone()
            revenue_mtd = round(float(row["rev"]), 2) if row else 0.0
    except Exception as e:
        log.warning("admin_stats: failed to fetch revenue_mtd", exc_info=True)

    return {
        "ok": True,
        "total_users": len(users),
        "active_hosts": len(active_hosts),
        "running_jobs": len(running),
        "revenue_mtd": revenue_mtd,
    }

@router.get("/api/admin/users", tags=["Admin"])
def api_admin_users(request: Request):
    """List all users for admin panel."""
    _require_admin(request)
    if _USE_PERSISTENT_AUTH:
        users = UserStore.list_users()
    else:
        users = list(_users_db.values())

    # Gather wallet balances, job counts, and last-activity timestamps in bulk
    wallet_map: dict[str, float] = {}
    job_count_map: dict[str, int] = {}
    last_activity_map: dict[str, float] = {}
    try:
        be = get_billing_engine()
        with be._conn() as conn:
            for wr in conn.execute("SELECT customer_id, balance_cad FROM wallets").fetchall():
                wallet_map[wr["customer_id"]] = float(wr["balance_cad"])
            for jr in conn.execute(
                "SELECT owner, COUNT(*) AS cnt, MAX(created_at) AS last_at FROM usage_meters GROUP BY owner"
            ).fetchall():
                job_count_map[jr["owner"]] = jr["cnt"]
                last_activity_map[jr["owner"]] = float(jr["last_at"]) if jr["last_at"] else 0.0
    except Exception as e:
        log.warning("admin_users: failed to fetch wallet/job data", exc_info=True)

    import datetime as _dt

    now_ts = time.time()
    active_threshold = now_ts - 30 * 86400  # 30 days

    safe_users = []
    for u in users:
        # Convert Unix timestamp to ISO string (fixes 1970 bug)
        raw_ts = u.get("created_at", "")
        if isinstance(raw_ts, (int, float)) and raw_ts > 0:
            created_iso = _dt.datetime.fromtimestamp(raw_ts, tz=_dt.timezone.utc).isoformat()
        elif isinstance(raw_ts, str) and raw_ts:
            created_iso = raw_ts
        else:
            created_iso = ""

        cid = u.get("customer_id", u.get("email", ""))
        email = u.get("email", "")
        last_at = last_activity_map.get(email, 0.0)
        safe_users.append({
            "email": email,
            "role": u.get("role", "submitter"),
            "is_admin": True if _is_platform_admin(u) else False,
            "is_active": last_at >= active_threshold,
            "created_at": created_iso,
            "wallet_balance_cad": wallet_map.get(cid, 0.0),
            "total_jobs": job_count_map.get(email, 0),
            "province": u.get("province", ""),
            "country": u.get("country", ""),
        })
    return {"ok": True, "users": safe_users}

@router.get("/api/admin/overview", tags=["Admin"])
def api_admin_overview(request: Request, days: int = 30):
    """Admin overview with KPIs and trend data for a configurable window."""
    _require_admin(request)
    import datetime as _dt

    days = max(1, min(days, 365))
    now = _dt.datetime.now(_dt.timezone.utc)
    now_ts = now.timestamp()
    thirty_days_ago = now_ts - days * 86400
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).timestamp()

    hosts = list_hosts(active_only=False)
    active_hosts = [h for h in hosts if h.get("status") == "active"]
    jobs = list_jobs()
    running = [j for j in jobs if j.get("status") in ("running", "assigned")]

    if _USE_PERSISTENT_AUTH:
        users = UserStore.list_users()
    else:
        users = list(_users_db.values())

    # Revenue + billing data
    revenue_mtd = 0.0
    revenue_total = 0.0
    total_gpu_hours = 0.0
    daily_revenue: list[dict] = []
    daily_signups: list[dict] = []
    daily_jobs: list[dict] = []
    try:
        be = get_billing_engine()
        with be._conn() as conn:
            # MTD revenue
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_cad), 0) AS rev FROM usage_meters WHERE created_at >= %s",
                (month_start,),
            ).fetchone()
            revenue_mtd = round(float(row["rev"]), 2) if row else 0.0

            # Total revenue
            row = conn.execute("SELECT COALESCE(SUM(total_cost_cad), 0) AS rev FROM usage_meters").fetchone()
            revenue_total = round(float(row["rev"]), 2) if row else 0.0

            # Total GPU hours
            row = conn.execute("SELECT COALESCE(SUM(gpu_seconds), 0) AS s FROM usage_meters").fetchone()
            total_gpu_hours = round(float(row["s"]) / 3600, 1) if row else 0.0

            # 30-day revenue trend
            for r in conn.execute(
                "SELECT to_char(to_timestamp(created_at), 'YYYY-MM-DD') AS day, "
                "ROUND(SUM(total_cost_cad)::numeric, 2) AS revenue, COUNT(*) AS jobs "
                "FROM usage_meters WHERE created_at >= %s GROUP BY day ORDER BY day",
                (thirty_days_ago,),
            ).fetchall():
                daily_revenue.append({"date": r["day"], "revenue": float(r["revenue"])})
                daily_jobs.append({"date": r["day"], "jobs": r["jobs"]})
    except Exception as e:
        log.warning("admin_overview: failed to fetch billing data", exc_info=True)

    # 30-day signup trend
    try:
        from db import auth_connection
        with auth_connection() as conn:
            for r in conn.execute(
                "SELECT to_char(to_timestamp(created_at), 'YYYY-MM-DD') AS day, COUNT(*) AS signups "
                "FROM users WHERE created_at >= %s GROUP BY day ORDER BY day",
                (thirty_days_ago,),
            ).fetchall():
                daily_signups.append({"date": r["day"], "signups": r["signups"]})
    except Exception as e:
        log.warning("admin_overview: failed to fetch signup data", exc_info=True)

    # Period-over-period trends (this period vs previous period)
    prev_thirty = thirty_days_ago - days * 86400
    prev_users = sum(1 for u in users if isinstance(u.get("created_at"), (int, float)) and prev_thirty <= u["created_at"] < thirty_days_ago)
    curr_users = sum(1 for u in users if isinstance(u.get("created_at"), (int, float)) and u["created_at"] >= thirty_days_ago)

    prev_revenue = 0.0
    try:
        be = get_billing_engine()
        with be._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(total_cost_cad), 0) AS rev FROM usage_meters WHERE created_at >= %s AND created_at < %s",
                (prev_thirty, thirty_days_ago),
            ).fetchone()
            prev_revenue = float(row["rev"]) if row else 0.0
    except Exception as e:
        log.debug("previous period revenue comparison failed: %s", e)

    def _pct(curr: float, prev: float) -> float:
        if prev == 0:
            return 100.0 if curr > 0 else 0.0
        return round((curr - prev) / prev * 100, 1)

    curr_revenue = sum(d["revenue"] for d in daily_revenue)

    # Host registration trends (registered_at is a unix timestamp)
    prev_hosts = sum(1 for h in hosts if isinstance(h.get("registered_at"), (int, float)) and prev_thirty <= h["registered_at"] < thirty_days_ago)
    curr_hosts = sum(1 for h in hosts if isinstance(h.get("registered_at"), (int, float)) and h["registered_at"] >= thirty_days_ago)

    # Job count trends from usage_meters
    prev_jobs = 0
    curr_jobs = sum(d.get("jobs", 0) for d in daily_jobs)  # daily_jobs already covers current 30d
    try:
        be = get_billing_engine()
        with be._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM usage_meters WHERE created_at >= %s AND created_at < %s",
                (prev_thirty, thirty_days_ago),
            ).fetchone()
            prev_jobs = int(row["cnt"]) if row else 0
    except Exception as e:
        log.debug("previous period jobs comparison failed: %s", e)

    trends = {
        "users_pct": _pct(curr_users, prev_users),
        "hosts_pct": _pct(curr_hosts, prev_hosts),
        "jobs_pct": _pct(curr_jobs, prev_jobs),
        "revenue_pct": _pct(curr_revenue, prev_revenue),
    }

    # Computed KPIs
    gpu_utilization = round(len(active_hosts) / len(hosts) * 100, 1) if hosts else 0.0
    job_failure_rate = 0.0
    try:
        be = get_billing_engine()
        with be._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE status = 'failed') AS failed, COUNT(*) AS total FROM usage_meters WHERE created_at >= %s",
                (thirty_days_ago,),
            ).fetchone()
            if row and row["total"] > 0:
                job_failure_rate = round(row["failed"] / row["total"] * 100, 1)
    except Exception as e:
        log.debug("job failure rate calculation failed: %s", e)
    arpu = round(curr_revenue / len(users), 2) if users else 0.0

    return {
        "ok": True,
        "days": days,
        "kpis": {
            "total_users": len(users),
            "active_hosts": len(active_hosts),
            "running_jobs": len(running),
            "total_jobs": len(jobs),
            "revenue_mtd": revenue_mtd,
            "revenue_total": revenue_total,
            "total_gpu_hours": total_gpu_hours,
            "gpu_utilization": gpu_utilization,
            "job_failure_rate": job_failure_rate,
            "arpu": arpu,
        },
        "trends": trends,
        "daily_revenue": daily_revenue,
        "daily_signups": daily_signups,
        "daily_jobs": daily_jobs,
    }

@router.post("/api/admin/users/{email}/role", tags=["Admin"])
def api_admin_set_user_role(email: str, request: Request, role: str = "submitter"):
    """Admin sets a user's role."""
    _require_admin(request)
    allowed_roles = {"submitter", "provider", "admin"}
    if role not in allowed_roles:
        raise HTTPException(400, f"Invalid role. Allowed: {', '.join(sorted(allowed_roles))}")
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        user = _users_db.get(email)
    if not user:
        raise HTTPException(404, "User not found")
    if _USE_PERSISTENT_AUTH:
        UserStore.update_user(email, {"role": role})
    else:
        _users_db[email]["role"] = role
    emit_event("user_role_changed", {"email": email, "role": role})
    return {"ok": True, "email": email, "role": role}

@router.post("/api/admin/users/{email}/toggle-admin", tags=["Admin"])
def api_admin_toggle_admin(email: str, request: Request):
    """Admin toggles another user's admin flag."""
    _require_admin(request)
    if _USE_PERSISTENT_AUTH:
        user = UserStore.get_user(email)
    else:
        user = _users_db.get(email)
    if not user:
        raise HTTPException(404, "User not found")
    current = _admin_flag(user.get("is_admin")) == 1
    new_val = 0 if current else 1
    if _USE_PERSISTENT_AUTH:
        UserStore.set_admin(email, new_val)
    else:
        _users_db[email]["is_admin"] = new_val
    emit_event("user_admin_toggled", {"email": email, "is_admin": new_val})
    return {"ok": True, "email": email, "is_admin": new_val}

@router.get("/api/admin/revenue", tags=["Admin"])
def api_admin_revenue(request: Request, days: int = 90):
    """Admin revenue breakdown — by day, GPU model, province, top customers/providers."""
    _require_admin(request)
    import datetime as _dt

    since = time.time() - days * 86400
    result: dict = {"ok": True, "days": days}

    try:
        be = get_billing_engine()
    except Exception as e:
        log.warning("admin_revenue: failed to get billing engine", exc_info=True)
        result.update(daily=[], by_gpu=[], by_province=[], top_customers=[], top_providers=[])
        return result

    # Daily revenue
    try:
        with be._conn() as conn:
            rows = conn.execute(
                "SELECT to_char(to_timestamp(created_at), 'YYYY-MM-DD') AS day, "
                "ROUND(SUM(total_cost_cad)::numeric, 2) AS revenue, COUNT(*) AS jobs, "
                "ROUND(SUM(gpu_seconds)::numeric / 3600, 1) AS gpu_hours "
                "FROM usage_meters WHERE created_at >= %s GROUP BY day ORDER BY day",
                (since,),
            ).fetchall()
            result["daily"] = [
                {"date": r["day"], "revenue": float(r["revenue"]), "jobs": r["jobs"], "gpu_hours": float(r["gpu_hours"])}
                for r in rows
            ]
    except Exception as e:
        log.warning("admin_revenue: failed to fetch daily revenue", exc_info=True)
        result["daily"] = []

    # By GPU model
    try:
        with be._conn() as conn:
            rows = conn.execute(
                "SELECT COALESCE(gpu_model, 'Unknown') AS gpu_model, ROUND(SUM(total_cost_cad)::numeric, 2) AS revenue, COUNT(*) AS jobs "
                "FROM usage_meters WHERE created_at >= %s GROUP BY COALESCE(gpu_model, 'Unknown') ORDER BY revenue DESC",
                (since,),
            ).fetchall()
            result["by_gpu"] = [{"gpu_model": r["gpu_model"], "revenue": float(r["revenue"]), "jobs": r["jobs"]} for r in rows]
    except Exception as e:
        log.warning("admin_revenue: failed to fetch revenue by GPU", exc_info=True)
        result["by_gpu"] = []

    # By province
    try:
        with be._conn() as conn:
            rows = conn.execute(
                "SELECT COALESCE(province, 'Unknown') AS province, ROUND(SUM(total_cost_cad)::numeric, 2) AS revenue, COUNT(*) AS jobs "
                "FROM usage_meters WHERE created_at >= %s GROUP BY COALESCE(province, 'Unknown') ORDER BY revenue DESC",
                (since,),
            ).fetchall()
            result["by_province"] = [{"province": r["province"], "revenue": float(r["revenue"]), "jobs": r["jobs"]} for r in rows]
    except Exception as e:
        log.warning("admin_revenue: failed to fetch revenue by province", exc_info=True)
        result["by_province"] = []

    # Top customers (by spend)
    try:
        with be._conn() as conn:
            rows = conn.execute(
                "SELECT owner AS email, ROUND(SUM(total_cost_cad)::numeric, 2) AS total_spend, COUNT(*) AS jobs "
                "FROM usage_meters WHERE created_at >= %s GROUP BY owner ORDER BY total_spend DESC LIMIT 10",
                (since,),
            ).fetchall()
            result["top_customers"] = [{"email": r["email"], "total_spend": float(r["total_spend"]), "jobs": r["jobs"]} for r in rows]
    except Exception as e:
        log.warning("admin_revenue: failed to fetch top customers", exc_info=True)
        result["top_customers"] = []

    # Top providers (by earnings)
    try:
        with be._conn() as conn:
            rows = conn.execute(
                "SELECT provider_id, ROUND(SUM(provider_payout_cad)::numeric, 2) AS earnings, COUNT(*) AS jobs "
                "FROM payout_ledger WHERE created_at >= %s GROUP BY provider_id ORDER BY earnings DESC LIMIT 10",
                (since,),
            ).fetchall()
            result["top_providers"] = [{"provider_id": r["provider_id"], "earnings": float(r["earnings"]), "jobs": r["jobs"]} for r in rows]
    except Exception as e:
        log.warning("admin_revenue: failed to fetch top providers", exc_info=True)
        result["top_providers"] = []

    return result

@router.get("/api/admin/infrastructure", tags=["Admin"])
def api_admin_infrastructure(request: Request):
    """Admin infrastructure view — hosts by state, GPU model, province, verification, reputation."""
    _require_admin(request)

    hosts = list_hosts(active_only=False)

    # By status
    state_counts: dict[str, int] = {}
    gpu_counts: dict[str, int] = {}
    province_counts: dict[str, int] = {}
    for h in hosts:
        st = h.get("status") or "unknown"
        state_counts[st] = state_counts.get(st, 0) + 1
        gpu = h.get("gpu_model") or "Unknown"
        gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
        prov = h.get("province") or "Unknown"
        province_counts[prov] = province_counts.get(prov, 0) + 1

    # Verification stats
    verification_stats: dict[str, int] = {}
    try:
        ve = get_verification_engine()
        with ve.store._conn() as conn:
            for r in conn.execute(
                "SELECT state, COUNT(*) AS cnt FROM host_verifications GROUP BY state"
            ).fetchall():
                verification_stats[r["state"]] = r["cnt"]
    except Exception as e:
        log.warning("admin_infrastructure: failed to fetch verification data", exc_info=True)

    # Reputation tiers
    reputation_tiers: dict[str, int] = {}
    try:
        from db import _get_pg_pool
        from psycopg.rows import dict_row
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            for r in conn.execute(
                "SELECT tier, COUNT(*) AS cnt FROM reputation_scores GROUP BY tier"
            ).fetchall():
                reputation_tiers[r["tier"]] = r["cnt"]
    except Exception as e:
        log.warning("admin_infrastructure: failed to fetch reputation data", exc_info=True)

    return {
        "ok": True,
        "total_hosts": len(hosts),
        "by_state": [{"state": k, "count": v} for k, v in state_counts.items()],
        "by_gpu": [{"gpu_model": k, "count": v} for k, v in gpu_counts.items()],
        "by_province": [{"province": k, "count": v} for k, v in province_counts.items()],
        "verification": [{"state": k, "count": v} for k, v in verification_stats.items()],
        "reputation_tiers": [{"tier": k, "count": v} for k, v in reputation_tiers.items()],
    }

@router.get("/api/admin/activity", tags=["Admin"])
def api_admin_activity(request: Request, days: int = 7, limit: int = 100):
    """Admin activity feed — recent events, job stats by day, events by type."""
    _require_admin(request)
    import datetime as _dt

    since = time.time() - days * 86400
    result: dict = {"ok": True, "days": days}

    # Recent events
    try:
        store = get_event_store()
        events = store.get_events(since=since, limit=limit)
        result["events"] = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "entity_type": e.entity_type,
                "entity_id": e.entity_id,
                "timestamp": _dt.datetime.fromtimestamp(e.timestamp, tz=_dt.timezone.utc).isoformat(),
                "actor": e.actor,
                "data": e.data,
            }
            for e in events[-limit:]
        ]
    except Exception as e:
        log.warning("admin_activity: failed to fetch events", exc_info=True)
        result["events"] = []

    # Events by type (last N days)
    try:
        store = get_event_store()
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) AS cnt FROM events WHERE timestamp >= %s GROUP BY event_type ORDER BY cnt DESC",
                (since,),
            ).fetchall()
            result["by_type"] = [{"event_type": r["event_type"], "count": r["cnt"]} for r in rows]
    except Exception as e:
        log.warning("admin_activity: failed to fetch events by type", exc_info=True)
        result["by_type"] = []

    # Daily job submissions + completions
    try:
        store = get_event_store()
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT to_char(to_timestamp(timestamp), 'YYYY-MM-DD') AS day, "
                "SUM(CASE WHEN event_type = 'job_submitted' THEN 1 ELSE 0 END) AS submitted, "
                "SUM(CASE WHEN event_type = 'job_completed' THEN 1 ELSE 0 END) AS completed, "
                "SUM(CASE WHEN event_type = 'job_failed' THEN 1 ELSE 0 END) AS failed "
                "FROM events WHERE timestamp >= %s AND event_type IN ('job_submitted', 'job_completed', 'job_failed') "
                "GROUP BY day ORDER BY day",
                (since,),
            ).fetchall()
            result["daily_jobs"] = [
                {"date": r["day"], "submitted": r["submitted"], "completed": r["completed"], "failed": r["failed"]}
                for r in rows
            ]
    except Exception as e:
        log.warning("admin_activity: failed to fetch daily jobs", exc_info=True)
        result["daily_jobs"] = []

    return result

@router.get("/api/admin/verification-queue", tags=["Admin"])
def api_admin_verification_queue(request: Request):
    """Get verification queue for admin panel."""
    _require_admin(request)
    import datetime as _dt
    ve = get_verification_engine()
    store = ve.store
    hosts_map = {h["host_id"]: h for h in list_hosts(active_only=False)}
    try:
        with store._conn() as conn:
            rows = conn.execute(
                "SELECT host_id, state, COALESCE(overall_score, 0) AS overall_score, last_check_at "
                "FROM host_verifications WHERE state IN ('pending', 'unverified') ORDER BY last_check_at DESC"
            ).fetchall()
        queue = []
        for r in rows:
            hid = r["host_id"]
            h = hosts_map.get(hid, {})
            last_ts = r["last_check_at"]
            queue.append({
                "host_id": hid,
                "state": r["state"],
                "overall_score": float(r["overall_score"]),
                "last_check_at": _dt.datetime.fromtimestamp(last_ts, tz=_dt.timezone.utc).isoformat() if last_ts else None,
                "gpu_model": h.get("gpu_model") or "Unknown",
                "province": h.get("province") or "Unknown",
                "cost_per_hour": float(h.get("cost_per_hour", 0)),
            })
    except Exception as e:
        log.warning("admin_verification_queue: failed to fetch queue", exc_info=True)
        queue = []
    return {"ok": True, "queue": queue}

