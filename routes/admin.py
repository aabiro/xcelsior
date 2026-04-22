"""Routes: admin."""

import time
import uuid

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
)
from db import NotificationStore, UserStore, WebPushSubscriptionStore, emit_event
from billing import get_billing_engine
from verification import get_verification_engine
from events import get_event_store
from web_push import get_web_push_observability_snapshot

router = APIRouter()


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
            "team_id": u.get("team_id") or None,
        })
    return {"ok": True, "users": safe_users}

@router.get("/api/admin/overview", tags=["Admin"])
def api_admin_overview(request: Request, days: int = 30):
    """Admin overview with KPIs and trend data for a configurable window."""
    user = _require_admin(request)
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
    web_push = {
        **get_web_push_observability_snapshot(),
        "notification_retained_total": 0,
        "current_user_subscription_count": 0,
    }
    try:
        web_push["notification_retained_total"] = NotificationStore.total_count()
        web_push["current_user_subscription_count"] = WebPushSubscriptionStore.count_active_for_user(
            user.get("email", ""),
        )
    except Exception as e:
        log.debug("admin_overview: failed to fetch web push data: %s", e)

    # Volume KPIs
    total_volumes = 0
    total_storage_gb = 0.0
    attached_volumes = 0
    volume_revenue = 0.0
    try:
        from volumes import VolumeEngine
        ve = VolumeEngine()
        with ve._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt, COALESCE(SUM(size_gb), 0) AS total_gb, "
                "COUNT(*) FILTER (WHERE status = 'attached') AS attached "
                "FROM volumes WHERE status != 'deleted'"
            ).fetchone()
            if row:
                total_volumes = row["cnt"]
                total_storage_gb = round(float(row["total_gb"]), 1)
                attached_volumes = row["attached"]
    except Exception as e:
        log.debug("admin_overview: failed to fetch volume KPIs: %s", e)
    try:
        be = get_billing_engine()
        with be._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(amount_cad), 0) AS rev FROM billing_cycles "
                "WHERE gpu_model = 'storage' AND tier = 'volume' AND status = 'charged'"
            ).fetchone()
            volume_revenue = round(float(row["rev"]), 2) if row else 0.0
    except Exception as e:
        log.debug("admin_overview: volume revenue query failed: %s", e)

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
            "total_volumes": total_volumes,
            "total_storage_gb": total_storage_gb,
            "attached_volumes": attached_volumes,
            "volume_revenue": volume_revenue,
        },
        "trends": trends,
        "daily_revenue": daily_revenue,
        "daily_signups": daily_signups,
        "daily_jobs": daily_jobs,
        "web_push": web_push,
    }


@router.post("/api/admin/web-push/test-notification", tags=["Admin"])
def api_admin_web_push_test_notification(request: Request):
    """Send a real web-push smoke notification to the current admin user."""
    user = _require_admin(request)
    user_email = user.get("email", "")
    if not user_email:
        raise HTTPException(400, "Current admin user email is unavailable")

    smoke_id = f"smoke-{uuid.uuid4().hex[:12]}"
    action_url = f"/dashboard/admin?push_smoke=1&smoke_id={smoke_id}"
    created_at = int(time.time())
    title = "Desktop push smoke test"
    body = "Open this notification to confirm desktop click-through on the admin dashboard."

    notification_id = NotificationStore.create(
        user_email,
        "admin_web_push_smoke",
        title,
        body,
        data={
            "smoke_test": True,
            "smoke_id": smoke_id,
            "triggered_by": user_email,
            "created_at": created_at,
        },
        action_url=action_url,
        entity_type="admin_web_push_smoke",
        entity_id=smoke_id,
        priority=10,
    )

    return {
        "ok": True,
        "notification_id": notification_id,
        "smoke_id": smoke_id,
        "title": title,
        "body": body,
        "action_url": action_url,
        "user_email": user_email,
        "active_subscription_count": WebPushSubscriptionStore.count_active_for_user(user_email),
        "web_push": get_web_push_observability_snapshot(),
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

@router.get("/api/admin/teams", tags=["Admin"])
def api_admin_teams(request: Request):
    """List all teams with their members. Platform admin only."""
    _require_admin(request)
    from db import get_db
    teams = []
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM teams ORDER BY created_at DESC").fetchall()
        for row in rows:
            team = dict(row)
            members = UserStore.list_team_members(team["team_id"])
            team["members"] = members
            teams.append(team)
    return {"ok": True, "teams": teams}

@router.delete("/api/admin/teams/{team_id}/members/{email}", tags=["Admin"])
def api_admin_remove_team_member(team_id: str, email: str, request: Request):
    """Remove a member from any team. Platform admin override."""
    _require_admin(request)
    team = UserStore.get_team(team_id)
    if not team:
        raise HTTPException(404, "Team not found")
    if email == team["owner_email"]:
        raise HTTPException(400, "Cannot remove team owner")
    members = UserStore.list_team_members(team_id)
    if not any(m["email"] == email for m in members):
        raise HTTPException(404, f"{email} is not a member of this team")
    UserStore.remove_team_member(team_id, email)
    emit_event("team_member_removed", {"team_id": team_id, "email": email})
    return {"ok": True, "message": f"{email} removed from team {team_id}"}

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


# ---------------------------------------------------------------------------
# AI Insights
# ---------------------------------------------------------------------------

@router.get("/api/admin/ai-stats", tags=["Admin"])
def api_admin_ai_stats(request: Request, days: int = 30):
    """Aggregate AI conversation statistics for the admin dashboard."""
    _require_admin(request)
    import datetime as _dt
    from db import _get_pg_pool
    from psycopg.rows import dict_row

    cutoff = time.time() - days * 86400
    pool = _get_pg_pool()
    with pool.connection() as conn:
        conn.row_factory = dict_row

        # --- per-source totals from ai_conversations ---
        source_rows = conn.execute(
            "SELECT source, COUNT(*) AS conversations, "
            "COALESCE(SUM(message_count), 0) AS messages, "
            "COALESCE(SUM(total_input_tokens), 0) AS input_tokens, "
            "COALESCE(SUM(total_output_tokens), 0) AS output_tokens "
            "FROM ai_conversations WHERE created_at >= %s "
            "GROUP BY source ORDER BY conversations DESC",
            (cutoff,),
        ).fetchall()

        # --- support (public chat) totals ---
        support_row = conn.execute(
            "SELECT COUNT(DISTINCT c.conversation_id) AS conversations, "
            "COUNT(m.id) AS messages "
            "FROM chat_conversations c "
            "LEFT JOIN chat_messages m ON m.conversation_id = c.conversation_id "
            "WHERE c.created_at >= %s",
            (cutoff,),
        ).fetchone()

        # --- daily breakdown for chart ---
        daily_rows = conn.execute(
            "SELECT FLOOR(created_at / 86400)::bigint AS day_epoch, source, COUNT(*) AS cnt "
            "FROM ai_conversations WHERE created_at >= %s "
            "GROUP BY day_epoch, source ORDER BY day_epoch",
            (cutoff,),
        ).fetchall()

        daily_support = conn.execute(
            "SELECT FLOOR(created_at / 86400)::bigint AS day_epoch, COUNT(*) AS cnt "
            "FROM chat_conversations WHERE created_at >= %s "
            "GROUP BY day_epoch ORDER BY day_epoch",
            (cutoff,),
        ).fetchall()

        # --- top users ---
        top_users = conn.execute(
            "SELECT user_id, COUNT(*) AS conversations, "
            "COALESCE(SUM(total_input_tokens + total_output_tokens), 0) AS total_tokens "
            "FROM ai_conversations WHERE created_at >= %s "
            "GROUP BY user_id ORDER BY conversations DESC LIMIT 10",
            (cutoff,),
        ).fetchall()

    # Build per-source summary (include support)
    by_source = {r["source"]: dict(r) for r in source_rows}
    support_convs = int(support_row["conversations"]) if support_row else 0
    support_msgs = int(support_row["messages"]) if support_row else 0
    by_source["support"] = {
        "source": "support",
        "conversations": support_convs,
        "messages": support_msgs,
        "input_tokens": 0,
        "output_tokens": 0,
    }

    total_conversations = sum(s["conversations"] for s in by_source.values())
    total_messages = sum(s["messages"] for s in by_source.values())
    total_input = sum(s["input_tokens"] for s in by_source.values())
    total_output = sum(s["output_tokens"] for s in by_source.values())
    estimated_cost = round(total_input * 0.000003 + total_output * 0.000015, 2)

    # Daily chart data
    daily_map: dict[int, dict] = {}
    for r in daily_rows:
        d = int(r["day_epoch"])
        if d not in daily_map:
            daily_map[d] = {"date": _dt.datetime.fromtimestamp(d * 86400, tz=_dt.timezone.utc).strftime("%Y-%m-%d")}
        daily_map[d][r["source"]] = int(r["cnt"])
    for r in daily_support:
        d = int(r["day_epoch"])
        if d not in daily_map:
            daily_map[d] = {"date": _dt.datetime.fromtimestamp(d * 86400, tz=_dt.timezone.utc).strftime("%Y-%m-%d")}
        daily_map[d]["support"] = int(r["cnt"])
    daily = sorted(daily_map.values(), key=lambda x: x["date"])

    return {
        "ok": True,
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "total_input_tokens": int(total_input),
        "total_output_tokens": int(total_output),
        "estimated_cost": estimated_cost,
        "by_source": list(by_source.values()),
        "daily": daily,
        "top_users": [dict(u) for u in top_users],
    }


@router.get("/api/admin/ai-conversations", tags=["Admin"])
def api_admin_ai_conversations(
    request: Request,
    source: str = "all",
    days: int = 7,
    search: str = "",
    page: int = 1,
    per_page: int = 30,
):
    """Paginated AI conversation list with messages for admin review."""
    _require_admin(request)
    import datetime as _dt
    from db import _get_pg_pool
    from psycopg.rows import dict_row
    from psycopg import sql as _sql

    cutoff = time.time() - days * 86400
    offset = (max(page, 1) - 1) * per_page
    pool = _get_pg_pool()

    conversations = []
    total = 0

    with pool.connection() as conn:
        conn.row_factory = dict_row

        # ---- AI assistant conversations ----
        if source in ("all", "xcel", "analytics", "wizard"):
            where_parts = ["c.created_at >= %s"]
            params: list = [cutoff]

            if source != "all":
                where_parts.append("c.source = %s")
                params.append(source)

            if search:
                where_parts.append(
                    "EXISTS (SELECT 1 FROM ai_messages m2 "
                    "WHERE m2.conversation_id = c.conversation_id "
                    "AND m2.content ILIKE %s)"
                )
                params.append(f"%{search}%")

            where_clause = " AND ".join(where_parts)

            # Count
            cnt = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM ai_conversations c WHERE {where_clause}",
                params,
            ).fetchone()
            ai_total = int(cnt["cnt"]) if cnt else 0

            # Fetch conversations
            ai_convs = conn.execute(
                f"SELECT c.conversation_id, c.user_id, c.title, c.source, "
                f"c.created_at, c.updated_at, c.message_count, "
                f"c.total_input_tokens, c.total_output_tokens "
                f"FROM ai_conversations c WHERE {where_clause} "
                f"ORDER BY c.updated_at DESC LIMIT %s OFFSET %s",
                params + [per_page, offset],
            ).fetchall()

            cids = [r["conversation_id"] for r in ai_convs]
            messages_map: dict[str, list] = {cid: [] for cid in cids}

            if cids:
                placeholders = ",".join(["%s"] * len(cids))
                msgs = conn.execute(
                    f"SELECT conversation_id, role, content, tool_name, "
                    f"tokens_in, tokens_out, created_at "
                    f"FROM ai_messages WHERE conversation_id IN ({placeholders}) "
                    f"ORDER BY created_at ASC",
                    cids,
                ).fetchall()
                for m in msgs:
                    messages_map[m["conversation_id"]].append({
                        "role": m["role"],
                        "content": m["content"],
                        "tool_name": m["tool_name"] or None,
                        "tokens_in": int(m["tokens_in"]),
                        "tokens_out": int(m["tokens_out"]),
                        "created_at": m["created_at"],
                    })

            for c in ai_convs:
                conversations.append({
                    "conversation_id": c["conversation_id"],
                    "source": c["source"],
                    "user": c["user_id"],
                    "title": c["title"],
                    "created_at": c["created_at"],
                    "updated_at": c["updated_at"],
                    "message_count": int(c["message_count"]),
                    "total_input_tokens": int(c["total_input_tokens"]),
                    "total_output_tokens": int(c["total_output_tokens"]),
                    "messages": messages_map.get(c["conversation_id"], []),
                })
            total += ai_total

        # ---- Public support chat ----
        if source in ("all", "support"):
            swhere = ["c.created_at >= %s"]
            sparams: list = [cutoff]

            if search:
                swhere.append(
                    "EXISTS (SELECT 1 FROM chat_messages m2 "
                    "WHERE m2.conversation_id = c.conversation_id "
                    "AND m2.content ILIKE %s)"
                )
                sparams.append(f"%{search}%")

            swhere_clause = " AND ".join(swhere)

            scnt = conn.execute(
                f"SELECT COUNT(*) AS cnt FROM chat_conversations c WHERE {swhere_clause}",
                sparams,
            ).fetchone()
            support_total = int(scnt["cnt"]) if scnt else 0

            s_offset = max(0, offset - total) if source == "all" else offset
            s_limit = max(0, per_page - len(conversations)) if source == "all" else per_page

            if s_limit > 0 and (source != "all" or len(conversations) < per_page):
                sconvs = conn.execute(
                    f"SELECT c.conversation_id, c.user_email, c.ip_hash, "
                    f"c.created_at, c.updated_at "
                    f"FROM chat_conversations c WHERE {swhere_clause} "
                    f"ORDER BY c.updated_at DESC LIMIT %s OFFSET %s",
                    sparams + [s_limit, s_offset],
                ).fetchall()

                scids = [r["conversation_id"] for r in sconvs]
                smsg_map: dict[str, list] = {cid: [] for cid in scids}

                if scids:
                    splaceholders = ",".join(["%s"] * len(scids))
                    smsgs = conn.execute(
                        f"SELECT conversation_id, role, content, created_at "
                        f"FROM chat_messages WHERE conversation_id IN ({splaceholders}) "
                        f"ORDER BY created_at ASC",
                        scids,
                    ).fetchall()
                    for m in smsgs:
                        smsg_map[m["conversation_id"]].append({
                            "role": m["role"],
                            "content": m["content"],
                            "tool_name": None,
                            "tokens_in": 0,
                            "tokens_out": 0,
                            "created_at": m["created_at"],
                        })

                for c in sconvs:
                    conversations.append({
                        "conversation_id": c["conversation_id"],
                        "source": "support",
                        "user": c["user_email"] or c["ip_hash"] or "anonymous",
                        "title": "",
                        "created_at": c["created_at"],
                        "updated_at": c["updated_at"],
                        "message_count": len(smsg_map.get(c["conversation_id"], [])),
                        "total_input_tokens": 0,
                        "total_output_tokens": 0,
                        "messages": smsg_map.get(c["conversation_id"], []),
                    })
            total += support_total

    # Sort merged results by updated_at descending
    conversations.sort(key=lambda x: x.get("updated_at", 0), reverse=True)

    return {
        "ok": True,
        "conversations": conversations[:per_page],
        "total": total,
        "page": page,
        "per_page": per_page,
    }



# ── P1.2 — Worker agent rolling self-update ────────────────────────────────
@router.post("/api/admin/agent/rollout", tags=["Admin"])
def api_admin_agent_rollout(request: Request, body: dict):
    """Enqueue an ``upgrade_agent`` directive across fleet (rolling batch).

    Body:
      version: str            — target version (stored in command args)
      sha256:  str            — expected sha256 (64 hex chars)
      url:     str | null     — defaults to https://xcelsior.ca/static/worker_agent.py
      host_ids: list[str] | null — restrict to specific hosts (default: all admitted)
      batch_pct: int           — 1–100; percentage of fleet to target this call (default 5)
      min_version: str | null  — agents at/above this are no-ops

    Returns the list of host_ids we enqueued, plus the generated command ids.
    This is deliberately stateless: call it repeatedly to roll waves (the
    dashboard drives pacing + health-checks between waves).
    """
    from routes.agent import enqueue_agent_command  # local import avoids cycle

    user = _require_admin(request)

    version = (body or {}).get("version", "").strip()
    sha256 = (body or {}).get("sha256", "").strip().lower()
    url = (body or {}).get("url") or "https://xcelsior.ca/static/worker_agent.py"
    host_ids = (body or {}).get("host_ids") or []
    batch_pct = int((body or {}).get("batch_pct") or 5)
    min_version = (body or {}).get("min_version") or version

    if not version or not sha256:
        raise HTTPException(400, "version and sha256 are required")
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise HTTPException(400, "sha256 must be 64 lowercase hex chars")
    if batch_pct < 1 or batch_pct > 100:
        raise HTTPException(400, "batch_pct must be 1..100")

    # Target set: explicit host_ids or all currently-admitted hosts
    all_hosts = list_hosts()
    admitted = [h for h in all_hosts if h.get("status") != "decommissioned"]
    if host_ids:
        targets = [h for h in admitted if h.get("host_id") in set(host_ids)]
    else:
        # Skip hosts already at the target sha to keep rollouts idempotent.
        targets = [h for h in admitted if (h.get("agent_sha256") or "") != sha256]

    # Rolling: take the first batch_pct% (deterministic ordering by host_id)
    targets.sort(key=lambda h: h.get("host_id", ""))
    batch_n = max(1, (len(targets) * batch_pct + 99) // 100)
    batch = targets[:batch_n]

    enqueued = []
    args = {"url": url, "sha256": sha256, "min_version": min_version}
    for h in batch:
        hid = h.get("host_id")
        if not hid:
            continue
        try:
            cmd_id = enqueue_agent_command(
                hid, "upgrade_agent", args=args, created_by=user.get("email", "admin"),
            )
            enqueued.append({"host_id": hid, "cmd_id": cmd_id})
        except Exception as e:
            log.warning("rollout: enqueue failed for host=%s: %s", hid, e)

    return {
        "ok": True,
        "version": version,
        "sha256": sha256,
        "batch_pct": batch_pct,
        "candidates": len(targets),
        "enqueued": enqueued,
    }
