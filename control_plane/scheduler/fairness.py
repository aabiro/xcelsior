"""Durable weighted service credit and operator-visible queue decisions.

PostgreSQL owns policy, time, ordering and service credit. Failed placement
transactions spend nothing. All reservations for a team take its state lock
before host/device locks, so two scheduler replicas cannot overrun its quota.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from psycopg import sql
from psycopg.rows import dict_row

POLICY_VERSION = "fairness/v1"
AGE_STEP_SECONDS = 300
MAX_AGE_BOOST = 20


def team_key_sql(alias="j"):
    # The alias is a source-code constant, never a caller value.
    if alias not in {"j", "other"}:
        raise ValueError("unsupported team-key alias")
    return (
        f"COALESCE('team:'||NULLIF({alias}.spec->>'team_id',''),"
        f"'tenant:'||NULLIF({alias}.spec->>'tenant_id',''),"
        f"'owner:'||NULLIF(COALESCE({alias}.spec->>'owner_id',{alias}.spec->>'owner'),''),"
        f"'job:'||{alias}.job_id)"
    )


@dataclass(frozen=True)
class TeamPolicy:
    key: str
    version: int = 0
    administrative_class: int = 0
    reservation_entitlement: int = 0
    weight: Decimal = Decimal(1)
    max_concurrent_jobs: int | None = None
    max_allocated_gpus: int | None = None
    preemption_enabled: bool = False
    grace_seconds: int = 30
    force_after_deadline: bool = False


class QuotaExceeded(Exception):
    code = "team_quota_exceeded"


def publish_policy(conn, policy: TeamPolicy, *, created_by: str) -> int:
    """Append a validated policy revision; never rewrite historical decisions."""
    if not policy.key.startswith(("team:", "tenant:", "owner:")) or not created_by.strip():
        raise ValueError("operator policy requires a scoped identity and actor")
    # Shared lock order with reservation; publishing cannot change an in-flight
    # quota decision. This is transactional accounting, not scheduler leadership.
    conn.execute(
        "SELECT virtual_time FROM scheduling_virtual_clock WHERE singleton FOR UPDATE"
    ).fetchone()
    conn.execute(
        "INSERT INTO scheduling_team_state(team_key) VALUES (%s) ON CONFLICT DO NOTHING",
        (policy.key,),
    )
    conn.execute(
        "SELECT 1 FROM scheduling_team_state WHERE team_key=%s FOR UPDATE", (policy.key,)
    ).fetchone()
    revision = conn.execute(
        "SELECT COALESCE(MAX(version),0)+1 AS version FROM scheduling_team_policies WHERE team_key=%s",
        (policy.key,),
    ).fetchone()
    if revision is None:
        raise RuntimeError("policy version query returned no row")
    version = int(revision["version"] if isinstance(revision, dict) else revision[0])
    conn.execute(
        """INSERT INTO scheduling_team_policies
        (team_key,version,administrative_class,reservation_entitlement,weight,
         max_concurrent_jobs,max_allocated_gpus,preemption_enabled,grace_seconds,
         force_after_deadline,created_by)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
        (
            policy.key,
            version,
            policy.administrative_class,
            policy.reservation_entitlement,
            policy.weight,
            policy.max_concurrent_jobs,
            policy.max_allocated_gpus,
            policy.preemption_enabled,
            policy.grace_seconds,
            policy.force_after_deadline,
            created_by,
        ),
    )
    return version


def claim_sql():
    """Calculate current ranks, then persist the chosen key with its claim.

    All eligible rows participate, so a stale diagnostic refresh cannot change
    fairness. SKIP LOCKED still permits independent replicas to claim different
    jobs. Reservation serializes service credit/quota before consuming devices.
    """
    key = team_key_sql()
    return f"""WITH ranks AS MATERIALIZED (
        SELECT j.job_id,{key} AS team_key,COALESCE(p.version,0) AS policy_revision,
            COALESCE(p.administrative_class,0) AS admin_class,
            COALESCE(p.reservation_entitlement,0) AS entitlement,
            COALESCE(p.weight,1) AS weight,
            GREATEST(0,LEAST(COALESCE(j.priority,0),979)) AS base_priority,
            GREATEST(0,LEAST({MAX_AGE_BOOST},floor(EXTRACT(EPOCH FROM
                (statement_timestamp()-j.queued_at))/{AGE_STEP_SECONDS})))::int AS age_boost,
            CASE WHEN j.spec->>'num_gpus' ~ '^[1-9][0-9]{{0,3}}$'
                THEN LEAST((j.spec->>'num_gpus')::int,1024) ELSE 1 END AS gpu_cost,
            GREATEST(COALESCE(s.virtual_finish,0),clock.virtual_time) AS virtual_start,
            p.max_concurrent_jobs,p.max_allocated_gpus,
            (j.spec->'preemptible'='true'::jsonb OR j.spec->'spot'='true'::jsonb
                OR j.spec->>'pricing_mode'='spot') IS TRUE AS preemptible
        FROM jobs j CROSS JOIN scheduling_virtual_clock clock
        LEFT JOIN scheduling_team_state s ON s.team_key={key}
        LEFT JOIN LATERAL (SELECT * FROM scheduling_team_policies p
            WHERE p.team_key={key} ORDER BY p.version DESC LIMIT 1) p ON TRUE
        WHERE j.phase='pending' AND j.desired_state='running'
          AND j.spec->>'execution_spec_version'='v1'
          AND (j.next_schedule_at IS NULL OR j.next_schedule_at<=statement_timestamp())
          AND (j.schedule_claim_expires_at IS NULL OR j.schedule_claim_expires_at<statement_timestamp())
          AND (%(scope_all)s OR lower(COALESCE(j.spec->>'gpu_model',''))=ANY(%(scope_models)s)
               OR lower(COALESCE(j.spec->>'scheduler',''))='v2')
    ), candidate AS (
        SELECT j.job_id FROM jobs j JOIN ranks r USING(job_id)
        ORDER BY r.admin_class*1000+r.base_priority+r.age_boost DESC,
            r.entitlement DESC,r.virtual_start+r.gpu_cost/r.weight ASC,
            j.queued_at ASC NULLS LAST,j.job_id ASC
        LIMIT 1 FOR UPDATE OF j SKIP LOCKED
    )
    UPDATE jobs j SET schedule_claim_owner=%(replica_id)s,
        schedule_claim_token=gen_random_uuid(),
        schedule_claim_expires_at=statement_timestamp()+make_interval(secs=>%(ttl_sec)s),
        schedule_attempt_count=j.schedule_attempt_count+1,version=j.version+1,
        updated_at=statement_timestamp(),scheduling_team_key=r.team_key,
        queue_policy_version=r.policy_revision,
        effective_priority=r.admin_class*1000+r.base_priority+r.age_boost,
        reservation_entitlement=r.entitlement,
        fair_share_finish=r.virtual_start+r.gpu_cost/r.weight,
        queue_calculated_at=statement_timestamp(),
        queue_components=jsonb_build_object(
            'policy','{POLICY_VERSION}','revision',r.policy_revision,
            'administrative_class',r.admin_class,'entitlement',r.entitlement,
            'base_priority',r.base_priority,'age_boost',r.age_boost,'age_boost_cap',{MAX_AGE_BOOST},
            'team',r.team_key,'weight',r.weight,'gpu_cost',r.gpu_cost,
            'virtual_start',r.virtual_start,'virtual_finish',r.virtual_start+r.gpu_cost/r.weight,
            'preemptible',r.preemptible,'max_concurrent_jobs',r.max_concurrent_jobs,
            'max_allocated_gpus',r.max_allocated_gpus)
    FROM candidate c JOIN ranks r USING(job_id) WHERE j.job_id=c.job_id
    RETURNING j.job_id,j.schedule_claim_token,j.schedule_claim_expires_at,
        j.schedule_attempt_count,j.version,j.generation,j.spec,j.spec_hash,j.payload,
        j.priority,j.effective_priority"""


def lock_team_policy(
    conn, *, job_id: str, requested_gpus: int, check_quota: bool = True
) -> TeamPolicy:
    """The caller already owns the job row; take this lock before its host.

    Policy publication takes the same state lock. A revision cannot change
    limits while an in-flight reservation checks and consumes capacity.
    """
    key = sql.SQL(team_key_sql())
    with conn.cursor(row_factory=dict_row) as cur:
        job = cur.execute(
            sql.SQL("SELECT {} AS team_key FROM jobs j WHERE j.job_id=%s").format(key), (job_id,)
        ).fetchone()
        if job is None:
            raise ValueError("team policy requires a present job")
        cur.execute(
            "SELECT virtual_time FROM scheduling_virtual_clock WHERE singleton FOR UPDATE"
        ).fetchone()
        group = job["team_key"]
        cur.execute(
            "INSERT INTO scheduling_team_state(team_key) VALUES (%s) ON CONFLICT DO NOTHING",
            (group,),
        )
        cur.execute(
            "SELECT virtual_finish FROM scheduling_team_state WHERE team_key=%s FOR UPDATE",
            (group,),
        ).fetchone()
        record = cur.execute(
            "SELECT * FROM scheduling_team_policies WHERE team_key=%s ORDER BY version DESC LIMIT 1",
            (group,),
        ).fetchone()
        fields = (
            {name: record[name] for name in TeamPolicy.__dataclass_fields__ if name != "key"}
            if record
            else {}
        )
        policy = TeamPolicy(key=group, **fields)
    if check_quota:
        require_team_quota(conn, policy, requested_gpus=requested_gpus)
    return policy


def require_team_quota(conn, policy, *, requested_gpus, replacing_attempts=()):
    """Caller holds team policy/accounting locks and every replacement job.

    Planning can credit only the exact eligible victims it will durably stop.
    Reservation calls this without credits and still waits for physical release.
    Other teams' allocations never contribute replacement credit.
    """
    key = team_key_sql()
    with conn.cursor(row_factory=dict_row) as cur:
        usage = cur.execute(
            f"""SELECT count(DISTINCT a.attempt_id) AS jobs,
                count(DISTINCT a.gpu_device_id) AS gpus
            FROM gpu_device_allocations a JOIN jobs j ON j.job_id=a.job_id
            WHERE a.released_at IS NULL AND {key}=%s
              AND NOT(a.attempt_id=ANY(%s::uuid[]))""",
            (policy.key, list(replacing_attempts)),
        ).fetchone()
        if usage is None:
            raise RuntimeError("team capacity query returned no row")
        if (
            policy.max_concurrent_jobs is not None
            and usage["jobs"] + 1 > policy.max_concurrent_jobs
        ):
            raise QuotaExceeded("team concurrent-job limit reached")
        if (
            policy.max_allocated_gpus is not None
            and usage["gpus"] + requested_gpus > policy.max_allocated_gpus
        ):
            raise QuotaExceeded("team GPU quota reached")


def spend_service_credit(conn, policy: TeamPolicy, *, gpu_count: int) -> Decimal:
    if gpu_count < 1:
        raise ValueError("service credit requires a positive GPU count")
    row = conn.execute(
        """UPDATE scheduling_team_state
        SET virtual_finish=GREATEST(virtual_finish,
            (SELECT virtual_time FROM scheduling_virtual_clock WHERE singleton))+%s/weight,
            updated_at=statement_timestamp()
        FROM (SELECT %s::numeric AS weight) p
        WHERE team_key=%s RETURNING virtual_finish""",
        (Decimal(gpu_count), policy.weight, policy.key),
    ).fetchone()
    if row is None:
        raise RuntimeError("service credit requires the locked team state")
    key = team_key_sql()
    conn.execute(f"""UPDATE scheduling_virtual_clock SET virtual_time=GREATEST(virtual_time,
        COALESCE((SELECT MIN(COALESCE(s.virtual_finish,virtual_time)) FROM jobs j
            LEFT JOIN scheduling_team_state s ON s.team_key={key}
            WHERE j.phase IN ('pending','scheduled','starting','running')
              AND j.desired_state='running' AND j.spec->>'execution_spec_version'='v1'),virtual_time))
        WHERE singleton""")
    return Decimal(row["virtual_finish"] if isinstance(row, dict) else row[0])
