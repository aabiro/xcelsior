# Xcelsior Billing & Metering — CAD-first
# Implements REPORT_FEATURE_FINAL.md + REPORT_MARKETING_FINAL.md:
#   - CAD pricing (competitors price in USD = procurement friction for CA buyers)
#   - Per-job metering
#   - Trust tier pricing multipliers
#   - Stripe Connect–ready payout structure
#   - Provider attestation bundle for compliance

import json
import logging
import os
import time

from money import cad_to_micros, micros_to_cad
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional, cast

log = logging.getLogger("xcelsior")


# ── Currency ─────────────────────────────────────────────────────────


class Currency(str, Enum):
    CAD = "CAD"
    USD = "USD"


# Default: price in CAD. This is a deliberate strategic decision per
# REPORT_MARKETING_FINAL.md — most competitors price in USD, which
# creates currency risk and procurement complexity for Canadian buyers.
DEFAULT_CURRENCY = Currency.CAD

# Approximate CAD/USD rate for display purposes (not for billing)
CAD_USD_RATE = float(os.environ.get("XCELSIOR_CAD_USD_RATE", "0.73"))


# ── Province-Aware GST/HST Rates ─────────────────────────────────────
# From REPORT_FEATURE_FINAL.md: "Canadian GST/HST rules for digital-economy
# distribution platform operators can impose collection obligations."
# Digital services in Canada are subject to GST/HST. Rates vary by province.

# How long an unresolved auto-top-up intent holds its wallet back from another
# charge. Longer than the 300s sweep so a confirmation that is merely slow never
# races a second charge, short enough that an intent abandoned by Stripe does not
# disable top-up for the rest of the day.
_TOPUP_INFLIGHT_SECONDS = 900

GST_RATE = 0.05  # Federal GST: 5%

PROVINCE_TAX_RATES = {
    # Province: (combined_rate, description)
    "AB": (0.05, "GST 5% (no PST)"),
    "BC": (0.12, "GST 5% + PST 7%"),
    "MB": (0.12, "GST 5% + RST 7%"),
    "NB": (0.15, "HST 15%"),
    "NL": (0.15, "HST 15%"),
    "NS": (0.15, "HST 15%"),
    "NT": (0.05, "GST 5% (no PST)"),
    "NU": (0.05, "GST 5% (no PST)"),
    "ON": (0.13, "HST 13%"),
    "PE": (0.15, "HST 15%"),
    "QC": (0.14975, "GST 5% + QST 9.975%"),
    "SK": (0.11, "GST 5% + PST 6%"),
    "YT": (0.05, "GST 5% (no PST)"),
}


def get_tax_rate_for_province(province: str) -> tuple[float, str]:
    """Get the combined GST/HST/PST rate for a Canadian province.

    Returns:
        (rate, description) — e.g., (0.13, "HST 13%")
        Falls back to GST-only (5%) for unknown provinces.
    """
    code = province.upper().strip()
    if code in PROVINCE_TAX_RATES:
        return PROVINCE_TAX_RATES[code]
    return (GST_RATE, f"GST {GST_RATE*100:.0f}% (province unknown)")


# ── Small-Supplier Threshold ─────────────────────────────────────────
# Per Excise Tax Act: a distribution platform operator must register for
# GST/HST once total taxable revenue exceeds $30,000 over any four
# consecutive calendar quarters.

GST_SMALL_SUPPLIER_THRESHOLD_CAD = 30_000.00


def resolve_job_pricing_mode(job: dict) -> str:
    """Normalized pricing mode for billing (on_demand | spot | reserved)."""
    mode = str(job.get("pricing_mode") or "").strip().lower()
    if mode in ("on_demand", "spot", "reserved"):
        return mode
    if job.get("preemptible") or job.get("spot") or job.get("tier") == "spot":
        return "spot"
    return "on_demand"


def resolve_compute_rate_cad(job: dict, host: dict | None = None) -> tuple[float, str]:
    """Effective CAD/hr for billing and pricing_mode label.

    Spot jobs bill at locked ``spot_rate_cad`` (per GPU) × ``num_gpus``.
    On-demand keeps the host ``cost_per_hour`` path unchanged.
    """
    pricing_mode = resolve_job_pricing_mode(job)
    num_gpus = max(1, int(job.get("num_gpus", 1) or 1))

    if pricing_mode == "spot":
        locked = job.get("spot_rate_cad")
        if locked is not None:
            return round(float(locked) * num_gpus, 6), "spot"
        gpu_model = (
            (job.get("gpu_model") or "")
            or (job.get("host_gpu_model") or "")
            or ((host or {}).get("gpu_model") or "")
            or "RTX 4090"
        )
        try:
            from spot_pricing import compute_live_spot_quote

            return round(compute_live_spot_quote(gpu_model).rate_cad * num_gpus, 6), "spot"
        except Exception:
            base = float((host or {}).get("cost_per_hour", 0.20))
            return round(base * 0.4 * num_gpus, 6), "spot"

    return round(float((host or {}).get("cost_per_hour", 0.20)), 6), "on_demand"


# ── Metering ─────────────────────────────────────────────────────────


@dataclass
class UsageMeter:
    """Per-job resource usage metering record."""

    meter_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    job_id: str = ""
    host_id: str = ""
    owner: str = ""
    # Placement attempt that owned the billed period (fenced work). NULL
    # for pure-legacy jobs that never had a job_attempts row.
    attempt_id: str | None = None

    # Time
    started_at: float = 0.0
    completed_at: float = 0.0
    duration_sec: float = 0.0
    gpu_seconds: float = 0.0  # Actual GPU utilization time

    # Resources
    gpu_model: str = ""
    vram_gb: float = 0.0
    gpu_utilization_pct: float = 0.0  # Average GPU util during job

    # Compute score (XCU)
    xcu_score: float = 0.0

    # Host location (invoice metadata only)
    country: str = ""
    province: str = ""

    # Trust tier
    trust_tier: str = "community"

    # Cost breakdown
    base_rate_per_hour: float = 0.0
    tier_multiplier: float = 1.0
    spot_discount: float = 0.0
    pricing_mode: str = "on_demand"
    total_cost_cad: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_meter_attempt_id(job: dict) -> str | None:
    """Resolve the placement attempt that owns this meter close.

    Preference order:
    1. Explicit attempt keys on the job dict (caller already knows authority)
    2. ``jobs.active_attempt_id`` when still bound
    3. Latest ``job_attempts`` row for the job (fenced history after clear)

    Returns None for pure-legacy jobs with no attempt history.
    """
    for key in ("attempt_id", "active_attempt_id", "_attempt_id"):
        raw = job.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()

    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        return None

    try:
        from db import _get_pg_pool

        pool = _get_pg_pool()
        with pool.connection() as conn:
            # Prefer live authority when present.
            row = conn.execute(
                "SELECT active_attempt_id FROM jobs WHERE job_id = %s",
                (job_id,),
            ).fetchone()
            if row is not None:
                active = row[0] if not isinstance(row, dict) else row.get("active_attempt_id")
                if active is not None and str(active).strip():
                    return str(active).strip()

            # Fenced history after active_attempt_id cleared (terminal settle).
            hist = conn.execute(
                """
                SELECT attempt_id FROM job_attempts
                 WHERE job_id = %s
                 ORDER BY attempt_number DESC NULLS LAST,
                          fencing_token DESC NULLS LAST
                 LIMIT 1
                """,
                (job_id,),
            ).fetchone()
            if hist is not None:
                aid = hist[0] if not isinstance(hist, dict) else hist.get("attempt_id")
                if aid is not None and str(aid).strip():
                    return str(aid).strip()
    except Exception as exc:
        log.debug("resolve_meter_attempt_id failed job=%s: %s", job_id, exc)
    return None


def _usage_meter_from_row(row: Any) -> UsageMeter:
    """Hydrate UsageMeter from a usage_meters DB row (dict preferred)."""
    if not isinstance(row, dict):
        # Best-effort sequence mapping for non-dict_row connections.
        keys = (
            "meter_id",
            "job_id",
            "host_id",
            "owner",
            "started_at",
            "completed_at",
            "duration_sec",
            "gpu_seconds",
            "gpu_model",
            "vram_gb",
            "gpu_utilization_pct",
            "xcu_score",
            "country",
            "province",
            "trust_tier",
            "base_rate_per_hour",
            "tier_multiplier",
            "spot_discount",
            "total_cost_cad",
            "created_at",
            "pricing_mode",
            "attempt_id",
        )
        row = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}

    attempt = row.get("attempt_id")
    if attempt is not None:
        attempt = str(attempt).strip() or None
    return UsageMeter(
        meter_id=str(row.get("meter_id") or ""),
        job_id=str(row.get("job_id") or ""),
        host_id=str(row.get("host_id") or ""),
        owner=str(row.get("owner") or ""),
        attempt_id=attempt,
        started_at=float(row.get("started_at") or 0),
        completed_at=float(row.get("completed_at") or 0),
        duration_sec=float(row.get("duration_sec") or 0),
        gpu_seconds=float(row.get("gpu_seconds") or 0),
        gpu_model=str(row.get("gpu_model") or ""),
        vram_gb=float(row.get("vram_gb") or 0),
        gpu_utilization_pct=float(row.get("gpu_utilization_pct") or 0),
        xcu_score=float(row.get("xcu_score") or 0),
        country=str(row.get("country") or ""),
        province=str(row.get("province") or ""),
        trust_tier=str(row.get("trust_tier") or "community"),
        base_rate_per_hour=float(row.get("base_rate_per_hour") or 0),
        tier_multiplier=float(row.get("tier_multiplier") or 1.0),
        spot_discount=float(row.get("spot_discount") or 0),
        pricing_mode=str(row.get("pricing_mode") or "on_demand"),
        # Derived from the integer column; the float twin was dropped in 087.
        total_cost_cad=micros_to_cad(row.get("total_cost_micros") or 0),
    )


# ── Invoice ──────────────────────────────────────────────────────────


@dataclass
class InvoiceLineItem:
    """Single line item on an invoice."""

    description: str = ""
    category: str = ""  # "compute", "storage", "monitoring", "security"
    quantity: float = 0.0
    unit: str = "GPU-hours"
    unit_price_cad: float = 0.0
    subtotal_cad: float = 0.0
    trust_tier: str = "community"
    job_id: str = ""
    host_id: str = ""
    province: str = ""
    line_type: str = "compute"  # compute | serverless | storage
    pricing_mode: str = ""
    gpu_model: str = ""
    stripe_product_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Invoice:
    """Customer invoice, broken out by cost category.

    Line items map to:
    - Core compute
    - Storage
    - Monitoring
    - Compute-specific security requirements
    """

    invoice_id: str = field(default_factory=lambda: f"INV-{int(time.time())}-{os.urandom(3).hex()}")
    customer_id: str = ""
    customer_name: str = ""
    currency: str = Currency.CAD

    # Period
    period_start: float = 0.0
    period_end: float = 0.0

    # Line items
    line_items: list = field(default_factory=list)

    # Totals
    subtotal_cad: float = 0.0
    tax_rate: float = 0.0  # GST/HST rate
    tax_amount_cad: float = 0.0
    total_cad: float = 0.0

    # The two fund_* columns exist so archived invoices raised
    # while the AI Compute Access Fund was running still round-trip; the program
    # has ended, so both are 0.0 on anything issued now.

    # Metadata
    created_at: float = field(default_factory=time.time)
    status: str = "draft"  # draft, issued, paid, void
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["line_items"] = [li if isinstance(li, dict) else li for li in self.line_items]
        return d


# ── Provider Attestation ─────────────────────────────────────────────
# From REPORT_MARKETING_FINAL.md: customers need supplier qualification
# evidence for procurement and compliance review.


@dataclass
class ProviderAttestation:
    """Supplier attestation bundle for procurement and compliance review."""

    attestation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    provider_name: str = "Xcelsior"
    incorporated_in: str = "Canada"
    registration_number: str = ""
    data_centers_in_canada: bool = True
    physical_infrastructure_canada: bool = True
    data_stays_in_canada: bool = True
    attested_at: float = field(default_factory=time.time)
    valid_until: float = 0.0

    # Compliance
    privacy_officer_designated: bool = False
    privacy_officer_contact: str = ""
    security_posture: str = "defense-in-depth"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Billing Engine ────────────────────────────────────────────────────


class BillingEngine:
    """Production billing engine with CAD pricing.

    Features:
    - Per-job metering
    - Trust tier pricing multipliers
    - Canadian / non-Canadian compute split on every invoice
    - Invoice generation with auditable line items
    - Stripe Connect–ready payout structure
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path  # Legacy compat — no longer used

    @contextmanager
    def _conn(self):
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

    def meter_job(
        self,
        job: dict,
        host: dict,
        host_location: Optional[dict] = None,
        trust_tier: str = "community",
    ) -> UsageMeter:
        """Create a metering record for a completed job.

        This is the source of truth for billing. Every completed job
        gets a usage meter that records exactly what resources were
        consumed, where they ran, and at what tier.

        Attempt-owned work stamps ``attempt_id`` and is idempotent under
        re-call for the same attempt (partial unique on attempt_id).
        Pure-legacy jobs meter by job/meter_id as before.
        """
        started = float(job.get("started_at", 0))
        completed = float(job.get("completed_at", 0))
        duration = completed - started if completed > started else 0

        # Host location is recorded on the invoice line. It never affects placement.
        country = ""
        province = ""
        if host_location:
            country = host_location.get("country", "")
            province = host_location.get("province", "")
        else:
            country = host.get("country", "")

        pricing_mode = resolve_job_pricing_mode(job)
        base_rate, _ = resolve_compute_rate_cad(job, host)
        # Price is the host's rate. Capacity is priced on what it is, never on
        # where it sits or what tier label it carries.
        multiplier = 1.0
        spot_discount = 0.0

        duration_hr = duration / 3600
        cost = round(duration_hr * base_rate * multiplier, 4)

        attempt_id = resolve_meter_attempt_id(job)

        meter = UsageMeter(
            job_id=job.get("job_id", ""),
            host_id=host.get("host_id", ""),
            owner=job.get("owner", ""),
            attempt_id=attempt_id,
            started_at=started,
            completed_at=completed,
            duration_sec=round(duration, 2),
            gpu_seconds=round(duration, 2),  # 1:1 for single GPU
            gpu_model=host.get("gpu_model", ""),
            vram_gb=float(job.get("vram_needed_gb", 0)),
            xcu_score=float(host.get("compute_score", 0)),
            country=country,
            province=province,
            trust_tier=trust_tier,
            base_rate_per_hour=base_rate,
            tier_multiplier=multiplier,
            spot_discount=spot_discount,
            pricing_mode=pricing_mode,
            total_cost_cad=cost,
        )

        # Persist — attempt-owned closes collapse on attempt_id uniqueness.
        with self._conn() as conn:
            if attempt_id:
                existing = conn.execute(
                    "SELECT * FROM usage_meters WHERE attempt_id = %s LIMIT 1",
                    (attempt_id,),
                ).fetchone()
                if existing is not None:
                    prior = _usage_meter_from_row(existing)
                    log.info(
                        "METERED job=%s attempt=%s idempotent_replay meter=%s cost=$%.4f",
                        prior.job_id,
                        attempt_id[:8],
                        prior.meter_id,
                        prior.total_cost_cad,
                    )
                    return prior

            try:
                conn.execute(
                    """INSERT INTO usage_meters
                       (meter_id, job_id, host_id, owner, started_at, completed_at,
                        duration_sec, gpu_seconds, gpu_model, vram_gb,
                        gpu_utilization_pct, xcu_score, country, province,
                        trust_tier, base_rate_per_hour,
                        tier_multiplier, spot_discount, pricing_mode,
                        total_cost_micros, created_at, attempt_id)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                               %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (meter_id) DO UPDATE SET
                         job_id = EXCLUDED.job_id, host_id = EXCLUDED.host_id,
                         owner = EXCLUDED.owner,
                         started_at = EXCLUDED.started_at,
                         completed_at = EXCLUDED.completed_at,
                         duration_sec = EXCLUDED.duration_sec,
                         gpu_seconds = EXCLUDED.gpu_seconds,
                         -- 087 dropped total_cost_cad and the trigger that
                         -- projected it; total_cost_micros is the only
                         -- stored representation.
                         total_cost_micros = EXCLUDED.total_cost_micros,
                         pricing_mode = EXCLUDED.pricing_mode,
                         attempt_id = COALESCE(usage_meters.attempt_id, EXCLUDED.attempt_id),
                         created_at = EXCLUDED.created_at""",
                    (
                        meter.meter_id,
                        meter.job_id,
                        meter.host_id,
                        meter.owner,
                        meter.started_at,
                        meter.completed_at,
                        meter.duration_sec,
                        meter.gpu_seconds,
                        meter.gpu_model,
                        meter.vram_gb,
                        meter.gpu_utilization_pct,
                        meter.xcu_score,
                        meter.country,
                        meter.province,
                        meter.trust_tier,
                        meter.base_rate_per_hour,
                        meter.tier_multiplier,
                        meter.spot_discount,
                        meter.pricing_mode,
                        cad_to_micros(meter.total_cost_cad),
                        time.time(),
                        attempt_id,
                    ),
                )
            except Exception as exc:
                # Race: two concurrent closes for the same attempt — unique
                # index wins; return the durable row already written.
                sqlstate = getattr(exc, "sqlstate", None)
                if attempt_id and sqlstate == "23505":
                    conn.rollback()
                    existing = conn.execute(
                        "SELECT * FROM usage_meters WHERE attempt_id = %s LIMIT 1",
                        (attempt_id,),
                    ).fetchone()
                    if existing is not None:
                        prior = _usage_meter_from_row(existing)
                        log.info(
                            "METERED job=%s attempt=%s race_idempotent meter=%s",
                            prior.job_id,
                            attempt_id[:8],
                            prior.meter_id,
                        )
                        return prior
                raise

        log.info(
            "METERED job=%s attempt=%s cost=$%.4f CAD mode=%s tier=%s",
            meter.job_id,
            (attempt_id or "")[:8] or "—",
            meter.total_cost_cad,
            pricing_mode,
            trust_tier,
        )
        return meter

    def generate_invoice(
        self,
        customer_id: str,
        customer_name: str,
        period_start: float,
        period_end: float,
        tax_rate: Optional[float] = None,  # None = auto-detect by province
        customer_province: str = "ON",  # Used for tax rate lookup
    ) -> Invoice:
        """Generate an itemized usage invoice for a billing period.

        Wallet debits happen continuously; this invoice is what customers see
        on request or on monthly summaries. Line items are grouped (not per-job
        pickers): GPU by model × tier × mode, serverless by GPU tier, storage
        by volume.
        """
        if tax_rate is None:
            tax_rate, _desc = get_tax_rate_for_province(customer_province)

        line_items: list[dict] = []
        subtotal_accum = 0.0

        def _mode_label(mode: str) -> str:
            labels = {
                "on_demand": "On-Demand",
                "spot": "Spot",
                "reserved_1mo": "Reserved 1 Month",
                "reserved_3mo": "Reserved 3 Months",
                "reserved_1yr": "Reserved 1 Year",
            }
            return labels.get(mode, mode.replace("_", " ").title())

        with self._conn() as conn:
            # ── GPU instances (grouped) ──
            gpu_rows = conn.execute(
                """SELECT
                    gpu_model,
                    trust_tier,
                    COALESCE(pricing_mode, 'on_demand') AS pricing_mode,
                    ROUND((SUM(duration_sec) / 3600.0)::numeric, 4) AS gpu_hours,
                    ROUND(COALESCE(SUM(total_cost_micros) / 1000000.0, 0)::numeric, 4) AS subtotal_cad,
                    ROUND(COALESCE(AVG(base_rate_per_hour * tier_multiplier), 0)::numeric, 4)
                        AS unit_price_cad,
                    MAX(province) AS province
                FROM usage_meters
                WHERE owner = %s
                  AND started_at >= %s
                  AND completed_at <= %s
                  AND total_cost_micros > 0
                GROUP BY gpu_model, trust_tier, COALESCE(pricing_mode, 'on_demand')
                HAVING SUM(total_cost_micros) / 1000000.0 > 0
                ORDER BY subtotal_cad DESC""",
                (customer_id, period_start, period_end),
            ).fetchall()

            for row in gpu_rows:
                cost = float(row["subtotal_cad"])
                mode = str(row["pricing_mode"])
                li = InvoiceLineItem(
                    description=(
                        f"{row['gpu_model']} — {_mode_label(mode)} "
                        f"({row['trust_tier']} tier)"
                    ),
                    category="compute",
                    quantity=float(row["gpu_hours"]),
                    unit="GPU-hours",
                    unit_price_cad=float(row["unit_price_cad"] or 0),
                    subtotal_cad=cost,
                    trust_tier=row["trust_tier"],
                    province=row["province"] or "",
                    line_type="compute",
                    pricing_mode=mode,
                    gpu_model=row["gpu_model"],
                )
                line_items.append(li.to_dict())
                subtotal_accum += cost

            # ── Serverless inference (grouped by GPU tier) ──
            sl_rows = conn.execute(
                """SELECT
                    gpu_model,
                    resource_type,
                    ROUND(COALESCE(SUM(duration_seconds), 0)::numeric, 0) AS worker_seconds,
                    ROUND(COALESCE(SUM(amount_micros), 0)::numeric / 1000000, 4) AS subtotal_cad,
                    ROUND(COALESCE(AVG(rate_per_hour), 0)::numeric, 4) AS rate_per_hour
                FROM billing_cycles
                WHERE customer_id = %s
                  AND resource_type IN ('serverless_gpu', 'serverless_gpu_cold_start')
                  AND status = 'charged'
                  AND period_start >= %s
                  AND period_end <= %s
                GROUP BY gpu_model, resource_type
                HAVING SUM(amount_micros) > 0
                ORDER BY subtotal_cad DESC""",
                (customer_id, period_start, period_end),
            ).fetchall()

            for row in sl_rows:
                cost = float(row["subtotal_cad"])
                gpu_tier = str(row["gpu_model"] or "serverless")
                rtype = str(row["resource_type"])
                cold = " (cold start)" if rtype == "serverless_gpu_cold_start" else ""
                seconds = int(row["worker_seconds"] or 0)
                li = InvoiceLineItem(
                    description=f"Serverless — {gpu_tier}{cold}",
                    category="compute",
                    quantity=round(seconds / 3600.0, 4),
                    unit="GPU-hours",
                    unit_price_cad=float(row["rate_per_hour"] or 0),
                    subtotal_cad=cost,
                    line_type="serverless",
                    gpu_model=gpu_tier,
                    pricing_mode="on_demand",
                )
                line_items.append(li.to_dict())
                subtotal_accum += cost

            # ── Persistent storage (per volume) ──
            vol_rows = conn.execute(
                """SELECT
                    bc.job_id,
                    COALESCE(v.name, bc.job_id) AS volume_name,
                    COALESCE(v.size_gb, 0) AS size_gb,
                    ROUND(COALESCE(SUM(bc.duration_seconds), 0)::numeric, 0) AS billed_seconds,
                    ROUND(COALESCE(SUM(bc.amount_micros), 0)::numeric / 1000000, 4) AS subtotal_cad
                FROM billing_cycles bc
                LEFT JOIN volumes v ON v.volume_id = bc.job_id
                WHERE bc.customer_id = %s
                  AND bc.resource_type = 'volume'
                  AND bc.status = 'charged'
                  AND bc.period_start >= %s
                  AND bc.period_end <= %s
                GROUP BY bc.job_id, v.name, v.size_gb
                HAVING SUM(bc.amount_micros) > 0
                ORDER BY subtotal_cad DESC""",
                (customer_id, period_start, period_end),
            ).fetchall()

            for row in vol_rows:
                cost = float(row["subtotal_cad"])
                size_gb = int(row["size_gb"] or 0)
                hours = round(float(row["billed_seconds"] or 0) / 3600.0, 4)
                li = InvoiceLineItem(
                    description=f"Storage — {row['volume_name']} ({size_gb} GB)",
                    category="storage",
                    quantity=hours,
                    unit="GB-hours",
                    unit_price_cad=round(cost / hours, 6) if hours > 0 else 0.0,
                    subtotal_cad=cost,
                    job_id=row["job_id"],
                    line_type="storage",
                    gpu_model="storage",
                )
                line_items.append(li.to_dict())
                subtotal_accum += cost

        try:
            from stripe_catalog import enrich_invoice_lines_with_catalog

            enrich_invoice_lines_with_catalog(line_items)
        except Exception as exc:
            log.debug("Stripe catalog enrichment skipped: %s", exc)

        subtotal = subtotal_accum
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)

        invoice = Invoice(
            customer_id=customer_id,
            customer_name=customer_name,
            period_start=period_start,
            period_end=period_end,
            line_items=line_items,
            subtotal_cad=round(subtotal, 2),
            tax_rate=tax_rate,
            tax_amount_cad=tax,
            total_cad=total,
        )

        # Persist
        with self._conn() as conn:
            from psycopg.types.json import Jsonb

            conn.execute(
                """INSERT INTO invoices
                   (invoice_id, customer_id, customer_name, currency,
                    period_start, period_end, line_items, subtotal_micros,
                    tax_rate, tax_amount_micros, total_micros,
                    created_at, status, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    invoice.invoice_id,
                    invoice.customer_id,
                    invoice.customer_name,
                    invoice.currency,
                    invoice.period_start,
                    invoice.period_end,
                    Jsonb(invoice.line_items),
                    # Integer micros in storage; the dataclass keeps CAD, which
                    # is what the API and the PDF renderer speak.
                    round(invoice.subtotal_cad * 1_000_000),
                    invoice.tax_rate,
                    round(invoice.tax_amount_cad * 1_000_000),
                    round(invoice.total_cad * 1_000_000),
                    invoice.created_at,
                    invoice.status,
                    invoice.notes,
                ),
            )

        log.info(
            "INVOICE %s customer=%s total=$%.2f CAD",
            invoice.invoice_id,
            customer_id,
            total,
        )
        return invoice

    def record_payout(
        self,
        provider_id: str,
        job_id: str,
        gross_amount_cad: float,
        platform_fee_pct: float = 0.15,  # 15% platform fee
    ) -> dict:
        """Record a payout to a compute provider (Stripe Connect ready).

        Platform takes a percentage; remainder goes to provider.
        """
        fee = round(gross_amount_cad * platform_fee_pct, 4)
        payout = round(gross_amount_cad - fee, 4)
        payout_id = f"PAY-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO payout_ledger
                   (payout_id, provider_id, job_id, amount_micros,
                    platform_fee_micros, provider_payout_micros, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    payout_id,
                    provider_id,
                    job_id,
                    round(gross_amount_cad * 1_000_000),
                    round(fee * 1_000_000),
                    round(payout * 1_000_000),
                    "pending",
                    time.time(),
                ),
            )

        log.info(
            "PAYOUT %s provider=%s job=%s gross=$%.4f fee=$%.4f payout=$%.4f",
            payout_id,
            provider_id,
            job_id,
            gross_amount_cad,
            fee,
            payout,
        )

        return {
            "payout_id": payout_id,
            "provider_id": provider_id,
            "job_id": job_id,
            "gross_amount_cad": gross_amount_cad,
            "platform_fee_cad": fee,
            "provider_payout_cad": payout,
            "status": "pending",
        }

    def get_usage_summary(
        self,
        customer_id: str,
        period_start: float,
        period_end: float,
    ) -> dict:
        """Usage summary for dashboard / reporting."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT
                    COUNT(*) as job_count,
                    SUM(duration_sec) as total_duration_sec,
                    SUM(gpu_seconds) as total_gpu_seconds,
                    SUM(total_cost_micros) / 1000000.0 AS total_cost_cad,
                    SUM(CASE WHEN COALESCE(pricing_mode, 'on_demand') = 'spot'
                        THEN total_cost_micros / 1000000.0 ELSE 0 END) as spot_cost,
                    SUM(CASE WHEN COALESCE(pricing_mode, 'on_demand') != 'spot'
                        THEN total_cost_micros / 1000000.0 ELSE 0 END) as on_demand_cost,
                    COUNT(DISTINCT host_id) as hosts_used,
                    COUNT(DISTINCT trust_tier) as tiers_used
                FROM usage_meters
                WHERE owner = %s AND started_at >= %s AND completed_at <= %s""",
                (customer_id, period_start, period_end),
            ).fetchone()

        return {
            "customer_id": customer_id,
            "period_start": period_start,
            "period_end": period_end,
            "job_count": rows["job_count"] or 0,
            "total_gpu_hours": round((rows["total_gpu_seconds"] or 0) / 3600, 2),
            "total_cost_cad": round(rows["total_cost_cad"] or 0, 2),
            "spot_spend_cad": round(rows["spot_cost"] or 0, 2),
            "on_demand_spend_cad": round(rows["on_demand_cost"] or 0, 2),
            "hosts_used": rows["hosts_used"] or 0,
            "tiers_used": rows["tiers_used"] or 0,
            "currency": "CAD",
        }

    # ── Reserved Commitments (UI-5.2) ─────────────────────────────────

    def record_reservation(self, c: dict) -> None:
        """Persist a reserved pricing commitment.

        Idempotent on ``commitment_id`` so a retried POST does not create
        duplicate commitments. Caller supplies the fully-priced commitment
        dict (see ``routes/billing.py:api_reserve_commitment``).

        Callers speak CAD; storage is integer micros. Converting here keeps the
        unit boundary in one place instead of at every call site.
        """
        c = {
            **c,
            "base_rate_micros": round(float(c["base_rate_cad"]) * 1_000_000),
            "discounted_rate_micros": round(float(c["discounted_rate_cad"]) * 1_000_000),
        }
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO reserved_commitments
                    (commitment_id, customer_id, commitment_type, gpu_model,
                     quantity, province, base_rate_micros, discounted_rate_micros,
                     discount_pct, min_hours_per_day, status, created_at,
                     start_at, end_at)
                   VALUES (%(commitment_id)s, %(customer_id)s, %(commitment_type)s,
                           %(gpu_model)s, %(quantity)s, %(province)s,
                           %(base_rate_micros)s, %(discounted_rate_micros)s,
                           %(discount_pct)s, %(min_hours_per_day)s, %(status)s,
                           %(created_at)s, %(start_at)s, %(end_at)s)
                   ON CONFLICT (commitment_id) DO NOTHING""",
                c,
            )

    def list_reservations(self, customer_id: str) -> list[dict]:
        """Return a customer's reserved commitments with realized savings.

        Realized savings are grounded in actual consumption: GPU-hours
        metered on the committed GPU model during the commitment window,
        times the per-hour gap between on-demand and discounted rates. A
        commitment past its ``end_at`` that is still ``active`` is reported
        as ``expired`` (the row is not mutated).
        """
        now = time.time()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT *, "
                "  base_rate_micros::double precision / 1000000.0 AS base_rate_cad, "
                "  discounted_rate_micros::double precision / 1000000.0 AS discounted_rate_cad "
                "FROM reserved_commitments WHERE customer_id = %s "
                "ORDER BY created_at DESC",
                (customer_id,),
            ).fetchall()
            out: list[dict] = []
            for row in rows:
                r = dict(row)
                used = conn.execute(
                    "SELECT COALESCE(SUM(gpu_seconds), 0) AS secs "
                    "FROM usage_meters "
                    "WHERE owner = %s AND gpu_model = %s "
                    "AND started_at >= %s AND started_at < %s",
                    (customer_id, r["gpu_model"], r["start_at"], r["end_at"]),
                ).fetchone()
                used_hours = float(used["secs"] or 0) / 3600.0
                per_hour_savings = max(
                    0.0, float(r["base_rate_cad"]) - float(r["discounted_rate_cad"])
                )
                r["realized_hours"] = round(used_hours, 2)
                r["realized_savings_cad"] = round(used_hours * per_hour_savings, 2)
                if r["status"] == "active" and now >= float(r["end_at"]):
                    r["status"] = "expired"
                r["is_active"] = r["status"] == "active"
                out.append(r)
            return out

    def generate_attestation(self) -> ProviderAttestation:
        """Generate a provider attestation bundle.

        This is the document customers attach to Fund claims
        to prove they used a Canadian compute provider.
        """
        return ProviderAttestation(
            valid_until=time.time() + 365 * 86400,  # 1 year validity
        )

    # ── Refund Logic (REPORT_FEATURE_1.md) ────────────────────────────

    def process_refund(self, job_id: str, exit_code: int, failure_reason: str = "") -> dict:
        """Determine and process refund for a failed job.

        From REPORT_FEATURE_1.md:
          - Hardware error → full refund
          - User OOM (exit 137) → zero refund
          - Network timeout → partial refund (50%)
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM usage_meters WHERE job_id = %s",
                (job_id,),
            ).fetchone()

        if not row:
            return {"refund": False, "reason": "No usage record found"}

        # SELECT * no longer yields a float twin; derive from the integer.
        cost = micros_to_cad(row["total_cost_micros"] or 0)

        # Classify failure type and determine refund
        if exit_code == 137:
            # User-side OOM — no refund
            refund_pct = 0.0
            classification = "user_oom"
        elif exit_code in (139, 134, 136):
            # SEGFAULT / SIGABRT / SIGFPE — likely user code error
            refund_pct = 0.0
            classification = "user_code_error"
        elif exit_code in (-1, 255) or "hardware" in failure_reason.lower():
            # Hardware error — full refund
            refund_pct = 1.0
            classification = "hardware_error"
        elif "timeout" in failure_reason.lower() or "network" in failure_reason.lower():
            # Network/timeout — partial refund
            refund_pct = 0.5
            classification = "network_error"
        elif "gpu" in failure_reason.lower() or "cuda" in failure_reason.lower():
            # GPU/CUDA error — full refund (host-side)
            refund_pct = 1.0
            classification = "gpu_error"
        else:
            # Unknown — partial refund, review needed
            refund_pct = 0.5
            classification = "unknown"

        refund_amount = round(cost * refund_pct, 4)

        result = {
            "job_id": job_id,
            "exit_code": exit_code,
            "failure_reason": failure_reason,
            "classification": classification,
            "original_cost_cad": cost,
            "refund_percentage": refund_pct,
            "refund_amount_cad": refund_amount,
            "refund": refund_amount > 0,
            "is_host_fault": classification in ("hardware_error", "gpu_error", "network_error"),
        }

        if refund_amount > 0:
            # Credit the refund to the user's wallet
            self._credit_wallet(
                row["owner"], refund_amount, f"Refund for job {job_id} ({classification})"
            )

        log.info(
            "REFUND job=%s classification=%s refund=$%.4f CAD (%.0f%%)",
            job_id,
            classification,
            refund_amount,
            refund_pct * 100,
        )
        return result

    # ── Credit/Wallet System (REPORT_FEATURE_1.md) ────────────────────

    def _ensure_wallet_table(self):
        pass  # Tables managed by Alembic migrations

    def get_wallet(self, customer_id: str) -> dict:
        """Get or create a customer wallet.

        Includes ``held_cad`` and ``available_cad`` (ledger − active holds)
        when the ``wallet_holds`` table is present.
        """
        self._ensure_wallet_table()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM wallets WHERE customer_id = %s",
                (customer_id,),
            ).fetchone()
            if not row:
                # Create new wallet
                now = time.time()
                conn.execute(
                    """INSERT INTO wallets
                       (customer_id, balance_micros, total_deposited_micros,
                        total_spent_micros, total_refunded_micros,
                        grace_until, status, created_at, updated_at)
                       VALUES (%s, 0, 0, 0, 0, 0, 'active', %s, %s)""",
                    (customer_id, now, now),
                )
                row = {
                    "customer_id": customer_id,
                    "balance_cad": 0.0,
                    "total_deposited_cad": 0.0,
                    "total_spent_cad": 0.0,
                    "total_refunded_cad": 0.0,
                    "grace_until": 0.0,
                    "status": "active",
                }
            else:
                row = dict(row)

            # The CAD keys are a presentation view derived from the integer
            # columns, not a second stored representation. Deriving them here
            # keeps every caller's row["balance_cad"] working while the
            # database holds exactly one authoritative number.
            for base in ("balance", "total_deposited", "total_spent", "total_refunded"):
                row[f"{base}_cad"] = micros_to_cad(row.get(f"{base}_micros") or 0)

            balance = float(row.get("balance_cad") or 0)
            held = self._active_holds_total(conn, customer_id)
            row["held_cad"] = held
            row["available_cad"] = round(max(0.0, balance - held), 4)
            # Low-balance UX signals (warning before hard stop at zero).
            warn_at = micros_to_cad(row.get("auto_topup_threshold_micros") or 0) or float(
                os.environ.get("XCELSIOR_LOW_BALANCE_WARN_CAD", "5.0")
            )
            row["low_balance_threshold_cad"] = warn_at
            row["low_balance"] = row["available_cad"] <= warn_at
            row["hard_stop"] = row["available_cad"] <= 0 or (row.get("status") == "suspended")
            return row

    def _active_holds_total(self, conn: Any, customer_id: str, *, now: float | None = None) -> float:
        """Sum of non-expired held amount for a customer (expires stale holds)."""
        now = time.time() if now is None else now
        try:
            conn.execute(
                """
                UPDATE wallet_holds
                   SET status = 'expired',
                       updated_at = %s,
                       released_at = COALESCE(released_at, %s)
                 WHERE customer_id = %s
                   AND status = 'held'
                   AND expires_at <= %s
                """,
                (now, now, customer_id, now),
            )
            row = conn.execute(
                """
                SELECT COALESCE(SUM(amount_micros) / 1000000.0, 0) AS held
                  FROM wallet_holds
                 WHERE customer_id = %s
                   AND status = 'held'
                   AND expires_at > %s
                """,
                (customer_id, now),
            ).fetchone()
        except Exception as exc:
            # Table missing mid-rollout: treat as zero holds.
            log.debug("wallet_holds sum skipped: %s", exc)
            return 0.0
        if row is None:
            return 0.0
        return float(row["held"] if isinstance(row, dict) else row[0] or 0)

    def available_balance_cad(self, customer_id: str) -> float:
        """Ledger balance minus durable active holds."""
        wallet = self.get_wallet(customer_id)
        return float(wallet.get("available_cad", wallet.get("balance_cad", 0)) or 0)

    def expire_stale_wallet_holds(self, *, limit: int = 500) -> int:
        """CAS held→expired for past-due holds. Idempotent; multi-replica safe.

        Returns the number of rows transitioned. Expired holds no longer
        reduce available balance. Safe to call from a durable scheduled
        task and from lazy balance reads (via ``_active_holds_total``).
        """
        now = time.time()
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """
                    UPDATE wallet_holds
                       SET status = 'expired',
                           updated_at = %s,
                           released_at = COALESCE(released_at, %s)
                     WHERE hold_id IN (
                         SELECT hold_id FROM wallet_holds
                          WHERE status = 'held'
                            AND expires_at <= %s
                          ORDER BY expires_at ASC
                          LIMIT %s
                          FOR UPDATE SKIP LOCKED
                     )
                     RETURNING hold_id
                    """,
                    (now, now, now, max(1, int(limit))),
                ).fetchall()
            n = len(rows or [])
            if n:
                log.info("WALLET HOLD EXPIRE count=%d", n)
            return n
        except Exception as exc:
            log.debug("expire_stale_wallet_holds skipped: %s", exc)
            return 0

    def wallet_has_available_funds(self, customer_id: str) -> dict:
        """Fund gate for start/restart: available_cad > 0 and not suspended.

        Returns ``{ok, reason?, available_cad, balance_cad, held_cad, status}``.
        """
        wallet = self.get_wallet(customer_id)
        status = wallet.get("status") or "active"
        balance = float(wallet.get("balance_cad") or 0)
        available = float(wallet.get("available_cad", balance) or 0)
        held = float(wallet.get("held_cad") or 0)
        if status == "suspended":
            return {
                "ok": False,
                "reason": "wallet_suspended",
                "available_cad": available,
                "balance_cad": balance,
                "held_cad": held,
                "status": status,
            }
        if available <= 0:
            return {
                "ok": False,
                "reason": "insufficient_available",
                "available_cad": available,
                "balance_cad": balance,
                "held_cad": held,
                "status": status,
            }
        return {
            "ok": True,
            "available_cad": available,
            "balance_cad": balance,
            "held_cad": held,
            "status": status,
        }

    def estimate_launch_hold_cad(
        self,
        *,
        pricing_mode: str = "on_demand",
        gpu_model: str | None = None,
        num_gpus: int = 1,
        host: dict | None = None,
    ) -> float:
        """CAD amount to reserve for one launch (≈ one hour of compute)."""
        n = max(1, int(num_gpus or 1))
        mode = (pricing_mode or "on_demand").strip().lower()
        if mode == "spot":
            model = (gpu_model or "").strip() or "RTX 4090"
            try:
                from spot_pricing import compute_live_spot_quote

                return round(max(0.01, compute_live_spot_quote(model).rate_cad * n), 4)
            except Exception:
                return round(max(0.01, 0.20 * n), 4)
        # On-demand: prefer host rate, else a conservative default per GPU.
        if host and host.get("cost_per_hour") is not None:
            return round(max(0.01, float(host["cost_per_hour"]) * n), 4)
        return round(max(0.01, 0.20 * n), 4)

    def create_wallet_hold(
        self,
        customer_id: str,
        amount_cad: float,
        *,
        idempotency_key: str | None = None,
        job_id: str | None = None,
        expires_in_sec: int = 3600,
        reason: str = "launch",
    ) -> dict:
        """Create a durable fund hold if available balance is sufficient.

        Locks the wallet row so concurrent creates see each other's holds.
        Returns ``{held, hold_id, amount_cad, available_cad, reason?}``.
        """
        amount = round(float(amount_cad), 4)
        if amount <= 0:
            return {
                "held": False,
                "reason": "invalid_amount",
                "available_cad": self.available_balance_cad(customer_id),
            }

        self._ensure_wallet_table()
        # Ensure wallet row exists before locking.
        self.get_wallet(customer_id)
        now = time.time()
        expires_at = now + max(60, int(expires_in_sec))
        hold_id = str(uuid.uuid4())
        idemp = (idempotency_key or "").strip() or None

        with self._conn() as conn:
            if idemp:
                prior = conn.execute(
                    """
                    SELECT hold_id, status, amount_micros / 1000000.0 AS amount_cad, expires_at
                      FROM wallet_holds
                     WHERE customer_id = %s AND idempotency_key = %s
                     LIMIT 1
                    """,
                    (customer_id, idemp),
                ).fetchone()
                if prior is not None:
                    st = prior["status"] if isinstance(prior, dict) else prior[1]
                    hid = str(prior["hold_id"] if isinstance(prior, dict) else prior[0])
                    if st == "held":
                        return {
                            "held": True,
                            "hold_id": hid,
                            "amount_cad": float(
                                prior["amount_cad"]
                                if isinstance(prior, dict)
                                else prior[2]
                            ),
                            "available_cad": self._available_locked(
                                conn, customer_id, now=now
                            ),
                            "idempotent_replay": True,
                        }
                    # Terminal prior (released/consumed/expired): free the
                    # idempotency key so a new attempt can reserve again.
                    conn.execute(
                        """
                        UPDATE wallet_holds
                           SET idempotency_key = NULL,
                               updated_at = %s
                         WHERE hold_id = %s::uuid
                           AND status <> 'held'
                        """,
                        (now, hid),
                    )

            wallet = conn.execute(
                """
                SELECT balance_micros / 1000000.0 AS balance_cad, status FROM wallets
                 WHERE customer_id = %s
                 FOR UPDATE
                """,
                (customer_id,),
            ).fetchone()
            if wallet is None:
                return {"held": False, "reason": "wallet_missing", "available_cad": 0.0}
            wstatus = wallet["status"] if isinstance(wallet, dict) else wallet[1]
            if wstatus == "suspended":
                return {
                    "held": False,
                    "reason": "wallet_suspended",
                    "available_cad": 0.0,
                }

            available = self._available_locked(conn, customer_id, now=now)
            if available + 1e-9 < amount:
                return {
                    "held": False,
                    "reason": "insufficient_available",
                    "available_cad": available,
                    "required_cad": amount,
                }

            try:
                conn.execute(
                    """
                    INSERT INTO wallet_holds
                        (hold_id, customer_id, amount_micros, status, job_id,
                         idempotency_key, created_at, expires_at, updated_at)
                    VALUES (%s, %s, %s, 'held', %s, %s, %s, %s, %s)
                    """,
                    (
                        hold_id,
                        customer_id,
                        cad_to_micros(amount),
                        job_id,
                        idemp,
                        now,
                        expires_at,
                        now,
                    ),
                )
            except Exception as exc:
                # Concurrent create with same key: replay the winner.
                if idemp and getattr(exc, "sqlstate", None) == "23505":
                    conn.rollback()
                    winner = conn.execute(
                        """
                        SELECT hold_id, status, amount_micros / 1000000.0 AS amount_cad
                          FROM wallet_holds
                         WHERE customer_id = %s AND idempotency_key = %s
                         LIMIT 1
                        """,
                        (customer_id, idemp),
                    ).fetchone()
                    if winner is not None:
                        st = winner["status"] if isinstance(winner, dict) else winner[1]
                        if st == "held":
                            whid = str(
                                winner["hold_id"]
                                if isinstance(winner, dict)
                                else winner[0]
                            )
                            return {
                                "held": True,
                                "hold_id": whid,
                                "amount_cad": float(
                                    winner["amount_cad"]
                                    if isinstance(winner, dict)
                                    else winner[2]
                                ),
                                "available_cad": self._available_locked(
                                    conn, customer_id, now=time.time()
                                ),
                                "idempotent_replay": True,
                            }
                raise
            remaining = round(available - amount, 4)

        log.info(
            "WALLET HOLD %s hold=%s amount=$%.4f reason=%s available_after=$%.4f",
            customer_id,
            hold_id[:8],
            amount,
            reason,
            remaining,
        )
        return {
            "held": True,
            "hold_id": hold_id,
            "amount_cad": amount,
            "available_cad": remaining,
            "expires_at": expires_at,
        }

    def _available_locked(self, conn: Any, customer_id: str, *, now: float) -> float:
        row = conn.execute(
            "SELECT balance_micros / 1000000.0 AS balance_cad FROM wallets WHERE customer_id = %s",
            (customer_id,),
        ).fetchone()
        balance = float(
            (row["balance_cad"] if isinstance(row, dict) else row[0]) if row else 0
        )
        held = self._active_holds_total(conn, customer_id, now=now)
        return round(max(0.0, balance - held), 4)

    def link_wallet_hold_to_job(self, hold_id: str, job_id: str) -> bool:
        """Stamp job.wallet_hold_id and hold.job_id (best-effort durable link)."""
        if not hold_id or not job_id:
            return False
        now = time.time()
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    UPDATE wallet_holds
                       SET job_id = %s, updated_at = %s
                     WHERE hold_id = %s::uuid
                       AND status = 'held'
                     RETURNING hold_id
                    """,
                    (job_id, now, hold_id),
                ).fetchone()
                if row is None:
                    return False
                conn.execute(
                    """
                    UPDATE jobs
                       SET wallet_hold_id = %s::uuid
                     WHERE job_id = %s
                    """,
                    (hold_id, job_id),
                )
            return True
        except Exception as exc:
            log.warning(
                "link_wallet_hold_to_job failed hold=%s job=%s: %s",
                hold_id[:8],
                job_id,
                exc,
            )
            return False

    def release_wallet_hold(self, hold_id: str, *, reason: str = "release") -> dict:
        """Release a held amount once (idempotent if already terminal)."""
        if not hold_id:
            return {"released": False, "reason": "missing_hold_id"}
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                """
                UPDATE wallet_holds
                   SET status = 'released',
                       released_at = %s,
                       updated_at = %s
                 WHERE hold_id = %s::uuid
                   AND status = 'held'
                 RETURNING hold_id, customer_id, amount_micros / 1000000.0 AS amount_cad, job_id
                """,
                (now, now, hold_id),
            ).fetchone()
            if row is None:
                existing = conn.execute(
                    "SELECT status FROM wallet_holds WHERE hold_id = %s::uuid",
                    (hold_id,),
                ).fetchone()
                if existing is None:
                    return {"released": False, "reason": "not_found"}
                st = existing["status"] if isinstance(existing, dict) else existing[0]
                return {
                    "released": True,
                    "already_terminal": True,
                    "status": st,
                    "hold_id": hold_id,
                }
            # Clear job pointer when present (does not fail release).
            jid = row["job_id"] if isinstance(row, dict) else row[3]
            if jid:
                conn.execute(
                    """
                    UPDATE jobs
                       SET wallet_hold_id = NULL
                     WHERE job_id = %s
                       AND wallet_hold_id = %s::uuid
                    """,
                    (jid, hold_id),
                )
        log.info("WALLET HOLD RELEASE hold=%s reason=%s", hold_id[:8], reason)
        return {"released": True, "hold_id": hold_id, "reason": reason}

    def release_wallet_hold_for_job(self, job_id: str, *, reason: str = "job_terminal") -> dict:
        """Release the hold linked to a job (by jobs.wallet_hold_id or hold.job_id).

        Idempotent: if the hold was already released/consumed/expired, still
        returns ``released=True`` with ``already_terminal``.
        """
        if not job_id:
            return {"released": False, "reason": "missing_job_id"}
        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT wallet_hold_id FROM jobs WHERE job_id = %s",
                    (job_id,),
                ).fetchone()
                hold_id = None
                if row is not None:
                    hold_id = row["wallet_hold_id"] if isinstance(row, dict) else row[0]
                if not hold_id:
                    alt = conn.execute(
                        """
                        SELECT hold_id FROM wallet_holds
                         WHERE job_id = %s
                         ORDER BY
                           CASE status
                             WHEN 'held' THEN 0
                             ELSE 1
                           END,
                           created_at DESC
                         LIMIT 1
                        """,
                        (job_id,),
                    ).fetchone()
                    if alt is not None:
                        hold_id = alt["hold_id"] if isinstance(alt, dict) else alt[0]
            if not hold_id:
                return {"released": False, "reason": "no_hold"}
            return self.release_wallet_hold(str(hold_id), reason=reason)
        except Exception as exc:
            log.debug("release_wallet_hold_for_job %s: %s", job_id, exc)
            return {"released": False, "reason": "error"}

    def deposit(
        self,
        customer_id: str,
        amount_cad: float,
        description: str = "Credit deposit",
        idempotency_key: str = "",
    ) -> dict:
        """Deposit credits into a customer wallet.

        If idempotency_key is provided, the deposit is deduplicated:
        a second call with the same key returns the original result.
        """
        self._ensure_wallet_table()

        # Idempotency check
        if idempotency_key:
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT tx_id, balance_after_micros / 1000000.0 AS balance_after_cad FROM wallet_transactions WHERE idempotency_key = %s",
                    (idempotency_key,),
                ).fetchone()
                if existing:
                    log.info(
                        "Idempotent deposit skipped (key=%s, existing tx=%s)",
                        idempotency_key,
                        existing["tx_id"],
                    )
                    return {
                        "tx_id": existing["tx_id"],
                        "balance_cad": existing["balance_after_cad"],
                        "dedup": True,
                    }

        wallet = self.get_wallet(customer_id)
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            # Atomic: increment balance and get new value in one statement
            row = conn.execute(
                # Integer minor units: the arithmetic itself must be exact
                # (companion §4.4 rule 6). 087 dropped the balance_cad column
                # and the trigger that projected it, so the float for legacy
                # readers is derived in the RETURNING clause instead — it is a
                # presentation value and never the accumulator.
                """UPDATE wallets
                   SET balance_micros = balance_micros + %s,
                       total_deposited_micros = total_deposited_micros + %s,
                       updated_at = %s
                   WHERE customer_id = %s
                   RETURNING balance_micros, balance_micros / 1000000.0 AS balance_cad""",
                (cad_to_micros(amount_cad), cad_to_micros(amount_cad),
                 time.time(), customer_id),
            ).fetchone()
            # Read the derived float for the legacy response shape; the
            # authoritative value is balance_micros.
            new_balance = (
                micros_to_cad(row["balance_micros"])
                if row and row.get("balance_micros") is not None
                else round(wallet["balance_cad"] + amount_cad, 4)
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_micros,
                    balance_after_micros, description, created_at, idempotency_key)
                   VALUES (%s, %s, 'deposit', %s, %s, %s, %s, %s)""",
                (
                    tx_id,
                    customer_id,
                    cad_to_micros(amount_cad),
                    cad_to_micros(new_balance),
                    description,
                    time.time(),
                    idempotency_key or "",
                ),
            )

        log.info("DEPOSIT %s +$%.2f CAD balance=$%.2f", customer_id, amount_cad, new_balance)
        return {"tx_id": tx_id, "balance_cad": new_balance}

    def low_balance_threshold_cad(self, customer_id: str) -> float:
        """Warn when available balance falls at or below this CAD amount.

        Uses auto-topup threshold when configured; otherwise $5 CAD default.
        """
        wallet = self.get_wallet(customer_id)
        threshold = micros_to_cad(wallet.get("auto_topup_threshold_micros") or 0)
        if threshold > 0:
            return threshold
        return float(os.environ.get("XCELSIOR_LOW_BALANCE_WARN_CAD", "5.0"))

    def maybe_warn_low_balance(self, customer_id: str, balance_cad: float) -> dict:
        """Emit a low-balance warning notification (rate-limited via grace_until stamp)."""
        threshold = self.low_balance_threshold_cad(customer_id)
        if balance_cad > threshold:
            return {"warned": False, "threshold_cad": threshold}
        now = time.time()
        wallet = self.get_wallet(customer_id)
        # Reuse grace_until as last-warn watermark when still positive balance.
        last = float(wallet.get("grace_until") or 0)
        if last and now - last < 6 * 3600 and balance_cad > 0:
            return {"warned": False, "threshold_cad": threshold, "reason": "rate_limited"}
        try:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE wallets SET grace_until = %s, updated_at = %s WHERE customer_id = %s",
                    (now, now, customer_id),
                )
            from db import NotificationStore

            NotificationStore.create(
                user_email=customer_id,
                notif_type="billing_low_balance",
                title="Low balance warning",
                body=(
                    f"Your wallet is ${balance_cad:.2f} CAD "
                    f"(warning at ${threshold:.2f} CAD). Top up to avoid hard stop."
                ),
                data={"balance_cad": balance_cad, "threshold_cad": threshold},
            )
        except Exception:
            pass
        return {"warned": True, "threshold_cad": threshold, "balance_cad": balance_cad}

    def charge(
        self,
        customer_id: str,
        amount_cad: float,
        job_id: str = "",
        description: str = "Compute charge",
    ) -> dict:
        """Charge a customer wallet from prepaid credits.

        Hard stop: never debit more than available balance (no free-ride grace).
        Low-balance warnings fire before zero so customers can top up.
        """
        self._ensure_wallet_table()
        wallet = self.get_wallet(customer_id)
        balance = float(wallet["balance_cad"] or 0)
        status = wallet.get("status") or "active"
        amount_cad = round(float(amount_cad), 4)

        if status == "suspended":
            return {
                "charged": False,
                "reason": "wallet_suspended",
                "balance_cad": balance,
                "action": "account_suspended",
            }

        # Pre-zero warning while still solvent.
        if balance > 0:
            self.maybe_warn_low_balance(customer_id, balance)

        if amount_cad <= 0:
            return {"charged": False, "reason": "invalid_amount", "balance_cad": balance}

        if balance + 1e-9 < amount_cad:
            # Hard stop at zero / insufficient funds — never run free compute.
            now = time.time()
            action = "hard_stop"
            if balance <= 0:
                try:
                    with self._conn() as conn:
                        conn.execute(
                            "UPDATE wallets SET status = 'suspended', updated_at = %s WHERE customer_id = %s",
                            (now, customer_id),
                        )
                    from db import NotificationStore

                    NotificationStore.create(
                        user_email=customer_id,
                        notif_type="billing_suspended",
                        title="Account suspended — zero balance",
                        body=(
                            "Your wallet balance is $0.00 CAD. Running workloads will be "
                            "stopped. Top up to reactivate."
                        ),
                        data={"balance_cad": balance, "job_id": job_id},
                    )
                except Exception:
                    pass
                action = "account_suspended"
            else:
                try:
                    from db import NotificationStore

                    NotificationStore.create(
                        user_email=customer_id,
                        notif_type="billing_insufficient",
                        title="Insufficient balance — charge blocked",
                        body=(
                            f"Charge of ${amount_cad:.2f} CAD blocked "
                            f"(balance ${balance:.2f} CAD). Top up to continue."
                        ),
                        data={
                            "balance_cad": balance,
                            "required_cad": amount_cad,
                            "job_id": job_id,
                        },
                    )
                except Exception:
                    pass
            log.warning(
                "WALLET %s hard-stop insufficient ($%.4f < $%.4f) job=%s action=%s",
                customer_id,
                balance,
                amount_cad,
                job_id,
                action,
            )
            return {
                "charged": False,
                "reason": "insufficient_balance",
                "balance_cad": balance,
                "required_cad": amount_cad,
                "action": action,
            }

        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"
        amount_micros = cad_to_micros(amount_cad)
        now = time.time()

        with self._conn() as conn:
            # Atomic floor: refuse concurrent races that would go negative.
            row = conn.execute(
                """UPDATE wallets
                   SET balance_micros = balance_micros - %s,
                       total_spent_micros = total_spent_micros + %s,
                       grace_until = 0,
                       updated_at = %s
                   WHERE customer_id = %s
                     AND balance_micros >= %s
                   RETURNING balance_micros, balance_micros / 1000000.0 AS balance_cad""",
                (amount_micros, amount_micros, now, customer_id, amount_micros),
            ).fetchone()
            if not row:
                # Concurrent winner already debited — re-read actual balance.
                log.warning(
                    "WALLET %s hard-stop race ($%.4f needed) job=%s",
                    customer_id,
                    amount_cad,
                    job_id,
                )
                fresh = self.get_wallet(customer_id)
                actual = float(fresh.get("balance_cad") or 0)
                suspended = (fresh.get("status") or "") == "suspended" or actual <= 0
                if actual <= 0 and (fresh.get("status") or "") != "suspended":
                    try:
                        with self._conn() as conn2:
                            conn2.execute(
                                "UPDATE wallets SET status = 'suspended', updated_at = %s "
                                "WHERE customer_id = %s AND balance_micros <= 0",
                                (time.time(), customer_id),
                            )
                    except Exception:
                        pass
                    suspended = True
                return {
                    "charged": False,
                    "reason": "insufficient_balance",
                    "balance_cad": actual,
                    "required_cad": amount_cad,
                    "action": "account_suspended" if suspended else "hard_stop",
                    "hard_stop": suspended or actual <= 0,
                    "low_balance": actual <= self.low_balance_threshold_cad(customer_id),
                }
            new_balance = (
                micros_to_cad(row["balance_micros"])
                if row.get("balance_micros") is not None
                else round(balance - amount_cad, 4)
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_micros,
                    balance_after_micros, description, job_id, created_at)
                   VALUES (%s, %s, 'charge', %s, %s, %s, %s, %s)""",
                (
                    tx_id,
                    customer_id,
                    -cad_to_micros(amount_cad),
                    cad_to_micros(new_balance),
                    description,
                    job_id,
                    now,
                ),
            )

        log.info(
            "CHARGE %s -$%.4f CAD job=%s balance=$%.4f",
            customer_id,
            amount_cad,
            job_id,
            new_balance,
        )
        # Non-blocking Stripe Billing Meter dual-write (wallet remains SoT).
        try:
            from stripe_meters import enqueue_usage_from_charge

            enqueue_usage_from_charge(
                customer_id=customer_id,
                amount_cad=amount_cad,
                job_id=job_id or "",
                description=description or "",
                tx_id=tx_id,
            )
        except Exception as exc:
            log.warning("meter dual-write enqueue failed (charge kept): %s", exc)
        if new_balance > 0:
            self.maybe_warn_low_balance(customer_id, new_balance)
        elif new_balance <= 0:
            # Exact zero after charge → hard-stop state for UI/consumers.
            try:
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE wallets SET status = 'suspended', updated_at = %s "
                        "WHERE customer_id = %s AND balance_micros <= 0",
                        (time.time(), customer_id),
                    )
            except Exception:
                pass
        return {
            "charged": True,
            "tx_id": tx_id,
            "balance_cad": new_balance,
            "hard_stop": new_balance <= 0,
            "low_balance": new_balance <= self.low_balance_threshold_cad(customer_id),
        }

    def _credit_wallet(
        self, customer_id: str, amount_cad: float, description: str = "Refund credit"
    ):
        """Internal: credit a wallet (for refunds)."""
        self._ensure_wallet_table()
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            row = conn.execute(
                """UPDATE wallets
                   SET balance_micros = balance_micros + %s,
                       total_refunded_micros = total_refunded_micros + %s,
                       updated_at = %s
                   WHERE customer_id = %s
                   RETURNING balance_micros, balance_micros / 1000000.0 AS balance_cad""",
                (cad_to_micros(amount_cad), cad_to_micros(amount_cad),
                 time.time(), customer_id),
            ).fetchone()
            new_balance = (
                micros_to_cad(row["balance_micros"])
                if row and row.get("balance_micros") is not None
                else amount_cad
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_micros,
                    balance_after_micros, description, created_at)
                   VALUES (%s, %s, 'refund', %s, %s, %s, %s)""",
                (
                    tx_id,
                    customer_id,
                    cad_to_micros(amount_cad),
                    cad_to_micros(new_balance),
                    description,
                    time.time(),
                ),
            )

    def get_wallet_history(self, customer_id: str, limit: int = 50) -> list:
        """Get transaction history for a wallet."""
        self._ensure_wallet_table()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM wallet_transactions
                   WHERE customer_id = %s
                   ORDER BY created_at DESC LIMIT %s""",
                (customer_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def reset_wallet_testing_state(self, customer_id: str) -> dict:
        """Reset a wallet to a clean promo-testing state.

        This intentionally clears wallet transactions so any one-time promo
        idempotency markers, such as the signup credit, can be exercised again.
        """
        self._ensure_wallet_table()
        self.get_wallet(customer_id)

        with self._conn() as conn:
            cleared = (
                conn.execute(
                    "DELETE FROM wallet_transactions WHERE customer_id = %s",
                    (customer_id,),
                ).rowcount
                or 0
            )
            conn.execute(
                """UPDATE wallets
                   -- Write the integer columns. The _cad pair is a projection
                   -- maintained by wallets_project_money; every other money
                   -- path already writes micros, and writing the float here
                   -- was the last place application code touched one.
                   SET balance_micros = 0,
                       total_deposited_micros = 0,
                       total_spent_micros = 0,
                       total_refunded_micros = 0,
                       grace_until = 0,
                       status = 'active',
                       updated_at = %s
                   WHERE customer_id = %s""",
                (time.time(), customer_id),
            )

        wallet = self.get_wallet(customer_id)
        log.info(
            "WALLET RESET %s cleared_transactions=%s",
            customer_id,
            cleared,
        )
        return {
            "wallet": wallet,
            "cleared_transactions": cleared,
            "promo_available": True,
        }

    # ── Instance Pause / Resume (REMOVED) ─────────────────────────────
    #
    # User-facing pause/resume was removed in favour of a RunPod-style
    # stop/start lifecycle: stopped instances preserve their container
    # (via the internal pause_container docker primitive) and can be
    # started again by the owner at any time. Low-balance auto-stops now
    # set ``payload.stop_reason = 'low_balance'`` instead of a distinct
    # ``paused_low_balance`` status, so the UI has one consistent
    # "stopped" surface. See alembic migration 031_drop_pause_resume_state.

    def _removed_pause_resume_stub(self, job_id: str, *args, **kwargs) -> dict:
        """Placeholder — pause/resume removed; callers must use stop/start."""
        raise RuntimeError(
            "pause_instance/resume_instance were removed; use stop_instance/start_instance"
        )

    # ── Instance Lifecycle: Stop / Start / Restart / Terminate ───────

    _VALID_STOP_REASONS = frozenset(
        {"user_stopped", "low_balance", "billing_suspended", "paused_low_balance"}
    )

    def stop_instance(self, job_id: str, reason: str = "user_stopped") -> dict:
        """Gracefully stop a running instance. Container is preserved for restart.

        Attempt-owned: atomic fenced lifecycle (``request_fenced_stop``) —
        no pre-mark raw SQL dual-write. Worker ACK projects to ``stopped``.
        Legacy: guarded ``update_job_status`` + pause_container agent queue.
        """
        if reason not in self._VALID_STOP_REASONS:
            return {
                "stopped": False,
                "reason": f"invalid_reason: must be one of {sorted(self._VALID_STOP_REASONS)}",
            }

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        if reason in ("paused_low_balance", "low_balance"):
            stop_reason_tag = "low_balance"
        elif reason == "billing_suspended":
            stop_reason_tag = "billing_suspended"
        else:
            stop_reason_tag = "user"
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id, active_attempt_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name
                   FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"stopped": False, "reason": "not_running"}

            owner = job.get("owner") or ""
            host_id = job.get("host_id") or ""
            active_attempt_id = job.get("active_attempt_id")
            container_name = job.get("container_name") or f"xcl-{job_id}"
            # Release lock before fenced controller / agent enqueue.
            conn.commit()

        # ── Attempt-owned: single fenced domain path (no dual raw SQL) ──
        if active_attempt_id:
            from control_plane.lifecycle import request_fenced_stop

            fenced = request_fenced_stop(
                job_id=job_id,
                created_by="billing_stop",
                container_name=container_name,
                reason_tag=stop_reason_tag,
            )
            if not fenced.ok:
                return {
                    "stopped": False,
                    "reason": fenced.reason or "enqueue_failed",
                    "job_id": job_id,
                    "status": fenced.status or None,
                }
            if fenced.command_created:
                cycle_id = f"BC-stop-{int(now)}-{os.urandom(3).hex()}"
                with pool.connection() as conn:
                    conn.execute(
                        """INSERT INTO billing_cycles
                           (cycle_id, job_id, customer_id, host_id, resource_type,
                            period_start, period_end, duration_seconds, rate_per_hour,
                            gpu_model, tier, tier_multiplier, amount_micros, status,
                            created_at)
                           VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0,
                                   0, 'stopped', %s)""",
                        (cycle_id, job_id, owner, host_id, now, now, now),
                    )
                    conn.commit()
            try:
                from db import NotificationStore

                NotificationStore.create(
                    user_email=owner,
                    notif_type="instance_stopped",
                    title=f"Instance stopped: {job.get('name', job_id)}",
                    body=(
                        "Your instance has been stopped. Storage continues to be billed. "
                        "Start it again anytime."
                    ),
                    data={"job_id": job_id},
                )
            except Exception:
                pass
            log.info(
                "STOP fenced job=%s reason=%s owner=%s attempt=%s",
                job_id,
                reason,
                owner,
                (fenced.attempt_id or "")[:8],
            )
            # Keep response shape stable for API/clients; extras are optional.
            out = {"stopped": True, "job_id": job_id, "status": "stopping"}
            if fenced.attempt_id:
                out["attempt_id"] = fenced.attempt_id
            if fenced.command_id:
                out["command_id"] = fenced.command_id
            return out

        # ── Legacy (no active attempt): guarded status + pause_container ──
        from scheduler import update_job_status

        marked = update_job_status(
            job_id,
            "stopping",
            expected_status="running",
            stop_reason=stop_reason_tag,
            stopping_at=now,
        )
        if marked is None:
            return {"stopped": False, "reason": "not_running"}

        stop_queued = False
        if host_id:
            try:
                from routes.agent import enqueue_agent_command
                from scheduler import _validate_name

                _validate_name(container_name, "container name")
                enqueue_agent_command(
                    host_id,
                    "pause_container",
                    {"container_name": container_name, "job_id": job_id},
                    created_by="billing_stop",
                )
                log.info("STOP pause_container queued: %s on %s", container_name, host_id)
                stop_queued = True
            except Exception as e:
                log.warning("STOP container stop enqueue failed for %s: %s", job_id, e)

        # Legacy commands have no ACK path — optimistic projection to stopped.
        final_status = "stopped" if stop_queued else "running"
        if stop_queued:
            update_job_status(
                job_id,
                "stopped",
                expected_status="stopping",
                stop_reason=stop_reason_tag,
                stopped_at=now,
            )
            with pool.connection() as conn:
                cycle_id = f"BC-stop-{int(now)}-{os.urandom(3).hex()}"
                conn.execute(
                    """INSERT INTO billing_cycles
                       (cycle_id, job_id, customer_id, host_id, resource_type,
                        period_start, period_end, duration_seconds, rate_per_hour,
                        gpu_model, tier, tier_multiplier, amount_micros, status,
                        created_at)
                       VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0,
                               0, 'stopped', %s)""",
                    (cycle_id, job_id, owner, host_id, now, now, now),
                )
                conn.commit()
        else:
            # Roll intermediate projection back so UI is not stuck on stopping.
            update_job_status(
                job_id,
                "running",
                expected_status="stopping",
            )
            log.error("STOP failed for job=%s — could not enqueue pause_container", job_id)
            return {"stopped": False, "reason": "enqueue_failed", "job_id": job_id}

        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="instance_stopped",
                title=f"Instance stopped: {job.get('name', job_id)}",
                body=(
                    "Your instance has been stopped. Storage continues to be billed. "
                    "Start it again anytime."
                ),
                data={"job_id": job_id},
            )
        except Exception:
            pass

        log.info("STOP job=%s reason=%s owner=%s", job_id, reason, owner)
        return {"stopped": True, "job_id": job_id, "status": final_status}

    def start_instance(self, job_id: str) -> dict:
        """Start a stopped instance.

        Legacy (no fenced history): docker start via agent queue.
        Fenced history: re-admit for a **new** placement attempt (never
        revive the old attempt/fence labels with start_container).
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id, active_attempt_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name,
                          EXISTS (
                              SELECT 1 FROM job_attempts a
                               WHERE a.job_id = jobs.job_id
                          ) AS has_fenced_history
                   FROM jobs
                   WHERE job_id = %s
                     AND status IN ('stopped', 'queued')
                   FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"started": False, "reason": "not_stopped"}

            owner = job.get("owner") or ""
            fund = self.wallet_has_available_funds(owner)
            if not fund.get("ok"):
                return {
                    "started": False,
                    "reason": (
                        "wallet_suspended"
                        if fund.get("reason") == "wallet_suspended"
                        else "insufficient_balance"
                    ),
                    "available_cad": fund.get("available_cad"),
                    "held_cad": fund.get("held_cad"),
                }

            # Fenced history: fresh-attempt resume (requeue), not container revive.
            if job.get("has_fenced_history"):
                conn.commit()
                from control_plane.lifecycle import request_fresh_attempt_resume

                resumed = request_fresh_attempt_resume(
                    job_id=job_id,
                    created_by="billing_start",
                    intent="resume",
                )
                if not resumed.ok:
                    return {
                        "started": False,
                        "reason": resumed.reason or "fenced_resume_failed",
                        "job_id": job_id,
                        "status": resumed.status or None,
                    }
                log.info("START fenced resume → queued job=%s", job_id)
                return {
                    "started": True,
                    "job_id": job_id,
                    "status": "queued",
                    "fresh_attempt": True,
                }

            if job.get("status") != "stopped":
                return {"started": False, "reason": "not_stopped"}

            host_id = job.get("host_id") or ""
            container_name = job.get("container_name") or f"xcl-{job_id}"
            conn.commit()

        # Legacy: guarded status transitions (CAS + outbox) — no bare SQL.
        from scheduler import update_job_status

        marked = update_job_status(
            job_id,
            "restarting",
            expected_status="stopped",
            restarting_at=now,
        )
        if marked is None:
            return {"started": False, "reason": "not_stopped"}

        start_queued = False
        if host_id:
            try:
                from routes.agent import enqueue_agent_command
                from scheduler import _validate_name

                _validate_name(container_name, "container name")
                enqueue_agent_command(
                    host_id,
                    "start_container",
                    {"container_name": container_name, "job_id": job_id},
                    created_by="billing_start",
                )
                start_queued = True
                log.info("START start_container queued: %s on %s", container_name, host_id)
            except Exception as e:
                log.warning("START container start enqueue failed for %s: %s", job_id, e)

        if not start_queued:
            update_job_status(
                job_id,
                "stopped",
                expected_status="restarting",
            )
            log.error("START failed for job=%s — could not enqueue start_container", job_id)
            return {"started": False, "reason": "enqueue_failed", "job_id": job_id}

        projected = update_job_status(
            job_id,
            "running",
            expected_status="restarting",
            started_at=now,
            stopped_at=0,
        )
        if projected is None:
            return {"started": False, "reason": "status_race", "job_id": job_id}

        with pool.connection() as conn:
            cycle_id = f"BC-start-{int(now)}-{os.urandom(3).hex()}"
            conn.execute(
                """INSERT INTO billing_cycles
                   (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                    duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                    amount_micros, status, created_at)
                   VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'started', %s)""",
                (cycle_id, job_id, owner, host_id, now, now, now),
            )
            conn.commit()

        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="instance_started",
                title=f"Instance started: {job.get('name', job_id)}",
                body="Your instance is running again.",
                data={"job_id": job_id},
            )
        except Exception:
            pass

        log.info("START job=%s owner=%s", job_id, owner)
        return {"started": True, "job_id": job_id, "status": "running"}

    def restart_instance(self, job_id: str) -> dict:
        """Restart a running or stopped instance.

        Legacy: pause_container + start_container via agent queue.
        Fenced / attempt-owned: never unfenced docker start. Running jobs
        get a fenced stop_attempt (intent=restart); worker ACK re-admits
        the job as ``queued`` for a new attempt. Already-stopped fenced
        jobs requeue immediately via fresh-attempt resume.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id, active_attempt_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name,
                          EXISTS (
                              SELECT 1 FROM job_attempts a
                               WHERE a.job_id = jobs.job_id
                          ) AS has_fenced_history
                   FROM jobs
                   WHERE job_id = %s
                     AND status IN ('running', 'stopped', 'stopping', 'queued')
                   FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"restarted": False, "reason": "not_restartable"}

            owner = job.get("owner") or ""
            fund = self.wallet_has_available_funds(owner)
            if not fund.get("ok"):
                return {
                    "restarted": False,
                    "reason": (
                        "wallet_suspended"
                        if fund.get("reason") == "wallet_suspended"
                        else "insufficient_balance"
                    ),
                    "available_cad": fund.get("available_cad"),
                    "held_cad": fund.get("held_cad"),
                }

            has_fenced = bool(job.get("has_fenced_history") or job.get("active_attempt_id"))
            if has_fenced:
                conn.commit()
                status = job.get("status")
                # Live attempt: fenced tear-down; ACK projects to queued.
                if job.get("active_attempt_id") and status in ("running", "stopping"):
                    from control_plane.lifecycle import request_fenced_stop_remove

                    fenced = request_fenced_stop_remove(
                        job_id=job_id,
                        intent="restart",
                        created_by="billing_restart",
                        container_name=job.get("container_name") or f"xcl-{job_id}",
                        reason_tag="user_restart",
                    )
                    if not fenced.ok:
                        return {
                            "restarted": False,
                            "reason": fenced.reason or "fenced_restart_failed",
                            "job_id": job_id,
                            "status": fenced.status or None,
                        }
                    log.info(
                        "RESTART fenced stop_attempt job=%s attempt=%s",
                        job_id,
                        (fenced.attempt_id or "")[:8],
                    )
                    return {
                        "restarted": True,
                        "job_id": job_id,
                        "status": "stopping",
                        "attempt_id": fenced.attempt_id,
                        "command_id": fenced.command_id,
                        "fresh_attempt": True,
                    }

                # Already stopped / queued / authority released: re-admit.
                from control_plane.lifecycle import request_fresh_attempt_resume

                resumed = request_fresh_attempt_resume(
                    job_id=job_id,
                    created_by="billing_restart",
                    intent="restart",
                )
                if not resumed.ok:
                    return {
                        "restarted": False,
                        "reason": resumed.reason or "fenced_restart_failed",
                        "job_id": job_id,
                        "status": resumed.status or None,
                    }
                log.info("RESTART fenced resume → queued job=%s", job_id)
                return {
                    "restarted": True,
                    "job_id": job_id,
                    "status": "queued",
                    "fresh_attempt": True,
                }

            if job["status"] not in ("running", "stopped"):
                return {"restarted": False, "reason": "not_restartable"}

            was_running = job["status"] == "running"
            prior_status = job["status"]
            host_id = job.get("host_id") or ""
            container_name = job.get("container_name") or f"xcl-{job_id}"
            conn.commit()

        # Legacy: guarded status transitions (CAS + outbox) — no bare SQL.
        from scheduler import update_job_status

        marked = update_job_status(
            job_id,
            "restarting",
            expected_status=prior_status,
            restarting_at=now,
        )
        if marked is None:
            return {"restarted": False, "reason": "not_restartable"}

        restart_queued = False
        enqueue_error: str | None = None
        if not host_id:
            enqueue_error = "no_host"
            log.warning(
                "RESTART failed for job=%s — no host_id on job (instance was never assigned?)",
                job_id,
            )
        else:
            try:
                from routes.agent import enqueue_agent_command
                from scheduler import _validate_name

                _validate_name(container_name, "container name")
                if was_running:
                    enqueue_agent_command(
                        host_id,
                        "pause_container",
                        {"container_name": container_name, "job_id": job_id},
                        created_by="billing_restart_stop",
                    )
                enqueue_agent_command(
                    host_id,
                    "start_container",
                    {"container_name": container_name, "job_id": job_id},
                    created_by="billing_restart_start",
                )
                restart_queued = True
                log.info(
                    "RESTART queued (was_running=%s): %s on %s",
                    was_running,
                    container_name,
                    host_id,
                )
            except Exception as e:
                enqueue_error = type(e).__name__
                log.warning("RESTART enqueue failed for %s: %s: %s", job_id, enqueue_error, e)

        if not restart_queued:
            # Project to stopped so UI is not stuck on restarting.
            update_job_status(
                job_id,
                "stopped",
                expected_status="restarting",
            )
            log.error(
                "RESTART failed for job=%s — marking stopped (reason=%s)",
                job_id,
                enqueue_error or "unknown",
            )
            return {
                "restarted": False,
                "reason": enqueue_error or "enqueue_failed",
                "job_id": job_id,
            }

        projected = update_job_status(
            job_id,
            "running",
            expected_status="restarting",
            restarted_at=now,
        )
        if projected is None:
            return {
                "restarted": False,
                "reason": "status_race",
                "job_id": job_id,
            }

        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="instance_restarted",
                title=f"Instance restarted: {job.get('name', job_id)}",
                body="Your instance has been restarted successfully.",
                data={"job_id": job_id},
            )
        except Exception:
            pass

        log.info("RESTART job=%s owner=%s", job_id, owner)
        return {"restarted": True, "job_id": job_id, "status": "running"}

    def terminate_instance(self, job_id: str) -> dict:
        """Hard-kill and remove a container. This is irreversible.

        The container and its anonymous volumes are permanently destroyed.
        Named/NFS volumes are preserved. No restart is possible after termination.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id, active_attempt_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name
                   FROM jobs
                   WHERE job_id = %s
                     AND status NOT IN ('terminated', 'completed', 'failed', 'preempted', 'cancelled')
                   FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"terminated": False, "reason": "already_terminal_or_not_found"}

            # Snapshot under lock, then release before fenced enqueue / legacy work.
            active_attempt_id = job.get("active_attempt_id")
            owner = job.get("owner") or ""
            host_id = job.get("host_id") or ""
            container_name = job.get("container_name") or f"xcl-{job_id}"
            conn.commit()

        # Attempt-owned: fenced stop/remove via lifecycle controller.
        # Do NOT mark terminal or detach volumes here — that races the
        # worker. Intermediate ``stopping`` + durable stop_attempt
        # (preserve=False, intent=terminate); ACK projects terminated.
        if active_attempt_id:
            from control_plane.lifecycle import request_fenced_stop_remove

            fenced = request_fenced_stop_remove(
                job_id=job_id,
                intent="terminate",
                created_by="billing_terminate",
                container_name=container_name,
                reason_tag="user_terminated",
            )
            if not fenced.ok:
                return {
                    "terminated": False,
                    "reason": fenced.reason or "fenced_terminate_failed",
                    "job_id": job_id,
                    "status": fenced.status or None,
                }
            # Billing anchor once per attempt: only when this call first
            # owns the durable command (idempotent re-entry reuses the
            # command without a second BC-term row). Deterministic cycle_id
            # is a second line of defense against concurrent double-insert.
            if fenced.command_created and fenced.attempt_id:
                cycle_id = f"BC-term-{fenced.attempt_id}"
                with pool.connection() as bconn:
                    bconn.execute(
                        """INSERT INTO billing_cycles
                           (cycle_id, job_id, customer_id, host_id, resource_type,
                            period_start, period_end, duration_seconds, rate_per_hour,
                            gpu_model, tier, tier_multiplier, amount_micros, status,
                            created_at)
                           VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0,
                                   0, 'terminated', %s)
                           ON CONFLICT (cycle_id) DO NOTHING""",
                        (cycle_id, job_id, owner, host_id, now, now, now),
                    )
                    bconn.commit()
            log.info(
                "TERMINATE fenced stop_attempt queued job=%s attempt=%s created=%s",
                job_id,
                (fenced.attempt_id or "")[:8],
                fenced.command_created,
            )
            return {
                "terminated": True,
                "job_id": job_id,
                "status": "stopping",
                "attempt_id": fenced.attempt_id,
                "command_id": fenced.command_id,
            }

        # Legacy (non-attempt-owned): guarded status path, detach, direct kill.
        from scheduler import update_job_status

        prior_status = job.get("status")
        terminated = update_job_status(
            job_id,
            "terminated",
            expected_status=prior_status,
            terminated_at=now,
        )
        if terminated is None:
            return {"terminated": False, "reason": "already_terminal_or_not_found"}
        with pool.connection() as conn:
            cycle_id = f"BC-term-{int(now)}-{os.urandom(3).hex()}"
            conn.execute(
                """INSERT INTO billing_cycles
                   (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                    duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                    amount_micros, status, created_at)
                   VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'terminated', %s)""",
                (cycle_id, job_id, owner, host_id, now, now, now),
            )
            conn.commit()

        try:
            from volumes import get_volume_engine

            get_volume_engine().detach_all_for_instance(job_id)
        except Exception as e:
            log.warning("Volume detach failed for %s: %s", job_id, e)

        if host_id:
            try:
                from scheduler import terminate_job as _terminate_job, list_hosts, _validate_name

                _validate_name(container_name, "container name")
                hosts = list_hosts()
                hmap = {h["host_id"]: h for h in hosts}
                host = hmap.get(host_id)
                if host:
                    _terminate_job({"job_id": job_id, "container_name": container_name}, host)
            except Exception as e:
                log.warning(
                    "TERMINATE container removal failed for %s: %s — already gone?", job_id, e
                )

        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="instance_terminated",
                title=f"Instance terminated: {job.get('name', job_id)}",
                body="Your instance has been permanently terminated.",
                data={"job_id": job_id},
            )
        except Exception:
            pass

        log.info("TERMINATE job=%s owner=%s", job_id, owner)
        return {"terminated": True, "job_id": job_id, "status": "terminated"}

    # ── Wallet Lifecycle ──────────────────────────────────────────────

    def reactivate_wallet(self, customer_id: str) -> dict:
        """Reactivate a suspended wallet (after successful deposit)."""
        with self._conn() as conn:
            conn.execute(
                """UPDATE wallets
                   SET status = 'active', grace_until = 0, updated_at = %s
                   WHERE customer_id = %s AND status = 'suspended'""",
                (time.time(), customer_id),
            )
        log.info("WALLET %s reactivated", customer_id)
        return {"customer_id": customer_id, "status": "active"}

    def configure_auto_topup(
        self,
        customer_id: str,
        enabled: bool,
        amount_cad: float = 50.0,
        threshold_cad: float = 10.0,
        stripe_payment_method_id: str = "",
    ) -> dict:
        """Configure auto-top-up for a customer wallet."""
        wallet = self.get_wallet(customer_id)
        with self._conn() as conn:
            conn.execute(
                """UPDATE wallets
                   SET auto_topup_enabled = %s,
                       auto_topup_amount_micros = %s,
                       auto_topup_threshold_micros = %s,
                       stripe_payment_method_id = %s,
                       updated_at = %s
                   WHERE customer_id = %s""",
                (
                    enabled,
                    # Convert once, at the edge. Storing money as float lets
                    # rounding drift accumulate every time it is read back.
                    cad_to_micros(amount_cad),
                    cad_to_micros(threshold_cad),
                    stripe_payment_method_id,
                    time.time(),
                    customer_id,
                ),
            )
        log.info(
            "Auto-topup configured for %s: enabled=%s amount=$%.2f threshold=$%.2f",
            customer_id,
            enabled,
            amount_cad,
            threshold_cad,
        )
        return {
            "customer_id": customer_id,
            "auto_topup_enabled": enabled,
            # The API contract stays in CAD; only the stored representation
            # became integer micros.
            "auto_topup_amount_cad": amount_cad,
            "auto_topup_threshold_cad": threshold_cad,
        }

    def ensure_stripe_customer(self, customer_id: str, email: str = "") -> str:
        """Return the Stripe customer id for a wallet, creating it on first use.

        Persisted on ``wallets.stripe_customer_id`` so off-session auto-top-up
        charges target a real Stripe Customer. Raises RuntimeError if Stripe is
        not configured.
        """
        from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

        if not STRIPE_ENABLED or not _stripe_mod:
            raise RuntimeError("Stripe is not configured")

        wallet = self.get_wallet(customer_id)
        existing = (wallet.get("stripe_customer_id") or "").strip()
        if existing:
            return existing

        create_kwargs: dict = {
            "metadata": {"xcelsior_customer_id": customer_id},
            "preferred_locales": ["en-CA", "en"],
        }
        if email:
            create_kwargs["email"] = email

        # Invoice PDF/email/hosted page defaults from brand config (footer keeps
        # support phone + address; custom fields surface contact on the header).
        from pathlib import Path

        dash_path = Path(__file__).resolve().parent / "config" / "stripe_dashboard.json"
        if dash_path.exists():
            try:
                dash = json.loads(dash_path.read_text())
                inv_settings: dict = {}
                if dash.get("invoice_footer"):
                    inv_settings["footer"] = dash["invoice_footer"]
                if dash.get("invoice_custom_fields"):
                    inv_settings["custom_fields"] = dash["invoice_custom_fields"]
                inv_settings["rendering_options"] = {
                    "amount_tax_display": "include_inclusive_tax",
                }
                if inv_settings:
                    create_kwargs["invoice_settings"] = inv_settings
            except Exception as exc:
                log.debug("Could not load invoice branding defaults: %s", exc)

        cust = _stripe_mod.Customer.create(**create_kwargs)

        with self._conn() as conn:
            conn.execute(
                "UPDATE wallets SET stripe_customer_id = %s, updated_at = %s "
                "WHERE customer_id = %s",
                (cust.id, time.time(), customer_id),
            )
        log.info("Created Stripe customer %s for %s", cust.id, customer_id)
        return cust.id

    def create_billing_portal_session(self, customer_id: str, email: str = "") -> dict:
        """Open Stripe Customer Portal (invoices, payment methods, profile)."""
        import json
        from pathlib import Path

        from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

        if not STRIPE_ENABLED or not _stripe_mod:
            raise RuntimeError("Stripe is not configured")

        stripe_customer_id = self.ensure_stripe_customer(customer_id, email)
        base_url = os.environ.get("XCELSIOR_BASE_URL", "https://xcelsior.ca").rstrip("/")
        kwargs: dict = {
            "customer": stripe_customer_id,
            "return_url": f"{base_url}/dashboard/billing",
        }
        dash_path = Path(__file__).resolve().parent / "config" / "stripe_dashboard.json"
        if dash_path.exists():
            try:
                portal_cfg = json.loads(dash_path.read_text()).get(
                    "billing_portal_configuration_id"
                )
                if portal_cfg:
                    kwargs["configuration"] = portal_cfg
            except Exception as exc:
                log.debug("Could not load portal configuration: %s", exc)

        session = _stripe_mod.billing_portal.Session.create(**kwargs)
        return {
            "url": session.url,
            "session_id": session.id,
            "stripe_customer_id": stripe_customer_id,
        }

    def create_setup_intent(self, customer_id: str, email: str = "") -> dict:
        """Create a Stripe SetupIntent so the client can save a card off-session."""
        from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

        if not STRIPE_ENABLED or not _stripe_mod:
            raise RuntimeError("Stripe is not configured")

        stripe_customer_id = self.ensure_stripe_customer(customer_id, email)
        si = _stripe_mod.SetupIntent.create(
            customer=stripe_customer_id,
            usage="off_session",
            automatic_payment_methods={"enabled": True},
            metadata={"xcelsior_customer_id": customer_id},
        )
        return {
            "client_secret": si.client_secret,
            "setup_intent_id": si.id,
            "stripe_customer_id": stripe_customer_id,
        }

    def list_payment_methods(self, customer_id: str) -> list[dict]:
        """List the saved card payment methods for a customer wallet."""
        from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

        if not STRIPE_ENABLED or not _stripe_mod:
            return []

        wallet = self.get_wallet(customer_id)
        stripe_customer_id = (wallet.get("stripe_customer_id") or "").strip()
        if not stripe_customer_id:
            return []

        default_pm = (wallet.get("stripe_payment_method_id") or "").strip()
        methods = _stripe_mod.PaymentMethod.list(customer=stripe_customer_id, type="card")
        out: list[dict] = []
        # **Subscript, not `.get()`.** The previous comment here read "Stripe
        # objects are dict-like at runtime but typed without .get()", and that
        # is false for the installed SDK (15.3.1): `StripeObject.__getattr__`
        # raises `AttributeError: get`.
        #
        # The shape of that bug is why it survived. With **zero** saved cards
        # the loop body never runs and this returns `[]` harmlessly; it only
        # raises for a customer who actually has a card. So every caller saw
        # "no saved cards" — including the manual top-up and the auto-top-up
        # sweep, both of which resolve a payment method through here. P1's
        # headline behaviour, a top-up on a saved card, could not work at all.
        #
        # Subscript access is what the SDK supports on both `StripeObject` and
        # the plain dicts a test double is likely to supply.
        for pm in cast("list[Any]", methods.data):
            card = pm["card"] if "card" in pm else {}
            out.append(
                {
                    "id": pm["id"],
                    "brand": card["brand"] if "brand" in card else "",
                    "last4": card["last4"] if "last4" in card else "",
                    "exp_month": card["exp_month"] if "exp_month" in card else None,
                    "exp_year": card["exp_year"] if "exp_year" in card else None,
                    "is_default": pm["id"] == default_pm,
                }
            )
        return out

    def resolve_payment_method(
        self,
        customer_id: str,
        *,
        payment_method_id: str = "",
        last4: str = "",
        brand: str = "",
    ) -> dict:
        """Pick the card the caller meant, or explain why it cannot.

        Nobody says `pm_1QxYz...`. They say "my Visa" or "the one ending 4242",
        so the selector has to be resolvable from what `list_payment_methods`
        already returns: brand, last four, and which is default.

        Resolution is server-side rather than left to the model. If a user says
        "Visa" and two Visas are on file, the right answer is to ask which —
        not to pick one. Charging the wrong card is not undone by an apology.

        Order:

        1. An explicit `payment_method_id`, which must belong to this customer.
           A card id that is not on the caller's account is refused rather than
           passed to Stripe.
        2. A `last4` and/or `brand` selector, which must match exactly one.
        3. Nothing — use the account's default card, which is the sensible
           reading of "top up my account".

        Returns `{"ok": True, "payment_method": {...}}` or `{"ok": False,
        "reason": ..., "candidates": [...]}` so the caller can tell the user
        what to choose between.
        """
        methods = self.list_payment_methods(customer_id)
        if not methods:
            return {
                "ok": False,
                "reason": "no_saved_cards",
                "message": (
                    "No cards are saved on this account. Add one in the "
                    "dashboard first — cards cannot be added through the API."
                ),
                "candidates": [],
            }

        wanted_id = (payment_method_id or "").strip()
        if wanted_id:
            match = next((m for m in methods if m.get("id") == wanted_id), None)
            if not match:
                # Deliberately does not distinguish "no such card" from "not
                # yours": both answer the same way, so this cannot be used to
                # probe whether a payment method id exists.
                return {
                    "ok": False,
                    "reason": "unknown_payment_method",
                    "message": "That payment method is not saved on this account.",
                    "candidates": methods,
                }
            return {"ok": True, "payment_method": match}

        wanted_last4 = (last4 or "").strip()[-4:]
        wanted_brand = (brand or "").strip().lower()
        if wanted_last4 or wanted_brand:
            candidates = [
                m
                for m in methods
                if (not wanted_last4 or str(m.get("last4") or "") == wanted_last4)
                and (not wanted_brand or str(m.get("brand") or "").lower() == wanted_brand)
            ]
            if not candidates:
                return {
                    "ok": False,
                    "reason": "no_matching_card",
                    "message": "No saved card matches that description.",
                    "candidates": methods,
                }
            if len(candidates) > 1:
                return {
                    "ok": False,
                    "reason": "ambiguous_card",
                    "message": (
                        "More than one saved card matches. Say which one — the "
                        "last four digits are enough."
                    ),
                    "candidates": candidates,
                }
            return {"ok": True, "payment_method": candidates[0]}

        default = next((m for m in methods if m.get("is_default")), None)
        if default:
            return {"ok": True, "payment_method": default}
        if len(methods) == 1:
            # One card and no default flag is unambiguous.
            return {"ok": True, "payment_method": methods[0]}
        return {
            "ok": False,
            "reason": "no_default_card",
            "message": (
                "No default card is set and more than one is saved. Say which "
                "one to use — the last four digits are enough."
            ),
            "candidates": methods,
        }

    def detach_payment_method(self, customer_id: str, payment_method_id: str) -> dict:
        """Detach a saved card. If it was the auto-top-up default, disable auto-top-up."""
        from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

        if not STRIPE_ENABLED or not _stripe_mod:
            raise RuntimeError("Stripe is not configured")

        wallet = self.get_wallet(customer_id)
        stripe_customer_id = (wallet.get("stripe_customer_id") or "").strip()

        # Verify the payment method belongs to this customer BEFORE detaching.
        # Stripe's detach() works on any pm_ id regardless of owner, so without
        # this check a user could detach another customer's card by id (IDOR).
        # retrieve() also raises for unknown ids — convert to a clean ValueError
        # (404 at the route) instead of letting it bubble up as a 500.
        try:
            pm = _stripe_mod.PaymentMethod.retrieve(payment_method_id)
        except Exception as e:
            raise ValueError("Payment method not found") from e
        pm_customer = getattr(pm, "customer", None)
        if not stripe_customer_id or pm_customer != stripe_customer_id:
            raise ValueError("Payment method not found")

        _stripe_mod.PaymentMethod.detach(payment_method_id)

        if (wallet.get("stripe_payment_method_id") or "").strip() == payment_method_id:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE wallets SET stripe_payment_method_id = '', "
                    "auto_topup_enabled = false, updated_at = %s WHERE customer_id = %s",
                    (time.time(), customer_id),
                )
            log.info("Detached default payment method for %s; auto-topup disabled", customer_id)
        return {"ok": True, "payment_method_id": payment_method_id}

    # ── Auto-Billing Cycle (Running Instances) ────────────────────────

    def _bill_gpu_period(
        self,
        conn,
        job_row: dict,
        period_start: float,
        period_end: float,
        *,
        min_seconds: float = 60.0,
        description_prefix: str = "Auto-billing",
    ) -> dict | None:
        """Charge one GPU billing period and insert billing_cycles row."""
        from psycopg.rows import dict_row

        conn.row_factory = dict_row

        job_id = job_row["job_id"]
        customer_id = job_row.get("owner") or ""
        if not customer_id:
            return None

        if period_end - period_start < min_seconds:
            return None

        duration_sec = period_end - period_start
        host_id = job_row.get("host_id") or ""
        gpu_model = job_row.get("gpu_model") or ""
        tier = job_row.get("tier", "free")

        host = conn.execute(
            "SELECT payload FROM hosts WHERE host_id = %s",
            (host_id,),
        ).fetchone()
        host_payload = {}
        if host and host.get("payload"):
            raw = host["payload"]
            host_payload = raw if isinstance(raw, dict) else json.loads(raw)

        job_payload = {
            "job_id": job_id,
            "pricing_mode": job_row.get("pricing_mode")
            or job_row.get("payload_pricing_mode"),
            "spot_rate_cad": job_row.get("spot_rate_cad"),
            "preemptible": job_row.get("preemptible"),
            "spot": job_row.get("spot"),
            "tier": tier,
            "gpu_model": gpu_model,
            "host_gpu_model": job_row.get("host_gpu_model"),
            "num_gpus": job_row.get("num_gpus") or 1,
        }
        rate_per_hour, pricing_mode = resolve_compute_rate_cad(job_payload, host_payload)

        tier_multiplier = 1.0

        amount_cad = round((duration_sec / 3600) * rate_per_hour * tier_multiplier, 4)
        if amount_cad <= 0:
            return None

        charge_result = self.charge(
            customer_id,
            amount_cad,
            job_id=job_id,
            description=f"{description_prefix}: {gpu_model} ({duration_sec/60:.1f}min)",
        )

        now = time.time()
        cycle_id = f"BC-{int(now)}-{os.urandom(3).hex()}"
        status = "charged" if charge_result.get("charged") else "failed"

        conn.execute(
            """INSERT INTO billing_cycles
               (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                amount_cad, status, pricing_mode, created_at)
               VALUES (%s, %s, %s, %s, 'gpu', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (
                cycle_id,
                job_id,
                customer_id,
                host_id,
                period_start,
                period_end,
                duration_sec,
                rate_per_hour,
                gpu_model,
                tier,
                tier_multiplier,
                amount_cad,
                status,
                pricing_mode,
                now,
            ),
        )
        return {
            "job_id": job_id,
            "amount_cad": amount_cad,
            "charge_result": charge_result,
            "pricing_mode": pricing_mode,
        }

    def bill_running_period(
        self,
        job_id: str,
        period_end: float | None = None,
        *,
        min_seconds: float = 0.0,
    ) -> dict:
        """Bill a running job from its last cycle (or started_at) through period_end.

        Used on preemption to close compute billing pro-rata.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        period_end = period_end or time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id,
                          payload->>'owner' AS owner,
                          (payload->>'started_at')::double precision AS started_at,
                          payload->>'gpu_model' AS gpu_model,
                          payload->>'host_gpu_model' AS host_gpu_model,
                          COALESCE((payload->>'num_gpus')::int, 1) AS num_gpus,
                          COALESCE(payload->>'pricing_mode', 'on_demand') AS pricing_mode,
                          (payload->>'spot_rate_cad')::double precision AS spot_rate_cad,
                          payload->>'preemptible' AS preemptible,
                          payload->>'spot' AS spot,
                          COALESCE(payload->>'tier', 'free') AS tier
                   FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"billed": False, "reason": "not_running"}

            last = conn.execute(
                """SELECT period_end FROM billing_cycles
                   WHERE job_id = %s ORDER BY period_end DESC LIMIT 1""",
                (job_id,),
            ).fetchone()
            period_start = last["period_end"] if last else float(job["started_at"] or 0)
            if period_start <= 0:
                return {"billed": False, "reason": "no_start_time"}

            result = self._bill_gpu_period(
                conn,
                job,
                period_start,
                period_end,
                min_seconds=min_seconds,
                description_prefix="Preempt billing",
            )
            conn.commit()
            if not result:
                return {"billed": False, "reason": "below_minimum"}
            return {"billed": True, **result}

    def auto_billing_cycle(self) -> dict:
        """Bill all running instances for the current billing period.

        Called periodically (every 5 minutes) by the background scheduler.
        For each running job, computes the charge since the last billing
        cycle and creates a billing_cycles record.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        billed = 0
        suspended = 0
        errors = 0

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            # Find all running jobs
            running = conn.execute(
                """SELECT j.job_id,
                          j.payload->>'owner' AS owner,
                          (j.payload->>'started_at')::double precision AS started_at,
                          j.host_id,
                          j.payload->>'gpu_model' AS gpu_model,
                          j.payload->>'host_gpu_model' AS host_gpu_model,
                          COALESCE((j.payload->>'num_gpus')::int, 1) AS num_gpus,
                          COALESCE(j.payload->>'pricing_mode', 'on_demand') AS pricing_mode,
                          (j.payload->>'spot_rate_cad')::double precision AS spot_rate_cad,
                          j.payload->>'preemptible' AS preemptible,
                          j.payload->>'spot' AS spot,
                          COALESCE(j.payload->>'tier', 'free') AS tier
                   FROM jobs j
                   WHERE j.status = 'running'
                     AND (j.payload->>'started_at')::double precision > 0""",
            ).fetchall()

        for job in running:
            try:
                job_id = job["job_id"]
                customer_id = job["owner"]
                if not customer_id:
                    log.warning("AUTO-BILLING: skipping job %s — no owner set", job_id)
                    continue
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    locked = conn.execute(
                        "SELECT job_id FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE SKIP LOCKED",
                        (job_id,),
                    ).fetchone()
                    if not locked:
                        continue

                    last = conn.execute(
                        """SELECT period_end FROM billing_cycles
                           WHERE job_id = %s ORDER BY period_end DESC LIMIT 1""",
                        (job_id,),
                    ).fetchone()

                    period_start = last["period_end"] if last else float(job["started_at"])
                    period_end = now

                    result = self._bill_gpu_period(
                        conn,
                        job,
                        period_start,
                        period_end,
                        min_seconds=60.0,
                    )
                    if not result:
                        continue
                    charge_result = result["charge_result"]
                    conn.commit()

                billed += 1

                # Low-balance notification at $2 (dedup: once per 24h per customer)
                # Skip notification if balance is zero AND no charge occurred (new account, never spent)
                if charge_result.get("charged"):
                    new_balance = charge_result.get("balance_cad", 0)
                    # Only notify if the user actually spent money (charged > 0) and balance dropped low
                    # This prevents firing for brand-new $0 wallets that have never run a job
                    amount_charged = (
                        charge_result.get("amount_charged", 0)
                        or charge_result.get("billed_usd", 0)
                        or 0
                    )
                    if new_balance < 2.0 and amount_charged > 0:
                        try:
                            from db import NotificationStore

                            # Check if we already sent a low-balance notif in the last 24h
                            recent = pool.connection()
                            with recent as rc:
                                rc.row_factory = dict_row
                                existing = rc.execute(
                                    """SELECT id FROM notifications
                                       WHERE user_email = %s AND type = 'billing'
                                         AND title LIKE 'Low balance%%'
                                         AND created_at > %s LIMIT 1""",
                                    (customer_id, now - 86400),
                                ).fetchone()
                            if not existing:
                                NotificationStore.create(
                                    user_email=customer_id,
                                    notif_type="billing",
                                    title=f"Low balance: ${new_balance:.2f} CAD",
                                    body="Your balance is running low. Top up to avoid service interruption.",
                                    data={"balance_cad": new_balance},
                                )
                                # Also send email to the user
                                try:
                                    from scheduler import send_email
                                    import threading

                                    threading.Thread(
                                        target=send_email,
                                        args=(
                                            f"Low balance: ${new_balance:.2f} CAD",
                                            f"Hi,\n\nYour Xcelsior balance is ${new_balance:.2f} CAD.\n\n"
                                            "Your running instances may be suspended if your balance reaches $0.\n\n"
                                            "Top up at https://xcelsior.ca/dashboard/billing\n\n"
                                            "— Xcelsior",
                                        ),
                                        kwargs={"to_email": customer_id},
                                        daemon=True,
                                    ).start()
                                except Exception:
                                    pass
                        except Exception:
                            pass

                # If charge failed with grace_expired → suspend and STOP the job
                if (
                    not charge_result.get("charged")
                    and charge_result.get("action") == "account_suspended"
                ):
                    suspended += 1
                    try:
                        stop_result = self.stop_instance(
                            job_id, reason="billing_suspended"
                        )
                        if not stop_result.get("stopped"):
                            log.error(
                                "BILLING: Failed to stop job %s on suspension: %s",
                                job_id,
                                stop_result.get("reason"),
                            )
                    except Exception as kill_err:
                        log.error(
                            "BILLING: Failed to kill job %s on suspension: %s", job_id, kill_err
                        )

            except Exception as e:
                errors += 1
                log.error("Auto-billing error for job %s: %s", job.get("job_id", "?"), e)

        # ── Bill active volumes (real-time storage charges) ──────────
        volume_billed = 0
        try:
            from volumes import get_volume_engine

            ve = get_volume_engine()

            # Sweep stale provisioning/deleting volumes before billing
            try:
                ve.cleanup_stale_volumes()
            except Exception as e:
                log.warning("Stale volume cleanup error: %s", e)

            # Reconcile orphaned attachments (volumes attached to dead instances)
            try:
                ve.reconcile_orphaned_attachments()
            except Exception as e:
                log.warning("Orphan volume reconciliation error: %s", e)

            # Fetch suspended wallets to skip their volumes
            suspended_owners: set[str] = set()
            _skip_volume_billing = False
            try:
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    rows = conn.execute(
                        "SELECT DISTINCT customer_id FROM wallets WHERE status = 'suspended'"
                    ).fetchall()
                    suspended_owners = {r["customer_id"] for r in rows}
            except Exception as e:
                # Fail-closed: if we can't check suspended wallets, skip all volume
                # billing this cycle rather than accidentally charging suspended users.
                log.error(
                    "Suspended wallet lookup failed — skipping volume billing this cycle: %s", e
                )
                _skip_volume_billing = True

            active_volumes = []
            if not _skip_volume_billing:
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    active_volumes = conn.execute(
                        """SELECT volume_id, owner_id, name, size_gb, created_at
                           FROM volumes WHERE status IN ('available', 'attached')""",
                    ).fetchall()

            for vol in active_volumes:
                try:
                    vid = vol["volume_id"]
                    vol_owner = vol["owner_id"]
                    size_gb = vol.get("size_gb", 0)
                    if size_gb <= 0:
                        continue
                    if vol_owner in suspended_owners:
                        log.debug("Skipping volume billing for %s: wallet suspended", vid)
                        continue

                    # Single transaction with row lock prevents double-billing
                    # from concurrent billing ticks (mirrors GPU billing pattern)
                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        # Lock the volume row so a concurrent tick skips it
                        locked = conn.execute(
                            "SELECT volume_id FROM volumes WHERE volume_id = %s AND status IN ('available', 'attached') FOR UPDATE SKIP LOCKED",
                            (vid,),
                        ).fetchone()
                        if not locked:
                            continue  # Another tick already processing this volume

                        last_vc = conn.execute(
                            """SELECT period_end FROM billing_cycles
                               WHERE job_id = %s ORDER BY period_end DESC LIMIT 1""",
                            (vid,),
                        ).fetchone()

                        vperiod_start = (
                            last_vc["period_end"] if last_vc else float(vol["created_at"])
                        )
                        vperiod_end = now

                        if vperiod_end - vperiod_start < 60:
                            continue

                        vduration_sec = vperiod_end - vperiod_start

                        from volumes import VOLUME_PRICE_PER_GB_MONTH_CAD

                        HOURS_PER_MONTH = 730  # industry standard (365.25 × 24 / 12)
                        rate_per_sec = (VOLUME_PRICE_PER_GB_MONTH_CAD * size_gb) / (
                            HOURS_PER_MONTH * 3600
                        )
                        vamount = round(rate_per_sec * vduration_sec, 4)

                        if vamount <= 0:
                            continue

                        # Charge the wallet
                        vcharge = self.charge(
                            vol_owner,
                            vamount,
                            job_id=vid,
                            description=f"Volume storage: {vol.get('name', vid)} ({size_gb} GB, {vduration_sec/60:.1f}min)",
                        )

                        vcycle_id = f"VC-{int(now)}-{os.urandom(3).hex()}"
                        vstatus = "charged" if vcharge.get("charged") else "failed"

                        # Record the billing cycle (inside same locked transaction)
                        conn.execute(
                            """INSERT INTO billing_cycles
                               (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                                duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                                amount_micros, status, created_at)
                               VALUES (%s, %s, %s, %s, 'volume', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (
                                vcycle_id,
                                vid,
                                vol_owner,
                                "",
                                vperiod_start,
                                vperiod_end,
                                vduration_sec,
                                round(VOLUME_PRICE_PER_GB_MONTH_CAD * size_gb / HOURS_PER_MONTH, 6),
                                "storage",
                                "volume",
                                1.0,
                                round(vamount * 1_000_000),
                                vstatus,
                                now,
                            ),
                        )
                        conn.commit()

                    volume_billed += 1
                except Exception as e:
                    errors += 1
                    log.error("Volume billing error for %s: %s", vol.get("volume_id", "?"), e)
        except Exception as e:
            log.error("Volume billing scan error: %s", e)

        # ── Bill active serverless workers (incremental GPU-seconds) ──
        inference_billed = 0
        try:
            from serverless.metering import bill_active_serverless_workers

            inference_billed = bill_active_serverless_workers(self, now=now)
        except Exception as e:
            log.error("Serverless billing scan error: %s", e)

        # ── Bill stopped instances for storage ───────────────────────
        # Charges per GB per hour based on storage_type. Requires the
        # storage_billing_rates table from migration 019. Gracefully
        # skips if the table doesn't exist yet.
        storage_billed = 0
        try:
            with pool.connection() as conn:
                conn.row_factory = dict_row
                stopped_jobs = conn.execute(
                    """SELECT j.job_id,
                              j.host_id,
                              j.payload->>'owner' AS owner,
                              COALESCE((j.payload->>'storage_gb')::double precision, 0) AS storage_gb,
                              COALESCE(j.payload->>'storage_type', 'hdd') AS storage_type,
                              COALESCE((j.payload->>'storage_rate_cad_per_gb_hr')::double precision, 0) AS cached_rate,
                              (j.payload->>'stopped_at')::double precision AS stopped_at
                       FROM jobs j
                       WHERE j.status = 'stopped'
                         AND COALESCE((j.payload->>'storage_gb')::double precision, 0) > 0""",
                ).fetchall()

            for sjob in stopped_jobs:
                try:
                    sjob_id = sjob["job_id"]
                    sowner = sjob["owner"]
                    storage_gb = float(sjob["storage_gb"] or 0)
                    storage_type = sjob["storage_type"] or "hdd"
                    cached_rate = float(sjob["cached_rate"] or 0)

                    if not sowner or storage_gb <= 0:
                        continue

                    # Look up current rate from storage_billing_rates table
                    try:
                        with pool.connection() as conn:
                            conn.row_factory = dict_row
                            rate_row = conn.execute(
                                "SELECT rate_cad_per_gb_hr FROM storage_billing_rates WHERE storage_type = %s",
                                (storage_type,),
                            ).fetchone()
                        rate = float(rate_row["rate_cad_per_gb_hr"]) if rate_row else cached_rate
                    except Exception:
                        rate = cached_rate  # table not yet created

                    if rate <= 0:
                        continue

                    # Find last storage billing cycle for this job
                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        last_sc = conn.execute(
                            """SELECT period_end FROM billing_cycles
                               WHERE job_id = %s AND status IN ('storage', 'stopped', 'started')
                               ORDER BY period_end DESC LIMIT 1""",
                            (sjob_id,),
                        ).fetchone()

                    stopped_at = float(sjob["stopped_at"] or 0)
                    speriod_start = (
                        last_sc["period_end"]
                        if last_sc
                        else (stopped_at if stopped_at > 0 else now)
                    )
                    speriod_end = now

                    if speriod_end - speriod_start < 60:
                        continue

                    sduration_sec = speriod_end - speriod_start
                    samount = round((sduration_sec / 3600) * rate * storage_gb, 6)

                    if samount <= 0:
                        continue

                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        slocked = conn.execute(
                            "SELECT job_id FROM jobs WHERE job_id = %s AND status = 'stopped' FOR UPDATE SKIP LOCKED",
                            (sjob_id,),
                        ).fetchone()
                        if not slocked:
                            continue

                        scharge = self.charge(
                            sowner,
                            samount,
                            job_id=sjob_id,
                            description=f"Storage: {storage_gb:.0f}GB {storage_type} ({sduration_sec/60:.1f}min)",
                        )
                        scycle_id = f"SC-{int(now)}-{os.urandom(3).hex()}"
                        sstatus = "storage" if scharge.get("charged") else "storage_failed"
                        conn.execute(
                            """INSERT INTO billing_cycles
                               (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                                duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                                amount_micros, status, created_at)
                               VALUES (%s, %s, %s, %s, 'gpu', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (
                                scycle_id,
                                sjob_id,
                                sowner,
                                sjob.get("host_id", ""),
                                speriod_start,
                                speriod_end,
                                sduration_sec,
                                rate * storage_gb,
                                storage_type,
                                "storage",
                                1.0,
                                round(samount * 1_000_000),
                                sstatus,
                                now,
                            ),
                        )
                        conn.commit()

                    storage_billed += 1
                except Exception as e:
                    errors += 1
                    log.error("Storage billing error for job %s: %s", sjob.get("job_id", "?"), e)
        except Exception as e:
            log.error("Storage billing scan error: %s", e)

        if billed or suspended or errors or volume_billed or inference_billed or storage_billed:
            log.info(
                "AUTO-BILLING: %d compute, %d storage, %d volumes, %d inference, %d suspended, %d errors",
                billed,
                storage_billed,
                volume_billed,
                inference_billed,
                suspended,
                errors,
            )

        return {
            "billed": billed,
            "storage_billed": storage_billed,
            "volume_billed": volume_billed,
            "inference_billed": inference_billed,
            "suspended": suspended,
            "errors": errors,
        }

    def _register_payment_intent(
        self,
        *,
        intent_id: str,
        customer_id: str,
        amount_cents: int,
        stripe_intent_id: str,
        description: str,
        created_at: float,
        status: str = "created",
    ) -> bool:
        """Record an intent so the confirmation webhook knows who to credit.

        `_handle_payment_succeeded` matches the Stripe intent id against this
        table and credits the wallet it finds. Without the row the customer is
        charged and the credit is silently dropped — the defect `601cb05`
        fixed, which existed because the only writer of this table was the
        dashboard path and auto-top-up called Stripe directly.

        `ON CONFLICT DO NOTHING` makes a repeated registration harmless, which
        matters because a retried charge returns Stripe's *original* intent for
        the same idempotency key.

        **Returns whether the row was new**, which is the only signal available
        that Stripe replayed a charge rather than making one. Stripe answers a
        repeated idempotency key with the original PaymentIntent and no
        indication that it is doing so, so a caller cannot otherwise tell "I
        charged the card" from "I was handed the receipt for an earlier
        charge". Discarding it is how a second top-up can look successful while
        no second charge exists.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            cur = conn.execute(
                """INSERT INTO payment_intents
                     (intent_id, customer_id, amount_cents, currency,
                      status, stripe_intent_id, description, created_at)
                   VALUES (%s, %s, %s, 'cad', %s, %s, %s, %s)
                   ON CONFLICT (stripe_intent_id)
                     WHERE stripe_intent_id IS NOT NULL
                       AND stripe_intent_id <> ''
                   DO NOTHING""",
                (
                    intent_id,
                    customer_id,
                    amount_cents,
                    status,
                    stripe_intent_id,
                    description,
                    created_at,
                ),
            )
            # 0 means the conflict fired: this Stripe intent was already
            # registered, so the charge that produced it happened earlier.
            inserted = cur.rowcount == 1
            conn.commit()
        return inserted

    def charge_saved_card(
        self,
        customer_id: str,
        amount_micros: int,
        *,
        stripe_customer_id: str,
        payment_method_id: str,
        idempotency_key: str,
        description: str,
        metadata: dict | None = None,
    ) -> dict:
        """Charge a card already on file, off-session, and register the intent.

        One implementation for both callers — the auto-top-up sweep and the
        manual top-up route. A second copy is how `601cb05` happened: the
        charge and the `payment_intents` registration have to stay together,
        and two copies means two places to forget the second half.

        **Order matters and is asserted.** The intent is registered immediately
        after Stripe accepts the charge and before this returns, so a
        confirmation webhook that arrives promptly finds the row rather than
        falling through and dropping the credit.

        Raises whatever Stripe raises. The callers' recovery differs — the
        sweep counts failures and eventually disables unattended top-up; a
        manual charge does neither — so the decision belongs to them, not here.
        """
        from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

        if not STRIPE_ENABLED or not _stripe_mod:
            raise RuntimeError("Stripe is not enabled on this deployment")

        # 10_000 micros == 1 cent. Integer division keeps the charged amount
        # exact; float * 100 did not.
        amount_cents = int(amount_micros) // 10_000
        if amount_cents <= 0:
            raise ValueError("amount must be greater than zero")

        # Off-session saved PM: do not hardcode payment_method_types.
        #
        # The idempotency key is the last line of defence against
        # double-charging: Stripe returns the *original* intent for a repeated
        # key, so a retry cannot produce a second charge.
        try:
            intent = _stripe_mod.PaymentIntent.create(
                amount=amount_cents,
                currency="cad",
                customer=stripe_customer_id,
                payment_method=payment_method_id,
                off_session=True,
                confirm=True,
                metadata={
                    "customer_id": customer_id,
                    "product_type": "wallet_deposit",
                    "xcelsior_sku": "xcelsior-compute-credits",
                    **(metadata or {}),
                },
                idempotency_key=idempotency_key,
            )
        except Exception as exc:
            # An `authentication_required` decline is not the end of the
            # charge — Stripe has *already created* the PaymentIntent and
            # attached it to the error. The cardholder can complete the
            # challenge in a browser, at which point Stripe fires
            # `payment_intent.succeeded` for that same intent id.
            #
            # `_handle_payment_succeeded` credits the wallet only if it can
            # match that id to a `payment_intents` row. Without this
            # registration the customer completes the challenge, is charged,
            # and the credit is silently dropped — `601cb05`'s defect arriving
            # by the SCA path instead of the auto-top-up one.
            #
            # Registered here rather than in either caller so the sweep gets it
            # too: §0.3 of the plan is about auto-top-up silently not working
            # for any card whose issuer demands SCA.
            pending = getattr(getattr(exc, "error", None), "payment_intent", None)
            pending_id = ""
            if pending is not None:
                pending_id = str(
                    (pending.get("id") if hasattr(pending, "get") else getattr(pending, "id", ""))
                    or ""
                )
            if pending_id:
                self._register_payment_intent(
                    intent_id=f"pi_topup_{uuid.uuid4().hex[:16]}",
                    customer_id=customer_id,
                    amount_cents=amount_cents,
                    stripe_intent_id=pending_id,
                    description=f"{description} (awaiting cardholder verification)",
                    created_at=time.time(),
                    # Not `created`. This intent is not in flight — it is
                    # stopped, waiting for the cardholder to satisfy their
                    # bank's challenge, and it will stay that way until they
                    # do. Recording it as `created` made an SCA decline
                    # indistinguishable from an ordinary new charge, so
                    # "which of my payments need me to act?" had no answer,
                    # and the wallet UI had nothing to show a pending state
                    # from. `check_low_balance_and_topup` already treats
                    # `requires_action` as in-flight for suppression, so the
                    # value is understood elsewhere in this file.
                    status="requires_action",
                )
                log.warning(
                    "Charge for %s requires cardholder verification; intent %s "
                    "registered so the credit lands if they complete it",
                    customer_id,
                    pending_id,
                )
            raise

        inserted = self._register_payment_intent(
            intent_id=f"pi_topup_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            amount_cents=amount_cents,
            stripe_intent_id=intent.id,
            description=description,
            created_at=time.time(),
        )

        return {
            "ok": True,
            "stripe_intent_id": intent.id,
            "amount_cents": amount_cents,
            "status": getattr(intent, "status", "") or "",
            # This intent was already on file, so Stripe answered a repeated
            # idempotency key with the original charge instead of making a new
            # one. Nothing failed and nothing was charged twice — but the
            # caller has not moved any additional money, and a response that
            # does not say so reads as if it has.
            "replayed": not inserted,
        }

    def check_low_balance_and_topup(self) -> dict:
        """Check all wallets for low balance and trigger auto-top-up if configured.

        Called periodically by the background scheduler.
        For wallets with auto_topup_enabled and balance below threshold,
        creates a Stripe PaymentIntent to charge the saved payment method.

        Retry schedule (Phase 1.4): 1min, 5min, 30min, then disable auto-topup.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        # Backoff schedule in seconds: attempt 1→60s, 2→300s, 3→1800s
        TOPUP_BACKOFF_SCHEDULE = [60, 300, 1800]
        inflight_cutoff = time.time() - _TOPUP_INFLIGHT_SECONDS

        topped_up = 0
        warnings = 0
        errors = 0

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            # A charge does not raise the balance — only Stripe's confirmation
            # does, minutes later. So a wallet that was just topped up still
            # satisfies every condition below, and the next sweep would charge
            # the card again. At a 300s cadence that is 288 charges a day
            # against one threshold breach. `NOT EXISTS` holds the wallet back
            # while an intent is unresolved; the cutoff releases it if an intent
            # is abandoned so a stuck row cannot disable top-up forever.
            wallets = conn.execute(
                """SELECT w.* FROM wallets w
                   WHERE w.status = 'active'
                     AND w.auto_topup_enabled = true
                     AND w.balance_micros <= w.auto_topup_threshold_micros
                     AND w.stripe_payment_method_id != ''
                     AND w.stripe_customer_id != ''
                     AND w.auto_topup_failures < 3
                     AND NOT EXISTS (
                           SELECT 1 FROM payment_intents p
                            WHERE p.customer_id = w.customer_id
                              AND p.status IN ('created', 'processing',
                                               'requires_action')
                              AND p.created_at > %s
                         )""",
                (inflight_cutoff,),
            ).fetchall()

        now = time.time()
        for w in wallets:
            customer_id = w["customer_id"]
            failures = w.get("auto_topup_failures", 0) or 0
            last_attempt = w.get("last_topup_attempt_at", 0) or 0

            # Exponential backoff: wait required interval before retrying
            if failures > 0 and failures <= len(TOPUP_BACKOFF_SCHEDULE):
                required_wait = TOPUP_BACKOFF_SCHEDULE[failures - 1]
                if now - last_attempt < required_wait:
                    continue

            try:
                from stripe_connect import STRIPE_ENABLED

                if not STRIPE_ENABLED:
                    continue

                amount_cents = int(w["auto_topup_amount_micros"]) // 10_000
                # Off-session saved PM: do not hardcode payment_method_types.
                #
                # The idempotency key is the last line of defence against
                # double-charging. The balance does not move until Stripe
                # confirms, so a wallet stays eligible for the whole interval
                # between the charge and the confirmation. In-flight suppression
                # in the query above is the primary guard; this makes a
                # duplicate impossible even if that guard is bypassed, because
                # Stripe returns the *original* intent for a repeated key.
                # Bucketed by sweep interval so a genuinely later top-up gets a
                # new key.
                # The shared charge, so a fix to the charge-and-register pair
                # lands on both this sweep and the manual top-up route. The
                # idempotency key is bucketed by sweep interval, so a genuinely
                # later top-up gets a new key while a retry within the interval
                # returns Stripe's original intent.
                charge = self.charge_saved_card(
                    customer_id,
                    int(w["auto_topup_amount_micros"]),
                    stripe_customer_id=w["stripe_customer_id"],
                    payment_method_id=w["stripe_payment_method_id"],
                    idempotency_key=(
                        f"autotopup:{customer_id}:{amount_cents}:"
                        f"{int(now // _TOPUP_INFLIGHT_SECONDS)}"
                    ),
                    description="Automatic wallet top-up",
                    metadata={"xcelsior_auto_topup": "true"},
                )
                log.info(
                    "Auto-topup PaymentIntent created for %s: %s ($%.2f)",
                    customer_id,
                    charge["stripe_intent_id"],
                    micros_to_cad(w["auto_topup_amount_micros"]),
                )
                topped_up += 1

                # The `payment_intents` row is written by `charge_saved_card`,
                # which is what marks the charge in-flight so the next sweep
                # leaves this wallet alone. Only the sweep's own bookkeeping
                # remains here.
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    conn.execute(
                        "UPDATE wallets SET last_topup_attempt_at = %s, auto_topup_failures = 0 WHERE customer_id = %s",
                        (now, customer_id),
                    )
                    conn.commit()

            except Exception as e:
                errors += 1
                new_failures = failures + 1
                log.error(
                    "Auto-topup failed for %s (attempt %d/3): %s", customer_id, new_failures, e
                )

                jobs_to_stop: list[str] = []
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    if new_failures >= 3:
                        # Max retries exhausted — disable auto-topup + pause instances
                        conn.execute(
                            """UPDATE wallets
                               SET auto_topup_failures = %s,
                                   auto_topup_enabled = false,
                                   last_topup_attempt_at = %s
                               WHERE customer_id = %s""",
                            (new_failures, now, customer_id),
                        )
                        log.warning(
                            "Auto-topup DISABLED for %s after 3 failures — stopping instances",
                            customer_id,
                        )
                        # Snapshot targets while updating the wallet, then
                        # invoke the shared lifecycle service after commit.
                        running = conn.execute(
                            """SELECT job_id
                               FROM jobs WHERE payload->>'owner' = %s AND status = 'running'""",
                            (customer_id,),
                        ).fetchall()
                        jobs_to_stop = [str(job["job_id"]) for job in running]
                    else:
                        conn.execute(
                            """UPDATE wallets
                               SET auto_topup_failures = %s,
                                   last_topup_attempt_at = %s
                               WHERE customer_id = %s""",
                            (new_failures, now, customer_id),
                        )
                    conn.commit()

                for job_id in jobs_to_stop:
                    stop_result = self.stop_instance(job_id, reason="low_balance")
                    if not stop_result.get("stopped"):
                        log.error(
                            "Auto-topup exhaustion failed to stop job=%s reason=%s",
                            job_id,
                            stop_result.get("reason"),
                        )

        if topped_up or errors:
            log.info("AUTO-TOPUP: %d topped up, %d errors", topped_up, errors)

        return {"topped_up": topped_up, "warnings": warnings, "errors": errors}

    # ── FINTRAC Compliance ────────────────────────────────────────────

    def fintrac_check_transaction(
        self, customer_id: str, amount_cad: float, currency: str = "CAD"
    ) -> Optional[dict]:
        """Check if a transaction triggers FINTRAC reporting requirements.

        Per REPORT_FEATURE_FINAL.md:
        - LVCTR (Large Value Cash Transaction Report): >= $10,000 CAD
        - STR (Suspicious Transaction Report): unusual patterns
        - 24-hour aggregate rule: multiple transactions totaling >= $10,000

        Returns report dict if threshold triggered, None otherwise.
        """
        LVCTR_THRESHOLD = 10_000.0
        report = None

        if amount_cad >= LVCTR_THRESHOLD:
            report = self._create_fintrac_report(
                customer_id=customer_id,
                report_type="LVCTR",
                trigger_amount=amount_cad,
                trigger_currency=currency,
            )

        # 24-hour aggregate check
        now = time.time()
        window_start = now - 86400
        with self._conn() as conn:
            row = conn.execute(
                """SELECT COALESCE(SUM(ABS(amount_micros / 1000000.0)), 0) as total_24h
                   FROM wallet_transactions
                   WHERE customer_id = %s AND created_at >= %s AND tx_type = 'deposit'""",
                (customer_id, window_start),
            ).fetchone()

        total_24h = float(row["total_24h"]) if row else 0.0
        if total_24h + amount_cad >= LVCTR_THRESHOLD and amount_cad < LVCTR_THRESHOLD:
            report = self._create_fintrac_report(
                customer_id=customer_id,
                report_type="LVCTR",
                trigger_amount=total_24h + amount_cad,
                trigger_currency=currency,
                notes=f"24-hour aggregate: {total_24h + amount_cad:.2f} CAD",
            )

        return report

    def _create_fintrac_report(
        self,
        customer_id: str,
        report_type: str,
        trigger_amount: float,
        trigger_currency: str = "CAD",
        notes: str = "",
    ) -> dict:
        now = time.time()
        report_id = f"FIN-{int(now)}-{os.urandom(3).hex()}"
        with self._conn() as conn:
            from psycopg.types.json import Jsonb

            conn.execute(
                """INSERT INTO fintrac_reports
                   (report_id, customer_id, report_type, trigger_amount_micros,
                    trigger_currency, aggregate_window_start, aggregate_window_end,
                    status, created_at, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)""",
                (
                    report_id,
                    customer_id,
                    report_type,
                    round(trigger_amount * 1_000_000),
                    trigger_currency,
                    now - 86400,
                    now,
                    now,
                    notes,
                ),
            )
        log.warning(
            "FINTRAC %s report created: %s customer=%s amount=$%.2f %s",
            report_type,
            report_id,
            customer_id,
            trigger_amount,
            trigger_currency,
        )
        return {
            "report_id": report_id,
            "report_type": report_type,
            "customer_id": customer_id,
            "trigger_amount_cad": trigger_amount,
            "status": "pending",
        }

    def charge_serverless_execution(
        self,
        worker: dict,
        endpoint: dict,
        *,
        period_end: float | None = None,
        final: bool = False,
    ) -> dict:
        """Bill an unbilled serverless worker uptime slice (Novita: $/s × running seconds)."""
        from serverless.metering import charge_serverless_execution as _charge
        from serverless.repo import ServerlessRepo

        return _charge(
            self,
            ServerlessRepo(),
            worker,
            endpoint,
            period_end=period_end,
            final=final,
        )

    def stop_jobs_for_suspended_wallets(self) -> int:
        """Find suspended wallets and stop their running jobs.

        Called by the billing cycle background task. When a wallet is
        suspended (grace period expired), all running jobs for that
        customer must be stopped.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        stopped = 0
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            suspended = conn.execute(
                "SELECT customer_id FROM wallets WHERE status = 'suspended'",
            ).fetchall()
            running_job_ids: list[tuple[str, str]] = []
            for w in suspended:
                cid = w["customer_id"]
                running = conn.execute(
                    """SELECT job_id
                       FROM jobs WHERE payload->>'owner' = %s AND status = 'running'""",
                    (cid,),
                ).fetchall()
                for job in running:
                    running_job_ids.append((str(job["job_id"]), str(cid)))

        # One lifecycle service owns both v1 and v2 stops. In particular,
        # attempt-owned jobs receive a durable fenced command instead of a
        # raw status write plus an unfenced legacy remove.
        for job_id, cid in running_job_ids:
            result = self.stop_instance(job_id, reason="billing_suspended")
            if result.get("stopped"):
                stopped += 1
                log.warning("Stopped job %s for suspended wallet %s", job_id, cid)
            else:
                log.warning(
                    "Suspended-wallet stop failed job=%s reason=%s",
                    job_id,
                    result.get("reason"),
                )

        # Deprovision serverless workers owned by suspended wallets
        try:
            from serverless.repo import ServerlessRepo
            from serverless.service import get_serverless_service

            sl_repo = ServerlessRepo()
            sl_svc = get_serverless_service()
            for w in suspended:
                cid = w["customer_id"]
                for ep in sl_repo.list_endpoints(cid):
                    for worker in sl_repo.list_workers(str(ep["endpoint_id"])):
                        state = str(worker.get("state") or "")
                        if state in ("booting", "ready", "idle", "draining"):
                            sl_svc.deprovision_worker(str(worker["worker_id"]), charge=False)
                            stopped += 1
        except Exception as e:
            log.warning("Serverless suspend enforcement failed: %s", e)

        if stopped:
            log.info("ENFORCEMENT: Stopped %d jobs for suspended wallets", stopped)
        return stopped

    # ── Time-to-Zero Depletion Projection ─────────────────────────────

    def time_to_zero(self, customer_id: str) -> dict:
        """Compute real-time balance depletion projection.

        Per Phase 1.3: `balance_cad / current_burn_rate_per_second`
        Returns seconds until zero, burn rate, and alert thresholds.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        wallet = self.get_wallet(customer_id)
        balance = wallet["balance_cad"]

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            running = conn.execute(
                """SELECT j.job_id, j.host_id, j.payload->>'gpu_model' AS gpu_model, COALESCE(j.payload->>'tier', 'free') AS tier
                   FROM jobs j
                   WHERE j.payload->>'owner' = %s AND j.status = 'running'""",
                (customer_id,),
            ).fetchall()

        burn_per_hour = 0.0
        instance_burns = []
        for job in running:
            host_id = job.get("host_id", "")
            with pool.connection() as conn:
                conn.row_factory = dict_row
                host = conn.execute(
                    "SELECT payload->>'cost_per_hour' AS cost_per_hour FROM hosts WHERE host_id = %s",
                    (host_id,),
                ).fetchone()
            rate = float(host["cost_per_hour"]) if host else 0.20
            burn_per_hour += rate
            instance_burns.append(
                {
                    "job_id": job["job_id"],
                    "gpu_model": job.get("gpu_model", ""),
                    "rate_per_hour": rate,
                }
            )

        burn_per_second = burn_per_hour / 3600 if burn_per_hour > 0 else 0
        seconds_to_zero = balance / burn_per_second if burn_per_second > 0 else float("inf")

        # Alert thresholds
        alert_30min = seconds_to_zero <= 1800
        alert_5min = seconds_to_zero <= 300
        alert_depleted = seconds_to_zero <= 0

        return {
            "customer_id": customer_id,
            "balance_cad": balance,
            "burn_rate_per_hour": round(burn_per_hour, 4),
            "burn_rate_per_second": round(burn_per_second, 6),
            "seconds_to_zero": (
                round(seconds_to_zero, 1) if seconds_to_zero != float("inf") else None
            ),
            "running_instances": len(running),
            "instance_burns": instance_burns,
            "alert_30min": alert_30min,
            "alert_5min": alert_5min,
            "alert_depleted": alert_depleted,
        }


# ── Singleton ─────────────────────────────────────────────────────────

_billing_engine: Optional[BillingEngine] = None


def get_billing_engine() -> BillingEngine:
    global _billing_engine
    if _billing_engine is None:
        _billing_engine = BillingEngine()
    return _billing_engine
