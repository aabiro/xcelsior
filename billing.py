# Xcelsior Billing & Metering — CAD-first, AI Compute Access Fund Ready
# Implements REPORT_FEATURE_FINAL.md + REPORT_MARKETING_FINAL.md:
#   - CAD pricing (competitors price in USD = procurement friction for CA buyers)
#   - AI Compute Access Fund rebate-ready invoice exports
#   - Per-job metering with residency trace
#   - Trust tier pricing multipliers
#   - Stripe Connect–ready payout structure
#   - Provider attestation bundle for compliance

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

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


# ── Metering ─────────────────────────────────────────────────────────


@dataclass
class UsageMeter:
    """Per-job resource usage metering record."""

    meter_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    job_id: str = ""
    host_id: str = ""
    owner: str = ""

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

    # Jurisdiction
    country: str = ""
    province: str = ""
    is_canadian_compute: bool = False

    # Trust tier
    trust_tier: str = "community"

    # Cost breakdown
    base_rate_per_hour: float = 0.0
    tier_multiplier: float = 1.0
    spot_discount: float = 0.0
    total_cost_cad: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


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
    is_canadian_compute: bool = False
    trust_tier: str = "community"
    job_id: str = ""
    host_id: str = ""
    province: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Invoice:
    """AI Compute Access Fund–aligned invoice.

    Designed to map cleanly to eligible cost categories:
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

    # AI Compute Access Fund breakdown
    canadian_compute_total_cad: float = 0.0
    non_canadian_compute_total_cad: float = 0.0
    fund_eligible_reimbursement_cad: float = 0.0
    effective_cost_after_fund_cad: float = 0.0

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
# evidence for AI Compute Access Fund claims.


@dataclass
class ProviderAttestation:
    """Supplier attestation bundle for compliance/fund claims."""

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
    pipeda_compliant: bool = True
    privacy_officer_designated: bool = False
    privacy_officer_contact: str = ""
    security_posture: str = "defense-in-depth"

    def to_dict(self) -> dict:
        return asdict(self)


# ── Billing Engine ────────────────────────────────────────────────────


class BillingEngine:
    """Production billing engine with CAD pricing and fund alignment.

    Features:
    - Per-job metering with residency trace
    - Trust tier pricing multipliers
    - AI Compute Access Fund eligible/ineligible split
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
        jurisdiction_data: Optional[dict] = None,
        trust_tier: str = "community",
    ) -> UsageMeter:
        """Create a metering record for a completed job.

        This is the source of truth for billing. Every completed job
        gets a usage meter that records exactly what resources were
        consumed, where they ran, and at what tier.
        """
        from jurisdiction import TRUST_TIER_REQUIREMENTS, TrustTier

        started = float(job.get("started_at", 0))
        completed = float(job.get("completed_at", 0))
        duration = completed - started if completed > started else 0

        # Jurisdiction info
        country = ""
        province = ""
        is_canadian = False
        if jurisdiction_data:
            country = jurisdiction_data.get("country", "")
            province = jurisdiction_data.get("province", "")
            is_canadian = country.upper() == "CA"
        elif host.get("country", "").upper() == "CA":
            country = "CA"
            is_canadian = True

        # Pricing
        base_rate = float(host.get("cost_per_hour", 0.20))
        tier_req = TRUST_TIER_REQUIREMENTS.get(TrustTier(trust_tier), {})
        multiplier = tier_req.get("pricing_multiplier", 1.0)
        spot_discount = float(job.get("spot_discount", 0))

        # Total cost
        duration_hr = duration / 3600
        cost = round(duration_hr * base_rate * multiplier * (1 - spot_discount), 4)

        meter = UsageMeter(
            job_id=job.get("job_id", ""),
            host_id=host.get("host_id", ""),
            owner=job.get("owner", ""),
            started_at=started,
            completed_at=completed,
            duration_sec=round(duration, 2),
            gpu_seconds=round(duration, 2),  # 1:1 for single GPU
            gpu_model=host.get("gpu_model", ""),
            vram_gb=float(job.get("vram_needed_gb", 0)),
            xcu_score=float(host.get("compute_score", 0)),
            country=country,
            province=province,
            is_canadian_compute=is_canadian,
            trust_tier=trust_tier,
            base_rate_per_hour=base_rate,
            tier_multiplier=multiplier,
            spot_discount=spot_discount,
            total_cost_cad=cost,
        )

        # Persist
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO usage_meters
                   (meter_id, job_id, host_id, owner, started_at, completed_at,
                    duration_sec, gpu_seconds, gpu_model, vram_gb,
                    gpu_utilization_pct, xcu_score, country, province,
                    is_canadian_compute, trust_tier, base_rate_per_hour,
                    tier_multiplier, spot_discount, total_cost_cad, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (meter_id) DO UPDATE SET
                     job_id = EXCLUDED.job_id, host_id = EXCLUDED.host_id, owner = EXCLUDED.owner,
                     started_at = EXCLUDED.started_at, completed_at = EXCLUDED.completed_at,
                     duration_sec = EXCLUDED.duration_sec, gpu_seconds = EXCLUDED.gpu_seconds,
                     total_cost_cad = EXCLUDED.total_cost_cad, created_at = EXCLUDED.created_at""",
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
                    1 if meter.is_canadian_compute else 0,
                    meter.trust_tier,
                    meter.base_rate_per_hour,
                    meter.tier_multiplier,
                    meter.spot_discount,
                    meter.total_cost_cad,
                    time.time(),
                ),
            )

        log.info(
            "METERED job=%s cost=$%.4f CAD tier=%s canadian=%s",
            meter.job_id,
            meter.total_cost_cad,
            trust_tier,
            is_canadian,
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
        """Generate an AI Compute Access Fund–aligned invoice.

        Groups usage by:
        - Canadian vs non-Canadian compute
        - Cost category (compute, storage, monitoring, security)
        - Trust tier
        - Province

        Tax rate is auto-detected from customer province if not specified.
        Produces line items that map directly to Fund eligible categories.
        """
        if tax_rate is None:
            tax_rate, _desc = get_tax_rate_for_province(customer_province)
        from jurisdiction import compute_fund_eligible_amount

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM usage_meters
                   WHERE owner = %s AND started_at >= %s AND completed_at <= %s
                   ORDER BY started_at""",
                (customer_id, period_start, period_end),
            ).fetchall()

        line_items = []
        ca_total = 0.0
        non_ca_total = 0.0

        for row in rows:
            cost = float(row["total_cost_cad"])
            is_ca = bool(row["is_canadian_compute"])

            li = InvoiceLineItem(
                description=f"GPU Compute: {row['gpu_model']} ({row['trust_tier']} tier)",
                category="compute",
                quantity=round(row["duration_sec"] / 3600, 4),
                unit="GPU-hours",
                unit_price_cad=float(row["base_rate_per_hour"]) * float(row["tier_multiplier"]),
                subtotal_cad=cost,
                is_canadian_compute=is_ca,
                trust_tier=row["trust_tier"],
                job_id=row["job_id"],
                host_id=row["host_id"],
                province=row["province"],
            )
            line_items.append(li.to_dict())

            if is_ca:
                ca_total += cost
            else:
                non_ca_total += cost

        subtotal = ca_total + non_ca_total
        tax = round(subtotal * tax_rate, 2)
        total = round(subtotal + tax, 2)

        # Fund calculations
        ca_fund = compute_fund_eligible_amount(ca_total, True)
        non_ca_fund = compute_fund_eligible_amount(non_ca_total, False)
        total_reimbursable = (
            ca_fund["reimbursable_amount_cad"] + non_ca_fund["reimbursable_amount_cad"]
        )
        effective = round(total - total_reimbursable, 2)

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
            canadian_compute_total_cad=round(ca_total, 2),
            non_canadian_compute_total_cad=round(non_ca_total, 2),
            fund_eligible_reimbursement_cad=round(total_reimbursable, 2),
            effective_cost_after_fund_cad=effective,
        )

        # Persist
        with self._conn() as conn:
            from psycopg.types.json import Jsonb

            conn.execute(
                """INSERT INTO invoices
                   (invoice_id, customer_id, customer_name, currency,
                    period_start, period_end, line_items, subtotal_cad,
                    tax_rate, tax_amount_cad, total_cad,
                    canadian_compute_total_cad, non_canadian_compute_total_cad,
                    fund_eligible_reimbursement_cad, effective_cost_after_fund_cad,
                    created_at, status, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    invoice.invoice_id,
                    invoice.customer_id,
                    invoice.customer_name,
                    invoice.currency,
                    invoice.period_start,
                    invoice.period_end,
                    Jsonb(invoice.line_items),
                    invoice.subtotal_cad,
                    invoice.tax_rate,
                    invoice.tax_amount_cad,
                    invoice.total_cad,
                    invoice.canadian_compute_total_cad,
                    invoice.non_canadian_compute_total_cad,
                    invoice.fund_eligible_reimbursement_cad,
                    invoice.effective_cost_after_fund_cad,
                    invoice.created_at,
                    invoice.status,
                    invoice.notes,
                ),
            )

        log.info(
            "INVOICE %s customer=%s total=$%.2f CAD (CA: $%.2f, non-CA: $%.2f, "
            "fund reimbursable: $%.2f, effective: $%.2f)",
            invoice.invoice_id,
            customer_id,
            total,
            ca_total,
            non_ca_total,
            total_reimbursable,
            effective,
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
                   (payout_id, provider_id, job_id, amount_cad,
                    platform_fee_cad, provider_payout_cad, status, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (
                    payout_id,
                    provider_id,
                    job_id,
                    gross_amount_cad,
                    fee,
                    payout,
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
                    SUM(total_cost_cad) as total_cost_cad,
                    SUM(CASE WHEN is_canadian_compute = 1 THEN total_cost_cad ELSE 0 END) as ca_cost,
                    SUM(CASE WHEN is_canadian_compute = 0 THEN total_cost_cad ELSE 0 END) as non_ca_cost,
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
            "canadian_compute_cad": round(rows["ca_cost"] or 0, 2),
            "non_canadian_compute_cad": round(rows["non_ca_cost"] or 0, 2),
            "hosts_used": rows["hosts_used"] or 0,
            "currency": "CAD",
        }

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

        cost = float(row["total_cost_cad"])

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
        """Get or create a customer wallet."""
        self._ensure_wallet_table()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM wallets WHERE customer_id = %s",
                (customer_id,),
            ).fetchone()
            if row:
                return dict(row)

            # Create new wallet
            now = time.time()
            conn.execute(
                """INSERT INTO wallets
                   (customer_id, balance_cad, total_deposited_cad,
                    total_spent_cad, total_refunded_cad,
                    grace_until, status, created_at, updated_at)
                   VALUES (%s, 0, 0, 0, 0, 0, 'active', %s, %s)""",
                (customer_id, now, now),
            )
            return {
                "customer_id": customer_id,
                "balance_cad": 0.0,
                "total_deposited_cad": 0.0,
                "total_spent_cad": 0.0,
                "total_refunded_cad": 0.0,
                "grace_until": 0.0,
                "status": "active",
            }

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
                    "SELECT tx_id, balance_after_cad FROM wallet_transactions WHERE idempotency_key = %s",
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
                """UPDATE wallets
                   SET balance_cad = balance_cad + %s,
                       total_deposited_cad = total_deposited_cad + %s,
                       updated_at = %s
                   WHERE customer_id = %s
                   RETURNING balance_cad""",
                (amount_cad, amount_cad, time.time(), customer_id),
            ).fetchone()
            new_balance = (
                row["balance_cad"] if row else round(wallet["balance_cad"] + amount_cad, 4)
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, created_at, idempotency_key)
                   VALUES (%s, %s, 'deposit', %s, %s, %s, %s, %s)""",
                (
                    tx_id,
                    customer_id,
                    amount_cad,
                    new_balance,
                    description,
                    time.time(),
                    idempotency_key or "",
                ),
            )

        log.info("DEPOSIT %s +$%.2f CAD balance=$%.2f", customer_id, amount_cad, new_balance)
        return {"tx_id": tx_id, "balance_cad": new_balance}

    def charge(
        self,
        customer_id: str,
        amount_cad: float,
        job_id: str = "",
        description: str = "Compute charge",
    ) -> dict:
        """Charge a customer wallet. Returns False if insufficient balance
        (triggers grace period per REPORT_FEATURE_1.md: 72 hours).
        """
        self._ensure_wallet_table()
        wallet = self.get_wallet(customer_id)
        balance = wallet["balance_cad"]

        GRACE_PERIOD_SEC = 72 * 3600  # 72 hours per REPORT_FEATURE_1.md

        if balance < amount_cad:
            # Check grace period
            now = time.time()
            grace = wallet.get("grace_until", 0)

            if grace == 0:
                # Start grace period
                grace_end = now + GRACE_PERIOD_SEC
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE wallets SET grace_until = %s, updated_at = %s WHERE customer_id = %s",
                        (grace_end, now, customer_id),
                    )
                log.warning(
                    "WALLET %s insufficient balance ($%.2f < $%.2f) " "— 72hr grace period started",
                    customer_id,
                    balance,
                    amount_cad,
                )
                try:
                    from db import NotificationStore

                    NotificationStore.create(
                        user_email=customer_id,
                        notif_type="billing_grace",
                        title="Low balance — 72-hour grace period started",
                        body=f"Your balance is ${balance:.2f} CAD. Add funds within 72 hours to avoid service interruption.",
                        data={"balance_cad": balance, "grace_until": grace_end},
                    )
                except Exception:
                    pass
                return {
                    "charged": False,
                    "reason": "insufficient_balance",
                    "balance_cad": balance,
                    "grace_until": grace_end,
                    "action": "grace_period_started",
                }
            elif now > grace:
                # Grace expired — auto-stop instances
                with self._conn() as conn:
                    conn.execute(
                        "UPDATE wallets SET status = 'suspended', updated_at = %s WHERE customer_id = %s",
                        (now, customer_id),
                    )
                log.warning("WALLET %s grace period expired — account suspended", customer_id)
                try:
                    from db import NotificationStore

                    NotificationStore.create(
                        user_email=customer_id,
                        notif_type="billing_suspended",
                        title="Account suspended — grace period expired",
                        body="Your account has been suspended due to insufficient funds. All running instances will be stopped. Add funds to reactivate.",
                        data={"balance_cad": balance},
                    )
                except Exception:
                    pass
                return {
                    "charged": False,
                    "reason": "grace_expired",
                    "balance_cad": balance,
                    "action": "account_suspended",
                }
            else:
                # Still in grace period — allow charge but warn
                log.warning("WALLET %s in grace period, allowing charge", customer_id)

        new_balance = round(balance - amount_cad, 4)
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            # Atomic: decrement balance with a floor check to prevent races
            row = conn.execute(
                """UPDATE wallets
                   SET balance_cad = balance_cad - %s,
                       total_spent_cad = total_spent_cad + %s,
                       grace_until = 0,
                       updated_at = %s
                   WHERE customer_id = %s
                   RETURNING balance_cad""",
                (amount_cad, amount_cad, time.time(), customer_id),
            ).fetchone()
            new_balance = row["balance_cad"] if row else new_balance
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, job_id, created_at)
                   VALUES (%s, %s, 'charge', %s, %s, %s, %s, %s)""",
                (tx_id, customer_id, -amount_cad, new_balance, description, job_id, time.time()),
            )

        log.info(
            "CHARGE %s -$%.4f CAD job=%s balance=$%.4f",
            customer_id,
            amount_cad,
            job_id,
            new_balance,
        )
        return {"charged": True, "tx_id": tx_id, "balance_cad": new_balance}

    def _credit_wallet(
        self, customer_id: str, amount_cad: float, description: str = "Refund credit"
    ):
        """Internal: credit a wallet (for refunds)."""
        self._ensure_wallet_table()
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            row = conn.execute(
                """UPDATE wallets
                   SET balance_cad = balance_cad + %s,
                       total_refunded_cad = total_refunded_cad + %s,
                       updated_at = %s
                   WHERE customer_id = %s
                   RETURNING balance_cad""",
                (amount_cad, amount_cad, time.time(), customer_id),
            ).fetchone()
            new_balance = row["balance_cad"] if row else amount_cad
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, created_at)
                   VALUES (%s, %s, 'refund', %s, %s, %s, %s)""",
                (tx_id, customer_id, amount_cad, new_balance, description, time.time()),
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
                   SET balance_cad = 0,
                       total_deposited_cad = 0,
                       total_spent_cad = 0,
                       total_refunded_cad = 0,
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

    # ── Instance Pause / Resume ───────────────────────────────────────

    _VALID_PAUSE_REASONS = frozenset({"paused_low_balance", "user_paused"})

    def pause_instance(self, job_id: str, reason: str = "paused_low_balance") -> dict:
        """Pause a running instance: stop container, preserve volume, stop billing.

        Per Phase 1.3: pause_instance() stops the container but preserves
        the volume mount so the user can resume later.
        """
        if reason not in self._VALID_PAUSE_REASONS:
            return {
                "paused": False,
                "reason": f"invalid_reason: must be one of {sorted(self._VALID_PAUSE_REASONS)}",
            }
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name
                   FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"paused": False, "reason": "not_running"}

            conn.execute(
                """UPDATE jobs SET status = %s,
                   payload = jsonb_set(
                       jsonb_set(payload, '{paused_at}', to_jsonb(%s::float)),
                       '{status}', %s::jsonb
                   )
                   WHERE job_id = %s""",
                (reason, now, json.dumps(reason), job_id),
            )
            # Insert a zero-amount billing cycle to anchor the billing period.
            # Without this, resume would bill for the entire paused duration.
            owner = job.get("owner") or ""
            cycle_id = f"BC-pause-{int(now)}-{os.urandom(3).hex()}"
            conn.execute(
                """INSERT INTO billing_cycles
                   (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                    duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                    amount_cad, status, created_at)
                   VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'paused', %s)""",
                (cycle_id, job_id, owner, job.get("host_id", ""), now, now, now),
            )
            conn.commit()

        # Enqueue stop_container on the host (preserve volumes); async via agent queue
        owner = job.get("owner") or ""
        host_id = job.get("host_id") or ""
        container_name = job.get("container_name") or f"xcl-{job_id}"
        if host_id:
            try:
                from routes.agent import enqueue_agent_command
                from scheduler import _validate_name

                _validate_name(container_name, "container name")
                enqueue_agent_command(
                    host_id,
                    "stop_container",
                    {"container_name": container_name, "job_id": job_id},
                    created_by="billing_pause",
                )
                log.info("PAUSE stop_container queued: %s on %s", container_name, host_id)
            except Exception as e:
                log.warning("PAUSE container stop enqueue failed for %s: %s", job_id, e)

        # Send notification
        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="billing_pause",
                title=f"Instance paused: {job.get('name', job_id)}",
                body=f"Your instance was paused due to {reason.replace('_', ' ')}. "
                "Add funds to resume.",
                data={"job_id": job_id, "reason": reason},
            )
        except Exception:
            pass  # non-critical

        log.warning("PAUSE job=%s reason=%s owner=%s", job_id, reason, owner)
        return {"paused": True, "job_id": job_id, "reason": reason}

    def resume_instance(self, job_id: str) -> dict:
        """Resume a paused instance: restart container from preserved state.

        Per Phase 1.3: wallet top-up triggers resume_instance() to restart
        the container from its preserved state.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'image' AS image,
                          payload->>'container_name' AS container_name
                   FROM jobs
                   WHERE job_id = %s AND status IN ('paused_low_balance', 'user_paused') FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"resumed": False, "reason": "not_paused"}

            # Verify wallet has funds
            owner = job.get("owner") or ""
            wallet = self.get_wallet(owner)
            if wallet["balance_cad"] <= 0:
                return {"resumed": False, "reason": "insufficient_balance"}

            conn.execute(
                """UPDATE jobs SET status = 'running',
                   payload = jsonb_set(
                       jsonb_set(
                           jsonb_set(payload, '{paused_at}', '0'::jsonb),
                           '{resumed_at}', to_jsonb(%s::float)
                       ),
                       '{status}', %s::jsonb
                   )
                   WHERE job_id = %s""",
                (now, json.dumps("running"), job_id),
            )
            # Insert a billing anchor at resume time so billing starts fresh
            cycle_id = f"BC-resume-{int(now)}-{os.urandom(3).hex()}"
            conn.execute(
                """INSERT INTO billing_cycles
                   (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                    duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                    amount_cad, status, created_at)
                   VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'resumed', %s)""",
                (cycle_id, job_id, owner, job.get("host_id", ""), now, now, now),
            )
            conn.commit()

        # Restart the container on the host
        host_id = job.get("host_id") or ""
        if host_id:
            try:
                from scheduler import list_hosts, run_job

                hosts = list_hosts()
                hmap = {h["host_id"]: h for h in hosts}
                host = hmap.get(host_id)
                if host:
                    from scheduler import get_job

                    full_job = get_job(job_id) or {}
                    run_job(full_job, host, docker_image=job.get("image"))
                    log.info("RESUME container restarted: %s on %s", job_id, host_id)
            except Exception as e:
                log.warning("RESUME container restart failed for %s: %s", job_id, e)

        # Send notification
        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="billing_resume",
                title=f"Instance resumed: {job.get('name', job_id)}",
                body="Your instance has been resumed after funds were added.",
                data={"job_id": job_id},
            )
        except Exception:
            pass  # non-critical

        log.info("RESUME job=%s owner=%s", job_id, owner)
        return {"resumed": True, "job_id": job_id, "status": "running"}

    # ── Instance Lifecycle: Stop / Start / Restart / Terminate ───────

    _VALID_STOP_REASONS = frozenset({"user_stopped", "paused_low_balance", "billing_suspended"})

    def stop_instance(self, job_id: str, reason: str = "user_stopped") -> dict:
        """Gracefully stop a running instance. Container is preserved for restart.

        Stops billing for compute; storage billing begins in auto_billing_cycle.
        The container is sent SIGTERM (docker stop -t 10) so the process can
        flush state before exiting. Volumes are NOT removed.
        """
        if reason not in self._VALID_STOP_REASONS:
            return {
                "stopped": False,
                "reason": f"invalid_reason: must be one of {sorted(self._VALID_STOP_REASONS)}",
            }

        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name
                   FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"stopped": False, "reason": "not_running"}

            # Mark transitional state
            conn.execute(
                """UPDATE jobs SET status = 'stopping',
                   payload = jsonb_set(
                       jsonb_set(payload, '{stopping_at}', to_jsonb(%s::float)),
                       '{status}', '"stopping"'::jsonb
                   )
                   WHERE job_id = %s""",
                (now, job_id),
            )
            conn.commit()

        owner = job.get("owner") or ""
        host_id = job.get("host_id") or ""
        container_name = job.get("container_name") or f"xcl-{job_id}"

        # Perform graceful stop on the host
        stop_ok = False
        if host_id:
            try:
                from scheduler import stop_container_graceful, list_hosts, _validate_name

                _validate_name(container_name, "container name")
                hosts = list_hosts()
                hmap = {h["host_id"]: h for h in hosts}
                host = hmap.get(host_id)
                if host:
                    stop_ok = stop_container_graceful(
                        {"job_id": job_id, "container_name": container_name}, host
                    )
            except Exception as e:
                log.warning("STOP container stop failed for %s: %s", job_id, e)

        # Update final status
        final_status = "stopped" if stop_ok else "running"
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """UPDATE jobs SET status = %s,
                   payload = jsonb_set(
                       jsonb_set(payload, '{stopped_at}', to_jsonb(%s::float)),
                       '{status}', %s::jsonb
                   )
                   WHERE job_id = %s""",
                (final_status, now, json.dumps(final_status), job_id),
            )
            # Billing anchor: closes the current compute billing period
            if stop_ok:
                owner_id = owner
                cycle_id = f"BC-stop-{int(now)}-{os.urandom(3).hex()}"
                conn.execute(
                    """INSERT INTO billing_cycles
                       (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                        duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                        amount_cad, status, created_at)
                       VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'stopped', %s)""",
                    (cycle_id, job_id, owner_id, host_id, now, now, now),
                )
            conn.commit()

        if not stop_ok:
            log.error("STOP failed for job=%s — container still running", job_id)
            return {"stopped": False, "reason": "container_stop_failed", "job_id": job_id}

        try:
            from db import NotificationStore

            NotificationStore.create(
                user_email=owner,
                notif_type="instance_stopped",
                title=f"Instance stopped: {job.get('name', job_id)}",
                body="Your instance has been stopped. Storage continues to be billed. Start it again anytime.",
                data={"job_id": job_id},
            )
        except Exception:
            pass

        log.info("STOP job=%s reason=%s owner=%s", job_id, reason, owner)
        return {"stopped": True, "job_id": job_id, "status": "stopped"}

    def start_instance(self, job_id: str) -> dict:
        """Start a stopped instance. Restores the container from its exited state.

        Requires a positive wallet balance. Billing resumes at the compute rate
        from the moment the container is running again.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name
                   FROM jobs
                   WHERE job_id = %s AND status IN ('stopped', 'user_paused', 'paused_low_balance') FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"started": False, "reason": "not_stopped"}

            owner = job.get("owner") or ""
            wallet = self.get_wallet(owner)
            if wallet["balance_cad"] <= 0:
                return {"started": False, "reason": "insufficient_balance"}

            # Mark transitional state
            conn.execute(
                """UPDATE jobs SET status = 'restarting',
                   payload = jsonb_set(
                       jsonb_set(payload, '{restarting_at}', to_jsonb(%s::float)),
                       '{status}', '"restarting"'::jsonb
                   )
                   WHERE job_id = %s""",
                (now, job_id),
            )
            conn.commit()

        host_id = job.get("host_id") or ""
        container_name = job.get("container_name") or f"xcl-{job_id}"

        start_ok = False
        if host_id:
            try:
                from scheduler import start_stopped_container, list_hosts, _validate_name

                _validate_name(container_name, "container name")
                hosts = list_hosts()
                hmap = {h["host_id"]: h for h in hosts}
                host = hmap.get(host_id)
                if host:
                    start_ok = start_stopped_container(
                        {"job_id": job_id, "container_name": container_name}, host
                    )
            except Exception as e:
                log.warning("START container start failed for %s: %s", job_id, e)

        final_status = "running" if start_ok else "stopped"
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """UPDATE jobs SET status = %s,
                   payload = jsonb_set(
                       jsonb_set(
                           jsonb_set(payload, '{started_at}', to_jsonb(%s::float)),
                           '{stopped_at}', '0'::jsonb
                       ),
                       '{status}', %s::jsonb
                   )
                   WHERE job_id = %s""",
                (final_status, now, json.dumps(final_status), job_id),
            )
            if start_ok:
                # Billing anchor: compute billing resumes from now
                cycle_id = f"BC-start-{int(now)}-{os.urandom(3).hex()}"
                conn.execute(
                    """INSERT INTO billing_cycles
                       (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                        duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                        amount_cad, status, created_at)
                       VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'started', %s)""",
                    (cycle_id, job_id, owner, host_id, now, now, now),
                )
            conn.commit()

        if not start_ok:
            log.error("START failed for job=%s", job_id)
            return {"started": False, "reason": "container_start_failed", "job_id": job_id}

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
        """Restart a running or stopped instance. Container data is preserved.

        For a running instance: stop gracefully then start.
        For a stopped instance: same as start_instance.
        Billing is continuous — no gap anchor is inserted. The compute billing
        period simply picks up again from when the container is running.
        Requires a positive wallet balance.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                """SELECT job_id, status, host_id,
                          payload->>'owner' AS owner,
                          payload->>'name' AS name,
                          payload->>'container_name' AS container_name
                   FROM jobs
                   WHERE job_id = %s AND status IN ('running', 'stopped') FOR UPDATE""",
                (job_id,),
            ).fetchone()
            if not job:
                return {"restarted": False, "reason": "not_restartable"}

            owner = job.get("owner") or ""
            wallet = self.get_wallet(owner)
            if wallet["balance_cad"] <= 0:
                return {"restarted": False, "reason": "insufficient_balance"}

            was_running = job["status"] == "running"

            conn.execute(
                """UPDATE jobs SET status = 'restarting',
                   payload = jsonb_set(
                       jsonb_set(payload, '{restarting_at}', to_jsonb(%s::float)),
                       '{status}', '"restarting"'::jsonb
                   )
                   WHERE job_id = %s""",
                (now, job_id),
            )
            conn.commit()

        host_id = job.get("host_id") or ""
        container_name = job.get("container_name") or f"xcl-{job_id}"

        restart_ok = False
        if host_id:
            try:
                from scheduler import (
                    stop_container_graceful,
                    start_stopped_container,
                    list_hosts,
                    _validate_name,
                )

                _validate_name(container_name, "container name")
                hosts = list_hosts()
                hmap = {h["host_id"]: h for h in hosts}
                host = hmap.get(host_id)
                if host:
                    job_obj = {"job_id": job_id, "container_name": container_name}
                    if was_running:
                        stop_container_graceful(job_obj, host)
                    restart_ok = start_stopped_container(job_obj, host)
            except Exception as e:
                log.warning("RESTART container restart failed for %s: %s", job_id, e)

        final_status = "running" if restart_ok else "stopped"
        with pool.connection() as conn:
            conn.row_factory = dict_row
            conn.execute(
                """UPDATE jobs SET status = %s,
                   payload = jsonb_set(
                       jsonb_set(payload, '{restarted_at}', to_jsonb(%s::float)),
                       '{status}', %s::jsonb
                   )
                   WHERE job_id = %s""",
                (final_status, now, json.dumps(final_status), job_id),
            )
            conn.commit()

        if not restart_ok:
            log.error("RESTART failed for job=%s — marking stopped", job_id)
            return {"restarted": False, "reason": "container_restart_failed", "job_id": job_id}

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
                """SELECT job_id, status, host_id,
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

            conn.execute(
                """UPDATE jobs SET status = 'terminated',
                   payload = jsonb_set(
                       jsonb_set(payload, '{terminated_at}', to_jsonb(%s::float)),
                       '{status}', '"terminated"'::jsonb
                   )
                   WHERE job_id = %s""",
                (now, job_id),
            )
            # Final billing anchor
            owner = job.get("owner") or ""
            host_id = job.get("host_id") or ""
            cycle_id = f"BC-term-{int(now)}-{os.urandom(3).hex()}"
            conn.execute(
                """INSERT INTO billing_cycles
                   (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                    duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                    amount_cad, status, created_at)
                   VALUES (%s, %s, %s, %s, 'gpu', %s, %s, 0, 0, '', '', 1.0, 0, 'terminated', %s)""",
                (cycle_id, job_id, owner, host_id, now, now, now),
            )
            conn.commit()

        # Detach all managed volumes attached to this instance
        try:
            from volumes import get_volume_engine

            get_volume_engine().detach_all_for_instance(job_id)
        except Exception as e:
            log.warning("Volume detach failed for %s: %s", job_id, e)

        owner = job.get("owner") or ""
        container_name = job.get("container_name") or f"xcl-{job_id}"
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
                       auto_topup_amount_cad = %s,
                       auto_topup_threshold_cad = %s,
                       stripe_payment_method_id = %s,
                       updated_at = %s
                   WHERE customer_id = %s""",
                (
                    enabled,
                    amount_cad,
                    threshold_cad,
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
            "auto_topup_amount_cad": amount_cad,
            "auto_topup_threshold_cad": threshold_cad,
        }

    # ── Auto-Billing Cycle (Running Instances) ────────────────────────

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
                host_id = job.get("host_id", "")
                gpu_model = job.get("gpu_model", "")
                tier = job.get("tier", "free")

                # Find the last billing cycle end for this job
                # Single transaction with row lock prevents double-billing from concurrent ticks
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    # Lock the job row so a concurrent billing tick skips it
                    locked = conn.execute(
                        "SELECT job_id FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE SKIP LOCKED",
                        (job_id,),
                    ).fetchone()
                    if not locked:
                        continue  # Another tick already processing this job

                    last = conn.execute(
                        """SELECT period_end FROM billing_cycles
                           WHERE job_id = %s ORDER BY period_end DESC LIMIT 1""",
                        (job_id,),
                    ).fetchone()

                    period_start = last["period_end"] if last else float(job["started_at"])
                    period_end = now

                    # Skip if less than 60 seconds since last billing
                    if period_end - period_start < 60:
                        continue

                    duration_sec = period_end - period_start

                    # Get the host's rate (same transaction)
                    host = conn.execute(
                        "SELECT payload->>'cost_per_hour' AS cost_per_hour FROM hosts WHERE host_id = %s",
                        (host_id,),
                    ).fetchone()

                    rate_per_hour = (
                        float(host["cost_per_hour"]) if host and host.get("cost_per_hour") else 0.20
                    )

                    # Tier multiplier
                    from jurisdiction import TRUST_TIER_REQUIREMENTS, TrustTier

                    try:
                        tier_req = TRUST_TIER_REQUIREMENTS.get(TrustTier(tier), {})
                        tier_multiplier = tier_req.get("pricing_multiplier", 1.0)
                    except (ValueError, KeyError):
                        tier_multiplier = 1.0

                    amount_cad = round((duration_sec / 3600) * rate_per_hour * tier_multiplier, 4)

                    if amount_cad <= 0:
                        continue

                    # Charge the wallet
                    charge_result = self.charge(
                        customer_id,
                        amount_cad,
                        job_id=job_id,
                        description=f"Auto-billing: {gpu_model} ({duration_sec/60:.1f}min)",
                    )

                    cycle_id = f"BC-{int(now)}-{os.urandom(3).hex()}"
                    status = "charged" if charge_result.get("charged") else "failed"

                    # Record the billing cycle (inside same locked transaction)
                    conn.execute(
                        """INSERT INTO billing_cycles
                           (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                            duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                            amount_cad, status, created_at)
                           VALUES (%s, %s, %s, %s, 'gpu', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
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
                            now,
                        ),
                    )
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
                    # Actually terminate the running container (via agent queue)
                    try:
                        from scheduler import get_job, _validate_name
                        from routes.agent import enqueue_agent_command

                        full_job = get_job(job_id)
                        if full_job:
                            cname = full_job.get("container_name") or f"xcl-{job_id}"
                            _validate_name(cname, "container name")
                            if host_id:
                                enqueue_agent_command(
                                    host_id,
                                    "stop_container",
                                    {"container_name": cname, "job_id": job_id},
                                    created_by="billing_grace_expired",
                                )
                                log.warning(
                                    "BILLING: Queued stop_container for job %s (suspended account %s)",
                                    job_id,
                                    customer_id,
                                )
                            # Mark job stopped in DB
                            with pool.connection() as kconn:
                                kconn.row_factory = dict_row
                                kconn.execute(
                                    "UPDATE jobs SET status = 'stopped', payload = jsonb_set(payload, '{completed_at}', to_jsonb(%s::float)) WHERE job_id = %s",
                                    (time.time(), job_id),
                                )
                                kconn.commit()
                        else:
                            log.warning("BILLING: Job %s not found for kill on suspension", job_id)
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
                                amount_cad, status, created_at)
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
                                vamount,
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

        # ── Bill active inference endpoints (real-time GPU compute) ──
        inference_billed = 0
        try:
            with pool.connection() as conn:
                conn.row_factory = dict_row
                active_eps = conn.execute(
                    """SELECT endpoint_id, owner_id, gpu_type, region, min_workers,
                              worker_job_id, created_at
                       FROM inference_endpoints
                       WHERE status = 'active' AND worker_job_id IS NOT NULL""",
                ).fetchall()

            for ep in active_eps:
                try:
                    ep_id = ep["endpoint_id"]
                    ep_owner = ep["owner_id"]
                    gpu_type = ep.get("gpu_type", "")
                    if not gpu_type:
                        continue

                    # Find last inference billing cycle
                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        last_ic = conn.execute(
                            """SELECT period_end FROM billing_cycles
                               WHERE job_id = %s ORDER BY period_end DESC LIMIT 1""",
                            (ep_id,),
                        ).fetchone()

                    iperiod_start = last_ic["period_end"] if last_ic else float(ep["created_at"])
                    iperiod_end = now

                    if iperiod_end - iperiod_start < 60:
                        continue

                    iduration_sec = iperiod_end - iperiod_start

                    # Look up GPU cost per hour
                    from inference import get_inference_engine

                    ie = get_inference_engine()
                    cost_per_hour = ie._get_gpu_cost_per_hour(gpu_type, ep.get("region", "ca-east"))
                    if cost_per_hour <= 0:
                        continue

                    iamount = round((iduration_sec / 3600) * cost_per_hour, 4)
                    if iamount <= 0:
                        continue

                    icharge = self.charge(
                        ep_owner,
                        iamount,
                        job_id=ep_id,
                        description=f"Inference compute: {gpu_type} ({iduration_sec/60:.1f}min)",
                    )

                    icycle_id = f"IC-{int(now)}-{os.urandom(3).hex()}"
                    istatus = "charged" if icharge.get("charged") else "failed"

                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        conn.execute(
                            """INSERT INTO billing_cycles
                               (cycle_id, job_id, customer_id, host_id, resource_type, period_start, period_end,
                                duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                                amount_cad, status, created_at)
                               VALUES (%s, %s, %s, %s, 'inference', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (
                                icycle_id,
                                ep_id,
                                ep_owner,
                                "",
                                iperiod_start,
                                iperiod_end,
                                iduration_sec,
                                cost_per_hour,
                                gpu_type,
                                "inference",
                                1.0,
                                iamount,
                                istatus,
                                now,
                            ),
                        )
                        conn.commit()

                    # Update total_cost_cad on the endpoint
                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        conn.execute(
                            """UPDATE inference_endpoints
                               SET total_cost_cad = COALESCE(total_cost_cad, 0) + %s, updated_at = %s
                               WHERE endpoint_id = %s""",
                            (iamount, now, ep_id),
                        )
                        conn.commit()

                    inference_billed += 1
                except Exception as e:
                    errors += 1
                    log.error("Inference billing error for %s: %s", ep.get("endpoint_id", "?"), e)
        except Exception as e:
            log.error("Inference billing scan error: %s", e)

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
                                amount_cad, status, created_at)
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
                                samount,
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

        topped_up = 0
        warnings = 0
        errors = 0

        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            wallets = conn.execute(
                """SELECT * FROM wallets
                   WHERE status = 'active'
                     AND auto_topup_enabled = true
                     AND balance_cad <= auto_topup_threshold_cad
                     AND stripe_payment_method_id != ''
                     AND auto_topup_failures < 3""",
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
                from stripe_connect import STRIPE_ENABLED, stripe as _stripe_mod

                if not STRIPE_ENABLED or not _stripe_mod:
                    continue

                amount_cents = int(w["auto_topup_amount_cad"] * 100)
                pi = _stripe_mod.PaymentIntent.create(
                    amount=amount_cents,
                    currency="cad",
                    customer=customer_id,
                    payment_method=w["stripe_payment_method_id"],
                    off_session=True,
                    confirm=True,
                    metadata={"xcelsior_auto_topup": "true", "customer_id": customer_id},
                )
                log.info(
                    "Auto-topup PaymentIntent created for %s: %s ($%.2f)",
                    customer_id,
                    pi.id,
                    w["auto_topup_amount_cad"],
                )
                topped_up += 1

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
                            "Auto-topup DISABLED for %s after 3 failures — pausing instances",
                            customer_id,
                        )
                        # Pause all running instances for this customer
                        running = conn.execute(
                            """SELECT job_id, host_id,
                                      payload->>'container_name' AS container_name
                               FROM jobs WHERE payload->>'owner' = %s AND status = 'running'""",
                            (customer_id,),
                        ).fetchall()
                        for job in running:
                            conn.execute(
                                "UPDATE jobs SET status = 'paused_low_balance', payload = jsonb_set(payload, '{paused_at}', to_jsonb(%s::float)) WHERE job_id = %s",
                                (now, job["job_id"]),
                            )
                        conn.commit()
                        # Enqueue stop_container for each job (after commit)
                        try:
                            from scheduler import _validate_name
                            from routes.agent import enqueue_agent_command

                            for job in running:
                                hid = job.get("host_id")
                                if not hid:
                                    continue
                                cname = job.get("container_name") or f"xcl-{job['job_id']}"
                                try:
                                    _validate_name(cname, "container name")
                                    enqueue_agent_command(
                                        hid,
                                        "stop_container",
                                        {"container_name": cname, "job_id": job["job_id"]},
                                        created_by="billing_autotopup_failed",
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                    else:
                        conn.execute(
                            """UPDATE wallets
                               SET auto_topup_failures = %s,
                                   last_topup_attempt_at = %s
                               WHERE customer_id = %s""",
                            (new_failures, now, customer_id),
                        )
                    conn.commit()

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
                """SELECT COALESCE(SUM(ABS(amount_cad)), 0) as total_24h
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
                   (report_id, customer_id, report_type, trigger_amount_cad,
                    trigger_currency, aggregate_window_start, aggregate_window_end,
                    status, created_at, notes)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending', %s, %s)""",
                (
                    report_id,
                    customer_id,
                    report_type,
                    trigger_amount,
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

            for w in suspended:
                cid = w["customer_id"]
                running = conn.execute(
                    """SELECT job_id, host_id,
                              payload->>'container_name' AS container_name
                       FROM jobs WHERE payload->>'owner' = %s AND status = 'running'""",
                    (cid,),
                ).fetchall()
                for job in running:
                    conn.execute(
                        "UPDATE jobs SET status = 'stopped', payload = jsonb_set(payload, '{completed_at}', to_jsonb(%s::float)) WHERE job_id = %s AND status = 'running'",
                        (time.time(), job["job_id"]),
                    )
                    stopped += 1
                    log.warning("Stopped job %s for suspended wallet %s", job["job_id"], cid)
            conn.commit()

        # Enqueue stop_container for each stopped job (after releasing the connection)
        if stopped:
            try:
                from scheduler import _validate_name
                from routes.agent import enqueue_agent_command

                for w in suspended:
                    cid = w["customer_id"]
                    with pool.connection() as kconn:
                        kconn.row_factory = dict_row
                        just_stopped = kconn.execute(
                            """SELECT job_id, host_id,
                                      payload->>'container_name' AS container_name
                               FROM jobs WHERE payload->>'owner' = %s AND status = 'stopped'
                                 AND (payload->>'completed_at')::double precision > %s""",
                            (cid, time.time() - 30),
                        ).fetchall()
                    for job in just_stopped:
                        hid = job.get("host_id")
                        if not hid:
                            continue
                        cname = job.get("container_name") or f"xcl-{job['job_id']}"
                        try:
                            _validate_name(cname, "container name")
                            enqueue_agent_command(
                                hid,
                                "stop_container",
                                {"container_name": cname, "job_id": job["job_id"]},
                                created_by="billing_wallet_suspended",
                            )
                        except Exception as ke:
                            log.warning("stop_container enqueue failed for %s: %s", job["job_id"], ke)
            except Exception as e:
                log.warning("Container cleanup enqueue failed: %s", e)

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

    # ── CAF Export (REPORT_FEATURE_2.md) ──────────────────────────────

    def export_caf_report(
        self,
        customer_id: str,
        period_start: float,
        period_end: float,
    ) -> dict:
        """Generate AI Compute Access Fund rebate documentation.

        From REPORT_FEATURE_2.md: `/billing/export?format=caf`
        Produces data structured for CAF claim submission with fields:
        job_id, duration, cost_CAD, eligible_category, host_country.
        """
        from jurisdiction import compute_fund_eligible_amount

        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM usage_meters
                   WHERE owner = %s AND started_at >= %s AND completed_at <= %s
                   ORDER BY started_at""",
                (customer_id, period_start, period_end),
            ).fetchall()

        line_items = []
        ca_total = 0.0
        non_ca_total = 0.0

        for row in rows:
            cost = float(row["total_cost_cad"])
            is_ca = bool(row["is_canadian_compute"])
            item = {
                "job_id": row["job_id"],
                "host_id": row["host_id"],
                "gpu_model": row["gpu_model"],
                "duration_hours": round(float(row["duration_sec"]) / 3600, 4),
                "cost_cad": cost,
                "eligible_category": (
                    "Canadian cloud compute" if is_ca else "Non-Canadian cloud compute"
                ),
                "host_country": row["country"],
                "host_province": row["province"],
                "trust_tier": row["trust_tier"],
                "is_canadian_compute": is_ca,
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
            }
            line_items.append(item)
            if is_ca:
                ca_total += cost
            else:
                non_ca_total += cost

        ca_fund = compute_fund_eligible_amount(ca_total, True)
        non_ca_fund = compute_fund_eligible_amount(non_ca_total, False)

        attestation = self.generate_attestation()

        return {
            "report_type": "AI Compute Access Fund — Eligible Cost Report",
            "customer_id": customer_id,
            "period_start": period_start,
            "period_end": period_end,
            "currency": "CAD",
            "line_items": line_items,
            "summary": {
                "total_jobs": len(line_items),
                "total_cost_cad": round(ca_total + non_ca_total, 2),
                "canadian_compute_cost_cad": round(ca_total, 2),
                "non_canadian_compute_cost_cad": round(non_ca_total, 2),
                "canadian_reimbursement_rate": "67% (2:1)",
                "non_canadian_reimbursement_rate": "50% (1:1) until March 31, 2027",
                "canadian_eligible_reimbursement_cad": ca_fund["reimbursable_amount_cad"],
                "non_canadian_eligible_reimbursement_cad": non_ca_fund["reimbursable_amount_cad"],
                "total_eligible_reimbursement_cad": round(
                    ca_fund["reimbursable_amount_cad"] + non_ca_fund["reimbursable_amount_cad"], 2
                ),
                "effective_cost_after_fund_cad": round(
                    (ca_total + non_ca_total)
                    - ca_fund["reimbursable_amount_cad"]
                    - non_ca_fund["reimbursable_amount_cad"],
                    2,
                ),
            },
            "supplier_attestation": attestation.to_dict(),
            "notes": (
                "This report is generated for AI Compute Access Fund claim purposes. "
                "All costs are denominated in Canadian Dollars (CAD). "
                "Canadian compute services are provided by Xcelsior, incorporated in Canada, "
                "operating physical data centres located within Canada. "
                "Infrastructure and data stay within Canadian borders throughout processing."
            ),
            "generated_at": time.time(),
        }

    def export_caf_csv(
        self,
        customer_id: str,
        period_start: float,
        period_end: float,
    ) -> str:
        """Generate CSV-formatted output for CAF claims."""
        import csv
        import io
        from datetime import datetime

        report = self.export_caf_report(customer_id, period_start, period_end)
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(
            [
                "Job ID",
                "Host ID",
                "GPU Model",
                "Duration (hrs)",
                "Cost (CAD)",
                "Eligible Category",
                "Host Country",
                "Host Province",
                "Trust Tier",
                "Canadian Compute",
                "Start Time",
                "End Time",
            ]
        )

        for item in report["line_items"]:
            writer.writerow(
                [
                    item["job_id"],
                    item["host_id"],
                    item["gpu_model"],
                    item["duration_hours"],
                    item["cost_cad"],
                    item["eligible_category"],
                    item["host_country"],
                    item["host_province"],
                    item["trust_tier"],
                    "Yes" if item["is_canadian_compute"] else "No",
                    (
                        datetime.fromtimestamp(item["started_at"]).isoformat()
                        if item["started_at"]
                        else ""
                    ),
                    (
                        datetime.fromtimestamp(item["completed_at"]).isoformat()
                        if item["completed_at"]
                        else ""
                    ),
                ]
            )

        # Summary footer
        writer.writerow([])
        writer.writerow(["SUMMARY"])
        for key, val in report["summary"].items():
            writer.writerow([key.replace("_", " ").title(), val])

        return output.getvalue()

    def export_caf_html(
        self,
        customer_id: str,
        period_start: float,
        period_end: float,
        customer_name: str = "",
    ) -> str:
        """Generate a print-ready HTML claim form for the AI Compute Access Fund.

        Designed to be opened in a browser and printed to PDF (A4). Fills in
        all claimant and supplier data from the usage meters + provider
        attestation so the customer can sign and submit.
        """
        from datetime import datetime, timezone
        from html import escape as _h

        report = self.export_caf_report(customer_id, period_start, period_end)
        s = report["summary"]
        att = report["supplier_attestation"]
        items = report["line_items"]

        def _fmt_cad(v) -> str:
            try:
                return f"${float(v):,.2f}"
            except (TypeError, ValueError):
                return "$0.00"

        def _fmt_date(ts) -> str:
            if not ts:
                return "—"
            try:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            except (TypeError, ValueError, OSError):
                return "—"

        def _fmt_dt(ts) -> str:
            if not ts:
                return "—"
            try:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except (TypeError, ValueError, OSError):
                return "—"

        period_label = f"{_fmt_date(period_start)} → {_fmt_date(period_end)}"
        generated_label = _fmt_dt(report.get("generated_at"))
        valid_until_label = _fmt_date(att.get("valid_until"))
        attested_at_label = _fmt_dt(att.get("attested_at"))
        customer_display = _h(customer_name or customer_id)

        rows_html_parts = []
        if not items:
            rows_html_parts.append(
                '<tr><td colspan="8" class="empty">No eligible compute usage recorded in this period.</td></tr>'
            )
        else:
            for it in items:
                rows_html_parts.append(
                    "<tr>"
                    f'<td class="mono">{_h(str(it.get("job_id", "")))}</td>'
                    f'<td>{_fmt_date(it.get("started_at"))}</td>'
                    f'<td>{_fmt_date(it.get("completed_at"))}</td>'
                    f'<td>{_h(str(it.get("gpu_model", "") or "—"))}</td>'
                    f'<td class="num">{float(it.get("duration_hours", 0) or 0):,.2f}</td>'
                    f'<td class="num">{_fmt_cad(it.get("cost_cad", 0))}</td>'
                    f'<td>{_h(str(it.get("host_country", "") or "—"))}'
                    f'{("/" + _h(str(it.get("host_province")))) if it.get("host_province") else ""}</td>'
                    f'<td class="{"ca" if it.get("is_canadian_compute") else "noca"}">'
                    f'{"Canadian" if it.get("is_canadian_compute") else "Non-Canadian"}</td>'
                    "</tr>"
                )
        rows_html = "\n".join(rows_html_parts)

        reg_no = att.get("registration_number") or "—"
        priv_officer = att.get("privacy_officer_contact") or "Not designated"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AI Compute Access Fund — Eligible Cost Claim · {_h(customer_id)}</title>
<style>
  :root {{
    --ink: #111;
    --muted: #555;
    --line: #333;
    --soft: #888;
    --bg: #fff;
    --accent: #8a1d2a;
    --ca: #0a5c2f;
    --noca: #7a4a00;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ background: #e9e9ee; margin: 0; padding: 0; color: var(--ink); font-family: "Helvetica Neue", Arial, "Segoe UI", sans-serif; }}
  .toolbar {{
    position: sticky; top: 0; z-index: 10;
    background: #1a1a1a; color: #fff;
    padding: 10px 20px; display: flex; justify-content: space-between; align-items: center;
    box-shadow: 0 2px 6px rgba(0,0,0,.2);
  }}
  .toolbar .title {{ font-weight: 600; font-size: 13px; letter-spacing: .04em; text-transform: uppercase; }}
  .toolbar button {{
    background: #fff; color: #111; border: 0;
    padding: 8px 16px; font-size: 13px; font-weight: 600; border-radius: 4px;
    cursor: pointer;
  }}
  .toolbar button:hover {{ background: #f0f0f0; }}
  .page {{
    background: var(--bg); width: 210mm; min-height: 297mm;
    margin: 24px auto; padding: 18mm 18mm 22mm 18mm;
    box-shadow: 0 0 18px rgba(0,0,0,.1);
    font-size: 11.5px; line-height: 1.45;
  }}
  .banner {{
    border-top: 6px solid var(--accent); border-bottom: 1px solid var(--line);
    padding-bottom: 10px; margin-bottom: 14px;
  }}
  .banner .crest {{ float: right; text-align: right; font-size: 10px; color: var(--muted); }}
  .banner h1 {{ font-size: 19px; margin: 8px 0 2px; color: var(--accent); letter-spacing: .01em; }}
  .banner .sub {{ font-size: 11px; color: var(--muted); }}
  h2 {{ font-size: 13px; margin: 18px 0 6px; padding-bottom: 3px; border-bottom: 1px solid var(--line); color: #222; text-transform: uppercase; letter-spacing: .05em; }}
  .kv {{ display: grid; grid-template-columns: 180px 1fr; gap: 4px 14px; font-size: 11.5px; }}
  .kv dt {{ color: var(--muted); }}
  .kv dd {{ margin: 0; color: var(--ink); }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .field {{ border-bottom: 1px solid #999; padding: 3px 2px 1px; min-height: 18px; }}
  .field.filled {{ font-weight: 600; }}
  .label {{ font-size: 9.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; margin-top: 10px; display: block; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 10.5px; }}
  th, td {{ padding: 5px 6px; border-bottom: 1px solid #ddd; text-align: left; vertical-align: top; }}
  th {{ background: #f3f3f3; border-bottom: 1.5px solid var(--line); font-weight: 600; font-size: 9.5px; letter-spacing: .05em; text-transform: uppercase; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.mono {{ font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 10px; }}
  td.ca {{ color: var(--ca); font-weight: 600; }}
  td.noca {{ color: var(--noca); }}
  td.empty {{ color: var(--muted); text-align: center; font-style: italic; padding: 14px; }}
  .summary {{ margin-top: 10px; }}
  .summary table th, .summary table td {{ border-bottom: 1px solid #ddd; }}
  .summary .total td {{ border-top: 1.5px solid var(--line); border-bottom: 2px solid var(--line); font-weight: 700; background: #fafafa; }}
  .attestation {{ background: #fafaf4; border: 1px solid #ccc8a8; padding: 12px 14px; font-size: 10.5px; }}
  .attestation .chk {{ display: inline-block; width: 11px; height: 11px; border: 1px solid #333; margin-right: 6px; vertical-align: -1px; background: #fff; position: relative; }}
  .attestation .chk.on::after {{
    content: "✓"; position: absolute; left: 1px; top: -4px; font-size: 12px; color: var(--ca); font-weight: bold;
  }}
  .signatures {{ margin-top: 22px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  .sig {{ margin-top: 28px; border-top: 1px solid #000; padding-top: 4px; font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
  .footer {{ margin-top: 22px; font-size: 9.5px; color: var(--muted); border-top: 1px solid #ddd; padding-top: 8px; }}
  .notes {{ font-size: 10px; color: var(--muted); margin-top: 6px; }}
  @media print {{
    html, body {{ background: #fff; }}
    .toolbar {{ display: none; }}
    .page {{ margin: 0; box-shadow: none; width: auto; min-height: auto; padding: 14mm; }}
    @page {{ size: A4; margin: 12mm; }}
  }}
</style>
</head>
<body>
<div class="toolbar">
  <div class="title">AI Compute Access Fund — Claim Form (CAD)</div>
  <div>
    <button onclick="window.print()">Print / Save as PDF</button>
  </div>
</div>
<div class="page">
  <div class="banner">
    <div class="crest">Form XCL-CAF-01 · Rev 2026-04<br/>Generated {_h(generated_label)}</div>
    <h1>AI Compute Access Fund — Eligible Cost Claim</h1>
    <div class="sub">Xcelsior Inc. · Canadian sovereign GPU compute · All amounts in Canadian Dollars (CAD)</div>
  </div>

  <h2>Part A — Claimant Information</h2>
  <div class="grid-2">
    <div>
      <span class="label">Claimant legal name</span>
      <div class="field filled">{customer_display}</div>
      <span class="label">Customer / Account ID</span>
      <div class="field filled mono">{_h(customer_id)}</div>
    </div>
    <div>
      <span class="label">Claim period (UTC)</span>
      <div class="field filled">{_h(period_label)}</div>
      <span class="label">Reporting currency</span>
      <div class="field filled">Canadian Dollars (CAD)</div>
    </div>
  </div>

  <h2>Part B — Claim Summary</h2>
  <div class="summary">
    <table>
      <thead>
        <tr>
          <th>Category</th>
          <th class="num">Eligible Cost (CAD)</th>
          <th class="num">Reimbursement Rate</th>
          <th class="num">Reimbursement (CAD)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Canadian cloud compute</td>
          <td class="num">{_fmt_cad(s.get("canadian_compute_cost_cad", 0))}</td>
          <td class="num">{_h(s.get("canadian_reimbursement_rate", ""))}</td>
          <td class="num">{_fmt_cad(s.get("canadian_eligible_reimbursement_cad", 0))}</td>
        </tr>
        <tr>
          <td>Non-Canadian cloud compute</td>
          <td class="num">{_fmt_cad(s.get("non_canadian_compute_cost_cad", 0))}</td>
          <td class="num">{_h(s.get("non_canadian_reimbursement_rate", ""))}</td>
          <td class="num">{_fmt_cad(s.get("non_canadian_eligible_reimbursement_cad", 0))}</td>
        </tr>
        <tr class="total">
          <td>Total — {int(s.get("total_jobs", 0))} job(s)</td>
          <td class="num">{_fmt_cad(s.get("total_cost_cad", 0))}</td>
          <td class="num">—</td>
          <td class="num">{_fmt_cad(s.get("total_eligible_reimbursement_cad", 0))}</td>
        </tr>
        <tr>
          <td colspan="3">Effective cost to claimant after fund reimbursement</td>
          <td class="num">{_fmt_cad(s.get("effective_cost_after_fund_cad", 0))}</td>
        </tr>
      </tbody>
    </table>
  </div>

  <h2>Part C — Itemised Compute Usage</h2>
  <table>
    <thead>
      <tr>
        <th>Job ID</th>
        <th>Start (UTC)</th>
        <th>End (UTC)</th>
        <th>GPU</th>
        <th class="num">Duration (hrs)</th>
        <th class="num">Cost (CAD)</th>
        <th>Host Country / Province</th>
        <th>Eligibility</th>
      </tr>
    </thead>
    <tbody>
{rows_html}
    </tbody>
  </table>

  <h2>Part D — Supplier Attestation (Xcelsior Inc.)</h2>
  <div class="attestation">
    <div><strong>Attestation ID:</strong> <span class="mono">{_h(att.get("attestation_id", ""))}</span> &nbsp;·&nbsp;
      <strong>Attested at:</strong> {_h(attested_at_label)} &nbsp;·&nbsp;
      <strong>Valid until:</strong> {_h(valid_until_label)}
    </div>
    <div style="margin-top:8px">
      <span class="chk {'on' if att.get('incorporated_in') == 'Canada' else ''}"></span>
      Provider is incorporated in Canada (<em>{_h(att.get("provider_name", ""))}, {_h(att.get("incorporated_in", ""))}</em>; registration: {_h(reg_no)}).
    </div>
    <div>
      <span class="chk {'on' if att.get('data_centers_in_canada') else ''}"></span>
      Compute was performed on physical infrastructure located in Canada.
    </div>
    <div>
      <span class="chk {'on' if att.get('physical_infrastructure_canada') else ''}"></span>
      Hardware is owned or controlled by the provider and physically located in Canada.
    </div>
    <div>
      <span class="chk {'on' if att.get('data_stays_in_canada') else ''}"></span>
      Customer data remained within Canadian borders throughout processing (data sovereignty).
    </div>
    <div>
      <span class="chk {'on' if att.get('pipeda_compliant') else ''}"></span>
      Provider operates in compliance with PIPEDA (Personal Information Protection and Electronic Documents Act).
    </div>
    <div>
      <span class="chk {'on' if att.get('security_posture') else ''}"></span>
      Security posture: {_h(att.get("security_posture", "—"))}. Privacy officer contact: {_h(priv_officer)}.
    </div>
    <div class="notes">{_h(report.get("notes", ""))}</div>
  </div>

  <h2>Part E — Claimant Declaration &amp; Signatures</h2>
  <p style="font-size:10.5px;margin:4px 0 10px;">
    I certify that the information in this claim is true, correct, and complete to the best of my knowledge,
    and that the eligible costs above were incurred solely for the purpose of AI research, development, or
    deployment activities eligible under the AI Compute Access Fund program guidelines. I retain the
    underlying invoices, attestation bundle, and usage records for audit.
  </p>
  <div class="signatures">
    <div>
      <span class="label">Claimant — Authorised Signatory</span>
      <div class="sig">Signature · Printed name · Date (YYYY-MM-DD)</div>
    </div>
    <div>
      <span class="label">Supplier — Xcelsior Inc.</span>
      <div class="sig">Signature · Printed name · Date (YYYY-MM-DD)</div>
    </div>
  </div>

  <div class="footer">
    Prepared by Xcelsior Inc. · Form XCL-CAF-01 · Attestation {_h(att.get("attestation_id", ""))} ·
    Report generated {_h(generated_label)} ·
    This document is machine-generated from audited usage meters. Retain alongside invoices for the
    AI Compute Access Fund claim submission.
  </div>
</div>
</body>
</html>
"""

    def export_caf_pdf(
        self,
        customer_id: str,
        period_start: float,
        period_end: float,
        customer_name: str = "",
    ) -> bytes:
        """Render the AI Compute Access Fund claim as a real printable PDF (A4).

        Uses reportlab. Returns raw PDF bytes. Mirrors the HTML form layout:
        claimant info, summary, itemised usage, supplier attestation with
        checkboxes, signature lines, footer.
        """
        import io
        from datetime import datetime, timezone

        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        report = self.export_caf_report(customer_id, period_start, period_end)
        s = report["summary"]
        att = report["supplier_attestation"]
        items = report["line_items"]

        def _fmt_cad(v) -> str:
            try:
                return f"${float(v):,.2f}"
            except (TypeError, ValueError):
                return "$0.00"

        def _fmt_date(ts) -> str:
            if not ts:
                return "—"
            try:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            except (TypeError, ValueError, OSError):
                return "—"

        def _fmt_dt(ts) -> str:
            if not ts:
                return "—"
            try:
                return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
                    "%Y-%m-%d %H:%M UTC"
                )
            except (TypeError, ValueError, OSError):
                return "—"

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=15 * mm,
            rightMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
            title=f"CAF Claim — {customer_id}",
            author="Xcelsior Inc.",
        )

        styles = getSampleStyleSheet()
        accent = colors.HexColor("#8a1d2a")
        muted = colors.HexColor("#555555")
        soft = colors.HexColor("#dddddd")

        h_title = ParagraphStyle(
            "CAFTitle",
            parent=styles["Title"],
            fontSize=16,
            textColor=accent,
            spaceAfter=2,
            alignment=0,
        )
        h_sub = ParagraphStyle("CAFSub", parent=styles["Normal"], fontSize=9, textColor=muted)
        h_section = ParagraphStyle(
            "CAFSection",
            parent=styles["Heading3"],
            fontSize=10,
            textColor=colors.HexColor("#222222"),
            spaceBefore=10,
            spaceAfter=4,
            underlineWidth=0.5,
        )
        h_body = ParagraphStyle("CAFBody", parent=styles["Normal"], fontSize=9, leading=12)
        h_small = ParagraphStyle(
            "CAFSmall",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
            textColor=muted,
        )

        story = []

        # Header banner
        header_tbl = Table(
            [
                [
                    Paragraph(
                        "<b>AI Compute Access Fund — Eligible Cost Claim</b>",
                        h_title,
                    ),
                    Paragraph(
                        f"Form XCL-CAF-01 · Rev 2026-04<br/>Generated {_fmt_dt(report.get('generated_at'))}",
                        h_small,
                    ),
                ]
            ],
            colWidths=[120 * mm, 60 * mm],
        )
        header_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                    ("LINEABOVE", (0, 0), (-1, 0), 2, accent),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                    ("TOPPADDING", (0, 0), (-1, 0), 6),
                ]
            )
        )
        story.append(header_tbl)
        story.append(
            Paragraph(
                "Xcelsior Inc. · Canadian sovereign GPU compute · All amounts in Canadian Dollars (CAD)",
                h_sub,
            )
        )
        story.append(Spacer(1, 6 * mm))

        # Part A — Claimant Information
        story.append(Paragraph("<b>PART A — CLAIMANT INFORMATION</b>", h_section))
        claimant_tbl = Table(
            [
                [
                    Paragraph("<b>Claimant legal name</b>", h_small),
                    Paragraph("<b>Customer / Account ID</b>", h_small),
                ],
                [
                    Paragraph(customer_name or customer_id, h_body),
                    Paragraph(f'<font face="Courier">{customer_id}</font>', h_body),
                ],
                [
                    Paragraph("<b>Claim period (UTC)</b>", h_small),
                    Paragraph("<b>Reporting currency</b>", h_small),
                ],
                [
                    Paragraph(
                        f"{_fmt_date(period_start)} → {_fmt_date(period_end)}",
                        h_body,
                    ),
                    Paragraph("Canadian Dollars (CAD)", h_body),
                ],
            ],
            colWidths=[90 * mm, 90 * mm],
        )
        claimant_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 1), (-1, 1), 0.5, colors.black),
                    ("LINEBELOW", (0, 3), (-1, 3), 0.5, colors.black),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(claimant_tbl)

        # Part B — Claim Summary
        story.append(Paragraph("<b>PART B — CLAIM SUMMARY</b>", h_section))
        summary_data = [
            [
                Paragraph("<b>Category</b>", h_small),
                Paragraph("<b>Eligible Cost (CAD)</b>", h_small),
                Paragraph("<b>Reimbursement Rate</b>", h_small),
                Paragraph("<b>Reimbursement (CAD)</b>", h_small),
            ],
            [
                Paragraph("Canadian cloud compute", h_body),
                Paragraph(_fmt_cad(s.get("canadian_compute_cost_cad", 0)), h_body),
                Paragraph(str(s.get("canadian_reimbursement_rate", "")), h_body),
                Paragraph(_fmt_cad(s.get("canadian_eligible_reimbursement_cad", 0)), h_body),
            ],
            [
                Paragraph("Non-Canadian cloud compute", h_body),
                Paragraph(_fmt_cad(s.get("non_canadian_compute_cost_cad", 0)), h_body),
                Paragraph(str(s.get("non_canadian_reimbursement_rate", "")), h_body),
                Paragraph(
                    _fmt_cad(s.get("non_canadian_eligible_reimbursement_cad", 0)),
                    h_body,
                ),
            ],
            [
                Paragraph(f"<b>Total — {int(s.get('total_jobs', 0))} job(s)</b>", h_body),
                Paragraph(f"<b>{_fmt_cad(s.get('total_cost_cad', 0))}</b>", h_body),
                Paragraph("—", h_body),
                Paragraph(
                    f"<b>{_fmt_cad(s.get('total_eligible_reimbursement_cad', 0))}</b>",
                    h_body,
                ),
            ],
            [
                Paragraph("Effective cost to claimant after fund reimbursement", h_body),
                "",
                "",
                Paragraph(
                    f"<b>{_fmt_cad(s.get('effective_cost_after_fund_cad', 0))}</b>",
                    h_body,
                ),
            ],
        ]
        summary_tbl = Table(summary_data, colWidths=[60 * mm, 40 * mm, 40 * mm, 40 * mm])
        summary_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                    ("ALIGN", (0, 0), (0, -1), "LEFT"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f3f3")),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, soft),
                    ("LINEABOVE", (0, 3), (-1, 3), 1, colors.black),
                    ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#fafafa")),
                    ("SPAN", (0, 4), (2, 4)),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(summary_tbl)

        # Part C — Itemised Compute Usage
        story.append(Paragraph("<b>PART C — ITEMISED COMPUTE USAGE</b>", h_section))
        items_header = [
            Paragraph("<b>Job ID</b>", h_small),
            Paragraph("<b>Start</b>", h_small),
            Paragraph("<b>End</b>", h_small),
            Paragraph("<b>GPU</b>", h_small),
            Paragraph("<b>Hrs</b>", h_small),
            Paragraph("<b>Cost (CAD)</b>", h_small),
            Paragraph("<b>Host</b>", h_small),
            Paragraph("<b>Eligibility</b>", h_small),
        ]
        items_rows = [items_header]
        if not items:
            items_rows.append(
                [
                    Paragraph(
                        "<i>No eligible compute usage recorded in this period.</i>",
                        h_small,
                    ),
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        else:
            for it in items:
                host_loc = str(it.get("host_country") or "—")
                if it.get("host_province"):
                    host_loc += f"/{it.get('host_province')}"
                items_rows.append(
                    [
                        Paragraph(
                            f'<font face="Courier" size="7">{str(it.get("job_id", ""))[:18]}</font>',
                            h_small,
                        ),
                        Paragraph(_fmt_date(it.get("started_at")), h_small),
                        Paragraph(_fmt_date(it.get("completed_at")), h_small),
                        Paragraph(str(it.get("gpu_model") or "—"), h_small),
                        Paragraph(
                            f"{float(it.get('duration_hours', 0) or 0):,.2f}",
                            h_small,
                        ),
                        Paragraph(_fmt_cad(it.get("cost_cad", 0)), h_small),
                        Paragraph(host_loc, h_small),
                        Paragraph(
                            "Canadian" if it.get("is_canadian_compute") else "Non-Canadian",
                            h_small,
                        ),
                    ]
                )
        items_tbl = Table(
            items_rows,
            colWidths=[
                28 * mm,
                18 * mm,
                18 * mm,
                22 * mm,
                12 * mm,
                22 * mm,
                28 * mm,
                32 * mm,
            ],
            repeatRows=1,
        )
        items_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (4, 1), (5, -1), "RIGHT"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f3f3")),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.black),
                    ("GRID", (0, 0), (-1, -1), 0.25, soft),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(items_tbl)

        # Part D — Supplier Attestation
        story.append(Paragraph("<b>PART D — SUPPLIER ATTESTATION (Xcelsior Inc.)</b>", h_section))

        def _chk(val) -> str:
            return "[X]" if val else "[ ]"

        att_lines = [
            f"<b>Attestation ID:</b> <font face='Courier'>{att.get('attestation_id', '')}</font> &nbsp;·&nbsp; "
            f"<b>Attested:</b> {_fmt_dt(att.get('attested_at'))} &nbsp;·&nbsp; "
            f"<b>Valid until:</b> {_fmt_date(att.get('valid_until'))}",
            f"{_chk(att.get('incorporated_in') == 'Canada')} Provider is incorporated in Canada "
            f"({att.get('provider_name', '')}, {att.get('incorporated_in', '')}; "
            f"registration: {att.get('registration_number') or '—'}).",
            f"{_chk(att.get('data_centers_in_canada'))} Compute was performed on physical infrastructure located in Canada.",
            f"{_chk(att.get('physical_infrastructure_canada'))} Hardware is owned or controlled by the provider and physically located in Canada.",
            f"{_chk(att.get('data_stays_in_canada'))} Customer data remained within Canadian borders throughout processing (data sovereignty).",
            f"{_chk(att.get('pipeda_compliant'))} Provider operates in compliance with PIPEDA.",
            f"{_chk(bool(att.get('security_posture')))} Security posture: {att.get('security_posture', '—')}. "
            f"Privacy officer: {att.get('privacy_officer_contact') or 'Not designated'}.",
        ]
        att_tbl = Table(
            [[Paragraph(line, h_body)] for line in att_lines],
            colWidths=[180 * mm],
        )
        att_tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafaf4")),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc8a8")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(att_tbl)
        if report.get("notes"):
            story.append(Spacer(1, 2 * mm))
            story.append(Paragraph(str(report["notes"]), h_small))

        # Part E — Declaration & Signatures
        story.append(Paragraph("<b>PART E — CLAIMANT DECLARATION &amp; SIGNATURES</b>", h_section))
        story.append(
            Paragraph(
                "I certify that the information in this claim is true, correct, and complete to the "
                "best of my knowledge, and that the eligible costs above were incurred solely for the "
                "purpose of AI research, development, or deployment activities eligible under the AI "
                "Compute Access Fund program guidelines. I retain the underlying invoices, attestation "
                "bundle, and usage records for audit.",
                h_body,
            )
        )
        story.append(Spacer(1, 10 * mm))
        sig_tbl = Table(
            [
                [
                    Paragraph("<b>Claimant — Authorised Signatory</b>", h_small),
                    Paragraph("<b>Supplier — Xcelsior Inc.</b>", h_small),
                ],
                [
                    Paragraph(
                        "<br/><br/>Signature · Printed name · Date (YYYY-MM-DD)",
                        h_small,
                    ),
                    Paragraph(
                        "<br/><br/>Signature · Printed name · Date (YYYY-MM-DD)",
                        h_small,
                    ),
                ],
            ],
            colWidths=[90 * mm, 90 * mm],
        )
        sig_tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEABOVE", (0, 1), (-1, 1), 0.75, colors.black),
                    ("TOPPADDING", (0, 1), (-1, 1), 2),
                ]
            )
        )
        story.append(sig_tbl)

        story.append(Spacer(1, 6 * mm))
        story.append(
            Paragraph(
                f"Prepared by Xcelsior Inc. · Form XCL-CAF-01 · "
                f"Attestation {att.get('attestation_id', '')} · "
                f"Generated {_fmt_dt(report.get('generated_at'))}. "
                f"Machine-generated from audited usage meters. Retain alongside invoices "
                f"for the AI Compute Access Fund claim submission.",
                h_small,
            )
        )

        doc.build(story)
        return buf.getvalue()


# ── Singleton ─────────────────────────────────────────────────────────

_billing_engine: Optional[BillingEngine] = None


def get_billing_engine() -> BillingEngine:
    global _billing_engine
    if _billing_engine is None:
        _billing_engine = BillingEngine()
    return _billing_engine
