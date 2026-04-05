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
        self, customer_id: str, amount_cad: float, description: str = "Credit deposit",
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
                    log.info("Idempotent deposit skipped (key=%s, existing tx=%s)", idempotency_key, existing["tx_id"])
                    return {"tx_id": existing["tx_id"], "balance_cad": existing["balance_after_cad"], "dedup": True}

        wallet = self.get_wallet(customer_id)
        new_balance = round(wallet["balance_cad"] + amount_cad, 4)
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            conn.execute(
                """UPDATE wallets
                   SET balance_cad = %s,
                       total_deposited_cad = total_deposited_cad + %s,
                       updated_at = %s
                   WHERE customer_id = %s""",
                (new_balance, amount_cad, time.time(), customer_id),
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, created_at, idempotency_key)
                   VALUES (%s, %s, 'deposit', %s, %s, %s, %s, %s)""",
                (tx_id, customer_id, amount_cad, new_balance, description, time.time(),
                 idempotency_key or ""),
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
            conn.execute(
                """UPDATE wallets
                   SET balance_cad = %s,
                       total_spent_cad = total_spent_cad + %s,
                       grace_until = 0,
                       updated_at = %s
                   WHERE customer_id = %s""",
                (new_balance, amount_cad, time.time(), customer_id),
            )
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
        wallet = self.get_wallet(customer_id)
        new_balance = round(wallet["balance_cad"] + amount_cad, 4)
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            conn.execute(
                """UPDATE wallets
                   SET balance_cad = %s,
                       total_refunded_cad = total_refunded_cad + %s,
                       updated_at = %s
                   WHERE customer_id = %s""",
                (new_balance, amount_cad, time.time(), customer_id),
            )
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
            cleared = conn.execute(
                "DELETE FROM wallet_transactions WHERE customer_id = %s",
                (customer_id,),
            ).rowcount or 0
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

    def pause_instance(self, job_id: str, reason: str = "paused_low_balance") -> dict:
        """Pause a running instance: stop container, preserve volume, stop billing.

        Per Phase 1.3: pause_instance() stops the container but preserves
        the volume mount so the user can resume later.
        """
        from db import _get_pg_pool
        from psycopg.rows import dict_row

        now = time.time()
        pool = _get_pg_pool()
        with pool.connection() as conn:
            conn.row_factory = dict_row
            job = conn.execute(
                "SELECT * FROM jobs WHERE job_id = %s AND status = 'running' FOR UPDATE",
                (job_id,),
            ).fetchone()
            if not job:
                return {"paused": False, "reason": "not_running"}

            conn.execute(
                "UPDATE jobs SET status = %s, completed_at = %s WHERE job_id = %s",
                (reason, now, job_id),
            )
            conn.commit()

        log.warning("PAUSE job=%s reason=%s owner=%s", job_id, reason, job.get("owner"))
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
                "SELECT * FROM jobs WHERE job_id = %s AND status = 'paused_low_balance' FOR UPDATE",
                (job_id,),
            ).fetchone()
            if not job:
                return {"resumed": False, "reason": "not_paused"}

            # Verify wallet has funds
            wallet = self.get_wallet(job["owner"])
            if wallet["balance_cad"] <= 0:
                return {"resumed": False, "reason": "insufficient_balance"}

            conn.execute(
                "UPDATE jobs SET status = 'running', completed_at = 0, updated_at = %s WHERE job_id = %s",
                (now, job_id),
            )
            conn.commit()

        log.info("RESUME job=%s owner=%s", job_id, job.get("owner"))
        return {"resumed": True, "job_id": job_id, "status": "running"}

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
                (enabled, amount_cad, threshold_cad, stripe_payment_method_id, time.time(), customer_id),
            )
        log.info("Auto-topup configured for %s: enabled=%s amount=$%.2f threshold=$%.2f",
                 customer_id, enabled, amount_cad, threshold_cad)
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
                host_id = job.get("host_id", "")
                gpu_model = job.get("gpu_model", "")
                tier = job.get("tier", "free")

                # Find the last billing cycle end for this job
                with pool.connection() as conn:
                    conn.row_factory = dict_row
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

                # Get the host's rate
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    host = conn.execute(
                        "SELECT cost_per_hour FROM hosts WHERE host_id = %s",
                        (host_id,),
                    ).fetchone()

                rate_per_hour = float(host["cost_per_hour"]) if host else 0.20

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
                    customer_id, amount_cad, job_id=job_id,
                    description=f"Auto-billing: {gpu_model} ({duration_sec/60:.1f}min)",
                )

                cycle_id = f"BC-{int(now)}-{os.urandom(3).hex()}"
                status = "charged" if charge_result.get("charged") else "failed"

                # Record the billing cycle
                with pool.connection() as conn:
                    conn.row_factory = dict_row
                    conn.execute(
                        """INSERT INTO billing_cycles
                           (cycle_id, job_id, customer_id, host_id, period_start, period_end,
                            duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                            amount_cad, status, created_at)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        (cycle_id, job_id, customer_id, host_id, period_start, period_end,
                         duration_sec, rate_per_hour, gpu_model, tier, tier_multiplier,
                         amount_cad, status, now),
                    )
                    conn.commit()

                billed += 1

                # If charge failed with grace_expired → suspend and STOP the job
                if not charge_result.get("charged") and charge_result.get("action") == "account_suspended":
                    suspended += 1
                    # Actually terminate the running container
                    try:
                        from scheduler import kill_job as scheduler_kill
                        with pool.connection() as kconn:
                            kconn.row_factory = dict_row
                            jrow = kconn.execute(
                                "SELECT j.*, h.ip FROM jobs j JOIN hosts h ON j.host_id = h.host_id WHERE j.job_id = %s",
                                (job_id,),
                            ).fetchone()
                        if jrow:
                            scheduler_kill(dict(jrow), dict(jrow))
                            log.warning("BILLING: Killed job %s for suspended account %s", job_id, customer_id)
                        else:
                            log.warning("BILLING: Job %s not found for kill on suspension", job_id)
                    except Exception as kill_err:
                        log.error("BILLING: Failed to kill job %s on suspension: %s", job_id, kill_err)

            except Exception as e:
                errors += 1
                log.error("Auto-billing error for job %s: %s", job.get("job_id", "?"), e)

        # ── Bill active volumes (real-time storage charges) ──────────
        volume_billed = 0
        try:
            from volumes import VolumeEngine
            ve = VolumeEngine()
            with pool.connection() as conn:
                conn.row_factory = dict_row
                active_volumes = conn.execute(
                    """SELECT volume_id, owner_id, name, size_gb, created_at
                       FROM volumes WHERE status != 'deleted'""",
                ).fetchall()

            for vol in active_volumes:
                try:
                    vid = vol["volume_id"]
                    vol_owner = vol["owner_id"]
                    size_gb = vol.get("size_gb", 0)
                    if size_gb <= 0:
                        continue

                    # Find last volume billing cycle
                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        last_vc = conn.execute(
                            """SELECT period_end FROM billing_cycles
                               WHERE job_id = %s ORDER BY period_end DESC LIMIT 1""",
                            (vid,),
                        ).fetchone()

                    vperiod_start = last_vc["period_end"] if last_vc else float(vol["created_at"])
                    vperiod_end = now

                    if vperiod_end - vperiod_start < 60:
                        continue

                    vduration_sec = vperiod_end - vperiod_start
                    # $0.07/GB/month → per-second rate
                    rate_per_sec = (0.07 * size_gb) / (30 * 24 * 3600)
                    vamount = round(rate_per_sec * vduration_sec, 4)

                    if vamount <= 0:
                        continue

                    vcharge = self.charge(
                        vol_owner, vamount, job_id=vid,
                        description=f"Volume storage: {vol.get('name', vid)} ({size_gb} GB, {vduration_sec/60:.1f}min)",
                    )

                    vcycle_id = f"VC-{int(now)}-{os.urandom(3).hex()}"
                    vstatus = "charged" if vcharge.get("charged") else "failed"

                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        conn.execute(
                            """INSERT INTO billing_cycles
                               (cycle_id, job_id, customer_id, host_id, period_start, period_end,
                                duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                                amount_cad, status, created_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (vcycle_id, vid, vol_owner, "", vperiod_start, vperiod_end,
                             vduration_sec, round(0.07 * size_gb / 730, 6), "storage", "volume", 1.0,
                             vamount, vstatus, now),
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
                        ep_owner, iamount, job_id=ep_id,
                        description=f"Inference compute: {gpu_type} ({iduration_sec/60:.1f}min)",
                    )

                    icycle_id = f"IC-{int(now)}-{os.urandom(3).hex()}"
                    istatus = "charged" if icharge.get("charged") else "failed"

                    with pool.connection() as conn:
                        conn.row_factory = dict_row
                        conn.execute(
                            """INSERT INTO billing_cycles
                               (cycle_id, job_id, customer_id, host_id, period_start, period_end,
                                duration_seconds, rate_per_hour, gpu_model, tier, tier_multiplier,
                                amount_cad, status, created_at)
                               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            (icycle_id, ep_id, ep_owner, "", iperiod_start, iperiod_end,
                             iduration_sec, cost_per_hour, gpu_type, "inference", 1.0,
                             iamount, istatus, now),
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

        if billed or suspended or errors or volume_billed or inference_billed:
            log.info("AUTO-BILLING: %d jobs, %d volumes, %d inference, %d suspended, %d errors",
                     billed, volume_billed, inference_billed, suspended, errors)

        return {"billed": billed, "volume_billed": volume_billed,
                "inference_billed": inference_billed, "suspended": suspended, "errors": errors}

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
                log.info("Auto-topup PaymentIntent created for %s: %s ($%.2f)",
                         customer_id, pi.id, w["auto_topup_amount_cad"])
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
                log.error("Auto-topup failed for %s (attempt %d/3): %s", customer_id, new_failures, e)

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
                        log.warning("Auto-topup DISABLED for %s after 3 failures — pausing instances", customer_id)
                        # Pause all running instances for this customer
                        running = conn.execute(
                            "SELECT job_id FROM jobs WHERE owner = %s AND status = 'running'",
                            (customer_id,),
                        ).fetchall()
                        for job in running:
                            conn.execute(
                                "UPDATE jobs SET status = 'paused_low_balance', completed_at = %s WHERE job_id = %s",
                                (now, job["job_id"]),
                            )
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

    def fintrac_check_transaction(self, customer_id: str, amount_cad: float, currency: str = "CAD") -> Optional[dict]:
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
                (report_id, customer_id, report_type, trigger_amount,
                 trigger_currency, now - 86400, now, now, notes),
            )
        log.warning("FINTRAC %s report created: %s customer=%s amount=$%.2f %s",
                     report_type, report_id, customer_id, trigger_amount, trigger_currency)
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
                    "SELECT job_id FROM jobs WHERE owner = %s AND status = 'running'",
                    (cid,),
                ).fetchall()
                for job in running:
                    conn.execute(
                        "UPDATE jobs SET status = 'stopped', completed_at = %s WHERE job_id = %s AND status = 'running'",
                        (time.time(), job["job_id"]),
                    )
                    stopped += 1
                    log.warning("Stopped job %s for suspended wallet %s", job["job_id"], cid)
            conn.commit()

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
                """SELECT j.job_id, j.host_id, j.gpu_model, j.tier
                   FROM jobs j
                   WHERE j.owner = %s AND j.status = 'running'""",
                (customer_id,),
            ).fetchall()

        burn_per_hour = 0.0
        instance_burns = []
        for job in running:
            host_id = job.get("host_id", "")
            with pool.connection() as conn:
                conn.row_factory = dict_row
                host = conn.execute(
                    "SELECT cost_per_hour FROM hosts WHERE host_id = %s",
                    (host_id,),
                ).fetchone()
            rate = float(host["cost_per_hour"]) if host else 0.20
            burn_per_hour += rate
            instance_burns.append({
                "job_id": job["job_id"],
                "gpu_model": job.get("gpu_model", ""),
                "rate_per_hour": rate,
            })

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
            "seconds_to_zero": round(seconds_to_zero, 1) if seconds_to_zero != float("inf") else None,
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


# ── Singleton ─────────────────────────────────────────────────────────

_billing_engine: Optional[BillingEngine] = None


def get_billing_engine() -> BillingEngine:
    global _billing_engine
    if _billing_engine is None:
        _billing_engine = BillingEngine()
    return _billing_engine
