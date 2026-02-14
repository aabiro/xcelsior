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
import sqlite3
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
    gpu_seconds: float = 0.0       # Actual GPU utilization time

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
    category: str = ""           # "compute", "storage", "monitoring", "security"
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
    tax_rate: float = 0.0          # GST/HST rate
    tax_amount_cad: float = 0.0
    total_cad: float = 0.0

    # AI Compute Access Fund breakdown
    canadian_compute_total_cad: float = 0.0
    non_canadian_compute_total_cad: float = 0.0
    fund_eligible_reimbursement_cad: float = 0.0
    effective_cost_after_fund_cad: float = 0.0

    # Metadata
    created_at: float = field(default_factory=time.time)
    status: str = "draft"         # draft, issued, paid, void
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
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), "xcelsior_billing.db"
        )
        self._init_db()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS usage_meters (
                    meter_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    host_id TEXT DEFAULT '',
                    owner TEXT DEFAULT '',
                    started_at REAL DEFAULT 0,
                    completed_at REAL DEFAULT 0,
                    duration_sec REAL DEFAULT 0,
                    gpu_seconds REAL DEFAULT 0,
                    gpu_model TEXT DEFAULT '',
                    vram_gb REAL DEFAULT 0,
                    gpu_utilization_pct REAL DEFAULT 0,
                    xcu_score REAL DEFAULT 0,
                    country TEXT DEFAULT '',
                    province TEXT DEFAULT '',
                    is_canadian_compute INTEGER DEFAULT 0,
                    trust_tier TEXT DEFAULT 'community',
                    base_rate_per_hour REAL DEFAULT 0,
                    tier_multiplier REAL DEFAULT 1.0,
                    spot_discount REAL DEFAULT 0,
                    total_cost_cad REAL DEFAULT 0,
                    created_at REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_meters_job
                    ON usage_meters(job_id);
                CREATE INDEX IF NOT EXISTS idx_meters_owner
                    ON usage_meters(owner);
                CREATE INDEX IF NOT EXISTS idx_meters_time
                    ON usage_meters(started_at);

                CREATE TABLE IF NOT EXISTS invoices (
                    invoice_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    customer_name TEXT DEFAULT '',
                    currency TEXT DEFAULT 'CAD',
                    period_start REAL DEFAULT 0,
                    period_end REAL DEFAULT 0,
                    line_items TEXT DEFAULT '[]',
                    subtotal_cad REAL DEFAULT 0,
                    tax_rate REAL DEFAULT 0,
                    tax_amount_cad REAL DEFAULT 0,
                    total_cad REAL DEFAULT 0,
                    canadian_compute_total_cad REAL DEFAULT 0,
                    non_canadian_compute_total_cad REAL DEFAULT 0,
                    fund_eligible_reimbursement_cad REAL DEFAULT 0,
                    effective_cost_after_fund_cad REAL DEFAULT 0,
                    created_at REAL DEFAULT 0,
                    status TEXT DEFAULT 'draft',
                    notes TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_invoices_customer
                    ON invoices(customer_id);
                CREATE INDEX IF NOT EXISTS idx_invoices_status
                    ON invoices(status);

                CREATE TABLE IF NOT EXISTS payout_ledger (
                    payout_id TEXT PRIMARY KEY,
                    provider_id TEXT NOT NULL,
                    job_id TEXT DEFAULT '',
                    amount_cad REAL DEFAULT 0,
                    platform_fee_cad REAL DEFAULT 0,
                    provider_payout_cad REAL DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_payouts_provider
                    ON payout_ledger(provider_id);
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def meter_job(self, job: dict, host: dict,
                  jurisdiction_data: Optional[dict] = None,
                  trust_tier: str = "community") -> UsageMeter:
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
                """INSERT OR REPLACE INTO usage_meters
                   (meter_id, job_id, host_id, owner, started_at, completed_at,
                    duration_sec, gpu_seconds, gpu_model, vram_gb,
                    gpu_utilization_pct, xcu_score, country, province,
                    is_canadian_compute, trust_tier, base_rate_per_hour,
                    tier_multiplier, spot_discount, total_cost_cad, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    meter.meter_id, meter.job_id, meter.host_id, meter.owner,
                    meter.started_at, meter.completed_at, meter.duration_sec,
                    meter.gpu_seconds, meter.gpu_model, meter.vram_gb,
                    meter.gpu_utilization_pct, meter.xcu_score,
                    meter.country, meter.province,
                    1 if meter.is_canadian_compute else 0,
                    meter.trust_tier, meter.base_rate_per_hour,
                    meter.tier_multiplier, meter.spot_discount,
                    meter.total_cost_cad, time.time(),
                ),
            )

        log.info("METERED job=%s cost=$%.4f CAD tier=%s canadian=%s",
                 meter.job_id, meter.total_cost_cad, trust_tier, is_canadian)
        return meter

    def generate_invoice(
        self,
        customer_id: str,
        customer_name: str,
        period_start: float,
        period_end: float,
        tax_rate: Optional[float] = None,  # None = auto-detect by province
        customer_province: str = "ON",      # Used for tax rate lookup
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
                   WHERE owner = ? AND started_at >= ? AND completed_at <= ?
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
        total_reimbursable = ca_fund["reimbursable_amount_cad"] + non_ca_fund["reimbursable_amount_cad"]
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
            conn.execute(
                """INSERT INTO invoices
                   (invoice_id, customer_id, customer_name, currency,
                    period_start, period_end, line_items, subtotal_cad,
                    tax_rate, tax_amount_cad, total_cad,
                    canadian_compute_total_cad, non_canadian_compute_total_cad,
                    fund_eligible_reimbursement_cad, effective_cost_after_fund_cad,
                    created_at, status, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    invoice.invoice_id, invoice.customer_id, invoice.customer_name,
                    invoice.currency, invoice.period_start, invoice.period_end,
                    json.dumps(invoice.line_items), invoice.subtotal_cad,
                    invoice.tax_rate, invoice.tax_amount_cad, invoice.total_cad,
                    invoice.canadian_compute_total_cad, invoice.non_canadian_compute_total_cad,
                    invoice.fund_eligible_reimbursement_cad, invoice.effective_cost_after_fund_cad,
                    invoice.created_at, invoice.status, invoice.notes,
                ),
            )

        log.info("INVOICE %s customer=%s total=$%.2f CAD (CA: $%.2f, non-CA: $%.2f, "
                 "fund reimbursable: $%.2f, effective: $%.2f)",
                 invoice.invoice_id, customer_id, total, ca_total, non_ca_total,
                 total_reimbursable, effective)
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
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (payout_id, provider_id, job_id, gross_amount_cad,
                 fee, payout, "pending", time.time()),
            )

        log.info("PAYOUT %s provider=%s job=%s gross=$%.4f fee=$%.4f payout=$%.4f",
                 payout_id, provider_id, job_id, gross_amount_cad, fee, payout)

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
                WHERE owner = ? AND started_at >= ? AND completed_at <= ?""",
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

    def process_refund(self, job_id: str, exit_code: int,
                       failure_reason: str = "") -> dict:
        """Determine and process refund for a failed job.

        From REPORT_FEATURE_1.md:
          - Hardware error → full refund
          - User OOM (exit 137) → zero refund
          - Network timeout → partial refund (50%)
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM usage_meters WHERE job_id = ?",
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
            self._credit_wallet(row["owner"], refund_amount,
                                f"Refund for job {job_id} ({classification})")

        log.info("REFUND job=%s classification=%s refund=$%.4f CAD (%.0f%%)",
                 job_id, classification, refund_amount, refund_pct * 100)
        return result

    # ── Credit/Wallet System (REPORT_FEATURE_1.md) ────────────────────

    def _ensure_wallet_table(self):
        """Create wallet tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS wallets (
                    customer_id TEXT PRIMARY KEY,
                    balance_cad REAL DEFAULT 0,
                    total_deposited_cad REAL DEFAULT 0,
                    total_spent_cad REAL DEFAULT 0,
                    total_refunded_cad REAL DEFAULT 0,
                    grace_until REAL DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    created_at REAL DEFAULT 0,
                    updated_at REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS wallet_transactions (
                    tx_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    tx_type TEXT NOT NULL,
                    amount_cad REAL DEFAULT 0,
                    balance_after_cad REAL DEFAULT 0,
                    description TEXT DEFAULT '',
                    job_id TEXT DEFAULT '',
                    created_at REAL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_wallet_tx_customer
                    ON wallet_transactions(customer_id);
                CREATE INDEX IF NOT EXISTS idx_wallet_tx_time
                    ON wallet_transactions(created_at);
            """)

    def get_wallet(self, customer_id: str) -> dict:
        """Get or create a customer wallet."""
        self._ensure_wallet_table()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM wallets WHERE customer_id = ?",
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
                   VALUES (?, 0, 0, 0, 0, 0, 'active', ?, ?)""",
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

    def deposit(self, customer_id: str, amount_cad: float,
                description: str = "Credit deposit") -> dict:
        """Deposit credits into a customer wallet."""
        self._ensure_wallet_table()
        wallet = self.get_wallet(customer_id)
        new_balance = round(wallet["balance_cad"] + amount_cad, 4)
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            conn.execute(
                """UPDATE wallets
                   SET balance_cad = ?,
                       total_deposited_cad = total_deposited_cad + ?,
                       updated_at = ?
                   WHERE customer_id = ?""",
                (new_balance, amount_cad, time.time(), customer_id),
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, created_at)
                   VALUES (?, ?, 'deposit', ?, ?, ?, ?)""",
                (tx_id, customer_id, amount_cad, new_balance,
                 description, time.time()),
            )

        log.info("DEPOSIT %s +$%.2f CAD balance=$%.2f", customer_id, amount_cad, new_balance)
        return {"tx_id": tx_id, "balance_cad": new_balance}

    def charge(self, customer_id: str, amount_cad: float,
               job_id: str = "", description: str = "Compute charge") -> dict:
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
                        "UPDATE wallets SET grace_until = ?, updated_at = ? WHERE customer_id = ?",
                        (grace_end, now, customer_id),
                    )
                log.warning("WALLET %s insufficient balance ($%.2f < $%.2f) "
                            "— 72hr grace period started",
                            customer_id, balance, amount_cad)
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
                        "UPDATE wallets SET status = 'suspended', updated_at = ? WHERE customer_id = ?",
                        (now, customer_id),
                    )
                log.warning("WALLET %s grace period expired — account suspended",
                            customer_id)
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
                   SET balance_cad = ?,
                       total_spent_cad = total_spent_cad + ?,
                       grace_until = 0,
                       updated_at = ?
                   WHERE customer_id = ?""",
                (new_balance, amount_cad, time.time(), customer_id),
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, job_id, created_at)
                   VALUES (?, ?, 'charge', ?, ?, ?, ?, ?)""",
                (tx_id, customer_id, -amount_cad, new_balance,
                 description, job_id, time.time()),
            )

        log.info("CHARGE %s -$%.4f CAD job=%s balance=$%.4f",
                 customer_id, amount_cad, job_id, new_balance)
        return {"charged": True, "tx_id": tx_id, "balance_cad": new_balance}

    def _credit_wallet(self, customer_id: str, amount_cad: float,
                       description: str = "Refund credit"):
        """Internal: credit a wallet (for refunds)."""
        self._ensure_wallet_table()
        wallet = self.get_wallet(customer_id)
        new_balance = round(wallet["balance_cad"] + amount_cad, 4)
        tx_id = f"TX-{int(time.time())}-{os.urandom(3).hex()}"

        with self._conn() as conn:
            conn.execute(
                """UPDATE wallets
                   SET balance_cad = ?,
                       total_refunded_cad = total_refunded_cad + ?,
                       updated_at = ?
                   WHERE customer_id = ?""",
                (new_balance, amount_cad, time.time(), customer_id),
            )
            conn.execute(
                """INSERT INTO wallet_transactions
                   (tx_id, customer_id, tx_type, amount_cad,
                    balance_after_cad, description, created_at)
                   VALUES (?, ?, 'refund', ?, ?, ?, ?)""",
                (tx_id, customer_id, amount_cad, new_balance,
                 description, time.time()),
            )

    def get_wallet_history(self, customer_id: str, limit: int = 50) -> list:
        """Get transaction history for a wallet."""
        self._ensure_wallet_table()
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM wallet_transactions
                   WHERE customer_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (customer_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

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
                   WHERE owner = ? AND started_at >= ? AND completed_at <= ?
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
                "eligible_category": "Canadian cloud compute" if is_ca else "Non-Canadian cloud compute",
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
        writer.writerow([
            "Job ID", "Host ID", "GPU Model", "Duration (hrs)",
            "Cost (CAD)", "Eligible Category", "Host Country",
            "Host Province", "Trust Tier", "Canadian Compute",
            "Start Time", "End Time",
        ])

        for item in report["line_items"]:
            writer.writerow([
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
                datetime.fromtimestamp(item["started_at"]).isoformat() if item["started_at"] else "",
                datetime.fromtimestamp(item["completed_at"]).isoformat() if item["completed_at"] else "",
            ])

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
