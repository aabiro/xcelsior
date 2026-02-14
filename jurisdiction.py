# Xcelsior Jurisdiction-Aware Scheduling
# Implements REPORT_FEATURE_FINAL.md + REPORT_MARKETING_FINAL.md:
#   - Canada-only, province-specific routing
#   - Compliance constraints as first-class scheduling inputs
#   - Trust tiers: Residency → Sovereignty → Regulated
#   - AI Compute Access Fund alignment (Canadian vs non-Canadian split)
#
# Provincial constraints modeled:
#   - Québec Law 25: cross-border PIA required, up to $25M / 4% penalty
#   - Nova Scotia PIIDPA: stored/accessed only in Canada (public bodies)
#   - BC FOIPPA: risk-based, not blanket restriction (post-2021 Bill 22)
#   - Ontario PHIPA: audit log provision drafted but not yet in force

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional

log = logging.getLogger("xcelsior")


# ── Canadian Provinces ────────────────────────────────────────────────

class Province(str, Enum):
    AB = "AB"  # Alberta
    BC = "BC"  # British Columbia
    MB = "MB"  # Manitoba
    NB = "NB"  # New Brunswick
    NL = "NL"  # Newfoundland & Labrador
    NS = "NS"  # Nova Scotia
    NT = "NT"  # Northwest Territories
    NU = "NU"  # Nunavut
    ON = "ON"  # Ontario
    PE = "PE"  # Prince Edward Island
    QC = "QC"  # Québec
    SK = "SK"  # Saskatchewan
    YT = "YT"  # Yukon


# Province-specific compliance notes (for UI/docs, not enforcement)
PROVINCE_COMPLIANCE = {
    Province.QC: {
        "regime": "Law 25 (private sector)",
        "key_rule": "Cross-border transfer requires PIA + written agreement",
        "max_penalty": "$25M or 4% worldwide turnover",
        "scheduling_impact": "quebec_only routing recommended for personal info workloads",
    },
    Province.NS: {
        "regime": "PIIDPA (public bodies)",
        "key_rule": "Stored AND accessed only in Canada",
        "scheduling_impact": "canada_only enforced for public body workloads",
    },
    Province.BC: {
        "regime": "FOIPPA (post-Bill 22, risk-based)",
        "key_rule": "Extra PIA for sensitive PI stored outside Canada",
        "scheduling_impact": "canada_preferred for sensitive workloads",
    },
    Province.ON: {
        "regime": "PHIPA (health, s.10.1 not yet in force)",
        "key_rule": "Electronic audit log drafted but not proclaimed",
        "scheduling_impact": "audit_tier available as premium feature",
    },
}


# ── Trust Tiers ──────────────────────────────────────────────────────
# From REPORT_MARKETING_FINAL.md:
#   1. Residency — workload runs on machines physically in Canada
#   2. Sovereignty — Canadian-jurisdiction operator, no foreign control
#   3. Regulated — deeper auditability for health/public sector

class TrustTier(str, Enum):
    COMMUNITY = "community"     # Open marketplace, no guarantees
    RESIDENCY = "residency"     # Canada-only physical location
    SOVEREIGNTY = "sovereignty" # Canadian-incorporated, no foreign control
    REGULATED = "regulated"     # Audit trail, health/public sector ready


TRUST_TIER_REQUIREMENTS = {
    TrustTier.COMMUNITY: {
        "label": "Community",
        "description": "Open marketplace. No jurisdiction guarantees.",
        "requires_canada": False,
        "requires_verified": False,
        "requires_sovereignty_vetting": False,
        "requires_audit_trail": False,
        "pricing_multiplier": 1.0,
    },
    TrustTier.RESIDENCY: {
        "label": "Canada Residency",
        "description": "Workload scheduled on machines physically in Canada.",
        "requires_canada": True,
        "requires_verified": True,
        "requires_sovereignty_vetting": False,
        "requires_audit_trail": False,
        "pricing_multiplier": 1.15,   # 15% premium for residency guarantee
    },
    TrustTier.SOVEREIGNTY: {
        "label": "Canada Sovereignty",
        "description": "Canadian-incorporated operator. No foreign jurisdictional exposure.",
        "requires_canada": True,
        "requires_verified": True,
        "requires_sovereignty_vetting": True,
        "requires_audit_trail": False,
        "pricing_multiplier": 1.35,   # 35% premium for sovereignty
    },
    TrustTier.REGULATED: {
        "label": "Regulated",
        "description": "Full audit trail. Suitable for health/public sector workloads.",
        "requires_canada": True,
        "requires_verified": True,
        "requires_sovereignty_vetting": True,
        "requires_audit_trail": True,
        "pricing_multiplier": 1.60,   # 60% premium for regulated tier
    },
}


# ── Host Jurisdiction Metadata ───────────────────────────────────────

@dataclass
class HostJurisdiction:
    """Jurisdiction and compliance metadata for a GPU host."""
    host_id: str = ""
    country: str = ""               # ISO 3166-1 alpha-2 (CA, US, etc.)
    province: str = ""              # Province/state code
    city: str = ""
    data_center_name: str = ""
    data_center_class: str = ""     # "commercial", "residential", "colocation"

    # Sovereignty vetting
    operator_name: str = ""
    operator_incorporated_in: str = ""  # Country of incorporation
    operator_registered_canada: bool = False
    foreign_control: bool = False   # Subject to foreign jurisdiction?
    sovereignty_vetted: bool = False
    sovereignty_vetted_at: Optional[float] = None

    # Compliance capabilities
    supports_audit_trail: bool = False
    supports_data_residency: bool = False

    # Trust tier
    max_trust_tier: str = TrustTier.COMMUNITY

    def to_dict(self) -> dict:
        return asdict(self)


# ── Jurisdiction-Aware Scheduling ────────────────────────────────────

@dataclass
class JurisdictionConstraint:
    """Scheduling constraint for jurisdiction-aware job routing."""
    canada_only: bool = False
    province: Optional[str] = None          # Restrict to specific province
    trust_tier: str = TrustTier.COMMUNITY   # Minimum trust tier
    exclude_countries: list = field(default_factory=list)
    require_verified: bool = False
    require_sovereignty: bool = False
    require_audit_trail: bool = False
    data_sensitivity: str = "standard"     # standard, sensitive, regulated

    def to_dict(self) -> dict:
        return asdict(self)


def host_meets_constraint(
    host: dict,
    jurisdiction: Optional[HostJurisdiction],
    constraint: JurisdictionConstraint,
) -> tuple[bool, str]:
    """Check if a host meets jurisdiction constraints.

    Returns:
        (meets: bool, reason: str) — reason explains why it was rejected
    """
    # Country check
    host_country = (jurisdiction.country if jurisdiction else
                    host.get("country", "")).upper()

    if constraint.canada_only and host_country != "CA":
        return False, f"Canada-only required, host is in {host_country or 'unknown'}"

    if host_country in [c.upper() for c in constraint.exclude_countries]:
        return False, f"Host country {host_country} is excluded"

    # Province check
    if constraint.province:
        host_province = (jurisdiction.province if jurisdiction else
                        host.get("province", "")).upper()
        if host_province != constraint.province.upper():
            return False, f"Province {constraint.province} required, host is in {host_province or 'unknown'}"

    # Verification check
    if constraint.require_verified:
        verified = host.get("verified", False)
        if jurisdiction:
            verified = True  # Has jurisdiction metadata = at least partially vetted
        if not verified:
            return False, "Host verification required"

    # Sovereignty check
    if constraint.require_sovereignty:
        if not jurisdiction or not jurisdiction.sovereignty_vetted:
            return False, "Sovereignty vetting required"
        if jurisdiction.foreign_control:
            return False, "Host operator is subject to foreign jurisdiction"

    # Audit trail check
    if constraint.require_audit_trail:
        if not jurisdiction or not jurisdiction.supports_audit_trail:
            return False, "Audit trail capability required"

    # Data sensitivity enforcement
    # REPORT_MARKETING_FINAL.md: encode sensitivity as scheduling input
    if constraint.data_sensitivity == "sensitive":
        # Sensitive data requires at least Residency tier (Canada-only)
        if not jurisdiction or jurisdiction.country.upper() != "CA":
            return False, "Sensitive data requires Canadian-resident host"
        if not host.get("verified", False) and not (jurisdiction and jurisdiction.sovereignty_vetted):
            return False, "Sensitive data requires verified host"
    elif constraint.data_sensitivity == "regulated":
        # Regulated data requires Sovereignty + audit trail
        if not jurisdiction or not jurisdiction.sovereignty_vetted:
            return False, "Regulated data requires sovereignty-vetted host"
        if not jurisdiction.supports_audit_trail:
            return False, "Regulated data requires audit trail capability"

    # Trust tier check
    tier_order = [TrustTier.COMMUNITY, TrustTier.RESIDENCY,
                  TrustTier.SOVEREIGNTY, TrustTier.REGULATED]
    required_idx = tier_order.index(TrustTier(constraint.trust_tier))
    host_tier = TrustTier(jurisdiction.max_trust_tier if jurisdiction else TrustTier.COMMUNITY)
    host_idx = tier_order.index(host_tier)

    if host_idx < required_idx:
        return False, (
            f"Trust tier {constraint.trust_tier} required, "
            f"host qualifies for {host_tier.value}"
        )

    return True, "OK"


def filter_hosts_by_jurisdiction(
    hosts: list[dict],
    jurisdictions: dict,  # host_id -> HostJurisdiction
    constraint: JurisdictionConstraint,
) -> list[dict]:
    """Filter hosts that meet jurisdiction constraints.

    Args:
        hosts: List of host dicts from scheduler
        jurisdictions: Map of host_id to HostJurisdiction
        constraint: The scheduling constraint to enforce

    Returns:
        Filtered list of eligible hosts
    """
    eligible = []
    for host in hosts:
        host_id = host.get("host_id", "")
        jur = jurisdictions.get(host_id)
        meets, reason = host_meets_constraint(host, jur, constraint)
        if meets:
            eligible.append(host)
        else:
            log.debug("HOST EXCLUDED %s: %s", host_id, reason)

    log.info("JURISDICTION FILTER: %d/%d hosts eligible (canada_only=%s, tier=%s, province=%s)",
             len(eligible), len(hosts), constraint.canada_only,
             constraint.trust_tier, constraint.province)
    return eligible


def classify_host_trust_tier(jurisdiction: Optional[HostJurisdiction]) -> TrustTier:
    """Determine the maximum trust tier a host qualifies for."""
    if not jurisdiction:
        return TrustTier.COMMUNITY

    if jurisdiction.country.upper() != "CA":
        return TrustTier.COMMUNITY

    # At minimum, a Canadian host qualifies for Residency
    tier = TrustTier.RESIDENCY

    # Sovereignty requires Canadian incorporation + no foreign control
    if (jurisdiction.sovereignty_vetted and
            jurisdiction.operator_registered_canada and
            not jurisdiction.foreign_control):
        tier = TrustTier.SOVEREIGNTY

        # Regulated requires audit trail capability on top of sovereignty
        if jurisdiction.supports_audit_trail:
            tier = TrustTier.REGULATED

    return tier


# ── AI Compute Access Fund Tracking ──────────────────────────────────
# From REPORT_MARKETING_FINAL.md:
#   - 67% (2:1) for Canadian compute costs
#   - 50% (1:1) for non-Canadian compute until March 31, 2027
#   - Non-Canadian no longer eligible after April 1, 2027

FUND_CANADIAN_RATE = 0.6667    # 2/3 coverage
FUND_NON_CANADIAN_RATE = 0.50  # 1/2 coverage
FUND_NON_CANADIAN_CUTOFF = 1743465600  # April 1, 2027 UTC


def compute_fund_eligible_amount(
    total_cost_cad: float,
    is_canadian_compute: bool,
    timestamp: Optional[float] = None,
) -> dict:
    """Calculate AI Compute Access Fund eligible reimbursement.

    Returns dict with fund details for invoicing/reporting.
    """
    ts = timestamp or time.time()

    if is_canadian_compute:
        rate = FUND_CANADIAN_RATE
        eligible = True
        label = "Canadian AI Compute (67% eligible)"
    else:
        if ts >= FUND_NON_CANADIAN_CUTOFF:
            rate = 0.0
            eligible = False
            label = "Non-Canadian Compute (no longer eligible after Apr 1, 2027)"
        else:
            rate = FUND_NON_CANADIAN_RATE
            eligible = True
            label = "Non-Canadian Compute (50% eligible until Mar 31, 2027)"

    reimbursable = round(total_cost_cad * rate, 2) if eligible else 0.0
    effective_cost = round(total_cost_cad - reimbursable, 2)

    return {
        "total_cost_cad": total_cost_cad,
        "is_canadian_compute": is_canadian_compute,
        "fund_eligible": eligible,
        "fund_rate": rate,
        "fund_label": label,
        "reimbursable_amount_cad": reimbursable,
        "effective_cost_cad": effective_cost,
        "computed_at": ts,
    }


def generate_residency_trace(
    job_id: str,
    host_id: str,
    jurisdiction: Optional[HostJurisdiction],
    started_at: float,
    completed_at: float,
) -> dict:
    """Generate a workload residency trace for compliance reporting.

    This is the artifact customers use to prove where their
    workload ran — for AI Compute Access Fund claims and compliance.
    """
    return {
        "job_id": job_id,
        "host_id": host_id,
        "country": jurisdiction.country if jurisdiction else "unknown",
        "province": jurisdiction.province if jurisdiction else "unknown",
        "city": jurisdiction.city if jurisdiction else "unknown",
        "data_center": jurisdiction.data_center_name if jurisdiction else "unknown",
        "data_center_class": jurisdiction.data_center_class if jurisdiction else "unknown",
        "operator": jurisdiction.operator_name if jurisdiction else "unknown",
        "operator_incorporated_in": jurisdiction.operator_incorporated_in if jurisdiction else "unknown",
        "operator_canadian_registered": jurisdiction.operator_registered_canada if jurisdiction else False,
        "sovereignty_vetted": jurisdiction.sovereignty_vetted if jurisdiction else False,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_sec": round(completed_at - started_at, 2),
        "is_canadian_compute": (jurisdiction.country.upper() == "CA") if jurisdiction else False,
        "trace_generated_at": time.time(),
    }
