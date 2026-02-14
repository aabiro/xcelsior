"""Tests for Xcelsior jurisdiction — trust tiers, residency trace, fund eligibility, constraints."""

import os
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")

from jurisdiction import (
    FUND_CANADIAN_RATE,
    FUND_NON_CANADIAN_CUTOFF,
    FUND_NON_CANADIAN_RATE,
    HostJurisdiction,
    JurisdictionConstraint,
    Province,
    TRUST_TIER_REQUIREMENTS,
    TrustTier,
    classify_host_trust_tier,
    compute_fund_eligible_amount,
    filter_hosts_by_jurisdiction,
    generate_residency_trace,
    host_meets_constraint,
)


# ── Trust Tier Classification ─────────────────────────────────────────


class TestTrustTiers:
    """Classify host into trust tiers."""

    def test_no_jurisdiction_is_community(self):
        assert classify_host_trust_tier(None) == TrustTier.COMMUNITY

    def test_non_canadian_is_community(self):
        jur = HostJurisdiction(country="US")
        assert classify_host_trust_tier(jur) == TrustTier.COMMUNITY

    def test_canadian_host_is_residency(self):
        jur = HostJurisdiction(country="CA", province="ON")
        assert classify_host_trust_tier(jur) == TrustTier.RESIDENCY

    def test_sovereignty_vetted_canadian(self):
        jur = HostJurisdiction(
            country="CA",
            sovereignty_vetted=True,
            operator_registered_canada=True,
            foreign_control=False,
        )
        assert classify_host_trust_tier(jur) == TrustTier.SOVEREIGNTY

    def test_regulated_with_audit_trail(self):
        jur = HostJurisdiction(
            country="CA",
            sovereignty_vetted=True,
            operator_registered_canada=True,
            foreign_control=False,
            supports_audit_trail=True,
        )
        assert classify_host_trust_tier(jur) == TrustTier.REGULATED

    def test_foreign_control_blocks_sovereignty(self):
        jur = HostJurisdiction(
            country="CA",
            sovereignty_vetted=True,
            operator_registered_canada=True,
            foreign_control=True,
        )
        tier = classify_host_trust_tier(jur)
        assert tier == TrustTier.RESIDENCY  # Not sovereignty because foreign control

    def test_all_tiers_have_requirements(self):
        for tier in TrustTier:
            assert tier in TRUST_TIER_REQUIREMENTS
            assert "pricing_multiplier" in TRUST_TIER_REQUIREMENTS[tier]


# ── Host Meets Constraint ────────────────────────────────────────────


class TestHostMeetsConstraint:
    """Jurisdiction constraint checking."""

    def test_unconstrained_host_passes(self):
        host = {"host_id": "h1", "country": "US"}
        constraint = JurisdictionConstraint()
        meets, reason = host_meets_constraint(host, None, constraint)
        assert meets is True

    def test_canada_only_rejects_us(self):
        host = {"host_id": "h1", "country": "US"}
        constraint = JurisdictionConstraint(canada_only=True)
        meets, reason = host_meets_constraint(host, None, constraint)
        assert meets is False
        assert "Canada-only" in reason

    def test_canada_only_accepts_ca(self):
        host = {"host_id": "h1", "country": "CA"}
        jur = HostJurisdiction(country="CA")
        constraint = JurisdictionConstraint(canada_only=True)
        meets, _ = host_meets_constraint(host, jur, constraint)
        assert meets is True

    def test_province_constraint(self):
        host = {"host_id": "h1"}
        jur = HostJurisdiction(country="CA", province="ON")
        constraint = JurisdictionConstraint(province="QC")
        meets, reason = host_meets_constraint(host, jur, constraint)
        assert meets is False
        assert "Province" in reason

    def test_province_match(self):
        host = {"host_id": "h1"}
        jur = HostJurisdiction(country="CA", province="QC")
        constraint = JurisdictionConstraint(province="QC")
        meets, _ = host_meets_constraint(host, jur, constraint)
        assert meets is True

    def test_exclude_countries(self):
        host = {"host_id": "h1", "country": "CN"}
        constraint = JurisdictionConstraint(exclude_countries=["CN", "RU"])
        meets, reason = host_meets_constraint(host, None, constraint)
        assert meets is False

    def test_sovereignty_required_no_jur_fails(self):
        host = {"host_id": "h1"}
        constraint = JurisdictionConstraint(require_sovereignty=True)
        meets, reason = host_meets_constraint(host, None, constraint)
        assert meets is False

    def test_audit_trail_required(self):
        host = {"host_id": "h1"}
        jur = HostJurisdiction(country="CA", supports_audit_trail=False)
        constraint = JurisdictionConstraint(require_audit_trail=True)
        meets, _ = host_meets_constraint(host, jur, constraint)
        assert meets is False

    def test_sensitive_data_requires_canada(self):
        host = {"host_id": "h1", "country": "US"}
        constraint = JurisdictionConstraint(data_sensitivity="sensitive")
        meets, reason = host_meets_constraint(host, None, constraint)
        assert meets is False
        assert "Canadian" in reason or "Sensitive" in reason


# ── Filter Hosts ─────────────────────────────────────────────────────


class TestFilterHosts:
    """Bulk host filtering by jurisdiction."""

    def test_filter_canada_only(self):
        hosts = [
            {"host_id": "h-ca", "country": "CA"},
            {"host_id": "h-us", "country": "US"},
        ]
        jurs = {
            "h-ca": HostJurisdiction(country="CA"),
            "h-us": HostJurisdiction(country="US"),
        }
        constraint = JurisdictionConstraint(canada_only=True)
        eligible = filter_hosts_by_jurisdiction(hosts, jurs, constraint)
        assert len(eligible) == 1
        assert eligible[0]["host_id"] == "h-ca"

    def test_filter_no_constraint_returns_all(self):
        hosts = [{"host_id": "h1"}, {"host_id": "h2"}]
        eligible = filter_hosts_by_jurisdiction(hosts, {}, JurisdictionConstraint())
        assert len(eligible) == 2


# ── AI Compute Access Fund ───────────────────────────────────────────


class TestComputeFundEligibility:
    """AI Compute Access Fund — 67% CA, 50% international."""

    def test_canadian_67pct(self):
        result = compute_fund_eligible_amount(100.0, is_canadian_compute=True)
        assert abs(result["reimbursable_amount_cad"] - 100 * FUND_CANADIAN_RATE) < 0.01

    def test_non_canadian_50pct_before_cutoff(self):
        result = compute_fund_eligible_amount(
            100.0, is_canadian_compute=False,
            timestamp=FUND_NON_CANADIAN_CUTOFF - 86400,
        )
        assert abs(result["reimbursable_amount_cad"] - 100 * FUND_NON_CANADIAN_RATE) < 0.01

    def test_non_canadian_zero_after_cutoff(self):
        result = compute_fund_eligible_amount(
            100.0, is_canadian_compute=False,
            timestamp=FUND_NON_CANADIAN_CUTOFF + 86400,
        )
        assert result["reimbursable_amount_cad"] == 0.0
        assert result["fund_eligible"] is False

    def test_zero_amount(self):
        result = compute_fund_eligible_amount(0.0, is_canadian_compute=True)
        assert result["reimbursable_amount_cad"] == 0.0

    def test_effective_cost_is_reduced(self):
        result = compute_fund_eligible_amount(100.0, is_canadian_compute=True)
        assert result["effective_cost_cad"] < 100.0


# ── Residency Trace ──────────────────────────────────────────────────


class TestResidencyTrace:
    """Workload residency trace for compliance reporting."""

    def test_trace_with_jurisdiction(self):
        jur = HostJurisdiction(
            host_id="h1", country="CA", province="ON",
            city="Toronto", data_center_name="Equinix TR1",
            operator_name="Xcelsior Inc",
            operator_incorporated_in="CA",
            operator_registered_canada=True,
        )
        now = time.time()
        trace = generate_residency_trace("j1", "h1", jur, now - 3600, now)
        assert trace["country"] == "CA"
        assert trace["province"] == "ON"
        assert trace["is_canadian_compute"] is True
        assert trace["duration_sec"] == pytest.approx(3600, abs=1)

    def test_trace_without_jurisdiction(self):
        trace = generate_residency_trace("j2", "h2", None, 0, 100)
        assert trace["country"] == "unknown"
        assert trace["is_canadian_compute"] is False


# ── Province Enum ─────────────────────────────────────────────────────


class TestProvinceEnum:
    """All 13 provinces/territories in the enum."""

    def test_13_members(self):
        assert len(Province) == 13

    def test_ontario_present(self):
        assert Province.ON.value == "ON"

    def test_quebec_present(self):
        assert Province.QC.value == "QC"
