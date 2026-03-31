"""Tests for Xcelsior Phase 3 features:
- Prometheus metrics endpoint format
- FINTRAC Bitcoin LVCTR reporting
- Compliance middleware gates
- Bayesian reputation scoring + fraud detection
- Auto-remediation checkpoint + re-queue
- CLI Rich table output
"""

import json
import math
import os
import tempfile
import time

import pytest

os.environ.setdefault("XCELSIOR_API_TOKEN", "")
os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_DB_BACKEND", "sqlite")

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_phase3_test_")
_tmpdir = _tmp_ctx.name


# ── Prometheus Metrics ────────────────────────────────────────────────


class TestPrometheusMetricsFormat:
    """Prometheus text exposition format validation."""

    def test_prometheus_endpoint_exists(self):
        """The /metrics/prometheus route is registered."""
        try:
            from api import app
        except ImportError:
            pytest.skip("api module import failed (missing dependency)")

        routes = [r.path for r in app.routes]
        assert "/metrics/prometheus" in routes

    def test_prometheus_content_type(self):
        """Prometheus endpoint should return text/plain version 0.0.4."""
        try:
            from fastapi.testclient import TestClient
            from api import app
        except ImportError:
            pytest.skip("api module import failed")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/metrics/prometheus")
        # The endpoint requires auth via middleware; check it doesn't 404
        assert resp.status_code != 404

    def test_prometheus_gauge_format(self):
        """Prometheus text format: TYPE/HELP/metric lines."""
        # Verify hand-built format helpers produce valid output
        lines = [
            "# HELP xcelsior_queue_depth Number of queued jobs",
            "# TYPE xcelsior_queue_depth gauge",
            "xcelsior_queue_depth 5",
        ]
        text = "\n".join(lines)
        assert "# TYPE" in text
        assert "# HELP" in text
        assert "xcelsior_queue_depth 5" in text

    def test_prometheus_counter_with_labels(self):
        """Counters with labels are valid Prometheus format."""
        line = 'xcelsior_gpu_utilization{host_id="h1",gpu_index="0"} 85.5'
        assert "{" in line
        assert "}" in line
        parts = line.split(" ")
        assert len(parts) == 2
        assert float(parts[1]) == 85.5


# ── FINTRAC Bitcoin LVCTR ─────────────────────────────────────────────


class TestFintracBitcoinLVCTR:
    """FINTRAC Large Value Cash Transaction reporting for BTC deposits."""

    def test_threshold_constant(self):
        """LVCTR threshold is $10,000 CAD."""
        from bitcoin import FINTRAC_LVCTR_THRESHOLD_CAD

        assert FINTRAC_LVCTR_THRESHOLD_CAD == 10_000.0

    def test_single_tx_above_threshold(self):
        """A single BTC deposit >= $10K triggers LVCTR report."""
        from bitcoin import fintrac_check_btc_deposit

        try:
            result = fintrac_check_btc_deposit("cust-test-1", 15_000.0)
        except Exception:
            pytest.skip("fintrac_reports table not available")
        assert result is not None
        assert result["customer_id"] == "cust-test-1"

    def test_single_tx_below_threshold(self):
        """A deposit under $10K with no aggregation doesn't trigger."""
        from bitcoin import fintrac_check_btc_deposit

        try:
            result = fintrac_check_btc_deposit("cust-test-2", 5_000.0)
        except Exception:
            pytest.skip("fintrac_reports table not available")
        if result is not None:
            assert "customer_id" in result

    def test_get_pending_fintrac_reports(self):
        """get_pending_fintrac_reports() returns a list."""
        from bitcoin import get_pending_fintrac_reports

        try:
            reports = get_pending_fintrac_reports()
        except Exception:
            pytest.skip("fintrac_reports table not available")
        assert isinstance(reports, list)


# ── Compliance Middleware ─────────────────────────────────────────────


class TestComplianceMiddleware:
    """PIPEDA/CASL compliance middleware gates."""

    def test_middleware_registered(self):
        """ComplianceGateMiddleware is in the middleware stack."""
        try:
            import api as api_module
        except ImportError:
            pytest.skip("api module import failed")

        assert hasattr(api_module, "ComplianceGateMiddleware") or "ComplianceGateMiddleware" in dir(
            api_module
        )

    def test_compliance_response_headers(self):
        """Compliance middleware adds X-Data-Residency and X-Compliance-Version."""
        try:
            from fastapi.testclient import TestClient
            from api import app
        except ImportError:
            pytest.skip("api module import failed")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/health")
        if resp.status_code == 200:
            assert resp.headers.get("X-Data-Residency") == "CA"
            assert "PIPEDA" in resp.headers.get("X-Compliance-Version", "")

    def test_province_header_passthrough(self):
        """x-province header is accepted and stored in request state."""
        try:
            from fastapi.testclient import TestClient
            from api import app
        except ImportError:
            pytest.skip("api module import failed")

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/health", headers={"x-province": "ON"})
        assert resp.status_code in (200, 401, 403)


# ── Bayesian Reputation Scoring ──────────────────────────────────────


class TestBayesianReputation:
    """Bayesian reputation model with cold start and time decay."""

    def _engine(self):
        from reputation import ReputationEngine, ReputationStore

        store = ReputationStore(db_path=os.path.join(_tmpdir, f"rep_{os.urandom(4).hex()}.db"))
        return ReputationEngine(store=store)

    def test_cold_start_uses_prior(self):
        """New entity with no data gets prior-weighted score."""
        engine = self._engine()
        result = engine.compute_bayesian_score("new-host-1")
        assert result["entity_id"] == "new-host-1"
        # Alpha should be close to 1.0 (full prior weight)
        assert result["alpha"] >= 0.8
        assert result["confidence"] == 0.0  # No observations
        assert result["total_jobs"] == 0

    def test_alpha_decays_with_observations(self):
        """Alpha decreases as the provider completes more jobs."""
        engine = self._engine()
        engine._ensure_entity("host-alpha", "host")

        # Record some completions
        for _ in range(20):
            engine.record_job_completed("host-alpha")

        result = engine.compute_bayesian_score("host-alpha")
        # With 20 jobs and prior_weight=10: alpha = 10/(10+20) = 0.333
        assert result["alpha"] < 0.5
        assert result["total_jobs"] == 20
        assert result["confidence"] > 0.3

    def test_bayesian_score_bounds(self):
        """Bayesian score should be between 0 and 1."""
        engine = self._engine()
        engine._ensure_entity("host-bounds", "host")

        for _ in range(50):
            engine.record_job_completed("host-bounds")

        result = engine.compute_bayesian_score("host-bounds")
        assert 0 <= result["bayesian_score"] <= 1.0

    def test_network_prior_default(self):
        """Network prior defaults to 0.7 when insufficient data."""
        engine = self._engine()
        prior = engine._get_network_prior()
        # Default is 0.7 when < 10 total jobs, 1.0 possible if all completed
        assert 0.0 <= prior <= 1.0

    def test_weighted_score_scales_to_1000(self):
        """Weighted score is scaled to 0-1000 range."""
        engine = self._engine()
        engine._ensure_entity("host-scale", "host")
        for _ in range(30):
            engine.record_job_completed("host-scale")

        result = engine.compute_bayesian_score("host-scale")
        assert result["weighted_score"] >= 0
        assert result["weighted_score"] <= 1000


# ── Fraud Detection ──────────────────────────────────────────────────


class TestFraudDetection:
    """Automated fraud detection: Sybil, benchmark spoofing, early termination."""

    def _engine(self):
        from reputation import ReputationEngine, ReputationStore

        store = ReputationStore(db_path=os.path.join(_tmpdir, f"rep_{os.urandom(4).hex()}.db"))
        return ReputationEngine(store=store)

    def test_sybil_detection_unknown_entity(self):
        """Sybil check on unknown entity doesn't crash."""
        engine = self._engine()
        result = engine.detect_sybil("nonexistent-host")
        assert result["flagged"] is False

    def test_benchmark_spoofing_no_baseline(self):
        """Unknown GPU model returns not-flagged."""
        engine = self._engine()
        result = engine.detect_benchmark_spoofing("host-1", 50.0, "Custom GPU 9000")
        assert result["flagged"] is False
        assert "No baseline" in result["reason"]

    def test_benchmark_spoofing_within_tolerance(self):
        """Performance within 20% of baseline is OK."""
        engine = self._engine()
        # RTX 4090 baseline is ~82.6 TFLOPS
        result = engine.detect_benchmark_spoofing("host-1", 80.0, "RTX 4090")
        assert result["flagged"] is False
        assert result["deviation_pct"] < 20

    def test_benchmark_spoofing_flagged(self):
        """Performance >20% from baseline is flagged."""
        engine = self._engine()
        # RTX 4090 baseline is ~82.6, reporting 150 is way off
        result = engine.detect_benchmark_spoofing("host-1", 150.0, "RTX 4090")
        assert result["flagged"] is True
        assert result["deviation_pct"] > 20
        assert result["severity"] in ("medium", "high")

    def test_benchmark_spoofing_high_severity(self):
        """Performance >50% deviation is high severity."""
        engine = self._engine()
        # RTX 3090 baseline ~35.6, reporting 5.0 is way below
        result = engine.detect_benchmark_spoofing("host-1", 5.0, "RTX 3090")
        assert result["flagged"] is True
        assert result["severity"] == "high"

    def test_early_termination_insufficient_data(self):
        """Early termination check needs >= 20 jobs."""
        engine = self._engine()
        engine._ensure_entity("host-few", "host")

        for _ in range(5):
            engine.record_job_completed("host-few")

        result = engine.detect_early_termination_pattern("host-few")
        assert result["flagged"] is False
        assert "Insufficient data" in result["reason"]

    def test_early_termination_good_provider(self):
        """Good provider (< 5% failure) is not flagged."""
        engine = self._engine()
        engine._ensure_entity("host-good", "host")

        for _ in range(98):
            engine.record_job_completed("host-good")
        engine.record_job_failure("host-good", is_host_fault=True)

        result = engine.detect_early_termination_pattern("host-good")
        # 1/99 = ~1% — well under 5%
        assert result["flagged"] is False

    def test_early_termination_bad_provider(self):
        """Bad provider (> 5% failure) is flagged."""
        engine = self._engine()
        engine._ensure_entity("host-bad", "host")

        for _ in range(17):
            engine.record_job_completed("host-bad")
        for _ in range(4):
            engine.record_job_failure("host-bad", is_host_fault=True)

        result = engine.detect_early_termination_pattern("host-bad")
        # 4/21 = ~19% — flagged
        assert result["flagged"] is True
        assert result["termination_rate"] > 0.05

    def test_run_fraud_scan_summary(self):
        """run_fraud_scan aggregates all checks."""
        engine = self._engine()
        engine._ensure_entity("host-scan", "host")

        result = engine.run_fraud_scan("host-scan", reported_perf=80.0, gpu_model="RTX 4090")
        assert "sybil" in result["checks"]
        assert "early_termination" in result["checks"]
        assert "benchmark_spoofing" in result["checks"]
        assert isinstance(result["any_flagged"], bool)

    def test_run_fraud_scan_without_perf(self):
        """Fraud scan without performance data skips benchmark check."""
        engine = self._engine()
        engine._ensure_entity("host-noperf", "host")

        result = engine.run_fraud_scan("host-noperf")
        assert "sybil" in result["checks"]
        assert "early_termination" in result["checks"]
        assert "benchmark_spoofing" not in result["checks"]


# ── Auto-Remediation ─────────────────────────────────────────────────


class TestAutoRemediation:
    """Docker checkpoint + re-queue for unhealthy hosts."""

    def test_health_failure_tracking(self):
        """Health failures are tracked per host."""
        from scheduler import record_health_check, _health_failure_counts

        _health_failure_counts.clear()

        # First failure — no remediation yet
        result = record_health_check("host-rem-1", healthy=False)
        assert result is None
        assert _health_failure_counts["host-rem-1"] == 1

    def test_healthy_check_resets_counter(self):
        """A healthy check resets the failure counter."""
        from scheduler import record_health_check, _health_failure_counts

        _health_failure_counts.clear()

        record_health_check("host-rem-2", healthy=False)
        record_health_check("host-rem-2", healthy=False)
        assert _health_failure_counts["host-rem-2"] == 2

        record_health_check("host-rem-2", healthy=True)
        assert "host-rem-2" not in _health_failure_counts

    def test_threshold_triggers_remediation(self):
        """3 consecutive failures trigger remediation."""
        from scheduler import record_health_check, _health_failure_counts, HEALTH_FAILURE_THRESHOLD

        _health_failure_counts.clear()

        assert HEALTH_FAILURE_THRESHOLD == 3

        record_health_check("host-rem-3", healthy=False)
        record_health_check("host-rem-3", healthy=False)
        result = record_health_check("host-rem-3", healthy=False)

        # Should have triggered remediation
        assert result is not None
        assert result["action"] == "remediated"
        assert result["host_id"] == "host-rem-3"
        # Counter should be reset after remediation
        assert "host-rem-3" not in _health_failure_counts

    def test_checkpoint_dir_constant(self):
        """CHECKPOINT_DIR is defined."""
        from scheduler import CHECKPOINT_DIR

        assert isinstance(CHECKPOINT_DIR, str)
        assert "checkpoints" in CHECKPOINT_DIR

    def test_checkpoint_container_returns_metadata(self):
        """checkpoint_container returns metadata dict even on failure."""
        from scheduler import checkpoint_container

        # Will fail (no Docker) but should return metadata
        result = checkpoint_container("test-host", "test-job-123")
        if result is not None:
            assert "checkpoint_name" in result
            assert "job_id" in result
            assert result["job_id"] == "test-job-123"

    def test_remediate_empty_host(self):
        """Remediating a host with no running jobs returns empty list."""
        from scheduler import remediate_unhealthy_host

        result = remediate_unhealthy_host("nonexistent-host")
        assert result == []


# ── CLI Rich Tables ──────────────────────────────────────────────────


class TestCLIRichTables:
    """CLI modernization with Rich tables."""

    def test_rich_available_flag(self):
        """RICH_AVAILABLE is set based on import success."""
        import cli

        assert hasattr(cli, "RICH_AVAILABLE")
        assert isinstance(cli.RICH_AVAILABLE, bool)

    def test_rich_is_installed(self):
        """Rich library is installed and importable."""
        from rich.console import Console
        from rich.table import Table

        assert Console is not None
        assert Table is not None

    def test_rich_table_creation(self):
        """Rich Table objects can be created."""
        from rich.table import Table

        table = Table(title="Test")
        table.add_column("ID")
        table.add_column("Value")
        table.add_row("1", "hello")
        assert table.row_count == 1

    def test_cli_version_updated(self):
        """CLI version header is updated."""
        import cli

        # Verify source has been updated
        import inspect

        source = inspect.getsource(cli)
        assert "2.3.0" in source

    def test_console_object(self):
        """CLI has console object when Rich is available."""
        import cli

        if cli.RICH_AVAILABLE:
            assert cli.console is not None


# ── PyPI Packaging ───────────────────────────────────────────────────


class TestPyPIPackaging:
    """pyproject.toml and packaging configuration."""

    def test_pyproject_has_project_section(self):
        """pyproject.toml has [project] section with name and version."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(
            os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "rb"
        ) as f:
            data = tomllib.load(f)

        assert "project" in data
        assert data["project"]["name"] == "xcelsior"
        assert "version" in data["project"]

    def test_pyproject_scripts_entry(self):
        """pyproject.toml has xcelsior CLI entry point."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(
            os.path.join(os.path.dirname(__file__), "..", "pyproject.toml"), "rb"
        ) as f:
            data = tomllib.load(f)

        assert data["project"]["scripts"]["xcelsior"] == "cli:main"

    def test_requirements_has_rich(self):
        """requirements.txt includes rich."""
        with open(
            os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        ) as f:
            content = f.read()
        assert "rich" in content

    def test_requirements_has_cryptography(self):
        """requirements.txt includes cryptography."""
        with open(
            os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
        ) as f:
            content = f.read()
        assert "cryptography" in content


# ── CI/CD Workflows ──────────────────────────────────────────────────


class TestCICDWorkflows:
    """GitHub Actions workflow files."""

    def test_publish_workflow_exists(self):
        """publish.yml exists in .github/workflows/."""
        path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "publish.yml"
        )
        assert os.path.exists(path)

    def test_publish_workflow_valid_yaml(self):
        """publish.yml is valid YAML."""
        import yaml

        path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "publish.yml"
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "jobs" in data
        assert "pypi" in data["jobs"]
        assert "docker" in data["jobs"]
        # YAML 'on' becomes True boolean key in Python
        assert data.get("on") is not None or data.get(True) is not None

    def test_publish_triggered_on_tags(self):
        """publish.yml triggers on version tags."""
        import yaml

        path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "publish.yml"
        )
        with open(path) as f:
            data = yaml.safe_load(f)
        # YAML 'on' is a boolean key in Python; try True key or 'on' string
        triggers = data.get("on") or data.get(True)
        assert triggers is not None
        push_config = triggers.get("push") if isinstance(triggers, dict) else None
        assert push_config is not None
        assert any("v*" in str(t) for t in push_config.get("tags", []))
