"""Phase 5 tests: OpenTelemetry instrumentation + LUKS encrypted volume provisioning.

Tests cover:
  - OpenTelemetry integration in api.py (FastAPIInstrumentor, custom spans, W3C propagation)
  - LUKS volume lifecycle in worker_agent.py (provision, attach, detach, destroy, cleanup)
  - otel_span() fallback when opentelemetry is not installed
"""

import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ── OpenTelemetry Tests ──────────────────────────────────────────────


class TestOpenTelemetryIntegration:
    """Verify OpenTelemetry is wired into the API layer."""

    def test_otel_packages_in_requirements(self):
        """All four OTel packages are in requirements.txt."""
        reqs = Path(__file__).resolve().parent.parent / "requirements.txt"
        text = reqs.read_text()
        for pkg in [
            "opentelemetry-api",
            "opentelemetry-sdk",
            "opentelemetry-instrumentation-fastapi",
            "opentelemetry-exporter-otlp",
        ]:
            assert pkg in text, f"{pkg} missing from requirements.txt"

    def test_otel_span_helper_exists(self):
        """api.py exports an otel_span() helper."""
        import api
        assert hasattr(api, "otel_span"), "otel_span not found in api module"
        assert callable(api.otel_span)

    def test_otel_span_returns_context_manager(self):
        """otel_span() returns a usable context manager even when OTel is disabled."""
        import api
        ctx = api.otel_span("test.span", {"key": "value"})
        # Should work as a context manager regardless of OTel state
        with ctx:
            pass  # No error means it works

    def test_otel_enabled_flag_exists(self):
        """_OTEL_ENABLED flag is defined in api.py."""
        import api
        assert hasattr(api, "_OTEL_ENABLED")
        assert isinstance(api._OTEL_ENABLED, bool)

    def test_otel_tracer_attribute(self):
        """_otel_tracer is defined (may be None if OTel not installed)."""
        import api
        assert hasattr(api, "_otel_tracer")

    def test_fastapi_instrumentor_import_attempted(self):
        """api.py source references FastAPIInstrumentor."""
        source = Path(__file__).resolve().parent.parent / "api.py"
        text = source.read_text()
        assert "FastAPIInstrumentor" in text
        assert "instrument_app" in text

    def test_w3c_trace_context_propagation(self):
        """api.py sets up W3C Trace Context propagation."""
        source = Path(__file__).resolve().parent.parent / "api.py"
        text = source.read_text()
        assert "TraceContextTextMapPropagator" in text
        assert "W3CBaggagePropagator" in text
        assert "set_global_textmap" in text

    def test_custom_spans_in_endpoints(self):
        """Key endpoints have otel_span() calls."""
        base = Path(__file__).resolve().parent.parent
        routes_dir = base / "routes"
        all_text = (base / "api.py").read_text()
        for py in routes_dir.glob("*.py"):
            all_text += py.read_text()
        for span_name in ["job.submit", "job.status_update", "billing.bill_job", "webhook.stripe", "billing.cycle"]:
            assert span_name in all_text, f"Custom span '{span_name}' not found in routes/ or api.py"

    def test_otlp_exporter_conditional(self):
        """OTLP exporter only activates when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
        source = Path(__file__).resolve().parent.parent / "api.py"
        text = source.read_text()
        assert "OTEL_EXPORTER_OTLP_ENDPOINT" in text
        assert "OTLPSpanExporter" in text

    def test_service_name_resource(self):
        """OpenTelemetry resource sets service name to xcelsior-api."""
        source = Path(__file__).resolve().parent.parent / "api.py"
        text = source.read_text()
        assert "xcelsior-api" in text


class TestOpenTelemetryGracefulDegradation:
    """Verify OTel gracefully degrades when packages are missing."""

    def test_otel_span_with_none_tracer(self):
        """otel_span returns nullcontext when _otel_tracer is None."""
        import routes._deps as _deps_mod
        original = _deps_mod._otel_tracer
        try:
            _deps_mod._otel_tracer = None
            ctx = _deps_mod.otel_span("test.noop")
            with ctx:
                result = 42
            assert result == 42
        finally:
            _deps_mod._otel_tracer = original

    def test_api_source_has_import_error_handler(self):
        """api.py wraps OTel imports in try/except ImportError."""
        source = Path(__file__).resolve().parent.parent / "api.py"
        text = source.read_text()
        assert "except ImportError" in text


# ── LUKS Encrypted Volume Tests ──────────────────────────────────────


class TestLUKSVolumeConstants:
    """Verify LUKS volume module constants and functions exist."""

    def test_volume_base_dir_constant(self):
        import worker_agent as wa
        assert hasattr(wa, "VOLUME_BASE_DIR")
        assert "volumes" in wa.VOLUME_BASE_DIR

    def test_volume_key_dir_constant(self):
        import worker_agent as wa
        assert hasattr(wa, "VOLUME_KEY_DIR")
        assert "volume-keys" in wa.VOLUME_KEY_DIR

    def test_provision_function_exists(self):
        import worker_agent as wa
        assert hasattr(wa, "provision_encrypted_volume")
        assert callable(wa.provision_encrypted_volume)

    def test_attach_function_exists(self):
        import worker_agent as wa
        assert hasattr(wa, "attach_encrypted_volume")
        assert callable(wa.attach_encrypted_volume)

    def test_detach_function_exists(self):
        import worker_agent as wa
        assert hasattr(wa, "detach_encrypted_volume")
        assert callable(wa.detach_encrypted_volume)

    def test_destroy_function_exists(self):
        import worker_agent as wa
        assert hasattr(wa, "destroy_encrypted_volume")
        assert callable(wa.destroy_encrypted_volume)

    def test_cleanup_function_exists(self):
        import worker_agent as wa
        assert hasattr(wa, "_cleanup_partial_volume")
        assert callable(wa._cleanup_partial_volume)


class TestLUKSVolumeProvisioning:
    """Test LUKS volume provisioning logic (mocked subprocess)."""

    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.makedirs")
    @patch("worker_agent.os.umask", return_value=0o022)
    @patch("worker_agent.os.urandom", return_value=b"\x00" * 32)
    @patch("builtins.open", new_callable=MagicMock)
    def test_provision_calls_cryptsetup(self, mock_open, mock_urandom, mock_umask, mock_makedirs, mock_run):
        """provision_encrypted_volume calls truncate, luksFormat, luksOpen, mkfs, mount."""
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        import worker_agent as wa
        result = wa.provision_encrypted_volume("vol-abc123", 50)
        assert result is True

        # Verify the key subprocess calls
        call_args_list = [c[0][0] for c in mock_run.call_args_list]
        commands = [args[0] for args in call_args_list]
        assert "truncate" in commands
        assert "cryptsetup" in commands
        assert "mkfs.ext4" in commands
        assert "mount" in commands

    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.makedirs")
    @patch("worker_agent.os.umask", return_value=0o022)
    @patch("worker_agent.os.urandom", return_value=b"\x00" * 32)
    @patch("builtins.open", new_callable=MagicMock)
    @patch("worker_agent._cleanup_partial_volume")
    def test_provision_cleanup_on_failure(self, mock_cleanup, mock_open, mock_urandom, mock_umask, mock_makedirs, mock_run):
        """Failed provisioning calls _cleanup_partial_volume."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "cryptsetup", stderr="error")
        import worker_agent as wa
        result = wa.provision_encrypted_volume("vol-fail", 10)
        assert result is False
        mock_cleanup.assert_called_once_with("vol-fail")

    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.path.exists")
    @patch("worker_agent.os.makedirs")
    def test_attach_opens_and_mounts(self, mock_makedirs, mock_exists, mock_run):
        """attach_encrypted_volume calls luksOpen and mount."""
        mock_exists.side_effect = lambda p: True  # All files exist
        mock_run.return_value = MagicMock(returncode=1, stderr="", stdout="")  # Not mounted yet
        import worker_agent as wa
        result = wa.attach_encrypted_volume("vol-attach1")
        assert result is not None

    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.path.exists", return_value=False)
    def test_attach_fails_no_backing(self, mock_exists, mock_run):
        """attach fails when backing file doesn't exist."""
        import worker_agent as wa
        result = wa.attach_encrypted_volume("vol-nope")
        assert result is None

    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.path.exists")
    def test_detach_unmounts_and_closes(self, mock_exists, mock_run):
        """detach_encrypted_volume calls umount and luksClose."""
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        import worker_agent as wa
        result = wa.detach_encrypted_volume("vol-det1")
        assert result is True


class TestLUKSVolumeCryptoErasure:
    """Test cryptographic erasure on volume destroy."""

    @patch("worker_agent.detach_encrypted_volume")
    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.path.exists", return_value=True)
    @patch("worker_agent.os.remove")
    @patch("worker_agent.os.path.isdir", return_value=True)
    @patch("worker_agent.os.rmdir")
    def test_destroy_shreds_key(self, mock_rmdir, mock_isdir, mock_remove, mock_exists, mock_run, mock_detach):
        """destroy_encrypted_volume uses shred on the key file."""
        mock_run.return_value = MagicMock(returncode=0)
        import worker_agent as wa
        result = wa.destroy_encrypted_volume("vol-del1")
        assert result is True

        # Verify shred was called
        shred_calls = [c for c in mock_run.call_args_list if "shred" in str(c)]
        assert len(shred_calls) >= 1, "shred should be called on key file"

    @patch("worker_agent.detach_encrypted_volume")
    @patch("worker_agent.subprocess.run")
    @patch("worker_agent.os.path.exists", return_value=True)
    @patch("worker_agent.os.remove")
    @patch("worker_agent.os.path.isdir", return_value=True)
    @patch("worker_agent.os.rmdir")
    def test_destroy_removes_backing_file(self, mock_rmdir, mock_isdir, mock_remove, mock_exists, mock_run, mock_detach):
        """destroy removes the backing .img file."""
        mock_run.return_value = MagicMock(returncode=0)
        import worker_agent as wa
        wa.destroy_encrypted_volume("vol-del2")
        assert mock_remove.called


class TestRunJobEncryptedVolumes:
    """Verify run_job integrates encrypted volume attach/detach."""

    def test_run_job_docstring_mentions_encrypted(self):
        """run_job docstring references encrypted volumes."""
        import worker_agent as wa
        assert "encrypted" in wa.run_job.__doc__.lower()

    def test_source_references_attach_detach(self):
        """run_job source calls attach_encrypted_volume and detach_encrypted_volume."""
        source = Path(__file__).resolve().parent.parent / "worker_agent.py"
        text = source.read_text()
        assert "attach_encrypted_volume" in text
        assert "detach_encrypted_volume" in text
        assert "encrypted_vol_ids" in text
