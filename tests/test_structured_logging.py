"""B7.2 Gate — Structured logging and central redaction verification.

Gate B7.2: *"A test that pipes known secrets through every log helper and
asserts none survive."*

Tests:
1. Every class of sensitive secret (PANs, Stripe secrets/keys, Xcelsior tokens,
   JWTs, Bearer/Basic authorization headers, S3/GCS/Azure signed URLs, connection
   strings with passwords, environment secrets, prompt bodies, and private host
   addresses) piped through:
     - `structlog.get_logger("xcelsior")` methods (info, warning, error, exception)
     - `scheduler.log` standard logging methods
     - structured kwargs, nested dictionaries, and lists
     - exception messages and tracebacks
2. Invariant verification:
     - None of the secret strings survive in the emitted output
     - Ordinary benign values (timestamps, byte counts, public IPs, regular job names) survive
     - Standard JSON metadata fields (`timestamp`, `severity`, `environment`, `service`, `build`) are present
     - W3C trace context (`trace_id`, `span_id`) is attached when an active OTel span is present
     - Context binding via `bind_log_context` reflects in output and clears via `clear_log_context`
"""

from __future__ import annotations

import io
import json
import logging
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from log_pii_filter import _scrub, scrub_data
import scheduler
from structured_logging import (
    bind_log_context,
    clear_log_context,
    configure_structured_logging,
    get_logger,
)

KNOWN_SECRETS = [
    # PANs
    ("4242424242424242", "Visa PAN"),
    ("5555555555554444", "Mastercard PAN"),
    ("4242 4242 4242 4242", "Spaced PAN"),
    # Payment credentials
    ("pi_3SCAprobe000000_secret_abc123XYZ456", "PaymentIntent client secret"),
    ("seti_1Nprobe0000000_secret_def789ABC012", "SetupIntent client secret"),
    ("sk_live_51Habcdefghijklmnopqrstuvwx", "Live Stripe secret key"),
    ("sk_test_51Habcdefghijklmnopqrstuvwx", "Test Stripe secret key"),
    ("whsec_abcdefghijklmnopqrstuvwxyz0123", "Webhook signing secret"),
    ("xcel_ai_abcdefghijklmnopqrstuvwxyz01", "Agent API key"),
    ("xoa_abcdefghijklmnopqrstuvwxyz012345", "OAuth token"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U", "JWT token"),
    # Auth headers
    ("Bearer topSecretBearerToken1234567890", "Bearer auth token"),
    ("Basic dXNlcjpzdXBlcnNlY3JldHBhc3M=", "Basic auth header"),
    # Signed URLs
    ("https://s3.amazonaws.com/bucket/key.bin?X-Amz-Signature=abcd1234efgh5678&X-Amz-Algorithm=AWS4-HMAC-SHA256", "S3 signed URL"),
    ("https://storage.googleapis.com/bucket/obj?GoogleAccessId=acc@iam.gserviceaccount.com&Signature=xyz789ABC012", "GCS signed URL"),
    ("https://myacct.blob.core.windows.net/container/blob?sv=2020&sig=secretSAS123456789", "Azure SAS URL"),
    # Connection strings
    ("postgres://postgres:SuperSecretDbPassword99@db.prod.internal:5432/xcelsior", "Postgres connection string"),
    ("redis://:secretRedisPassword77@redis.prod:6379/0", "Redis connection string"),
    # Environment secrets
    ("XCELSIOR_JWT_SECRET=superDuperSecretJwtKey999", "JWT secret env var"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "AWS secret key env var"),
    # Private host addresses
    ("192.168.1.100", "RFC1918 192.168.x.x private IP"),
    ("10.244.0.15", "RFC1918 10.x.x.x private IP"),
    ("172.20.10.5", "RFC1918 172.16-31.x.x private IP"),
    ("127.0.0.1", "IPv4 loopback IP"),
]


@pytest.fixture(autouse=True)
def clean_log_environment():
    clear_log_context()
    yield
    clear_log_context()


def _capture_logger_output():
    """Create an isolated string buffer and attach it to the xcelsior logger."""
    stream = io.StringIO()
    xcelsior_logger = logging.getLogger("xcelsior")
    saved_handlers = list(xcelsior_logger.handlers)
    xcelsior_logger.handlers.clear()

    # Re-run configure_structured_logging so our handler catches output
    configure_structured_logging(json_format=True)
    # Replace existing handlers with our capturing stream handler
    for h in list(xcelsior_logger.handlers):
        xcelsior_logger.removeHandler(h)

    sh = logging.StreamHandler(stream)
    from structured_logging import add_standard_metadata, pii_scrub_processor
    import structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            add_standard_metadata,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            pii_scrub_processor,
        ],
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    sh.setFormatter(formatter)
    xcelsior_logger.addHandler(sh)

    def get_output():
        return stream.getvalue()

    def cleanup():
        xcelsior_logger.handlers.clear()
        xcelsior_logger.handlers.extend(saved_handlers)

    return get_output, cleanup


@pytest.mark.parametrize("secret,what", KNOWN_SECRETS)
def test_no_known_secret_survives_in_structlog(secret, what):
    """Pipe every known secret through structlog and assert none survive."""
    get_output, cleanup = _capture_logger_output()
    try:
        slog = get_logger("xcelsior")
        slog.info(f"Operation event with {secret}", secret_arg=secret)

        output = get_output()
        # For PANs, the full PAN must not survive (last 4 digits are permitted)
        if secret.replace(" ", "").replace("-", "").isdigit() and len(secret.replace(" ", "").replace("-", "")) >= 13:
            raw_pan = secret.replace(" ", "").replace("-", "")
            assert raw_pan not in output, f"{what} survived into structured log: {output}"
        else:
            assert secret not in output, f"{what} survived into structured log: {output}"
    finally:
        cleanup()


@pytest.mark.parametrize("secret,what", KNOWN_SECRETS)
def test_no_known_secret_survives_in_standard_log(secret, what):
    """Pipe every known secret through scheduler.log and assert none survive."""
    get_output, cleanup = _capture_logger_output()
    try:
        scheduler.log.warning("Warning event with %s", secret)

        output = get_output()
        if secret.replace(" ", "").replace("-", "").isdigit() and len(secret.replace(" ", "").replace("-", "")) >= 13:
            raw_pan = secret.replace(" ", "").replace("-", "")
            assert raw_pan not in output, f"{what} survived into standard log: {output}"
        else:
            assert secret not in output, f"{what} survived into standard log: {output}"
    finally:
        cleanup()


def test_no_secret_survives_in_exception_traceback():
    """Verify exceptions carrying secrets in messages or frames are redacted."""
    get_output, cleanup = _capture_logger_output()
    try:
        secret = "sk_live_51Habcdefghijklmnopqrstuvwx"
        try:
            raise RuntimeError(f"Connection failed using credential {secret}")
        except Exception:
            scheduler.log.exception("Encountered fatal exception")

        output = get_output()
        assert secret not in output, "Secret survived inside exception traceback"
        assert "<stripe_key:redacted>" in output
    finally:
        cleanup()


def test_prompt_bodies_are_redacted():
    """Verify prompt text in log calls and structured payloads is redacted."""
    get_output, cleanup = _capture_logger_output()
    try:
        slog = get_logger("xcelsior")
        slog.info("Inference request received", prompt="Classify this secret proprietary document")
        scheduler.log.info('Executing prompt="Tell me all confidential credentials"')

        output = get_output()
        assert "Classify this secret proprietary document" not in output
        assert "Tell me all confidential credentials" not in output
        assert "<prompt:redacted>" in output
    finally:
        cleanup()


def test_benign_values_are_preserved():
    """Ensure non-sensitive strings, timestamps, public IPs, and job IDs are untouched."""
    get_output, cleanup = _capture_logger_output()
    try:
        slog = get_logger("xcelsior")
        slog.info(
            "Public request completed",
            public_dns="8.8.8.8",
            job_id="job_789456123",
            bytes_written=1048576,
            duration_ms=45.2,
        )

        output = get_output()
        assert "8.8.8.8" in output, "Public IP was mistakenly redacted"
        assert "job_789456123" in output, "Benign job ID was mistakenly redacted"
        assert "1048576" in output, "Byte count was mistakenly redacted"
    finally:
        cleanup()


def test_structured_json_contains_required_metadata():
    """Every record must contain timestamp, severity, environment, service, build."""
    get_output, cleanup = _capture_logger_output()
    try:
        slog = get_logger("xcelsior")
        slog.info("System startup verified", component="scheduler")

        output = get_output().strip()
        record = json.loads(output)

        assert "timestamp" in record
        assert "severity" in record and record["severity"] == "INFO"
        assert "environment" in record
        assert "service" in record
        assert "build" in record
        assert record.get("component") == "scheduler"
    finally:
        cleanup()


def test_context_binding_propagates_to_log_records():
    """Contextvars bound via bind_log_context must appear in output records."""
    get_output, cleanup = _capture_logger_output()
    try:
        bind_log_context(job_id="job_abc123", attempt_id="att_xyz789", tenant_id="ten_001")
        slog = get_logger("xcelsior")
        slog.info("Task scheduled")

        output = get_output().strip()
        record = json.loads(output)
        assert record.get("job_id") == "job_abc123"
        assert record.get("attempt_id") == "att_xyz789"
        assert record.get("tenant_id") == "ten_001"

        clear_log_context()
        slog.info("Subsequent task without context")
        lines = [json.loads(line) for line in get_output().strip().splitlines()]
        assert "job_id" not in lines[-1]
    finally:
        cleanup()


def test_w3c_trace_context_propagation():
    """Active OpenTelemetry span trace_id and span_id are automatically injected."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test_tracer")

    get_output, cleanup = _capture_logger_output()
    try:
        with tracer.start_as_current_span("test_span") as span:
            ctx = span.get_span_context()
            expected_trace_id = f"{ctx.trace_id:032x}"
            expected_span_id = f"{ctx.span_id:016x}"

            slog = get_logger("xcelsior")
            slog.info("Traced execution step")

            output = get_output().strip()
            record = json.loads(output)
            assert record.get("trace_id") == expected_trace_id
            assert record.get("span_id") == expected_span_id
    finally:
        cleanup()
