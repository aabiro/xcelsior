"""Live SPIRE mesh: identity binding, gateway config, and readiness.

Blueprint §19.2 (worker identity), §19.3 (agent ingress), §21.3 (health
semantics), §22.3 (agent gateway requirements).

The previous posture accepted *any* SPIFFE-shaped string that happened to
contain ``/worker/`` — including one minted in an attacker-chosen trust
domain — and never bound it to the host it claimed. These tests pin the
replacement contract:

    spiffe://<trust-domain>/worker/host/<host_id>

and prove the three places that must agree on it (the API verifier, the
Envoy gateway, and the SPIRE registration script) actually do.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

from control_plane import identity

ROOT = Path(__file__).resolve().parent.parent
SPIRE = ROOT / "infra" / "spire"
ENVOY = ROOT / "infra" / "envoy" / "agent-gateway.yaml"


# ── The identity contract ─────────────────────────────────────────────


def test_canonical_spiffe_id_shape(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    assert identity.spiffe_id_for_host("gpu-01") == "spiffe://xcelsior.ca/worker/host/gpu-01"
    # Path-unsafe characters are folded deterministically, so issuer and
    # verifier cannot disagree about the same host.
    assert identity.spiffe_id_for_host("gpu/01:a") == "spiffe://xcelsior.ca/worker/host/gpu-01-a"


def test_round_trip_id_matches_its_host(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    for host_id in ("gpu-01", "aaryn-tuf-rtx2060", "host.with.dots", "a_b-c"):
        sid = identity.spiffe_id_for_host(host_id)
        assert identity.spiffe_id_matches_host(sid, host_id), host_id


@pytest.mark.parametrize(
    "spiffe_id,reason",
    [
        # The bug this replaced: any trust domain, as long as the path
        # mentioned /worker/.
        ("spiffe://attacker.example/worker/host/gpu-01", "foreign trust domain"),
        ("spiffe://xcelsior.ca/service/api", "not a worker path"),
        ("spiffe://xcelsior.ca/worker/gpu-01", "missing /host/ segment"),
        ("spiffe://xcelsior.ca/worker/host/", "empty host component"),
        ("spiffe://xcelsior.ca/worker/host/gpu-01/extra", "nested path"),
        ("https://xcelsior.ca/worker/host/gpu-01", "not a spiffe scheme"),
        ("worker/host/gpu-01", "not a URI"),
        ("", "empty"),
    ],
)
def test_non_worker_identities_are_rejected(monkeypatch, spiffe_id, reason):
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    assert identity.parse_worker_spiffe_id(spiffe_id) is None, reason
    assert not identity.spiffe_id_matches_host(spiffe_id, "gpu-01"), reason


def test_a_valid_id_for_one_host_does_not_authenticate_another(monkeypatch):
    """The whole point of binding: SVIDs are not interchangeable."""
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    sid = identity.spiffe_id_for_host("gpu-01")
    assert identity.spiffe_id_matches_host(sid, "gpu-01")
    assert not identity.spiffe_id_matches_host(sid, "gpu-02")


def test_trust_domain_is_configurable_and_enforced(monkeypatch):
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "staging.xcelsior.ca")
    sid = identity.spiffe_id_for_host("gpu-01")
    assert sid.startswith("spiffe://staging.xcelsior.ca/")
    assert identity.spiffe_id_matches_host(sid, "gpu-01")
    # An ID from production must not authenticate against staging.
    assert not identity.spiffe_id_matches_host("spiffe://xcelsior.ca/worker/host/gpu-01", "gpu-01")


# ── Gateway admission through the real resolver ───────────────────────


def _gateway_headers(**extra) -> dict:
    headers = {
        "x-xcelsior-agent-gateway": "1",
        "x-xcelsior-gateway-auth": "s3cr3t",
    }
    headers.update(extra)
    return headers


@pytest.fixture
def gateway_mode(monkeypatch):
    monkeypatch.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", "s3cr3t")
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    monkeypatch.setenv("XCELSIOR_SPIFFE_STRICT", "1")
    return monkeypatch


def test_gateway_identity_requires_a_matching_spiffe_id(gateway_mode):
    good = identity.resolve_gateway_identity(
        _gateway_headers(
            **{
                "x-worker-host-id": "gpu-01",
                "x-worker-spiffe-id": "spiffe://xcelsior.ca/worker/host/gpu-01",
            }
        ),
        required_host_id="gpu-01",
    )
    assert good is not None
    assert good.host_id == "gpu-01"
    assert good.source == "gateway_header"


def test_gateway_rejects_a_spiffe_id_for_a_different_host(gateway_mode):
    """Gateway and request disagreeing about who is calling = fail closed."""
    with pytest.raises(identity.IdentityAdmissionError) as excinfo:
        identity.resolve_gateway_identity(
            _gateway_headers(
                **{
                    "x-worker-host-id": "gpu-01",
                    "x-worker-spiffe-id": "spiffe://xcelsior.ca/worker/host/gpu-99",
                }
            ),
            required_host_id="gpu-01",
        )
    assert excinfo.value.code == "invalid_spiffe"
    assert excinfo.value.http_status == 403


def test_gateway_rejects_a_foreign_trust_domain(gateway_mode):
    with pytest.raises(identity.IdentityAdmissionError) as excinfo:
        identity.resolve_gateway_identity(
            _gateway_headers(
                **{
                    "x-worker-host-id": "gpu-01",
                    "x-worker-spiffe-id": "spiffe://evil.example/worker/host/gpu-01",
                }
            ),
            required_host_id="gpu-01",
        )
    assert excinfo.value.code == "invalid_spiffe"


def test_strict_mode_requires_the_gateway_to_present_an_svid(gateway_mode):
    """A gateway that says "trust me" without naming the workload is not enough."""
    with pytest.raises(identity.IdentityAdmissionError) as excinfo:
        identity.resolve_gateway_identity(
            _gateway_headers(**{"x-worker-host-id": "gpu-01"}),
            required_host_id="gpu-01",
        )
    assert excinfo.value.code == "missing_spiffe"
    assert excinfo.value.http_status == 401


def test_non_strict_mode_supports_the_interim_nginx_gateway(gateway_mode):
    """Nginx mTLS has a certificate DN, not an SVID — allowed only explicitly."""
    gateway_mode.setenv("XCELSIOR_SPIFFE_STRICT", "0")
    admitted = identity.resolve_gateway_identity(
        _gateway_headers(**{"x-worker-host-id": "gpu-01"}),
        required_host_id="gpu-01",
    )
    assert admitted is not None and admitted.spiffe_id is None
    # Even relaxed, a *present* SPIFFE ID must still bind correctly.
    with pytest.raises(identity.IdentityAdmissionError):
        identity.resolve_gateway_identity(
            _gateway_headers(
                **{
                    "x-worker-host-id": "gpu-01",
                    "x-worker-spiffe-id": "spiffe://xcelsior.ca/worker/host/gpu-99",
                }
            ),
            required_host_id="gpu-01",
        )


def test_strict_binding_defaults_on(monkeypatch):
    monkeypatch.delenv("XCELSIOR_SPIFFE_STRICT", raising=False)
    assert identity.spiffe_strict_binding() is True
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("XCELSIOR_SPIFFE_STRICT", off)
        assert identity.spiffe_strict_binding() is False


def test_gateway_without_a_secret_fails_closed(monkeypatch):
    """Enabling gateway mode without the secret would trust forgeable headers."""
    monkeypatch.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    monkeypatch.delenv("XCELSIOR_AGENT_GATEWAY_SECRET", raising=False)
    with pytest.raises(identity.IdentityAdmissionError) as excinfo:
        identity.resolve_gateway_identity(_gateway_headers(**{"x-worker-host-id": "gpu-01"}))
    assert excinfo.value.code == "gateway_secret_unconfigured"
    assert excinfo.value.http_status == 503


# ── Readiness gate (§21.3) ────────────────────────────────────────────


def test_readyz_refuses_a_half_configured_gateway(monkeypatch):
    """A misconfigured cutover must fail the deploy gate, not serve."""
    from fastapi.testclient import TestClient

    from api import app

    monkeypatch.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    monkeypatch.delenv("XCELSIOR_AGENT_GATEWAY_SECRET", raising=False)
    with TestClient(app) as c:
        resp = c.get("/readyz")
    assert resp.status_code == 503, resp.text
    assert "XCELSIOR_AGENT_GATEWAY_SECRET" in resp.text


def test_readyz_reports_identity_posture_when_configured(monkeypatch):
    from fastapi.testclient import TestClient

    from api import app

    monkeypatch.setenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", "1")
    monkeypatch.setenv("XCELSIOR_AGENT_GATEWAY_SECRET", "s3cr3t")
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    monkeypatch.setenv("XCELSIOR_SPIFFE_STRICT", "1")
    with TestClient(app) as c:
        resp = c.get("/readyz")
    if resp.status_code != 200:  # pragma: no cover - unrelated dependency down
        pytest.skip(f"readyz unavailable for an unrelated reason: {resp.text[:120]}")
    body = resp.json()["identity"]
    assert body == {
        "agent_gateway": True,
        "spiffe_trust_domain": "xcelsior.ca",
        "spiffe_strict": True,
    }


def test_readyz_is_ready_without_a_gateway(monkeypatch):
    from fastapi.testclient import TestClient

    from api import app

    monkeypatch.delenv("XCELSIOR_TRUSTED_AGENT_GATEWAY", raising=False)
    with TestClient(app) as c:
        resp = c.get("/readyz")
    if resp.status_code != 200:  # pragma: no cover
        pytest.skip(f"readyz unavailable for an unrelated reason: {resp.text[:120]}")
    assert resp.json()["identity"] == {"agent_gateway": False}


# ── The gateway must enforce the same contract Envoy-side ─────────────


def test_envoy_gateway_validates_worker_svids_only():
    cfg = yaml.safe_load(ENVOY.read_text())
    listener = cfg["static_resources"]["listeners"][0]
    tls = listener["filter_chains"][0]["transport_socket"]["typed_config"]

    assert tls["require_client_certificate"] is True
    common = tls["common_tls_context"]
    # Trust bundle and server cert both come from SPIRE over SDS — no
    # static key material, so rotation is continuous.
    assert common["tls_certificate_sds_secret_configs"][0]["name"].startswith("spiffe://")
    validation = common["combined_validation_context"]
    assert validation["validation_context_sds_secret_config"]["name"] == "spiffe://xcelsior.ca"

    matchers = validation["default_validation_context"]["match_typed_subject_alt_names"]
    uri_prefixes = [m["matcher"]["prefix"] for m in matchers if m["san_type"] == "URI"]
    assert uri_prefixes == [
        "spiffe://xcelsior.ca/worker/host/"
    ], "Envoy must accept only worker SVIDs — the same prefix the API verifies"


def test_envoy_prefix_matches_the_python_verifier(monkeypatch):
    """One drift guard for the two implementations of the same rule."""
    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    cfg = yaml.safe_load(ENVOY.read_text())
    tls = cfg["static_resources"]["listeners"][0]["filter_chains"][0]["transport_socket"][
        "typed_config"
    ]
    prefix = tls["common_tls_context"]["combined_validation_context"]["default_validation_context"][
        "match_typed_subject_alt_names"
    ][0]["matcher"]["prefix"]

    sample = identity.spiffe_id_for_host("gpu-01")
    assert sample.startswith(prefix)
    # And the Lua filter derives the host id from that same prefix.
    lua = _envoy_lua_source(cfg)
    assert f'local PREFIX = "{prefix}"' in lua


def _envoy_lua_source(cfg: dict) -> str:
    hcm = cfg["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"]
    for filt in hcm["http_filters"]:
        if filt["name"] == "envoy.filters.http.lua":
            return filt["typed_config"]["default_source_code"]["inline_string"]
    raise AssertionError("no lua filter in the agent gateway")


def test_envoy_strips_every_forgeable_identity_header():
    """Whatever a client sends must be removed before the API sees it."""
    cfg = yaml.safe_load(ENVOY.read_text())
    lua = _envoy_lua_source(cfg)
    for header in identity.UNTRUSTED_IDENTITY_HEADERS:
        assert f'"{header}"' in lua, f"gateway does not strip {header}"
    assert "headers:remove(name)" in lua


def test_envoy_sanitizes_client_supplied_xfcc():
    cfg = yaml.safe_load(ENVOY.read_text())
    hcm = cfg["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0]["typed_config"]
    # SANITIZE_SET: Envoy replaces XFCC with its own verified value, so a
    # client cannot smuggle a URI SAN through it.
    assert hcm["forward_client_cert_details"] == "SANITIZE_SET"
    assert hcm["set_current_client_cert_details"]["uri"] is True


def test_envoy_serves_only_the_agent_protocol():
    """§22.3: no frontend, no MCP, no general API proxying."""
    cfg = yaml.safe_load(ENVOY.read_text())
    vhost = cfg["static_resources"]["listeners"][0]["filter_chains"][0]["filters"][0][
        "typed_config"
    ]["route_config"]["virtual_hosts"][0]
    routed = [r for r in vhost["routes"] if "route" in r]
    assert [r["match"]["prefix"] for r in routed] == ["/agent/v2/"]
    catchall = [r for r in vhost["routes"] if "direct_response" in r]
    assert catchall and catchall[-1]["direct_response"]["status"] == 404


def test_envoy_gateway_secret_is_not_committed():
    raw = ENVOY.read_text()
    assert "${XCELSIOR_AGENT_GATEWAY_SECRET}" in raw, "secret must be substituted at deploy"
    # Nothing that looks like a rendered secret may be in the file.
    assert not re.search(r'gateway-auth",\s*"[A-Za-z0-9+/=]{16,}"', raw)


# ── SPIRE deployment assets ───────────────────────────────────────────


def test_spire_compose_brings_up_server_agent_and_gateway():
    cfg = yaml.safe_load((SPIRE / "docker-compose.spire.yml").read_text())
    services = cfg["services"]
    assert {"spire-server", "spire-agent", "agent-gateway"} <= set(services)

    # The gateway gets its SVID from the agent's Workload API socket,
    # not from files on disk.
    gw = services["agent-gateway"]
    assert any("/run/spire/sockets" in v for v in gw["volumes"])
    assert gw["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in gw["security_opt"]

    # Ordering matters: Envoy cannot start without a healthy agent, and
    # the agent cannot attest without a healthy server.
    assert gw["depends_on"]["spire-agent"]["condition"] == "service_healthy"
    assert services["spire-agent"]["depends_on"]["spire-server"]["condition"] == ("service_healthy")
    # A missing secret must fail compose, not silently render an empty one.
    assert "?" in cfg["services"]["agent-gateway"]["environment"]["XCELSIOR_AGENT_GATEWAY_SECRET"]


def test_spire_server_issues_short_lived_svids():
    conf = (SPIRE / "server.conf").read_text()
    assert 'trust_domain    = "xcelsior.ca"' in conf
    ttl = re.search(r'default_x509_svid_ttl\s*=\s*"([^"]+)"', conf)
    assert ttl and ttl.group(1) == "1h", "SVIDs must be short lived to be worth having"
    # The registration store must survive a restart with the control plane.
    assert 'database_type     = "postgres"' in conf


def test_register_host_script_emits_the_canonical_id(monkeypatch, tmp_path):
    """The registration script and the Python verifier must agree exactly.

    Runs the script's own sanitisation/formatting functions in bash and
    compares against `identity.spiffe_id_for_host`, so a divergence in
    either implementation fails here rather than in production as a
    host that attests but cannot authenticate.
    """
    import subprocess

    monkeypatch.setenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", "xcelsior.ca")
    source = (SPIRE / "register-host.sh").read_text()
    # Source only the pure functions; the script's main body requires a
    # live spire-server socket.
    harness = "\n".join(
        [
            "TRUST_DOMAIN=xcelsior.ca",
            _extract_bash_function(source, "sanitize_host_component"),
            _extract_bash_function(source, "spiffe_id_for_host"),
            'spiffe_id_for_host "$1"',
        ]
    )

    for host_id in ("gpu-01", "aaryn-tuf-rtx2060", "host.with.dots", "gpu/01:a"):
        out = subprocess.run(
            ["bash", "-c", harness, "_", host_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == identity.spiffe_id_for_host(host_id), host_id


def _extract_bash_function(source: str, name: str) -> str:
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}", source, re.S | re.M)
    assert match, f"function {name} not found in register-host.sh"
    return match.group(0)


def test_register_host_only_covers_admitted_hosts():
    """SPIRE must not widen who the control plane admitted."""
    script = (SPIRE / "register-host.sh").read_text()
    assert "admitted" in script
    assert "FROM hosts WHERE" in script


def test_spire_agent_binds_svids_to_the_workload():
    conf = (SPIRE / "agent.conf").read_text()
    # Workload attestation is what stops any container on a GPU host from
    # asking for that host's worker identity.
    assert 'WorkloadAttestor "unix"' in conf
    assert 'WorkloadAttestor "docker"' in conf
    assert 'socket_path        = "/run/spire/sockets/agent.sock"' in conf


def test_nginx_interim_gateway_does_not_fabricate_a_spiffe_id():
    """A certificate DN is not an SVID; synthesising one would be a lie."""
    conf = (ROOT / "nginx" / "agent-xcelsior.conf").read_text()
    assert "ssl_verify_client on" in conf
    # It may blank the header, but must never populate it from $ssl_client_s_dn.
    assert not re.search(r"proxy_set_header\s+X-Worker-Spiffe-Id\s+\$ssl_client_s_dn", conf)
    # Identity is only derived from a chain that actually verified.
    assert "$agent_verified_host_id" in conf
    assert "map $ssl_client_verify $agent_verified_host_id" in conf


def test_nginx_agent_gateway_has_its_own_limits():
    """§22.3: worker traffic must not share the user API's budget."""
    conf = (ROOT / "nginx" / "agent-xcelsior.conf").read_text()
    assert "limit_req_zone" in conf
    assert "client_max_body_size" in conf
    assert "location /agent/v2/" in conf
    # No frontend or general API proxying.
    assert re.search(r"location / \{\s*\n\s*return 404;", conf)
