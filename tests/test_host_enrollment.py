"""One-click host enrollment (issue #67): SPIRE join token + mesh key for an owner's host."""

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from control_plane import host_enrollment as he

ROOT = Path(__file__).resolve().parent.parent
REGISTER_HOST_SH = ROOT / "api" / "infra" / "spire" / "register-host.sh"


@pytest.fixture
def fake_spire(tmp_path, monkeypatch):
    """A spire-server stand-in that records argv and answers like the real CLI."""
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "spire-server"
    script.write_text(f"""#!/usr/bin/env python3
import json, sys
open({str(log)!r}, "a").write(json.dumps(sys.argv[1:]) + "\\n")
a = sys.argv[1:]
if a[:2] == ["token", "generate"]:
    print("Token: 4f2a9c1e-0000-4000-8000-000000000001")
elif a[:2] == ["entry", "show"]:
    print("Found 1 entry\\nEntry ID         : e-123" if "existing" in " ".join(a) else "Found 0 entries")
elif a[:2] == ["bundle", "show"]:
    print("-----BEGIN CERTIFICATE-----\\nMIIB\\n-----END CERTIFICATE-----")
""")
    script.chmod(0o755)
    monkeypatch.setenv("XCELSIOR_SPIRE_SERVER_CMD", str(script))
    monkeypatch.setenv("XCELSIOR_SPIRE_SERVER_ADDRESS", "100.64.0.1")
    monkeypatch.delenv("XCELSIOR_SPIFFE_TRUST_DOMAIN", raising=False)
    return lambda: [json.loads(line) for line in log.read_text().splitlines()]


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def test_spire_enrollment_matches_register_host_sh(fake_spire):
    got = he.issue_spire_enrollment("tower-rtx3060")
    calls = fake_spire()
    token, show, create, bundle = calls
    assert token[:2] == ["token", "generate"]
    assert _flag(token, "-spiffeID") == "spiffe://xcelsior.ca/node/host/tower-rtx3060"
    assert create[:2] == ["entry", "create"]
    assert _flag(create, "-spiffeID") == "spiffe://xcelsior.ca/worker/host/tower-rtx3060"
    assert _flag(create, "-parentID") == "spiffe://xcelsior.ca/node/host/tower-rtx3060"
    assert _flag(create, "-dns") == "tower-rtx3060"  # nginx maps the SVID's CN to the host
    # Same TTL and default selector as the operator script, so issuer and verifier agree.
    sh = REGISTER_HOST_SH.read_text()
    assert _flag(create, "-x509SVIDTTL") in re.search(r"-x509SVIDTTL (\d+)", sh).group(1)
    assert f'selectors=("{_flag(create, "-selector")}")' in sh
    assert got.join_token.startswith("4f2a9c1e") and got.server_address == "100.64.0.1"
    assert "BEGIN CERTIFICATE" in got.trust_bundle_pem
    assert all("-socketPath" in c for c in calls)


def test_existing_entry_is_updated_not_duplicated(fake_spire):
    he.issue_spire_enrollment("existing-host")
    verbs = [c[:2] for c in fake_spire()]
    assert ["entry", "update"] in verbs and ["entry", "create"] not in verbs


def test_spire_not_configured_is_unavailable(monkeypatch):
    monkeypatch.delenv("XCELSIOR_SPIRE_SERVER_ADDRESS", raising=False)
    with pytest.raises(he.EnrollmentUnavailable):
        he.issue_spire_enrollment("h1")


def test_spire_failure_is_reported(fake_spire, monkeypatch, tmp_path):
    broken = tmp_path / "broken"
    broken.write_text("#!/bin/sh\necho 'connection refused' >&2\nexit 1\n")
    broken.chmod(0o755)
    monkeypatch.setenv("XCELSIOR_SPIRE_SERVER_CMD", str(broken))
    with pytest.raises(he.EnrollmentFailed, match="connection refused"):
        he.issue_spire_enrollment("h1")


def test_mesh_key_is_single_use_and_short_lived(monkeypatch):
    sent = {}

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"preAuthKey": {"key": "hskey-123"}}).encode()

    def urlopen(req, timeout):
        sent["url"], sent["auth"] = req.full_url, req.get_header("Authorization")
        sent["body"] = json.loads(req.data)
        return Reply()

    monkeypatch.setattr(he.urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("XCELSIOR_HEADSCALE_URL", "https://hs.xcelsior.ca/")
    monkeypatch.setenv("XCELSIOR_HEADSCALE_API_KEY", "hskey-api")
    mesh = he.issue_mesh_enrollment("h1")
    assert mesh.auth_key == "hskey-123" and mesh.login_server == "https://hs.xcelsior.ca"
    assert sent["url"] == "https://hs.xcelsior.ca/api/v1/preauthkey"
    assert sent["auth"] == "Bearer hskey-api"
    assert sent["body"]["reusable"] is False and sent["body"]["ephemeral"] is False
    assert sent["body"]["user"] == "1"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", sent["body"]["expiration"])


def test_mesh_not_configured_is_unavailable(monkeypatch):
    monkeypatch.delenv("XCELSIOR_HEADSCALE_URL", raising=False)
    monkeypatch.delenv("XCELSIOR_HEADSCALE_API_KEY", raising=False)
    with pytest.raises(he.EnrollmentUnavailable):
        he.issue_mesh_enrollment("h1")


# ── route: only the owner, never an unowned row ─────────────────────────


@pytest.fixture
def route(monkeypatch):
    import routes._deps as deps
    import routes.hosts as hosts

    state = SimpleNamespace(user=None, hosts=[], issued=[])
    monkeypatch.setattr(deps, "_get_current_user", lambda request: state.user)
    monkeypatch.setattr(deps, "_check_host_enroll_rate_limit", lambda host_id: None)
    monkeypatch.setattr(hosts, "list_hosts", lambda active_only=False: state.hosts)
    monkeypatch.setattr(
        he,
        "enroll_host",
        lambda host_id: state.issued.append(host_id)
        or {"host_id": host_id, "spire": {"spiffe_id": "x"}},
    )
    state.call = lambda host_id: hosts.api_enroll_host(host_id, request=None)
    return state


def _user(uid, scopes=("hosts:write",)):
    # An agent API key: the credential install.sh and the wizard use, and a
    # machine principal, so its scopes are enforced (browser sessions are not).
    return {
        "user_id": uid,
        "email": f"{uid}@example.com",
        "scopes": list(scopes),
        "role": "user",
        "auth_type": "agent_api_key",
    }


def test_owner_can_enroll(route):
    route.user = _user("u1")
    route.hosts = [{"host_id": "h1", "owner": "u1"}]
    assert route.call("h1")["ok"] and route.issued == ["h1"]


@pytest.mark.parametrize(
    "hosts,why",
    [
        ([{"host_id": "h1", "owner": "someone-else"}], "another provider's host"),
        ([{"host_id": "h1", "owner": ""}], "an unowned row is not claimable here"),
        ([], "unknown host"),
    ],
)
def test_others_get_404(route, hosts, why):
    route.user = _user("u1")
    route.hosts = hosts
    with pytest.raises(HTTPException) as exc:
        route.call("h1")
    assert exc.value.status_code == 404, why
    assert route.issued == []


def test_requires_auth_and_hosts_write(route):
    route.hosts = [{"host_id": "h1", "owner": "u1"}]
    with pytest.raises(HTTPException) as exc:
        route.call("h1")
    assert exc.value.status_code == 401
    route.user = _user("u1", scopes=("hosts:read",))
    with pytest.raises(HTTPException) as exc:
        route.call("h1")
    assert exc.value.status_code == 403


def test_unconfigured_control_plane_answers_503(route, monkeypatch):
    route.user = _user("u1")
    route.hosts = [{"host_id": "h1", "owner": "u1"}]

    def unavailable(host_id):
        raise he.EnrollmentUnavailable("XCELSIOR_SPIRE_SERVER_ADDRESS is not configured")

    monkeypatch.setattr(he, "enroll_host", unavailable)
    with pytest.raises(HTTPException) as exc:
        route.call("h1")
    assert exc.value.status_code == 503 and "SPIRE" not in exc.value.detail  # no internals leak


# ── against a real SPIRE server (opt-in: XCELSIOR_SPIRE_IT=1, needs docker) ─


@pytest.mark.skipif(
    os.environ.get("XCELSIOR_SPIRE_IT") != "1" or not shutil.which("docker"),
    reason="real SPIRE integration test is opt-in",
)
def test_real_spire_server_issues_a_usable_enrollment(tmp_path, monkeypatch):
    name = f"spire-enroll-it-{os.getpid()}"
    conf = tmp_path / "server.conf"
    conf.write_text(
        'server { bind_address = "127.0.0.1"  bind_port = "18091"  socket_path = "/tmp/spire-server/private/api.sock"\n'
        '  trust_domain = "xcelsior.ca"  data_dir = "/opt/spire/data/server"  log_level = "WARN" }\n'
        'plugins { DataStore "sql" { plugin_data { database_type = "sqlite3"  connection_string = "/tmp/ds.sqlite3" } }\n'
        '  NodeAttestor "join_token" { plugin_data {} }  KeyManager "memory" { plugin_data {} } }\n'
    )
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--network",
            "host",
            "-v",
            f"{conf}:/s.conf:ro",
            "ghcr.io/spiffe/spire-server:1.11.2",
            "-config",
            "/s.conf",
        ],
        check=True,
        capture_output=True,
    )
    try:
        for _ in range(30):
            if (
                subprocess.run(
                    ["docker", "exec", name, "/opt/spire/bin/spire-server", "healthcheck"],
                    capture_output=True,
                ).returncode
                == 0
            ):
                break
            time.sleep(1)
        monkeypatch.setenv(
            "XCELSIOR_SPIRE_SERVER_CMD", f"docker exec {name} /opt/spire/bin/spire-server"
        )
        monkeypatch.setenv("XCELSIOR_SPIRE_SERVER_ADDRESS", "127.0.0.1")
        first = he.issue_spire_enrollment("it-host-1")
        second = he.issue_spire_enrollment("it-host-1")  # re-enrollment updates, never duplicates
        assert first.join_token != second.join_token
        entries = subprocess.run(
            [
                "docker",
                "exec",
                name,
                "/opt/spire/bin/spire-server",
                "entry",
                "show",
                "-spiffeID",
                "spiffe://xcelsior.ca/worker/host/it-host-1",
            ],
            capture_output=True,
            text=True,
        ).stdout
        assert entries.count("Entry ID") == 1
        assert "unix:user:xcelsior" in entries and "it-host-1" in entries
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
