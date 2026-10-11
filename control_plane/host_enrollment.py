"""One-click host enrollment: everything a new GPU host needs to reach the agent gateway.

Production refuses the public worker surface (``XCELSIOR_AGENT_PUBLIC_INGRESS=deny``),
so a new host can only talk to the control plane through ``agent.xcelsior.ca`` with
an identity the gateway verifies. Enrollment hands an owner's *registered* host the
two single-use credentials that bootstrap that identity (issue #67):

* a **SPIRE join token**. The host's spire-agent attests with it once. The worker
  then receives hourly X.509-SVIDs for ``spiffe://<td>/worker/host/<host_id>``
  with ``CN=<host_id>``, which is what ``nginx/agent-xcelsior.conf`` maps to the
  host identity. The registration entry is created exactly as
  ``infra/spire/register-host.sh`` creates it (same SPIFFE ID, DNS name, TTL and
  ``unix:user:xcelsior`` selector), so issuer and verifier cannot drift.
* a **Headscale pre-auth key** (single use, short-lived). The SPIRE server is
  reached over the tailnet (infra/spire/README.md, Note 3), and job traffic
  reaches hosts over it too.

Identity is not admission. A pending host needs an identity to heartbeat and to
submit the verification report that admission reads. Admission stays the
operator's decision (routes/host_admission.py), so enrollment does not widen it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from control_plane.identity import spiffe_host_component, spiffe_id_for_host, spiffe_trust_domain

log = logging.getLogger("xcelsior.control_plane.host_enrollment")

#: Same selector `infra/spire/register-host.sh` defaults to: the worker process
#: running as the dedicated service user that install.sh creates.
WORKER_SELECTOR = "unix:user:xcelsior"
WORKER_SVID_TTL_SECONDS = 3600


class EnrollmentUnavailable(Exception):
    """The control plane is not configured to enroll hosts (maps to 503)."""


class EnrollmentFailed(Exception):
    """SPIRE or Headscale refused or failed (maps to 502)."""


@dataclass(frozen=True)
class SpireEnrollment:
    server_address: str
    server_port: int
    trust_domain: str
    join_token: str
    join_token_ttl_seconds: int
    trust_bundle_pem: str
    spiffe_id: str


@dataclass(frozen=True)
class MeshEnrollment:
    login_server: str
    auth_key: str
    expires_at: str


def node_spiffe_id(host_id: str) -> str:
    """The join-token alias the host's spire-agent attests as (parent of the worker entry)."""
    return f"spiffe://{spiffe_trust_domain()}/node/host/{spiffe_host_component(host_id)}"


# ── SPIRE ──────────────────────────────────────────────────────────────


def _spire_cmd() -> list[str]:
    """`spire-server` invocation; a full command (e.g. via `docker exec`) is allowed."""
    raw = os.environ.get("XCELSIOR_SPIRE_SERVER_CMD", "spire-server").strip()
    return shlex.split(raw)


def _spire(*args: str, timeout: int = 30) -> str:
    socket_path = os.environ.get(
        "XCELSIOR_SPIRE_SERVER_SOCKET", "/tmp/spire-server/private/api.sock"
    ).strip()
    cmd = [*_spire_cmd(), *args, "-socketPath", socket_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnrollmentFailed(f"spire-server {args[0]} {args[1]}: {exc}") from exc
    if out.returncode != 0:
        raise EnrollmentFailed(
            f"spire-server {args[0]} {args[1]} failed: {(out.stderr or out.stdout).strip()[:300]}"
        )
    return out.stdout


def _upsert_worker_entry(host_id: str, parent_id: str) -> None:
    spiffe_id = spiffe_id_for_host(host_id)
    entry_args = [
        "-spiffeID", spiffe_id,
        "-parentID", parent_id,
        "-x509SVIDTTL", str(WORKER_SVID_TTL_SECONDS),
        "-dns", host_id,
        "-selector", WORKER_SELECTOR,
    ]  # fmt: skip
    shown = _spire("entry", "show", "-spiffeID", spiffe_id)
    match = re.search(r"Entry ID\s*:\s*(\S+)", shown)
    if match:
        _spire("entry", "update", "-entryID", match.group(1), *entry_args)
    else:
        _spire("entry", "create", *entry_args)


def issue_spire_enrollment(host_id: str) -> SpireEnrollment:
    address = os.environ.get("XCELSIOR_SPIRE_SERVER_ADDRESS", "").strip()
    if not address:
        raise EnrollmentUnavailable("XCELSIOR_SPIRE_SERVER_ADDRESS is not configured")
    port = int(os.environ.get("XCELSIOR_SPIRE_SERVER_PORT", "8091"))
    ttl = int(os.environ.get("XCELSIOR_SPIRE_JOIN_TOKEN_TTL", "3600"))
    parent_id = node_spiffe_id(host_id)

    out = _spire("token", "generate", "-spiffeID", parent_id, "-ttl", str(ttl))
    match = re.search(r"Token:\s*(\S+)", out)
    if not match:
        raise EnrollmentFailed("spire-server token generate printed no token")
    _upsert_worker_entry(host_id, parent_id)
    bundle = _spire("bundle", "show", "-format", "pem")
    if "BEGIN CERTIFICATE" not in bundle:
        raise EnrollmentFailed("spire-server bundle show returned no certificate")
    return SpireEnrollment(
        server_address=address,
        server_port=port,
        trust_domain=spiffe_trust_domain(),
        join_token=match.group(1),
        join_token_ttl_seconds=ttl,
        trust_bundle_pem=bundle,
        spiffe_id=spiffe_id_for_host(host_id),
    )


# ── Headscale (v0.26+ REST API: numeric user id) ───────────────────────


def issue_mesh_enrollment(host_id: str) -> MeshEnrollment:
    url = os.environ.get("XCELSIOR_HEADSCALE_URL", "").strip().rstrip("/")
    api_key = os.environ.get("XCELSIOR_HEADSCALE_API_KEY", "").strip()
    if not (url and api_key):
        raise EnrollmentUnavailable(
            "XCELSIOR_HEADSCALE_URL / XCELSIOR_HEADSCALE_API_KEY are not configured"
        )
    user_id = os.environ.get("XCELSIOR_HEADSCALE_USER_ID", "1").strip()
    ttl = int(os.environ.get("XCELSIOR_HEADSCALE_KEY_TTL", "3600"))
    expires = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body: dict = {"user": user_id, "reusable": False, "ephemeral": False, "expiration": expires}
    tags = [
        t.strip() for t in os.environ.get("XCELSIOR_HEADSCALE_ACL_TAGS", "").split(",") if t.strip()
    ]
    if tags:
        body["aclTags"] = tags
    req = urllib.request.Request(
        f"{url}/api/v1/preauthkey",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            reply = json.load(resp)
    except urllib.error.HTTPError as exc:
        raise EnrollmentFailed(f"headscale refused the pre-auth key: HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise EnrollmentFailed(f"headscale unreachable: {exc}") from exc
    key = (reply.get("preAuthKey") or {}).get("key")
    if not key:
        raise EnrollmentFailed("headscale returned no pre-auth key")
    return MeshEnrollment(login_server=url, auth_key=key, expires_at=expires)


def enroll_host(host_id: str) -> dict:
    """Both credentials plus where to send worker traffic. Each call issues fresh ones."""
    mesh = issue_mesh_enrollment(host_id)
    spire = issue_spire_enrollment(host_id)
    return {
        "host_id": host_id,
        "scheduler_url": os.environ.get("XCELSIOR_AGENT_GATEWAY_URL", "https://agent.xcelsior.ca"),
        "mesh": mesh.__dict__,
        "spire": spire.__dict__,
    }
