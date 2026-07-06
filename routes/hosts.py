"""Routes: hosts."""

import os
import re
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from routes._deps import (
    _caller_owner_ids,
    _is_platform_admin,
    _lookup_user_by_email,
    _require_admin,
    _require_auth,
    _require_scope,
    broadcast_sse,
    log,
)
from scheduler import (
    check_hosts,
    get_compute_score,
    list_hosts,
    list_jobs,
    log,
    register_host,
    remove_host,
    set_host_draining,
    update_host_spot_settings,
)
from db import UserStore
from verification import get_verification_engine
from security import admit_node
from reputation import VerificationType, get_reputation_engine
from host_metadata import (
    CANONICAL_GPU_MODELS,
    enrich_host_for_api,
    normalize_gpu_model,
    normalize_region,
    platform_rate_cad,
)

router = APIRouter()


def _host_owner_ids(host: dict | None) -> set[str]:
    """Identifiers on a registered host row that denote ownership."""
    if not host:
        return set()
    ids: set[str] = set()
    for key in ("owner", "provider_id", "user_id"):
        value = str(host.get(key) or "").strip()
        if value:
            ids.add(value)
    return ids


def _oauth_client_creator(user: dict) -> dict | None:
    """Resolve the interactive user behind a client_credentials principal."""
    auth_type = str(user.get("auth_type") or user.get("grant_type") or "")
    if auth_type != "client_credentials":
        return None
    client_id = str(user.get("client_id") or "").strip()
    if not client_id:
        return None
    from oauth_service import get_client

    client = get_client(client_id)
    if not client:
        return None
    email = str(client.get("created_by_email") or "").strip()
    if not email:
        return None
    return _lookup_user_by_email(email)


def _resolve_host_owner_from_user(user: dict) -> str:
    """Return the user identity that should own a worker-managed host."""
    creator = _oauth_client_creator(user)
    if creator:
        return str(creator.get("user_id") or creator.get("sub") or creator.get("email") or "")
    grant = str(user.get("grant_type") or user.get("auth_type") or "")
    if grant == "client_credentials":
        client_id = str(user.get("client_id") or "").strip()
        if client_id:
            return f"client:{client_id}"
    return str(user.get("user_id") or user.get("sub") or user.get("email") or "")


def _require_host_operator(user: dict, host_id: str) -> None:
    if _is_platform_admin(user):
        return

    host_id = (host_id or "").strip()
    if not host_id:
        raise HTTPException(403, "Forbidden")

    caller_ids = _caller_owner_ids(user)
    creator = _oauth_client_creator(user)
    if creator:
        caller_ids |= _caller_owner_ids(creator)
        creator_user_id = str(creator.get("user_id") or creator.get("sub") or "").strip()
        if creator_user_id:
            caller_ids.add(creator_user_id)

    if host_id in caller_ids:
        return

    entry = next((h for h in list_hosts(active_only=False) if h.get("host_id") == host_id), None)
    if entry and caller_ids & _host_owner_ids(entry):
        return

    # Worker OAuth clients (hosts:write) may register new hosts or claim rows
    # with no owner. Creator email lookup can fail for machine-owned clients
    # (e.g. created_by_email=api-token@xcelsior.ca), so gate on grant_type.
    grant = str(user.get("grant_type") or user.get("auth_type") or "")
    if grant == "client_credentials" and (entry is None or not _host_owner_ids(entry)):
        return

    raise HTTPException(403, "Forbidden")


def _resolve_host_id(host_id: str) -> tuple[str | None, list[dict]]:
    """Resolve a (possibly truncated) host_id to its full form.

    Returns (resolved_host_id, hosts_list). If resolved_host_id is None,
    no unique match was found. Accepts prefixes of 8+ chars.
    Frontend displays commonly truncate UUIDs to 8 or 12 chars; users
    sometimes copy the truncated form into URLs/CLIs.
    """
    hosts = list_hosts(active_only=False)
    # Exact match first
    if any(h["host_id"] == host_id for h in hosts):
        return host_id, hosts
    # Prefix match fallback (must be unambiguous and at least 8 chars)
    if len(host_id) >= 8:
        matches = [h for h in hosts if h["host_id"].startswith(host_id)]
        if len(matches) == 1:
            return matches[0]["host_id"], hosts
    return None, hosts


_BLOCKING_JOB_STATUSES = frozenset({"assigned", "leased", "running"})


def _interactive_host_jobs(host_id: str) -> list[dict]:
    """Interactive jobs that should block worker host maintenance."""
    jobs = []
    for job in list_jobs():
        if job.get("host_id") != host_id:
            continue
        if not job.get("interactive", False):
            continue
        if job.get("status") not in _BLOCKING_JOB_STATUSES:
            continue
        jobs.append(job)
    return jobs


def _serverless_host_jobs(host_id: str) -> list[dict]:
    """Serverless worker scheduler jobs that should block worker host maintenance."""
    jobs = []
    for job in list_jobs():
        if job.get("host_id") != host_id:
            continue
        if str(job.get("job_type") or "") != "serverless_worker":
            continue
        if job.get("status") not in _BLOCKING_JOB_STATUSES:
            continue
        jobs.append(job)
    return jobs


def _blocking_host_jobs(host_id: str) -> tuple[list[dict], list[dict]]:
    """Return (interactive, serverless) jobs that block maintenance on a host."""
    return _interactive_host_jobs(host_id), _serverless_host_jobs(host_id)


# ── Model: HostIn ──


class HostIn(BaseModel):
    host_id: str = Field(min_length=1, max_length=64)
    ip: str = Field(min_length=7, max_length=45)  # IPv4 min 7, IPv6 max 45
    gpu_model: str = Field(min_length=1, max_length=64)
    total_vram_gb: float = Field(ge=0, le=1024)
    free_vram_gb: float = Field(ge=0, le=1024)
    cost_per_hour: float = Field(default=0.20, ge=0, le=1000)
    country: str = Field(default="CA", min_length=2, max_length=2)  # ISO 3166-1 alpha-2
    province: str = Field(default="", max_length=10)  # CA province code (ON, QC, BC, etc.)
    region: str = Field(default="", max_length=32)
    # Optional: agent-reported versions for inline admission
    versions: dict | None = None  # {"runc": "1.2.4", "nvidia_ctk": "1.17.8", ...}
    # Canadian company fields (Report #1.B — Provider Onboarding)
    corporation_name: str = Field(default="", max_length=256)  # Legal corporation name
    business_number: str = Field(
        default="", max_length=64
    )  # CRA Business Number (BN), e.g. 123456789RC0001
    gst_hst_number: str = Field(default="", max_length=64)  # GST/HST registration number
    legal_name: str = Field(default="", max_length=256)  # Legal name of individual or company
    # Agent self-reporting (P1.2 — worker self-update). Optional so older
    # agents that haven't been rolled out yet still pass validation.
    agent_version: str | None = Field(default=None, max_length=32)
    agent_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$|^$")
    cpu_count: int | None = Field(default=None, ge=1, le=512)
    spot_enabled: bool | None = None
    spot_gpu_slots: int | None = Field(default=None, ge=0, le=64)
    spot_min_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    # CRIU checkpoint capability (S6 F4.2): docker-criu | gpu-criu | empty
    checkpoint_class: str | None = Field(default=None, max_length=32)
    capabilities: dict | None = None
    cuda_driver_version: str | None = Field(default=None, max_length=32)


# ── Model: JobIn ──


class JobIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    vram_needed_gb: float = Field(default=0, ge=0)
    priority: int = Field(default=0, ge=0, le=10)
    tier: str | None = None
    num_gpus: int = Field(default=1, ge=1, le=64)
    host_id: str | None = None  # Direct host assignment (marketplace)
    gpu_model: str | None = None  # Hint for VRAM lookup
    nfs_server: str | None = None
    nfs_path: str | None = None
    nfs_mount_point: str | None = None
    image: str | None = None
    interactive: bool = True
    command: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)


# ── Model: StatusUpdate ──


class StatusUpdate(BaseModel):
    status: str = Field(min_length=1, max_length=32)
    host_id: str | None = Field(None, max_length=64)
    container_id: str | None = Field(None, max_length=128)
    container_name: str | None = Field(None, max_length=128)


@router.put("/host", tags=["Hosts"])
def api_register_host(h: HostIn, request: Request):
    """Register or update a host with strict admission gating.

    Per REPORT_FEATURE_FINAL.md §62 and REPORT_FEATURE_2.md §37:
    - Hosts must pass node admission (version gating) before accepting work
    - Country/province recorded for jurisdiction-aware scheduling
    - Hosts register as 'pending' until agent completes benchmark + admission
    - If versions are provided inline, admission is checked immediately
    """
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    _require_host_operator(user, h.host_id)
    from security import admit_node

    gpu_model = normalize_gpu_model(h.gpu_model)
    total_vram = float(h.total_vram_gb or 0)
    if total_vram <= 0 and gpu_model:
        from host_metadata import default_vram_gb

        total_vram = default_vram_gb(gpu_model)
    cost_per_hour = float(h.cost_per_hour or 0)
    if cost_per_hour <= 0 and gpu_model:
        cost_per_hour = platform_rate_cad(gpu_model) or cost_per_hour

    # Register the host with country/province metadata
    entry = register_host(
        h.host_id,
        h.ip,
        gpu_model,
        total_vram,
        h.free_vram_gb,
        cost_per_hour,
        spot_enabled=h.spot_enabled,
        spot_gpu_slots=h.spot_gpu_slots,
        spot_min_cents=h.spot_min_cents,
        region=h.region,
    )
    owner_id = _resolve_host_owner_from_user(user)
    existing_owner = str(entry.get("owner") or "").strip()
    if existing_owner and existing_owner != h.host_id:
        entry["owner"] = existing_owner
    elif owner_id:
        entry["owner"] = owner_id
    entry["country"] = h.country.upper()
    entry["province"] = h.province.upper() if h.province else ""
    if h.region or h.province or not entry.get("region"):
        entry["region"] = normalize_region(h.region, country=entry["country"], province=entry["province"])
    # P1.2: record what the agent says about itself so rolling-upgrade logic
    # can target only out-of-date hosts and detect hashes that don't match
    # anything the control plane has signed off on.
    if h.agent_version:
        entry["agent_version"] = h.agent_version
    if h.agent_sha256:
        entry["agent_sha256"] = h.agent_sha256
    if h.cpu_count:
        entry["cpu_count"] = int(h.cpu_count)
    if h.checkpoint_class:
        entry["checkpoint_class"] = h.checkpoint_class.strip().lower()
    if h.capabilities:
        entry["capabilities"] = h.capabilities
    if h.cuda_driver_version:
        entry["cuda_driver_version"] = h.cuda_driver_version
    # Persist Canadian company info if provided
    if h.corporation_name:
        entry["corporation_name"] = h.corporation_name
    if h.business_number:
        entry["business_number"] = h.business_number
    if h.gst_hst_number:
        entry["gst_hst_number"] = h.gst_hst_number
    if h.legal_name:
        entry["legal_name"] = h.legal_name

    # Inline admission check if versions provided
    if h.versions:
        admitted, details = admit_node(h.host_id, h.versions, h.gpu_model)
        entry["admitted"] = admitted
        entry["admission_details"] = details
        entry["recommended_runtime"] = details.get("recommended_runtime", "runc")
        if not admitted:
            # Host is registered but marked as not-admitted — won't receive work
            entry["status"] = "pending"
            log.warning(
                "HOST %s registered but NOT ADMITTED: %s",
                h.host_id,
                details.get("rejection_reasons", []),
            )
    else:
        # No versions provided — only set pending for NEW hosts.
        # Existing hosts preserve their admission status from /agent/versions.
        if not entry.get("admitted"):
            entry.setdefault("admitted", False)
            entry["status"] = "pending"

    # Persist the updated entry (country, province, admitted status)
    from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed

    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        _upsert_host_row(conn, entry)

    # Auto-compute score and auto-list on marketplace
    from scheduler import estimate_compute_score, register_compute_score, list_rig

    score = estimate_compute_score(gpu_model)
    register_compute_score(h.host_id, gpu_model, score)
    entry["compute_score"] = score

    list_rig(
        h.host_id,
        gpu_model,
        total_vram,
        cost_per_hour,
        description=f"{gpu_model} ({total_vram}GB) in {h.country.upper()}",
        owner=entry.get("owner") or owner_id or h.host_id,
        region=entry.get("region", ""),
        province=entry.get("province", ""),
    )

    # ── Auto-create verification + reputation records ────────────────────
    # Ensures the host appears on the trust page immediately (as "unverified")
    # and gets a baseline reputation score.  Hardware verification scoring
    # happens later when the agent sends a full benchmark report.
    try:
        ve = get_verification_engine()
        if not ve.store.get_verification(h.host_id):
            from verification import HostVerification, HostVerificationState

            ve.store.save_verification(
                HostVerification(
                    verification_id=str(uuid.uuid4())[:12],
                    host_id=h.host_id,
                    state=HostVerificationState.UNVERIFIED,
                )
            )
            log.info("VERIFY RECORD created for new host %s", h.host_id)
        # Bootstrap reputation — email verification is implicit for registered users
        re = get_reputation_engine()
        re.add_verification(h.host_id, VerificationType.EMAIL)
    except Exception as e:
        log.exception("Non-fatal: could not bootstrap verification/reputation for %s", h.host_id)

    broadcast_sse(
        "host_update",
        {
            "host_id": h.host_id,
            "gpu_model": h.gpu_model,
            "admitted": entry.get("admitted", False),
            "country": entry.get("country", ""),
        },
    )
    return {"ok": True, "host": enrich_host_for_api(entry)}


# ── Model: RegisterHostRequest (web-facing registration form) ──

# Must match frontend gpu-models.ts / db._GPU_PRICING_BASE.
_VALID_GPU_MODELS = CANONICAL_GPU_MODELS

# ISO 3166-1 alpha-2 — every assigned country code
_VALID_COUNTRY_CODES = frozenset(
    {
        "AD",
        "AE",
        "AF",
        "AG",
        "AI",
        "AL",
        "AM",
        "AO",
        "AQ",
        "AR",
        "AS",
        "AT",
        "AU",
        "AW",
        "AX",
        "AZ",
        "BA",
        "BB",
        "BD",
        "BE",
        "BF",
        "BG",
        "BH",
        "BI",
        "BJ",
        "BL",
        "BM",
        "BN",
        "BO",
        "BQ",
        "BR",
        "BS",
        "BT",
        "BV",
        "BW",
        "BY",
        "BZ",
        "CA",
        "CC",
        "CD",
        "CF",
        "CG",
        "CH",
        "CI",
        "CK",
        "CL",
        "CM",
        "CN",
        "CO",
        "CR",
        "CU",
        "CV",
        "CW",
        "CX",
        "CY",
        "CZ",
        "DE",
        "DJ",
        "DK",
        "DM",
        "DO",
        "DZ",
        "EC",
        "EE",
        "EG",
        "EH",
        "ER",
        "ES",
        "ET",
        "FI",
        "FJ",
        "FK",
        "FM",
        "FO",
        "FR",
        "GA",
        "GB",
        "GD",
        "GE",
        "GF",
        "GG",
        "GH",
        "GI",
        "GL",
        "GM",
        "GN",
        "GP",
        "GQ",
        "GR",
        "GS",
        "GT",
        "GU",
        "GW",
        "GY",
        "HK",
        "HM",
        "HN",
        "HR",
        "HT",
        "HU",
        "ID",
        "IE",
        "IL",
        "IM",
        "IN",
        "IO",
        "IQ",
        "IR",
        "IS",
        "IT",
        "JE",
        "JM",
        "JO",
        "JP",
        "KE",
        "KG",
        "KH",
        "KI",
        "KM",
        "KN",
        "KP",
        "KR",
        "KW",
        "KY",
        "KZ",
        "LA",
        "LB",
        "LC",
        "LI",
        "LK",
        "LR",
        "LS",
        "LT",
        "LU",
        "LV",
        "LY",
        "MA",
        "MC",
        "MD",
        "ME",
        "MF",
        "MG",
        "MH",
        "MK",
        "ML",
        "MM",
        "MN",
        "MO",
        "MP",
        "MQ",
        "MR",
        "MS",
        "MT",
        "MU",
        "MV",
        "MW",
        "MX",
        "MY",
        "MZ",
        "NA",
        "NC",
        "NE",
        "NF",
        "NG",
        "NI",
        "NL",
        "NO",
        "NP",
        "NR",
        "NU",
        "NZ",
        "OM",
        "PA",
        "PE",
        "PF",
        "PG",
        "PH",
        "PK",
        "PL",
        "PM",
        "PN",
        "PR",
        "PS",
        "PT",
        "PW",
        "PY",
        "QA",
        "RE",
        "RO",
        "RS",
        "RU",
        "RW",
        "SA",
        "SB",
        "SC",
        "SD",
        "SE",
        "SG",
        "SH",
        "SI",
        "SJ",
        "SK",
        "SL",
        "SM",
        "SN",
        "SO",
        "SR",
        "SS",
        "ST",
        "SV",
        "SX",
        "SY",
        "SZ",
        "TC",
        "TD",
        "TF",
        "TG",
        "TH",
        "TJ",
        "TK",
        "TL",
        "TM",
        "TN",
        "TO",
        "TR",
        "TT",
        "TV",
        "TW",
        "TZ",
        "UA",
        "UG",
        "UM",
        "US",
        "UY",
        "UZ",
        "VA",
        "VC",
        "VE",
        "VG",
        "VI",
        "VN",
        "VU",
        "WF",
        "WS",
        "XK",  # Kosovo (user-assigned, universally recognized)
        "YE",
        "YT",
        "ZA",
        "ZM",
        "ZW",
        # Region identifiers used by frontend location selector
        "EU",
        "AP",
    }
)

_VALID_CA_PROVINCES = frozenset(
    {
        "AB",
        "BC",
        "MB",
        "NB",
        "NL",
        "NS",
        "NT",
        "NU",
        "ON",
        "PE",
        "QC",
        "SK",
        "YT",
    }
)


class RegisterHostRequest(BaseModel):
    """Human-facing host registration from the dashboard UI.

    Separate from HostIn which is the machine-to-machine agent schema.
    Fields that the agent reports (host_id, ip, free_vram_gb) are
    auto-generated or derived here.
    """

    hostname: str = Field(min_length=1, max_length=128, pattern=r"^[\w .\-]+$")
    gpu_model: str = Field(min_length=1, max_length=64)
    vram_gb: float = Field(gt=0, le=192)
    cost_per_hour: float = Field(default=0.20, ge=0.01, le=100.0)
    country: str = Field(default="CA", min_length=2, max_length=2)
    province: str = Field(default="", max_length=4)
    region: str = Field(default="", max_length=32)
    notes: str = Field(default="", max_length=1000)
    spot_enabled: bool = True
    spot_gpu_slots: int | None = Field(default=None, ge=0, le=64)
    spot_min_cents: int | None = Field(default=None, ge=0, le=1_000_000)

    @field_validator("hostname")
    @classmethod
    def hostname_normalize(cls, v: str) -> str:
        """Strip leading/trailing whitespace and reject all-blank hostnames."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("hostname must contain at least one non-space character")
        return stripped

    @field_validator("gpu_model")
    @classmethod
    def gpu_model_validate(cls, v: str) -> str:
        """Validate gpu_model against the canonical set of supported models.
        The set mirrors the frontend GPU_MODELS catalogue."""
        stripped = normalize_gpu_model(v)
        if not stripped:
            raise ValueError("gpu_model must not be blank")
        if stripped not in _VALID_GPU_MODELS:
            raise ValueError(
                f"'{stripped}' is not a recognized GPU model; "
                f"see the GPU selector for supported models"
            )
        return stripped

    @field_validator("country")
    @classmethod
    def country_validate(cls, v: str) -> str:
        """Normalize to uppercase and validate against ISO 3166-1 alpha-2
        codes plus the EU/AP region identifiers used by the frontend."""
        upper = v.strip().upper()
        if upper not in _VALID_COUNTRY_CODES:
            raise ValueError(f"'{upper}' is not a valid ISO 3166-1 alpha-2 country code")
        return upper

    @field_validator("province")
    @classmethod
    def province_normalize(cls, v: str) -> str:
        """Normalize to uppercase. Cross-field validation with country
        happens in the model_validator below."""
        return v.strip().upper()

    @field_validator("region")
    @classmethod
    def region_normalize(cls, v: str) -> str:
        return normalize_region(v, default="") if v.strip() else ""

    @field_validator("notes")
    @classmethod
    def notes_sanitize(cls, v: str) -> str:
        """Strip leading/trailing whitespace and reject control characters
        (except newlines and tabs which are legitimate in freeform notes)."""
        stripped = v.strip()
        if any(ord(c) < 32 and c not in ("\n", "\r", "\t") for c in stripped):
            raise ValueError("notes contains invalid control characters")
        return stripped

    @model_validator(mode="after")
    def province_matches_country(self) -> "RegisterHostRequest":
        """If country is CA and a province is provided, verify it's a real
        Canadian province/territory code."""
        if self.country == "CA" and self.province:
            if self.province not in _VALID_CA_PROVINCES:
                raise ValueError(
                    f"province '{self.province}' is not a valid Canadian province code; "
                    f"valid codes: {', '.join(sorted(_VALID_CA_PROVINCES))}"
                )
        return self


@router.post("/api/hosts/register", tags=["Hosts"])
def api_register_host_web(h: RegisterHostRequest, request: Request):
    """Register a host from the web dashboard UI.

    Generates host_id, derives IP from request, and maps the simplified
    web form fields to the internal register_host() function. The existing
    PUT /host endpoint is untouched — that's for the worker agent.
    """
    from routes._deps import _require_scope, _get_current_user

    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    _require_scope(user, "hosts:write")

    host_id = str(uuid.uuid4())[:12]
    client_ip = request.client.host if request.client else "0.0.0.0"

    # Register using the existing scheduler function
    from spot_pricing import suggested_spot_min_cents

    spot_min = (
        h.spot_min_cents
        if h.spot_min_cents is not None
        else suggested_spot_min_cents(h.gpu_model)
    )
    platform_rate = platform_rate_cad(h.gpu_model)
    cost_per_hour = h.cost_per_hour if h.cost_per_hour > 0 else platform_rate
    if cost_per_hour <= 0:
        cost_per_hour = 0.20

    entry = register_host(
        host_id,
        client_ip,
        h.gpu_model,
        h.vram_gb,
        h.vram_gb,  # free_vram_gb = total at registration
        cost_per_hour,
        country=h.country,
        province=h.province,
        region=h.region,
        spot_enabled=h.spot_enabled,
        spot_gpu_slots=h.spot_gpu_slots,
        spot_min_cents=spot_min,
    )

    # Store web-form-specific fields (validators already strip/normalize)
    entry["hostname"] = h.hostname
    if h.notes:
        entry["notes"] = h.notes
    entry["country"] = h.country
    entry["province"] = h.province
    entry["region"] = normalize_region(h.region, country=h.country, province=h.province)
    entry["owner"] = user.get("user_id", user.get("sub", ""))

    # New web-registered hosts start as pending until agent connects
    entry.setdefault("admitted", False)
    entry["status"] = "pending"

    # Persist the full entry
    from scheduler import _atomic_mutation, _upsert_host_row, _migrate_hosts_if_needed

    with _atomic_mutation() as conn:
        _migrate_hosts_if_needed(conn)
        _upsert_host_row(conn, entry)

    # Auto-compute score and auto-list on marketplace
    from scheduler import estimate_compute_score, register_compute_score, list_rig

    score = estimate_compute_score(h.gpu_model)
    register_compute_score(host_id, h.gpu_model, score)
    entry["compute_score"] = score

    list_rig(
        host_id,
        h.gpu_model,
        h.vram_gb,
        h.cost_per_hour,
        description=f"{h.gpu_model} ({h.vram_gb}GB) in {h.country.upper()}",
        owner=entry["owner"],
        region=entry.get("region", ""),
        province=entry.get("province", ""),
    )

    try:
        update_host_spot_settings(
            host_id,
            spot_enabled=entry.get("spot_enabled", True),
            spot_gpu_slots=entry.get("spot_gpu_slots"),
            spot_min_cents=entry.get("spot_min_cents", spot_min),
        )
    except Exception as exc:
        log.debug("Spot offer sync on web register skipped: %s", exc)

    # Auto-create verification + reputation records
    try:
        ve = get_verification_engine()
        if not ve.store.get_verification(host_id):
            from verification import HostVerification, HostVerificationState

            ve.store.save_verification(
                HostVerification(
                    verification_id=str(uuid.uuid4())[:12],
                    host_id=host_id,
                    state=HostVerificationState.UNVERIFIED,
                )
            )
            log.info("VERIFY RECORD created for new host %s", host_id)
        re = get_reputation_engine()
        re.add_verification(host_id, VerificationType.EMAIL)
    except Exception as e:
        log.exception("Non-fatal: could not bootstrap verification/reputation for %s", host_id)

    broadcast_sse(
        "host_registered",
        {
            "host_id": host_id,
            "hostname": h.hostname,
            "gpu_model": h.gpu_model,
            "country": entry.get("country", ""),
        },
    )
    return {"ok": True, "host": enrich_host_for_api(entry)}


class HostSpotSettingsUpdate(BaseModel):
    spot_enabled: bool | None = None
    spot_gpu_slots: int | None = Field(default=None, ge=0, le=64)
    spot_min_cents: int | None = Field(default=None, ge=0, le=1_000_000)


@router.patch("/api/hosts/{host_id}/spot-settings", tags=["Hosts"])
def api_update_host_spot_settings(host_id: str, body: HostSpotSettingsUpdate, request: Request):
    """Update provider spot controls for a host (scheduler + marketplace offer)."""
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    resolved, _hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    _require_host_operator(user, resolved)

    updated = update_host_spot_settings(
        resolved,
        spot_enabled=body.spot_enabled,
        spot_gpu_slots=body.spot_gpu_slots,
        spot_min_cents=body.spot_min_cents,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")

    from spot_pricing import effective_spot_rate_cad

    preview = effective_spot_rate_cad(
        str(updated.get("gpu_model") or ""),
        int(updated.get("spot_min_cents", 0) or 0),
    )
    broadcast_sse("host_update", {"host_id": resolved, "spot_enabled": updated.get("spot_enabled")})
    return {"ok": True, "host": updated, "spot_preview": preview}


@router.get("/api/hosts/{host_id}/spot-preview", tags=["Hosts"])
def api_host_spot_preview(host_id: str, request: Request, spot_min_cents: int | None = None):
    """Live spot rate preview for a host GPU and optional provider floor."""
    user = _require_auth(request)
    _require_scope(user, "hosts:read")
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host = next(h for h in hosts if h["host_id"] == resolved)
    from spot_pricing import effective_spot_rate_cad, suggested_spot_min_cents

    floor = (
        spot_min_cents
        if spot_min_cents is not None
        else int(host.get("spot_min_cents", suggested_spot_min_cents(host.get("gpu_model", ""))) or 0)
    )
    preview = effective_spot_rate_cad(str(host.get("gpu_model") or ""), floor)
    return {"ok": True, "host_id": resolved, "spot_preview": preview}


@router.get("/host/{host_id}", tags=["Hosts"])
def api_get_host(host_id: str, request: Request):
    """Get a single host by ID."""
    user = _require_auth(request)
    _require_scope(user, "hosts:read")
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host = next(h for h in hosts if h["host_id"] == resolved)
    return {"ok": True, "host": enrich_host_for_api(host)}


@router.get("/hosts", tags=["Hosts"])
def api_list_hosts(request: Request, active_only: bool = True):
    """List all hosts."""
    user = _require_auth(request)
    _require_scope(user, "hosts:read")
    return {"hosts": [enrich_host_for_api(h) for h in list_hosts(active_only=active_only)]}


@router.get("/host/{host_id}/maintenance", tags=["Hosts"])
def api_host_maintenance(host_id: str, request: Request):
    """Return maintenance readiness for a host."""
    _require_admin(request)
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host_id = resolved
    host = next(h for h in hosts if h["host_id"] == host_id)

    interactive_jobs, serverless_jobs = _blocking_host_jobs(host_id)
    interactive_summary = [
        {
            "job_id": job.get("job_id"),
            "name": job.get("name"),
            "status": job.get("status"),
            "owner": job.get("owner"),
        }
        for job in interactive_jobs
    ]
    serverless_summary = [
        {
            "job_id": job.get("job_id"),
            "name": job.get("name"),
            "status": job.get("status"),
            "owner": job.get("owner"),
            "job_type": job.get("job_type"),
        }
        for job in serverless_jobs
    ]
    blocking_count = len(interactive_summary) + len(serverless_summary)
    safe_to_maintain = host.get("status") == "draining" and blocking_count == 0

    return {
        "ok": True,
        "host_id": host_id,
        "status": host.get("status"),
        "draining": host.get("status") == "draining",
        "admitted": host.get("admitted", False),
        "active_interactive_instances": len(interactive_summary),
        "interactive_instances": interactive_summary,
        "active_serverless_workers": len(serverless_summary),
        "serverless_workers": serverless_summary,
        "safe_to_maintain": safe_to_maintain,
    }


@router.post("/host/{host_id}/drain", tags=["Hosts"])
def api_drain_host(host_id: str, request: Request):
    """Stop new placements on a host without evicting active instances."""
    _require_admin(request)
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host_id = resolved
    host = next(h for h in hosts if h["host_id"] == host_id)
    if host.get("status") == "dead":
        raise HTTPException(status_code=409, detail="Cannot drain a dead host")

    updated = set_host_draining(host_id, draining=True)
    from scheduler import run_drain_preemptions

    preempted = run_drain_preemptions(host_id)
    broadcast_sse("host_update", {"host_id": host_id, "status": "draining"})
    return {
        "ok": True,
        "host": updated,
        "preempted": [j["job_id"] for j in preempted],
        "maintenance": api_host_maintenance(host_id, request),
    }


@router.post("/host/{host_id}/undrain", tags=["Hosts"])
def api_undrain_host(host_id: str, request: Request):
    """Restore a drained host to active or pending status."""
    _require_admin(request)
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host_id = resolved
    host = next(h for h in hosts if h["host_id"] == host_id)
    if host.get("status") == "dead":
        raise HTTPException(status_code=409, detail="Cannot undrain a dead host")

    updated = set_host_draining(host_id, draining=False)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    broadcast_sse("host_update", {"host_id": host_id, "status": updated.get("status", "pending")})
    return {"ok": True, "host": updated}


@router.delete("/host/{host_id}", tags=["Hosts"])
def api_remove_host(host_id: str, request: Request):
    """Remove a host."""
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    _require_host_operator(user, host_id)
    resolved, _hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host_id = resolved
    remove_host(host_id)
    broadcast_sse("host_removed", {"host_id": host_id})
    return {"ok": True, "removed": host_id}


@router.post("/hosts/check", tags=["Hosts"])
def api_check_hosts(request: Request):
    """Ping all hosts and update status."""
    _require_admin(request)
    results = check_hosts()
    return {"results": results}


@router.get("/compute-score/{host_id}", tags=["Hosts"])
def api_get_compute_score(host_id: str):
    """Get the compute score (XCU) for a host."""
    score = get_compute_score(host_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"No compute score for host {host_id}")
    return {"ok": True, "host_id": host_id, "score": score}


@router.get("/compute-scores", tags=["Hosts"])
def api_list_compute_scores():
    """List compute scores for all hosts."""
    hosts = list_hosts(active_only=False)
    scores = {}
    for h in hosts:
        score = get_compute_score(h["host_id"])
        if score is not None:
            scores[h["host_id"]] = {
                "score": score,
                "gpu_model": h.get("gpu_model", "unknown"),
            }
    return {"ok": True, "scores": scores}
