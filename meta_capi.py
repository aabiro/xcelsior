"""Meta Conversions API for xcelsior.

Browser Pixel and this server sender share an event_id so Meta can deduplicate.
The access token stays in the environment. Nothing here raises into a request.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Any

import httpx

log = logging.getLogger("xcelsior.meta_capi")

GRAPH_VERSION = "v25.0"
_ATTR_CACHE: dict[str, tuple[str, str]] = {}
_FBP_RE = re.compile(r"^fb\.1\.\d{10,16}\.\d{5,20}$")
_FBC_RE = re.compile(r"^fb\.1\.\d{10,16}\.[A-Za-z0-9_\-]{8,200}$")


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def pixel_id() -> str:
    return _env("META_DATASET_ID") or _env("META_PIXEL_ID")


def configured() -> bool:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    return bool(pixel_id() and _env("META_CAPI_ACCESS_TOKEN"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_email(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if "@" not in normalized:
        return None
    return _sha256(normalized)


def hash_phone(value: str | None) -> str | None:
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) < 8:
        return None
    return _sha256(digits)


def hash_name(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z]", "", value.strip().lower())
    if not normalized:
        return None
    return _sha256(normalized)


def hash_country(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^a-z]", "", value.strip().lower())[:2]
    if len(normalized) != 2:
        return None
    return _sha256(normalized)


def hash_external_id(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return _sha256(normalized)


def split_name(full_name: str | None) -> tuple[str | None, str | None]:
    parts = [part for part in (full_name or "").strip().split() if part]
    if not parts:
        return None, None
    if len(parts) == 1:
        return parts[0], None
    return parts[0], parts[-1]


def _clean_click_id(value: str | None, kind: str) -> str | None:
    if not value:
        return None
    token = value.strip()
    if len(token) > 500:
        return None
    if kind == "fbp" and _FBP_RE.match(token):
        return token
    if kind == "fbc" and _FBC_RE.match(token):
        return token
    return None


def request_context(request: Any | None) -> dict[str, str]:
    if request is None:
        return {}
    headers = getattr(request, "headers", {}) or {}
    cookies = getattr(request, "cookies", {}) or {}

    def header(name: str) -> str:
        getter = getattr(headers, "get", None)
        if callable(getter):
            return str(getter(name) or getter(name.lower()) or "").strip()
        return ""

    fbp = _clean_click_id(header("x-fb-fbp") or cookies.get("_fbp"), "fbp")
    fbc = _clean_click_id(header("x-fb-fbc") or cookies.get("_fbc"), "fbc")
    event_id = header("x-event-id")[:128]
    forwarded = header("cf-connecting-ip") or header("x-forwarded-for").split(",")[0].strip()
    client = getattr(request, "client", None)
    client_host = getattr(client, "host", "") if client else ""
    ip = forwarded or str(client_host or "")
    ua = header("user-agent")[:512]
    referer = header("referer")[:1000]
    out: dict[str, str] = {}
    if fbp:
        out["fbp"] = fbp
    if fbc:
        out["fbc"] = fbc
    if event_id:
        out["event_id"] = event_id
    if ip:
        out["client_ip"] = ip
    if ua:
        out["user_agent"] = ua
    if referer:
        out["event_source_url"] = referer
    return out


def _stored_ads(user: dict | None) -> dict[str, str]:
    if not user:
        return {}
    raw = user.get("preferences") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = {}
    if not isinstance(raw, dict):
        return {}
    ads = raw.get("meta_ads") or {}
    return ads if isinstance(ads, dict) else {}


def build_user_data(
    *,
    email: str | None = None,
    phone: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    external_id: str | None = None,
    country: str | None = None,
    fbp: str | None = None,
    fbc: str | None = None,
    client_ip: str | None = None,
    client_user_agent: str | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    hashed_email = hash_email(email)
    hashed_phone = hash_phone(phone)
    hashed_fn = hash_name(first_name)
    hashed_ln = hash_name(last_name)
    hashed_external = hash_external_id(external_id)
    hashed_country = hash_country(country)
    if hashed_email:
        data["em"] = [hashed_email]
    if hashed_phone:
        data["ph"] = [hashed_phone]
    if hashed_fn:
        data["fn"] = [hashed_fn]
    if hashed_ln:
        data["ln"] = [hashed_ln]
    if hashed_external:
        data["external_id"] = [hashed_external]
    if hashed_country:
        data["country"] = [hashed_country]
    click_fbp = _clean_click_id(fbp, "fbp")
    click_fbc = _clean_click_id(fbc, "fbc")
    if click_fbp:
        data["fbp"] = click_fbp
    if click_fbc:
        data["fbc"] = click_fbc
    if client_ip:
        data["client_ip_address"] = client_ip
    if client_user_agent:
        data["client_user_agent"] = client_user_agent
    return data


def build_event(
    event_name: str,
    event_id: str,
    *,
    custom_data: dict[str, Any] | None = None,
    event_source_url: str | None = None,
    action_source: str = "website",
    **user_fields: Any,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "event_name": event_name,
        "event_time": int(time.time()),
        "event_id": event_id,
        "action_source": action_source,
        "user_data": build_user_data(**user_fields),
    }
    if event_source_url:
        event["event_source_url"] = event_source_url
    if custom_data:
        event["custom_data"] = {key: value for key, value in custom_data.items() if value is not None}
    return event


def _identity_from_user(user: dict | None, request: Any | None = None) -> dict[str, Any]:
    ctx = request_context(request)
    ads = _stored_ads(user)
    user = user or {}
    first, last = split_name(str(user.get("name") or ""))
    return {
        "email": user.get("email"),
        "phone": user.get("phone"),
        "first_name": first,
        "last_name": last,
        "external_id": user.get("user_id") or user.get("customer_id"),
        "country": user.get("country") or "CA",
        "fbp": ctx.get("fbp") or ads.get("fbp"),
        "fbc": ctx.get("fbc") or ads.get("fbc"),
        "client_ip": ctx.get("client_ip") or ads.get("ip"),
        "client_user_agent": ctx.get("user_agent") or ads.get("ua"),
        "event_source_url": ctx.get("event_source_url"),
        "event_id": ctx.get("event_id"),
    }


def capture_request_attribution(user: dict | None, request: Any | None) -> None:
    """Remember first-party click ids for a later webhook. Never raises."""
    if not user or request is None:
        return
    email = str(user.get("email") or "").strip().lower()
    if not email:
        return
    ctx = request_context(request)
    fbp = ctx.get("fbp") or ""
    fbc = ctx.get("fbc") or ""
    if not fbp and not fbc:
        return
    signature = (fbp, fbc)
    if _ATTR_CACHE.get(email) == signature:
        return
    _ATTR_CACHE[email] = signature
    try:
        from db import UserStore

        existing = UserStore.get_user(email) or {}
        raw = existing.get("preferences") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        ads = dict(raw.get("meta_ads") or {})
        if fbp:
            ads["fbp"] = fbp
        if fbc:
            ads["fbc"] = fbc
        if ctx.get("client_ip"):
            ads["ip"] = ctx["client_ip"]
        if ctx.get("user_agent"):
            ads["ua"] = ctx["user_agent"]
        raw["meta_ads"] = ads
        UserStore.update_user(email, {"preferences": raw})
    except Exception:
        log.debug("meta attribution persist failed", exc_info=True)


def deliver(events: list[dict[str, Any]]) -> dict[str, Any]:
    dataset = pixel_id()
    token = _env("META_CAPI_ACCESS_TOKEN")
    if not dataset or not token or not events:
        return {"skipped": True}
    payload: dict[str, Any] = {"data": events}
    test_code = _env("META_TEST_EVENT_CODE")
    if test_code:
        payload["test_event_code"] = test_code
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/{dataset}/events"
    last_error = "meta capi request failed"
    for attempt in range(3):
        try:
            response = httpx.post(url, params={"access_token": token}, json=payload, timeout=8.0)
        except httpx.HTTPError as exc:
            last_error = str(exc)
            time.sleep(0.2 * (2**attempt))
            continue
        if response.status_code == 429 or response.status_code >= 500:
            last_error = f"HTTP {response.status_code}"
            time.sleep(0.2 * (2**attempt))
            continue
        if response.status_code >= 400:
            log.warning("meta capi rejected %s", response.status_code)
            return {"ok": False, "status": response.status_code}
        body = response.json()
        log.info(
            "meta capi accepted events=%s fbtrace=%s",
            body.get("events_received"),
            body.get("fbtrace_id"),
        )
        return {"ok": True, **body}
    log.warning("meta capi gave up: %s", last_error)
    return {"ok": False, "error": last_error}


def track_event(
    event_name: str,
    event_id: str,
    *,
    user: dict | None = None,
    request: Any | None = None,
    custom_data: dict[str, Any] | None = None,
    action_source: str = "website",
    **overrides: Any,
) -> None:
    """Queue one event. Returns immediately and never raises."""
    try:
        capture_request_attribution(user, request)
        if not event_name or not event_id or not configured():
            return
        identity = _identity_from_user(user, request)
        identity.update({key: value for key, value in overrides.items() if value})
        event = build_event(
            event_name,
            str(event_id),
            custom_data=custom_data,
            event_source_url=identity.pop("event_source_url", None),
            action_source=action_source,
            email=identity.get("email"),
            phone=identity.get("phone"),
            first_name=identity.get("first_name"),
            last_name=identity.get("last_name"),
            external_id=identity.get("external_id"),
            country=identity.get("country"),
            fbp=identity.get("fbp"),
            fbc=identity.get("fbc"),
            client_ip=identity.get("client_ip"),
            client_user_agent=identity.get("client_user_agent"),
        )

        def _send() -> None:
            try:
                deliver([event])
            except Exception:
                log.debug("meta capi send failed", exc_info=True)

        threading.Thread(target=_send, name="meta-capi", daemon=True).start()
    except Exception:
        log.debug("meta capi schedule failed", exc_info=True)


def ads_metadata(request: Any | None) -> dict[str, str]:
    ctx = request_context(request)
    out: dict[str, str] = {}
    if ctx.get("fbp"):
        out["fbp"] = ctx["fbp"]
    if ctx.get("fbc"):
        out["fbc"] = ctx["fbc"]
    if ctx.get("client_ip"):
        out["client_ip"] = ctx["client_ip"]
    return out


def dataset_quality() -> dict[str, Any]:
    dataset = pixel_id()
    token = _env("META_CAPI_ACCESS_TOKEN")
    if not dataset or not token:
        return {"ok": False, "error": "meta_not_configured"}
    url = f"https://graph.facebook.com/{GRAPH_VERSION}/dataset_quality"
    try:
        response = httpx.get(
            url,
            params={
                "access_token": token,
                "dataset_id": dataset,
                "fields": "web{event_name,event_match_quality,dedupe_key_feedback,data_freshness}",
            },
            timeout=15.0,
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        body = response.json()
    except Exception:
        body = {"raw": response.text[:500]}
    if response.status_code >= 400:
        return {"ok": False, "status": response.status_code, "body": body}
    return {"ok": True, "dataset_id": dataset, "quality": body}
