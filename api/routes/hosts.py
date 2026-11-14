"""Routes: hosts."""

import logging
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
from security import check_node_versions, recommend_runtime
from reputation import VerificationType, get_reputation_engine
from host_metadata import (
    CANONICAL_GPU_MODELS,
    enrich_host_for_api,
    normalize_gpu_model,
    normalize_region,
    platform_rate_cad,
)

router = APIRouter()


@router.get("/api/locations", tags=["Hosts"])
def api_location_catalog(country: str = "", purpose: str = "host"):
    """Return named ISO countries, subdivisions, labels, and required address fields.

    ``purpose=address`` follows the country's postal format (no state where the
    post uses none); the default ``host`` keeps the subdivision a host's region needs.
    """
    from host_metadata import location_catalog

    try:
        return {"ok": True, **location_catalog(country, purpose)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


class AddressSearchRequest(BaseModel):
    country: str = ""
    text: str = Field(min_length=3, max_length=200)
    lang: str = Field(default="", max_length=8)  # the page's language, for place names


class AddressCheckRequest(BaseModel):
    """The same fields and limits the profile save accepts."""

    country: str = Field(default="", max_length=2)
    province: str = Field(default="", max_length=8)
    address_line1: str = Field(default="", max_length=200)
    address_line2: str = Field(default="", max_length=200)
    city: str = Field(default="", max_length=80)
    postal_code: str = Field(default="", max_length=32)


def _address_lookup_country(value: str) -> str:
    import pycountry

    country = value.strip().upper()
    if pycountry.countries.get(alpha_2=country) is None:
        raise HTTPException(422, "Select a valid country")
    return country


def _visitor_country_hint(request: Request) -> str:
    """Cloudflare's country for the visitor, used only to order a search with no country.

    It never leaves this server. A forged header can only reorder the caller's
    own suggestions. Cloudflare sends XX for unknown and T1 for Tor; neither is
    a country, so both are dropped.
    """
    import pycountry

    value = request.headers.get("cf-ipcountry", "").strip().upper()
    if len(value) != 2 or pycountry.countries.get(alpha_2=value) is None:
        return ""
    return value


# Countries that write the house number after the street name ("Hauptstraße 5").
_NUMBER_AFTER_STREET = frozenset(
    "AR AT BA BE BG BR CH CL CO CZ DE DK EE ES FI GR HR HU IS IT LT LU LV MX NL NO PE PL PT "
    "RO RS SE SI SK TR UY".split()
)
_ADDRESS_UNAVAILABLE = "Address search is unavailable. Enter your address below."


# komoot's public Photon blocks clients that send too much (one burst of testing
# got a whole IP refused), and every visitor's search leaves from this server's
# one address. So: answers are cached briefly, and after a failure the server
# stops asking for a minute and the form falls back to manual entry at once.
_PHOTON_CACHE: dict[tuple[tuple[str, str], ...], tuple[float, list[dict]]] = {}
_PHOTON_CACHE_SECONDS = 600.0
_PHOTON_CACHE_SIZE = 2048
_PHOTON_PAUSE_SECONDS = 60.0
_photon_paused_until = 0.0
# HERE Autocomplete includes the postal code in the suggestion. When HERE fails
# (a rejected key - the Maven repository token is not a REST key - the monthly
# quota spent, an outage) it is left alone for a minute, and search falls back
# to Photon instead of failing the form.
_HERE_CACHE: dict[tuple[tuple[str, str], ...], tuple[float, list[dict]]] = {}
_here_paused_until = 0.0
_HERE_AUTOCOMPLETE = "https://autocomplete.search.hereapi.com/v1/autocomplete"


class _HereUnavailable(Exception):
    """HERE gave no usable answer: a refused key, a spent quota, a timeout or an outage."""


def _photon_features(params: list[tuple[str, str]]) -> list[dict]:
    """Ask Photon (OpenStreetMap search) for address matches. Free; no key.

    XCELSIOR_PHOTON_URL points at a self-hosted Photon when there is one; the
    default is komoot's public instance, which allows only modest use. The
    visitor's IP and cookies stay here: only the query goes out.
    """
    import httpx

    global _photon_paused_until
    key = tuple(params)
    now = time.monotonic()
    cached = _PHOTON_CACHE.get(key)
    if cached is not None and now - cached[0] < _PHOTON_CACHE_SECONDS:
        return cached[1]
    if now < _photon_paused_until:
        raise HTTPException(502, _ADDRESS_UNAVAILABLE)
    base = (os.environ.get("XCELSIOR_PHOTON_URL") or "https://photon.komoot.io").rstrip("/")
    try:
        response = httpx.get(
            f"{base}/api/",
            params=key,  # repeated keys (layer=house&layer=street); a tuple satisfies httpx's types
            headers={"User-Agent": "Xcelsior address search (+https://xcelsior.ca)"},
            timeout=6.0,
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        _photon_paused_until = time.monotonic() + _PHOTON_PAUSE_SECONDS
        raise HTTPException(502, _ADDRESS_UNAVAILABLE) from None
    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise HTTPException(502, _ADDRESS_UNAVAILABLE)
    features = [
        f for f in features if isinstance(f, dict) and isinstance(f.get("properties"), dict)
    ]
    if len(_PHOTON_CACHE) >= _PHOTON_CACHE_SIZE:
        _PHOTON_CACHE.pop(next(iter(_PHOTON_CACHE)))  # oldest first: dicts keep insertion order
    _PHOTON_CACHE[key] = (now, features)
    return features


_HOUSE_NUMBER = re.compile(r"^\d+[A-Za-z]?(?:[-/]\d+[A-Za-z]?)*$")


def _typed_house_number(text: str) -> str:
    """The house number someone typed ("300 Dufferin…", "Via del Corso 300"), if any."""
    words = text.split(",", 1)[0].split()
    if len(words) < 2:
        return ""
    for word in (words[0], words[-1]):
        if _HOUSE_NUMBER.match(word):
            return word
    return ""


def _street_was_typed(street: str, text: str) -> bool:
    """Whether a street is the one being typed: some typed word starts one of its words."""
    from host_metadata import _fold

    street_words = _fold(street).replace("-", " ").split()
    typed = [
        w
        for w in _fold(text).replace(",", " ").split()
        if len(w) >= 3 and not _HOUSE_NUMBER.match(w)
    ]
    return any(word.startswith(t) for t in typed for word in street_words)


def _fully_typed(address: dict[str, str], text: str, number: str) -> bool:
    """Whether an address carries the typed number and every typed word starts one of its words."""
    from host_metadata import _fold

    words = (
        _fold(" ".join((address["address_line1"], address["city"])))
        .replace(",", " ")
        .replace("-", " ")
        .split()
    )
    line = re.split(r"[\s,]+", address["address_line1"])
    typed = [w for w in _fold(text).replace(",", " ").split() if len(w) >= 3 and w != _fold(number)]
    return number in line and all(any(word.startswith(t) for word in words) for t in typed)


def _non_latin(value: str) -> bool:
    """Chinese, Japanese or Korean script, where the number follows the street."""
    return any(ord(ch) >= 0x2E80 for ch in value)


def _photon_address(
    properties: dict, typed_number: str = "", typed: str = ""
) -> dict[str, str] | None:
    """One Photon result as the profile's address fields, or None if it isn't an address.

    ``typed_number`` is the house number in ``typed``, the search text. When
    OpenStreetMap knows the typed street but not that number, the street is
    offered with it, as Google Maps does. The street's postcode stays off that
    house: it belongs to one segment of the road, and copying it is how a
    suggestion arrives with the wrong ZIP. The confirm step shows the address
    back before it is saved.
    """
    from host_metadata import _fold, accepted_postal_code, resolve_subdivision, standard_postal_code

    def text(key: str) -> str:
        value = properties.get(key)
        return " ".join(str(value).split()) if isinstance(value, (str, int)) else ""

    country = text("countrycode").upper()
    if not country:
        return None
    kind = properties.get("type")
    street, number = text("street"), text("housenumber")
    if kind == "street" and not street:
        street = text("name")
    if (
        kind == "street"
        and not number
        and typed_number
        and country != "JP"
        and typed_number not in street.split()
        and _street_was_typed(street, typed)
    ):
        number = typed_number
    place_city = text("city") or text("locality") or text("district")
    city = place_city
    if country == "JP":
        # Japan numbers blocks, not streets: Marunouchi 1-chōme, block 9, building 1 is
        # "Marunouchi 1-9-1"; Photon puts the block in `street` and the chōme in `locality`.
        area, ward = text("locality"), text("district")
        block = "-".join(part for part in (street if street.isdigit() else "", number) if part)
        if area and block:
            line1 = f"{area}-{block}" if area[-1].isdigit() else f"{area}{block}"
        elif street and not street.isdigit():
            line1 = f"{street}{number}" if _non_latin(street) else f"{number} {street}".strip()
        else:
            return None
        if ward and _fold(ward) != _fold(city):
            if not city or _fold(city) in {_fold(text("state")), "tokyo"}:
                city = ward  # a Tokyo ward is the city of its address
            else:
                line1 = f"{ward}{line1}" if _non_latin(line1) else f"{line1}, {ward}"
    elif not street:
        # Villages that number houses without street names ("Lhota 15").
        place = text("locality") or text("district")
        if not (number and place):
            return None
        line1 = f"{place} {number}"
    elif not number:
        line1 = street
    elif country in _NUMBER_AFTER_STREET or _non_latin(street):
        line1 = f"{street} {number}"
    else:
        line1 = f"{number} {street}"
    # The segment code can still name the region (Montreal's H is Quebec) even
    # when it must not be offered as this building's ZIP.
    formatted = standard_postal_code(country, text("postcode"))
    postal = "" if kind == "street" else accepted_postal_code(country, formatted)
    return {
        "address_line1": line1,
        "address_line2": "",
        "city": city,
        "postal_code": postal,
        "country": country,
        # Only an exact match fills the region; otherwise the person picks it.
        "province": resolve_subdivision(
            country, state=text("state"), county=text("county"), city=place_city, postal_code=formatted
        ),
    }


_PHOTON_LANGUAGES = frozenset({"en", "fr", "de"})


def _region_in_postal_format(country: str) -> bool:
    from i18naddress import get_validation_rules

    return "%S" in get_validation_rules({"country_code": country}).address_format


def _here_api_key() -> str:
    return (os.environ.get("XCELSIOR_HERE_API_KEY") or "").strip()


def _here_items(params: list[tuple[str, str]]) -> list[dict]:
    """One HERE Autocomplete call. The key stays on this server and out of the cache.

    Autocomplete is the free tier with 30,000 transactions a month, and each
    item already carries the house number, street, city, region and postal code.
    """
    import httpx

    cache_key = tuple(params)
    now = time.monotonic()
    cached = _HERE_CACHE.get(cache_key)
    if cached is not None and now - cached[0] < _PHOTON_CACHE_SECONDS:
        return cached[1]
    try:
        response = httpx.get(
            _HERE_AUTOCOMPLETE,
            params=[*params, ("apiKey", _here_api_key())],
            headers={"User-Agent": "Xcelsior address search (+https://xcelsior.ca)"},
            timeout=6.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise _HereUnavailable(type(exc).__name__) from None
    try:
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise _HereUnavailable(f"HTTP {response.status_code}") from None
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise _HereUnavailable("no items in the answer")
    items = [item for item in items if isinstance(item, dict) and isinstance(item.get("address"), dict)]
    if len(_HERE_CACHE) >= _PHOTON_CACHE_SIZE:
        _HERE_CACHE.pop(next(iter(_HERE_CACHE)))
    _HERE_CACHE[cache_key] = (now, items)
    return items


def _here_address(item: dict) -> dict[str, str] | None:
    """One HERE suggestion as the profile's address fields.

    The postal code is HERE's, from its own address file, so it is filled in.
    A street-segment code from OpenStreetMap is a different problem and is not
    copied onto a house.
    """
    import pycountry
    from host_metadata import accepted_postal_code, resolve_subdivision, standard_postal_code

    address = item["address"]

    def text(key: str) -> str:
        value = address.get(key)
        return " ".join(str(value).split()) if isinstance(value, (str, int)) else ""

    country3 = text("countryCode").upper()
    found = pycountry.countries.get(alpha_3=country3) if len(country3) == 3 else None
    country = found.alpha_2 if found is not None else ""
    if not country:
        return None
    street, number = text("street"), text("houseNumber")
    city = text("city") or text("district")
    if not street and not number:
        return None
    if not street:
        line1 = f"{city} {number}".strip()
    elif not number:
        line1 = street
    elif country in _NUMBER_AFTER_STREET or _non_latin(street):
        line1 = f"{street} {number}"
    else:
        line1 = f"{number} {street}"
    formatted = standard_postal_code(country, text("postalCode"))
    state_code = text("stateCode")
    if state_code.upper().startswith(f"{country}-"):
        state_code = state_code.split("-", 1)[1]
    return {
        "address_line1": line1,
        "address_line2": text("building"),
        "city": city,
        "postal_code": accepted_postal_code(country, formatted),
        "country": country,
        "province": resolve_subdivision(
            country,
            state=text("state") or state_code,
            county=text("county"),
            city=text("city"),
            postal_code=formatted,
        ),
    }


def _here_suggestions(text: str, *, country: str, lang: str, visitor_country: str) -> list[dict]:
    """HERE Autocomplete, ranked like the Photon search: nearby first, then the world."""
    lang = lang.strip().lower()[:2]
    base = [("q", text), ("limit", "8"), ("types", "address")]
    if len(lang) == 2 and lang.isalpha():
        base.append(("lang", lang))
    near = country or visitor_country
    batches: list[list[dict]] = []
    if near:
        import pycountry

        found = pycountry.countries.get(alpha_2=near)
        if found is not None:
            batches.append(_here_items([*base, ("in", f"countryCode:{found.alpha_3}")]))
    if not country and (not batches or len(batches[0]) < 4):
        try:
            batches.append(_here_items(base))
        except _HereUnavailable:
            if not batches:
                raise
    addresses: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for batch in batches:
        for item in batch:
            address = _here_address(item)
            if address is None or (country and address["country"] != country):
                continue
            door = (address["address_line1"], address["city"], address["country"])
            if door in seen:
                continue
            seen.add(door)
            addresses.append(address)
            if len(addresses) == 8:
                break
        if len(addresses) == 8:
            break
    return _suggestion_rows(addresses, id_prefix="here")


@router.post("/api/locations/search", tags=["Account"])
def api_address_search(body: AddressSearchRequest, request: Request):
    """Suggest addresses while someone types, the way Google Maps does.

    With XCELSIOR_HERE_API_KEY set to a HERE REST key, suggestions come from
    HERE Autocomplete and include that service's postal code. The Maven
    repository token is not that key. When HERE fails or finds nothing, and
    when no key is set, results come from Photon (OpenStreetMap). Each
    suggestion carries its whole address, so picking one costs nothing more.
    The form shows the result back before it is saved.
    """
    from routes._deps import _require_user_grant

    _require_user_grant(request)
    country = _address_lookup_country(body.country) if body.country.strip() else ""
    text = body.text.strip()
    if len(text) < 3:
        raise HTTPException(422, "Enter at least three address characters")
    suggestions = _address_suggestions(
        text,
        country=country,
        lang=body.lang,
        visitor_country="" if country else _visitor_country_hint(request),
    )
    return {"ok": True, "suggestions": suggestions}


def _suggestion_rows(addresses: list[dict[str, str]], *, id_prefix: str) -> list[dict]:
    """The dropdown rows: the street on the first line, the rest beside it."""
    import pycountry
    from host_metadata import _fold, region_display

    suggestions = []
    for index, address in enumerate(addresses[:8]):
        place = pycountry.countries.get(alpha_2=address["country"])
        region = (
            region_display(address["country"], address["province"])
            if _region_in_postal_format(address["country"])
            else ""
        )
        if _fold(region) == _fold(address["city"]):
            region = ""
        description = ", ".join(
            part
            for part in (
                address["city"],
                region,
                address["postal_code"],
                place.name if place is not None else address["country"],
            )
            if part
        )
        suggestions.append(
            {
                "id": f"{id_prefix}-{index}",
                "text": address["address_line1"],
                "description": description,
                "address": address,
            }
        )
    return suggestions


def _one_postcode_per_door(addresses: list[dict[str, str]]) -> list[dict[str, str]]:
    """One row per door, and only a postcode every record agrees on.

    Photon often returns the same building twice: the house, and a shop or
    office inside it, each with a different ZIP. Offering both lets the click
    fill the wrong one. A blank code does not erase a real one. Two different
    codes are dropped, and the person types theirs.
    """
    folded: list[dict[str, str]] = []
    index: dict[tuple[str, str, str], int] = {}
    for address in addresses:
        door = (address["address_line1"], address["city"], address["country"])
        slot = index.get(door)
        if slot is None:
            index[door] = len(folded)
            folded.append(dict(address))
            continue
        current = folded[slot]
        codes = {code for code in (current["postal_code"], address["postal_code"]) if code}
        if len(codes) == 1:
            current["postal_code"] = next(iter(codes))
        elif len(codes) > 1:
            current["postal_code"] = ""
        if not current["province"] and address["province"]:
            current["province"] = address["province"]
    return folded


def _address_suggestions(text: str, *, country: str, lang: str, visitor_country: str) -> list[dict]:
    """Search, parse and rank: the part of address search that needs no request."""
    global _here_paused_until

    if _here_api_key() and time.monotonic() >= _here_paused_until:
        try:
            found = _here_suggestions(
                text, country=country, lang=lang, visitor_country=visitor_country
            )
        except _HereUnavailable as exc:
            logging.getLogger(__name__).warning(
                "HERE address search unavailable (%s); using Photon for a minute", exc
            )
            _here_paused_until = time.monotonic() + _PHOTON_PAUSE_SECONDS
        else:
            if found:
                return found
    lang = lang.strip().lower()
    base = [("q", text), ("limit", "12"), ("layer", "house"), ("layer", "street")]
    if lang in _PHOTON_LANGUAGES:
        base.append(("lang", lang))
    near_country = country or visitor_country
    typed_number = _typed_house_number(text)
    batches: list[list[dict]] = []
    if near_country:
        batches.append(_photon_features(base + [("countrycode", near_country)]))
    # The world is searched only when no country was chosen and the visitor's
    # country gave fewer than four addresses, so typing a local address costs one
    # request, not two.
    if not country and (
        not batches
        or len([f for f in batches[0] if _photon_address(f["properties"], typed_number, text)]) < 4
    ):
        try:
            batches.append(_photon_features(base))
        except HTTPException:
            if not batches:
                raise
    addresses: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for position, batch in enumerate(batches):
        # A few nearby results first, then the world, so the best match anywhere still shows.
        room = 4 if len(batches) > 1 and position == 0 else 8
        taken = 0
        for feature in batch:
            address = _photon_address(feature["properties"], typed_number, text)
            if address is None or (country and address["country"] != country):
                continue
            # Several businesses at one door are one address.
            key = tuple(
                address[field] for field in ("address_line1", "city", "postal_code", "country")
            )
            if key in seen:
                continue
            seen.add(key)
            addresses.append(address)
            taken += 1
            if taken == room:
                break
    if typed_number and not any(_fully_typed(address, text, typed_number) for address in addresses):
        # Google Maps still offers the street when it has never heard of that
        # number; OpenStreetMap often returns nothing useful while the number is
        # in the query, so ask once more without it and put the number back.
        words = text.split(",", 1)[0].split()
        street_only = " ".join(word for word in words if word != typed_number)
        rest = text.split(",", 1)[1] if "," in text else ""
        retry = ", ".join(part for part in (street_only, rest.strip()) if part)
        if len(retry) >= 3:
            params = [("q", retry), *base[1:]] + (
                [("countrycode", near_country)] if country else []
            )
            try:
                again = _photon_features(params)
            except HTTPException:
                again = []
            numbered = []
            for feature in again:
                address = _photon_address(feature["properties"], typed_number, text)
                if address is None or (country and address["country"] != country):
                    continue
                if not _fully_typed(address, text, typed_number):
                    continue
                key = tuple(
                    address[field] for field in ("address_line1", "city", "postal_code", "country")
                )
                if key not in seen:
                    seen.add(key)
                    numbered.append(address)
            addresses = numbered + addresses
    addresses = _one_postcode_per_door(addresses)
    return _suggestion_rows(addresses, id_prefix="photon")


@router.post("/api/locations/check", tags=["Account"])
def api_address_check(body: AddressCheckRequest, request: Request):
    """Put a finished address into postal format and say what looks wrong. Free: no provider call.

    Errors block a save; warnings only ask the person to confirm. The profile
    save applies the same cleanup, so what this shows is what gets stored.
    """
    from routes._deps import _require_user_grant
    from host_metadata import review_profile_address

    _require_user_grant(request)
    try:
        review = review_profile_address(body.model_dump())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"ok": True, **review}


def _host_owner_ids(host: dict | None) -> set[str]:
    """Identifiers on a registered host row that denote ownership."""
    if not host:
        return set()
    ids: set[str] = set()
    for key in ("owner_id", "owner", "provider_id"):
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


#: Platform-wide host visibility. Operator authority, and only an admin may
#: delegate it — see `oauth_delegation.OPERATOR_SCOPES`.
FLEET_READ_SCOPE = "hosts:fleet"


def visible_hosts(user: dict, hosts: list[dict]) -> list[dict]:
    """The subset of *hosts* that *user* may see.

    `GET /hosts` returned the entire fleet to anyone holding `hosts:read` —
    capacity, GPU models, owners, admission state. That scope is granted to
    every provider so their worker agent can report its own admission status,
    which made competitors' capacity readable by anyone who registered a rig.

    Reclassifying `hosts:read` as operator authority breaks provider onboarding,
    because providers are not admins. The split is additive instead —
    `hosts:read` answers *your* hosts, and platform-wide visibility moved behind
    `hosts:fleet`, which only an admin may delegate.

    Admins keep fleet visibility without holding the scope, matching every other
    operator gate here ("admin, or a machine principal with the scope").

    Ownership resolves through the OAuth client's creator as well as the
    principal itself, so a worker credential still sees the host it was created
    to run — that path is what makes provisioning work.

    The failure this prevents is silent: no error and no refusal, just more rows
    than the caller should see.
    """
    if _is_platform_admin(user):
        return list(hosts)
    if FLEET_READ_SCOPE in set(user.get("scopes") or ()):
        return list(hosts)

    caller_ids = _caller_owner_ids(user)
    creator = _oauth_client_creator(user)
    if creator:
        caller_ids |= _caller_owner_ids(creator)
        creator_user_id = str(creator.get("user_id") or creator.get("sub") or "").strip()
        if creator_user_id:
            caller_ids.add(creator_user_id)

    # An unowned row is nobody's host, not everybody's.
    return [h for h in hosts if caller_ids & _host_owner_ids(h)]


def _require_host_visible(user: dict, host: dict) -> None:
    """404 unless *user* may see *host*.

    404 rather than 403: a 403 confirms the host exists, which is most of what
    fleet enumeration wanted in the first place.

    Filtering only the list endpoint leaves this open — `GET /host/{id}` and the
    spot preview each return a single row by id, and the spot preview discloses
    a provider's pricing floor, which is commercially sensitive on its own.
    """
    if not visible_hosts(user, [host]):
        raise HTTPException(status_code=404, detail="Host not found")


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
    # `.get`, not `[...]`: this is the funnel every host route resolves
    # through, so one malformed row here is a 500 for *every* host lookup in
    # the process, not just that row's. `db.load_hosts` now fills `host_id`
    # from the column so the key should always be present — this is the second
    # layer, because the cost of being wrong is disproportionate.
    if any(h.get("host_id") == host_id for h in hosts):
        return host_id, hosts
    # Prefix match fallback (must be unambiguous and at least 8 chars)
    if len(host_id) >= 8:
        matches = [h for h in hosts if str(h.get("host_id") or "").startswith(host_id)]
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
    # 0 / omitted: the GPU's Standard Stripe rate (platform_rate_cad).
    cost_per_hour: float = Field(default=0.0, ge=0, le=1000)
    country: str = Field(default="", max_length=2)  # Unresolved enrollment stays empty.
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
    ssh_user: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.-]{1,64}$|^$")
    spot_enabled: bool | None = None
    spot_gpu_slots: int | None = Field(default=None, ge=0, le=64)
    spot_min_cents: int | None = Field(default=None, ge=0, le=1_000_000)
    # CRIU checkpoint capability (S6 F4.2): docker-criu | gpu-criu | empty
    checkpoint_class: str | None = Field(default=None, max_length=32)
    capabilities: dict | None = None
    cuda_driver_version: str | None = Field(default=None, max_length=32)
    attestation: dict | None = None  # tee_evidence_jwt, nras_verify_status, attestation_tier


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
    """Register or update a host without trusting self-reported admission.

    Heartbeats and inline component versions are useful inventory and
    compatibility signals, but they are supplied by the provider-controlled
    client. New hosts therefore remain pending until a separate authoritative
    worker-verification path admits them.
    """
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    _require_host_operator(user, h.host_id)
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
        pending_until_admitted=True,
    )
    owner_id = _resolve_host_owner_from_user(user)
    existing_owner = str(entry.get("owner") or "").strip()
    if existing_owner and existing_owner != h.host_id:
        entry["owner"] = existing_owner
    elif owner_id:
        entry["owner"] = owner_id
    location_confirmed = bool(entry.get("location_confirmed_at"))
    if not location_confirmed and "country" in h.model_fields_set:
        entry["country"] = h.country.upper()
    if not location_confirmed and "province" in h.model_fields_set and h.province.strip():
        entry["province"] = h.province.upper()
    from host_metadata import require_location

    try:
        location = require_location(
            entry.get("country", ""),
            entry.get("province", ""),
            entry.get("region", "") if location_confirmed else h.region if h.region.strip() else "",
        )
    except ValueError:
        # Telemetry remains acceptable for pending/incomplete enrollment. It
        # cannot make unresolved geography visible in customer inventory.
        pass
    else:
        entry.update(location)
    # P1.2: record what the agent says about itself so rolling-upgrade logic
    # can target only out-of-date hosts and detect hashes that don't match
    # anything the control plane has signed off on.
    if h.agent_version:
        entry["agent_version"] = h.agent_version
    if h.agent_sha256:
        entry["agent_sha256"] = h.agent_sha256
        # Only an agent sends its own hash, so this is the agent's word that
        # it is running. The health check judges an agent host by this alone;
        # `last_seen` is also written by the health check and by the web form.
        entry["last_agent_heartbeat"] = time.time()
    if h.cpu_count:
        entry["cpu_count"] = int(h.cpu_count)
    if h.ssh_user:
        entry["ssh_user"] = h.ssh_user.strip()
    if h.checkpoint_class:
        entry["checkpoint_class"] = h.checkpoint_class.strip().lower()
    if h.capabilities:
        entry["capabilities"] = h.capabilities
    if h.cuda_driver_version:
        entry["cuda_driver_version"] = h.cuda_driver_version
    if h.attestation:
        from host_attestation import merge_attestation_into_admission, normalize_attestation

        entry["attestation"] = normalize_attestation(h.attestation)
    # Persist Canadian company info if provided
    if h.corporation_name:
        entry["corporation_name"] = h.corporation_name
    if h.business_number:
        entry["business_number"] = h.business_number
    if h.gst_hst_number:
        entry["gst_hst_number"] = h.gst_hst_number
    if h.legal_name:
        entry["legal_name"] = h.legal_name

    # Inline versions are a compatibility signal only. A provider-controlled
    # client must never be able to grant itself scheduler admission.
    if h.versions:
        compatible, reasons = check_node_versions(h.versions)
        runtime, runtime_reason = recommend_runtime(gpu_model or "unknown")
        entry["reported_versions"] = dict(h.versions)
        entry["compatibility_details"] = {
            "host_id": h.host_id,
            "compatible": compatible,
            "admission_applied": False,
            "versions": dict(h.versions),
            "rejection_reasons": reasons,
            "recommended_runtime": runtime,
            "runtime_reason": runtime_reason,
            "checked_at": time.time(),
        }
        entry["recommended_runtime"] = runtime

    if entry.get("admitted") is not True:
        entry["admitted"] = False
        entry["status"] = "pending"

    # Persist the updated entry (country, province, admitted status)
    from scheduler import _atomic_mutation, _upsert_host_row

    with _atomic_mutation() as conn:
        _upsert_host_row(conn, entry)

    # A deterministic score is safe to show while pending, but marketplace
    # publication is reserved for the future authoritative admission path.
    from scheduler import estimate_compute_score, register_compute_score

    score = estimate_compute_score(gpu_model)
    register_compute_score(h.host_id, gpu_model, score)
    entry["compute_score"] = score

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
        # Reputation must reflect an actually verified account email.
        re = get_reputation_engine()
        if user.get("email_verified") is True:
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

# The GPU models of the Stripe compute catalog (billing/stripe_catalog.json).
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
    # Omitted: the GPU's Standard Stripe rate (platform_rate_cad).
    cost_per_hour: float | None = Field(default=None, ge=0.01, le=100.0)
    country: str = Field(min_length=2, max_length=2)
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
        from host_metadata import require_location

        location = require_location(self.country, self.province, self.region)
        self.country = location["country"]
        self.province = location["province"]
        self.region = location["region"]
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
        h.spot_min_cents if h.spot_min_cents is not None else suggested_spot_min_cents(h.gpu_model)
    )
    # No asked rate means the GPU's Standard Stripe rate; there is no stand-in.
    cost_per_hour = h.cost_per_hour or platform_rate_cad(h.gpu_model)
    if cost_per_hour <= 0:
        raise HTTPException(503, detail=f"Stripe compute pricing is unavailable for {h.gpu_model}")

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
        pending_until_admitted=True,
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
    entry["admitted"] = False
    entry["status"] = "pending"

    # Persist the full entry
    from scheduler import _atomic_mutation, _upsert_host_row

    with _atomic_mutation() as conn:
        _upsert_host_row(conn, entry)

    # Manual registration records inventory and a display score only. It must
    # not publish provider-asserted hardware to the marketplace.
    from scheduler import estimate_compute_score, register_compute_score

    score = estimate_compute_score(h.gpu_model)
    register_compute_score(host_id, h.gpu_model, score)
    entry["compute_score"] = score

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
        if user.get("email_verified") is True:
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


def _require_location_owner(host_id: str, request: Request, scope: str):
    from routes._deps import _require_user_grant

    user = _require_auth(request)
    _require_scope(user, scope)
    if not _is_platform_admin(user):
        user = _require_user_grant(request)
    resolved, hosts = _resolve_host_id(host_id)
    entry = next((host for host in hosts if host.get("host_id") == resolved), None)
    if entry is None or resolved is None:
        raise HTTPException(404, "Host not found")
    _require_host_visible(user, entry)
    _require_host_operator(user, resolved)
    return user, resolved


class HostLocationUpdate(BaseModel):
    country: str = Field(min_length=2, max_length=2)
    province: str = Field(default="", max_length=32)
    expected_version: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_location(self):
        from host_metadata import require_location

        location = require_location(self.country, self.province)
        self.country, self.province = location["country"], location["province"]
        return self


@router.patch("/api/hosts/{host_id}/location", tags=["Hosts"])
def api_update_host_location(host_id: str, body: HostLocationUpdate, request: Request):
    """Confirm a host's physical site with owner authorization and version fencing."""
    from control_plane.host_lifecycle import set_location
    from control_plane.launch.service import PlanConflict
    from db import _get_pg_pool
    from routes._deps import _canonical_owner_id

    user, resolved = _require_location_owner(host_id, request, "hosts:write")
    try:
        with _get_pg_pool().connection() as conn:
            updated = set_location(
                conn,
                host_id=resolved,
                country=body.country,
                province=body.province,
                expected_version=body.expected_version,
                confirmed_by=_canonical_owner_id(user),
            )
    except PlanConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    if updated is None:
        raise HTTPException(404, "Host not found")
    broadcast_sse("host_update", {"host_id": resolved, "region": updated["region"]})
    return {"ok": True, "host": enrich_host_for_api(updated)}


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
    # Discloses a provider's pricing floor — sensitive independently of the
    # rest of the row.
    _require_host_visible(user, host)
    from spot_pricing import effective_spot_rate_cad, suggested_spot_min_cents

    floor = (
        spot_min_cents
        if spot_min_cents is not None
        else int(
            host.get("spot_min_cents", suggested_spot_min_cents(host.get("gpu_model", ""))) or 0
        )
    )
    preview = effective_spot_rate_cad(str(host.get("gpu_model") or ""), floor)
    return {"ok": True, "host_id": resolved, "spot_preview": preview}


# The five routes below are product API (console and administrators), but they
# were registered only under ``/host/``, which ``AgentIngressMiddleware`` treats
# as worker protocol. With the public ingress in ``deny`` (production since the
# gateway cutover) every one of them answered ``410 agent_ingress_retired``: the
# console's host page could not load a host or remove one (CVP-01, 2026-11-11).
# Each is now also served under ``/api/hosts/``, which the gate never touches;
# the ``/host/`` spelling stays for the agent gateway and older clients.
@router.get("/api/hosts/{host_id}", tags=["Hosts"])
@router.get("/host/{host_id}", tags=["Hosts"])
def api_get_host(host_id: str, request: Request):
    """Get a single host by ID."""
    user = _require_auth(request)
    _require_scope(user, "hosts:read")
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host = next(h for h in hosts if h["host_id"] == resolved)
    _require_host_visible(user, host)
    # V2 workers report observations instead of refreshing the legacy JSON
    # last_seen field. Expose the server receipt time for install verification.
    from db import pg_connection

    with pg_connection() as conn:
        row = conn.execute(
            "SELECT EXTRACT(EPOCH FROM last_observed_at) FROM hosts WHERE host_id = %s",
            (resolved,),
        ).fetchone()
    host = {**host, "last_observed_at": float(row[0]) if row and row[0] is not None else None}
    return {"ok": True, "host": enrich_host_for_api(host)}


@router.get("/hosts", tags=["Hosts"])
def api_list_hosts(request: Request, active_only: bool = True):
    """List hosts the caller may see."""
    user = _require_auth(request)
    _require_scope(user, "hosts:read")
    hosts = visible_hosts(user, list_hosts(active_only=active_only))
    return {"hosts": [enrich_host_for_api(h) for h in hosts]}


@router.get("/api/hosts/{host_id}/maintenance", tags=["Hosts"])
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


@router.post("/api/hosts/{host_id}/enrollment", tags=["Hosts"])
def api_enroll_host(host_id: str, request: Request):
    """One-time credentials that connect a registered host to the agent gateway.

    Returns a single-use SPIRE join token and a single-use mesh pre-auth key
    (control_plane/host_enrollment.py). The installer and the wizard call this
    right after registration; the provider does nothing by hand. Only the host's
    owner (or a platform admin) may enroll it, and identity is not admission:
    the host stays pending until an operator admits it.
    """
    from control_plane import host_enrollment
    from routes._deps import _check_host_enroll_rate_limit, _get_current_user, _require_scope

    user = _get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    _require_scope(user, "hosts:write")
    entry = next((h for h in list_hosts(active_only=False) if h.get("host_id") == host_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Host not found")
    if not _is_platform_admin(user):
        # Strictly the owner: unlike _require_host_operator, an unowned row is
        # not claimable here, because this hands out a credential for it.
        caller_ids = _caller_owner_ids(user)
        creator = _oauth_client_creator(user)
        if creator:
            caller_ids |= _caller_owner_ids(creator)
            creator_user_id = str(creator.get("user_id") or creator.get("sub") or "").strip()
            if creator_user_id:
                caller_ids.add(creator_user_id)
        if not caller_ids & _host_owner_ids(entry):
            raise HTTPException(status_code=404, detail="Host not found")
    _check_host_enroll_rate_limit(host_id)
    try:
        enrollment = host_enrollment.enroll_host(host_id)
    except host_enrollment.EnrollmentUnavailable as exc:
        log.error("host enrollment unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Host enrollment is not available yet. Contact support to connect this host.",
        ) from exc
    except host_enrollment.EnrollmentFailed as exc:
        log.error("host enrollment failed for %s: %s", host_id, exc)
        raise HTTPException(
            status_code=502, detail="Host enrollment failed; retry shortly."
        ) from exc
    log.info(
        "HOST ENROLLED host_id=%s by=%s spiffe_id=%s",
        host_id,
        user.get("user_id") or user.get("email") or user.get("client_id") or "?",
        enrollment["spire"]["spiffe_id"],
    )
    return {"ok": True, **enrollment}


@router.post("/api/hosts/{host_id}/drain", tags=["Hosts"])
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
    broadcast_sse("host_update", {"host_id": host_id, "status": "draining"})
    return {
        "ok": True,
        "host": updated,
        "preempted": [],
        "maintenance": api_host_maintenance(host_id, request),
    }


@router.post("/api/hosts/{host_id}/undrain", tags=["Hosts"])
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


def machine_credential_may_not_delete(user: dict, entry: dict | None) -> bool:
    """True when a machine credential is trying to delete an *admitted* host.

    Deleting a host cascades to `host_admission_decisions` and
    `host_admission_evidence`, so this endpoint erases the operator's signed
    decision and the evidence behind it. `worker_agent.py` called it on every
    graceful shutdown, so a `systemctl restart` silently revoked the host's own
    admission and it came back `pending` — mistaken twice for a database losing
    writes before the cause was found.

    The agent is fixed, but a client-side fix cannot bind a client and old
    binaries are still in the field, so the invariant lives here as well:
    revoking admission is an operator act and a host may not perform it on
    itself. Humans are deliberately unaffected — an admin or the host's owner
    may still remove it, admitted or not.
    """
    if entry is None:
        return False
    grant = str(user.get("grant_type") or user.get("auth_type") or "")
    if grant != "client_credentials":
        return False
    if _is_platform_admin(user):
        return False
    return str(entry.get("admission_state") or "").strip().lower() == "admitted"


@router.delete("/api/hosts/{host_id}", tags=["Hosts"])
@router.delete("/host/{host_id}", tags=["Hosts"])
def api_remove_host(host_id: str, request: Request):
    """Remove a host."""
    user = _require_auth(request)
    _require_scope(user, "hosts:write")
    _require_host_operator(user, host_id)
    resolved, hosts = _resolve_host_id(host_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Host {host_id} not found")
    host_id = resolved

    # A machine credential may not delete an *admitted* host.
    #
    # `host_admission_decisions` and `host_admission_evidence` are ON DELETE
    # CASCADE, so this endpoint erases the operator's signed decision and the
    # evidence behind it. `worker_agent.py` called it on every graceful
    # shutdown, which meant a `systemctl restart` silently revoked the host's
    # own admission and it came back `pending` — twice mistaken for a database
    # losing writes.
    #
    # The agent side is fixed, but old binaries in the field still have that
    # behaviour, and a client-side fix cannot bind a client. So the invariant
    # lives here too: revoking admission is an operator act, and a host may not
    # perform it on itself. Humans are unaffected — an admin or the host's owner
    # can still remove it, admitted or not.
    entry = next((h for h in hosts if h.get("host_id") == host_id), None)
    if machine_credential_may_not_delete(user, entry):
        raise HTTPException(
            409,
            "This host is admitted; a worker credential cannot delete it. "
            "Deleting would cascade away the operator's admission decision and "
            "evidence. Revoke admission first "
            "(POST /api/admin/hosts/{host_id}/admission-decisions, action=revoke), "
            "or remove the host as an operator.",
        )

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
