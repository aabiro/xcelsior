"""Normalize and enrich host records for API responses and worker heartbeats."""

from __future__ import annotations

import re
from typing import Any

from stripe_catalog import load_manifest

# Canonical GPU titles and their smallest VRAM, from the versioned Stripe
# catalog (billing/stripe_catalog.json) that deploy imports into gpu_pricing.
# A missing catalog fails here rather than leaving every name unrecognised.
_DEFAULT_VRAM_BY_MODEL: dict[str, int] = {}
for _row in load_manifest()["service_tier_catalog"]["prices"]:
    _model = str(_row["variant"]["gpu_model"])
    _vram = int(_row["variant"]["vram_gb"])
    if _vram < _DEFAULT_VRAM_BY_MODEL.get(_model, _vram + 1):
        _DEFAULT_VRAM_BY_MODEL[_model] = _vram
CANONICAL_GPU_MODELS: frozenset[str] = frozenset(_DEFAULT_VRAM_BY_MODEL)

_NVIDIA_PREFIX_RE = re.compile(r"^NVIDIA\s+(GeForce\s+)?", re.IGNORECASE)
_DEFAULT_REGION = ""
_CA_PROVINCES = frozenset(
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


def normalize_gpu_model(raw: str) -> str:
    """Map worker/dashboard aliases to canonical short GPU titles."""
    model = (raw or "").strip()
    if not model:
        return ""

    if model in CANONICAL_GPU_MODELS:
        return model

    model = _NVIDIA_PREFIX_RE.sub("", model).strip()

    # Legacy dashed IDs from an older dashboard validator (RTX-3060 → RTX 3060).
    if "-" in model and " " not in model:
        dashed_as_spaced = model.replace("-", " ")
        if dashed_as_spaced in CANONICAL_GPU_MODELS:
            return dashed_as_spaced

    # Case-insensitive match against catalogue, then the same match with
    # spacing removed (RTX4090, rtx_4090 → RTX 4090).
    lower = model.lower()
    for canonical in CANONICAL_GPU_MODELS:
        if canonical.lower() == lower:
            return canonical
    compact = re.sub(r"[\s\-_]+", "", lower)
    for canonical in CANONICAL_GPU_MODELS:
        if canonical.lower().replace(" ", "") == compact:
            return canonical

    return model


def default_vram_gb(gpu_model: str) -> float:
    model = normalize_gpu_model(gpu_model)
    return float(_DEFAULT_VRAM_BY_MODEL.get(model, 0))


def platform_rate_cad(gpu_model: str, *, pricing_mode: str = "on_demand") -> float:
    """The Standard Stripe rate for a GPU model's default variant, 0.0 when unpriced."""
    model = normalize_gpu_model(gpu_model)
    if not model:
        return 0.0
    from pricing_reference import get_rate

    return get_rate(model, pricing_mode)


def normalize_region(
    region: str | None = "",
    *,
    country: str | None = "",
    province: str | None = "",
    default: str = _DEFAULT_REGION,
) -> str:
    """Normalize stored host/offer regions for API selectors and placement."""
    value = str(region or "").strip().replace("_", "-")
    province_code = str(province or "").strip().upper()
    country_code = str(country or "").strip().upper()

    if value:
        upper = value.upper()
        if upper in _CA_PROVINCES:
            return f"ca-{upper.lower()}"
        if upper.startswith("CA-") and upper[3:] in _CA_PROVINCES:
            return f"ca-{upper[3:].lower()}"
        return value.lower()

    if country_code == "CA" and province_code in _CA_PROVINCES:
        return f"ca-{province_code.lower()}"
    if country_code in {"US", "EU", "AP"}:
        return country_code.lower()
    return default


def normalize_host_region(host: dict[str, Any]) -> str:
    """Return the canonical region for a host payload."""
    return normalize_region(
        str(host.get("region") or ""),
        country=str(host.get("country") or ""),
        province=str(host.get("province") or ""),
    )


def infer_gpu_from_host_id(host_id: str) -> str:
    """Best-effort GPU model from host_id tokens (e.g. gpu-3060-lab → RTX 3060)."""
    raw = (host_id or "").strip().lower()
    if not raw:
        return ""

    compact = re.sub(r"[\s\-_]+", "", raw)
    for canonical in sorted(CANONICAL_GPU_MODELS, key=len, reverse=True):
        key = canonical.lower().replace(" ", "")
        if key in compact:
            return canonical

    match = re.search(r"(?:rtx[\-_ ]?)?(\d{4})", raw)
    if match:
        candidate = f"RTX {match.group(1)}"
        if candidate in CANONICAL_GPU_MODELS:
            return candidate

    return ""


def _display_hostname(host: dict[str, Any]) -> str:
    explicit = str(host.get("hostname") or "").strip()
    if explicit:
        return explicit
    host_id = str(host.get("host_id") or "").strip()
    if not host_id:
        return ""
    # gpu-2060-aaryn-local → gpu 2060 aaryn local
    return host_id.replace("_", " ").replace("-", " ").strip()


def enrich_host_for_api(host: dict[str, Any]) -> dict[str, Any]:
    """Add dashboard-friendly aliases and fill missing catalogue fields."""
    out = dict(host)
    gpu = normalize_gpu_model(str(out.get("gpu_model") or ""))
    if not gpu:
        gpu = infer_gpu_from_host_id(str(out.get("host_id") or ""))
    if gpu:
        out["gpu_model"] = gpu

    total = float(out.get("total_vram_gb") or out.get("vram_gb") or 0)
    if total <= 0 and gpu:
        total = default_vram_gb(gpu)
    if total > 0:
        out["total_vram_gb"] = total
        out["vram_gb"] = total

    rate = float(out.get("cost_per_hour") or out.get("price_per_hour") or 0)
    if rate <= 0 and gpu:
        rate = platform_rate_cad(gpu)
    if rate > 0:
        out["cost_per_hour"] = rate
        out["price_per_hour"] = rate

    hostname = _display_hostname(out)
    if hostname:
        out["hostname"] = hostname

    try:
        import pycountry

        location = require_location(
            str(out.get("country") or ""),
            str(out.get("province") or ""),
            str(out.get("region") or ""),
        )
        country = pycountry.countries.get(alpha_2=location["country"])
        subdivision = _subdivision(code=location["region"].upper())
        out["location_complete"] = True
        out["location_label"] = ", ".join(
            item.name for item in (subdivision, country) if item is not None
        )
    except ValueError:
        out["location_complete"] = False
        out["location_label"] = ""

    out.setdefault("country", "")
    out.setdefault("province", "")
    out["region"] = normalize_host_region(out)

    return out


def merge_host_update(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge a worker heartbeat into a stored host without clobbering dashboard metadata."""
    merged = dict(incoming)
    ex = existing or {}

    # Location is an explicit owner declaration. An agent's old configuration
    # cannot move it, or forge the server's confirmation metadata.
    for field in ("location_confirmed_at", "location_confirmed_by"):
        merged.pop(field, None)
        if field in ex:
            merged[field] = ex[field]
    if ex.get("location_confirmed_at"):
        for field in ("country", "province", "region"):
            merged[field] = ex.get(field, "")

    for field in (
        "hostname",
        "notes",
        "owner",
        "provider_id",
        "user_id",
        "country",
        "province",
        "region",
        "corporation_name",
        "business_number",
        "gst_hst_number",
        "legal_name",
        "admitted",
        "admission_details",
        "recommended_runtime",
        "compute_score",
        "registered_at",
        "ssh_user",
        "checkpoint_class",
        "capabilities",
        "cuda_driver_version",
    ):
        if ex.get(field) and not merged.get(field):
            merged[field] = ex[field]

    incoming_gpu = normalize_gpu_model(str(merged.get("gpu_model") or ""))
    existing_gpu = normalize_gpu_model(str(ex.get("gpu_model") or ""))
    if incoming_gpu:
        merged["gpu_model"] = incoming_gpu
    elif existing_gpu:
        merged["gpu_model"] = existing_gpu
    else:
        inferred = infer_gpu_from_host_id(str(merged.get("host_id") or ex.get("host_id") or ""))
        if inferred:
            merged["gpu_model"] = inferred

    incoming_total = float(merged.get("total_vram_gb") or 0)
    existing_total = float(ex.get("total_vram_gb") or ex.get("vram_gb") or 0)
    if incoming_total > 0:
        merged["total_vram_gb"] = incoming_total
    elif existing_total > 0:
        merged["total_vram_gb"] = existing_total
    elif merged.get("gpu_model"):
        merged["total_vram_gb"] = default_vram_gb(str(merged["gpu_model"]))

    incoming_free = float(merged.get("free_vram_gb") or 0)
    if incoming_free <= 0 and merged.get("total_vram_gb"):
        merged["free_vram_gb"] = float(merged["total_vram_gb"])
    elif incoming_free > 0:
        merged["free_vram_gb"] = incoming_free
    elif ex.get("free_vram_gb"):
        merged["free_vram_gb"] = ex["free_vram_gb"]

    incoming_cost = float(merged.get("cost_per_hour") or 0)
    existing_cost = float(ex.get("cost_per_hour") or 0)
    if incoming_cost > 0:
        merged["cost_per_hour"] = incoming_cost
    elif existing_cost > 0:
        merged["cost_per_hour"] = existing_cost
    else:
        platform = platform_rate_cad(str(merged.get("gpu_model") or ""))
        if platform > 0:
            merged["cost_per_hour"] = platform

    if not merged.get("hostname"):
        merged["hostname"] = _display_hostname(ex) or _display_hostname(merged)

    return merged


def location_catalog(country: str = "", purpose: str = "host") -> dict:
    """ISO 3166 country/subdivision choices, shared by forms and API validation.

    ``purpose="address"`` is a person's postal address; ``"host"`` is where a machine runs.
    """
    import pycountry

    countries = sorted(
        ({"code": c.alpha_2, "name": c.name} for c in pycountry.countries),
        key=lambda c: c["name"],
    )
    country_code = str(country or "").strip().upper()
    if country_code and pycountry.countries.get(alpha_2=country_code) is None:
        raise ValueError("Select a valid country")
    subdivisions = (
        sorted(
            (
                {"code": s.code.split("-", 1)[1], "region": s.code.lower(), "name": s.name}
                for s in pycountry.subdivisions.get(country_code=country_code) or []
            ),
            key=lambda s: (s["name"], s["code"]),
        )
        if country_code
        else []
    )
    if purpose not in {"host", "address"}:
        raise ValueError("Unknown location purpose")
    # A host's subdivision is its scheduling region, so hosts always pick one where
    # ISO lists any. A postal address follows the post: the UK and Germany never use one.
    if purpose == "address" and country_code:
        mode = address_subdivision_mode(country_code)
        if mode == "none":
            subdivisions = []
    else:
        mode = "required" if subdivisions else "none"
    return {
        "countries": countries,
        "subdivisions": subdivisions,
        "requires_subdivision": mode == "required",
        "subdivision_mode": mode,
        "labels": address_labels(country_code) if country_code else address_labels(""),
        "required_address_fields": address_required_fields(country_code) if country_code else [],
    }


_SUBDIVISION_LABELS = {
    "province": "Province", "state": "State", "prefecture": "Prefecture", "county": "County",
    "department": "Department", "emirate": "Emirate", "island": "Island", "parish": "Parish",
    "oblast": "Region", "do_si": "Province", "district": "District", "area": "Area",
}
_SUBDIVISION_LABEL_OVERRIDES = {"CA": "Province or territory", "AU": "State or territory"}
_CITY_LABELS = {"post_town": "Town or city", "suburb": "Suburb", "district": "District"}
_POSTAL_LABELS = {"zip": "ZIP code", "pin": "PIN code", "eircode": "Eircode"}


def address_subdivision_mode(country: str) -> str:
    """How this country's postal addresses use a state or province: required, optional, or none.

    From Google's address data (the same rules the postal-code checks use), not
    from ISO: ISO lists subdivisions for almost every country, but 174 of them
    never write one on an envelope.
    """
    import pycountry
    from i18naddress import get_validation_rules

    country_code = str(country or "").strip().upper()
    if not country_code or not (pycountry.subdivisions.get(country_code=country_code) or []):
        return "none"
    rules = get_validation_rules({"country_code": country_code})
    if "country_area" in rules.required_fields:
        return "required"
    return "optional" if "%S" in rules.address_format else "none"


def address_labels(country: str) -> dict[str, str]:
    """What this country calls its address fields: "State", "Prefecture", "ZIP code", "Eircode"…"""
    from i18naddress import get_validation_rules

    country_code = str(country or "").strip().upper()
    if not country_code:
        return {"subdivision": "State, province, or territory", "city": "City or locality", "postal_code": "Postal code"}
    rules = get_validation_rules({"country_code": country_code})
    return {
        "subdivision": _SUBDIVISION_LABEL_OVERRIDES.get(country_code)
        or _SUBDIVISION_LABELS.get(rules.country_area_type, "State or province"),
        "city": _CITY_LABELS.get(rules.city_type, "City"),
        "postal_code": _POSTAL_LABELS.get(rules.postal_code_type, "Postal code"),
    }


def require_address_location(country: str, province: str = "") -> dict[str, str]:
    """`require_location` for a person's postal address, which follows the country's post.

    Where the post never uses a state (the UK, Germany, France…) the province is
    dropped; where it is optional a blank is fine. Elsewhere it must be exact.
    """
    import pycountry

    country_code = str(country or "").strip().upper()
    if pycountry.countries.get(alpha_2=country_code) is None:
        raise ValueError("Select a valid country")
    mode = address_subdivision_mode(country_code)
    if mode == "none" or (mode == "optional" and not str(province or "").strip()):
        return {"country": country_code, "province": "", "region": country_code.lower()}
    return require_location(country_code, province)


def _subdivision(code: str) -> Any:
    """One ISO 3166-2 subdivision by code, or None. pycountry types `get` loosely."""
    import pycountry

    return pycountry.subdivisions.get(code=code)


def require_location(country: str, province: str = "", region: str = "") -> dict[str, str]:
    """Resolve one exact registered location. Missing/ambiguous data is an error.

    Country and subdivision identify geography; broad legacy labels such as
    ca-east/ca-central are not evidence of a province. No fuzzy matching or
    implicit location is permitted.
    """
    import pycountry

    country_code = str(country or "").strip().upper()
    if pycountry.countries.get(alpha_2=country_code) is None:
        raise ValueError("Select a valid country")
    value = str(province or "").strip().upper()
    region_code = str(region or "").strip().replace("_", "-").upper()
    if not value and region_code:
        sub = _subdivision(code=region_code)
        if sub is not None and sub.country_code == country_code:
            value = sub.code.split("-", 1)[1]
    subdivisions = pycountry.subdivisions.get(country_code=country_code) or []
    if subdivisions:
        code = value if value.startswith(country_code + "-") else country_code + "-" + value
        sub = _subdivision(code=code)
        if sub is None or sub.country_code != country_code:
            raise ValueError("Select a valid state, province, or territory for the country")
        canonical_region = sub.code.lower()
        value = sub.code.split("-", 1)[1]
    else:
        if value:
            raise ValueError("This country has no ISO state or province subdivision")
        canonical_region = country_code.lower()
    if region_code and region_code.lower() != canonical_region:
        raise ValueError("Region must match the selected country and state or province")
    return {"country": country_code, "province": value, "region": canonical_region}


def address_required_fields(country: str) -> list[str]:
    """Country-specific postal requirements from packaged address metadata."""
    from i18naddress import get_validation_rules

    mapping = {"street_address": "address_line1", "city": "city", "postal_code": "postal_code"}
    rules = get_validation_rules({"country_code": country})
    return sorted(mapping[field] for field in rules.required_fields if field in mapping)


def _fold(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", value)
    return " ".join("".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold().split())


def subdivision_code(country: str, value: str) -> str:
    """The ISO code for a state or province given as a code ("QC") or a name ("Québec").

    Returns "" when nothing matches exactly, so a search result never picks a
    province the person did not see; they choose it themselves instead.
    """
    import pycountry

    country_code = str(country or "").strip().upper()
    wanted = _fold(str(value or ""))
    if not country_code or not wanted:
        return ""
    for sub in pycountry.subdivisions.get(country_code=country_code) or []:
        code = sub.code.split("-", 1)[1]
        if wanted in {code.casefold(), _fold(sub.name)}:
            return code
    return ""


def standard_postal_code(country: str, value: str) -> str:
    """The postal code as the country writes it: uppercase, with its usual separator.

    The separator comes from the country's published examples (CA ``N6B 1Z2``,
    US ``10001-1234``, GB ``SW1A 1AA``) and is applied only when the result is
    still a valid postal code there. Anything else comes back uppercased as typed.
    """
    from i18naddress import get_validation_rules

    text = " ".join(str(value or "").split()).upper()
    if not text or not country:
        return text
    rules = get_validation_rules({"country_code": country})
    compact = re.sub(r"[\s-]", "", text)
    for example in rules.postal_code_examples:
        shape = example.strip().upper()
        if len(re.sub(r"[\s-]", "", shape)) != len(compact):
            continue
        chars = iter(compact)
        candidate = "".join(ch if ch in " -" else next(chars) for ch in shape)
        if all(matcher.match(candidate) for matcher in rules.postal_code_matchers):
            return candidate
    return text


def _postal_label(country: str) -> str:
    from i18naddress import get_validation_rules

    kind = get_validation_rules({"country_code": country}).postal_code_type
    return {"zip": "ZIP code", "pin": "PIN code", "eircode": "Eircode"}.get(kind, "postal code")


def _postal_region_warning(location: dict[str, str], postal: str) -> dict[str, str]:
    """Say which state or province the postal code usually belongs to.

    A warning, never an error: real addresses cross these lines. Lloydminster,
    Alberta uses Saskatchewan's S9V codes, and some US ZIP codes span two states.
    """
    import pycountry
    from i18naddress import get_validation_rules

    country = location["country"]
    likely = []
    for sub in pycountry.subdivisions.get(country_code=country) or []:
        try:
            rules = get_validation_rules(
                {"country_code": country, "country_area": sub.code.split("-", 1)[1]}
            )
        except Exception:
            continue
        # Only an area with its own prefix says anything; the rest inherit the country's.
        if len(rules.postal_code_matchers) > 1 and all(
            matcher.match(postal) for matcher in rules.postal_code_matchers
        ):
            likely.append(sub.name)
    chosen = _subdivision(code=f"{country}-{location['province']}")
    chosen_name = chosen.name if chosen is not None else location["province"]
    label = _postal_label(country)
    if len(likely) == 1:
        message = f"{postal} is usually a {label} in {likely[0]}, not {chosen_name}."
    else:
        message = f"{postal} isn't usually a {label} in {chosen_name}."
    return {
        "field": "postal_code",
        "code": "postal_code_region",
        "message": message + " Check it, or keep it if your address is right.",
    }


def _address_review(
    profile: dict[str, Any],
) -> tuple[dict[str, str], list[str], list[dict[str, str]]]:
    """Clean an address into postal format, then sort problems into errors and warnings.

    Errors are things no address can be: a missing required field, or a postal
    code that isn't one in that country. Everything a real address might still
    be is a warning, so a correct but unusual address can always be saved.
    """
    from i18naddress import get_validation_rules

    location = require_address_location(
        str(profile.get("country") or ""), str(profile.get("province") or "")
    )
    cleaned = {
        key: " ".join(str(profile.get(key) or "").split()).strip(" ,")
        for key in ("address_line1", "address_line2", "city", "postal_code")
    }
    cleaned["postal_code"] = standard_postal_code(location["country"], cleaned["postal_code"])
    errors = [key for key in address_required_fields(location["country"]) if not cleaned[key]]
    warnings: list[dict[str, str]] = []
    postal = cleaned["postal_code"]
    country_rules = get_validation_rules({"country_code": location["country"]})
    if postal and not all(matcher.match(postal) for matcher in country_rules.postal_code_matchers):
        errors.append("postal_code")
    elif postal and location["province"]:
        area_rules = get_validation_rules(
            {"country_code": location["country"], "country_area": location["province"]}
        )
        if not all(matcher.match(postal) for matcher in area_rules.postal_code_matchers):
            warnings.append(_postal_region_warning(location, postal))
    if len(cleaned["address_line1"]) > 200 or len(cleaned["address_line2"]) > 200:
        errors.append("address_line1")
    if len(cleaned["city"]) > 80:
        errors.append("city")
    if len(postal) > 32:
        errors.append("postal_code")
    return {**location, **cleaned}, list(dict.fromkeys(errors)), warnings


def profile_address_values(profile: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Validate manual and captured addresses with the same country rules.

    Returns the cleaned address and its blocking errors. Warnings (a postal code
    from a neighbouring province, say) never block a save; the form shows them
    first through `review_profile_address`.
    """
    cleaned, errors, _warnings = _address_review(profile)
    return cleaned, errors


def postal_lines(address: dict[str, str]) -> list[str]:
    """The address laid out the way the country's post office prints it."""
    import pycountry
    from i18naddress import get_validation_rules

    rules = get_validation_rules({"country_code": address["country"]})
    street = "\n".join(
        value for value in (address.get("address_line1"), address.get("address_line2")) if value
    )
    values = {
        "A": street,
        "C": address.get("city") or "",
        "S": address.get("province") or "",
        "Z": address.get("postal_code") or "",
    }
    text = re.sub(r"%([A-Z])", lambda match: values.get(match.group(1), ""), rules.address_format)
    lines = [" ".join(line.split()).strip(" ,") for line in text.replace("%n", "\n").split("\n")]
    country = pycountry.countries.get(alpha_2=address["country"])
    return [line for line in lines if line] + ([country.name] if country is not None else [])


def review_profile_address(profile: dict[str, Any]) -> dict[str, Any]:
    """The free check run once, when someone saves an address: no provider is called.

    Returns the cleaned address, blocking errors and soft warnings as messages a
    person can act on, and the postal-format lines to show them before saving.
    """
    import pycountry

    cleaned, error_fields, warnings = _address_review(profile)
    country = pycountry.countries.get(alpha_2=cleaned["country"])
    country_name = country.name if country is not None else cleaned["country"]
    label = _postal_label(cleaned["country"])
    postal = cleaned["postal_code"]
    messages = {
        "address_line1": (
            "Address lines can be 200 characters at most."
            if cleaned["address_line1"]
            else "Enter your street address."
        ),
        "city": "City can be 80 characters at most." if cleaned["city"] else "Enter your city or locality.",
        "postal_code": (
            f"{postal} isn't a valid {label} in {country_name}." if postal else f"Enter your {label}."
        ),
    }
    fields = ("country", "province", "address_line1", "address_line2", "city", "postal_code")
    return {
        "address": {field: cleaned[field] for field in fields},
        "errors": [{"field": field, "message": messages[field]} for field in error_fields],
        "warnings": warnings,
        "lines": postal_lines(cleaned),
    }


def profile_missing_fields(profile: dict[str, Any]) -> list[str]:
    """Required account data is checked from stored values, never UI preferences."""
    missing = []
    if not str(profile.get("name") or "").strip():
        missing.append("name")
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", str(profile.get("email") or "")):
        missing.append("email")
    try:
        require_address_location(
            str(profile.get("country") or ""), str(profile.get("province") or "")
        )
    except ValueError:
        import pycountry

        country_code = str(profile.get("country") or "").strip().upper()
        missing.append(
            "country" if pycountry.countries.get(alpha_2=country_code) is None else "province"
        )
    if profile.get("account_type") not in {"individual", "business"}:
        missing.append("account_type")
    if (
        profile.get("account_type") == "business"
        and not str(profile.get("business_name") or "").strip()
    ):
        missing.append("business_name")
    try:
        _, address_errors = profile_address_values(profile)
        missing.extend(address_errors)
    except ValueError:
        if not str(profile.get("address_line1") or "").strip():
            missing.append("address_line1")
    return list(dict.fromkeys(missing))
