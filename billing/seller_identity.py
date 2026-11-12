"""Seller identity for Canadian marketplace reporting.

Providers are independent suppliers. The platform reports what each one earned,
so onboarding stores a legal name, a mailing address, and a tax identifier.
The identifier is encrypted. API responses keep only the last four characters.
"""

from __future__ import annotations

import re
import math

_GST_HST = re.compile(r"^\d{9}RT\d{4}$")
_TAX_KINDS = frozenset({"sin", "bn", "gst", "other"})

# Columns a provider response must never carry, even to the account owner.
SECRET_COLUMNS = ("tax_id_ciphertext",)


class SellerProfileRequired(Exception):
    """A new payout account needs a legal name, address, and tax identifier."""


def luhn_ok(digits: str) -> bool:
    """True when `digits` passes the Luhn check used by a Canadian SIN."""
    if not digits.isdigit():
        return False
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def normalize_gst_hst(raw: str) -> str:
    """Return a compact GST/HST number, or empty when the customer has none.

    A business number plus the RT program account, for example 123456789RT0001.
    """
    compact = re.sub(r"[\s-]", "", raw or "").upper()
    if not compact:
        return ""
    if not _GST_HST.match(compact):
        raise ValueError("GST/HST number must look like 123456789RT0001")
    return compact


def normalize_business_name(raw: str) -> str:
    name = " ".join((raw or "").split())
    if len(name) > 200:
        raise ValueError("Business name must be 200 characters or fewer")
    return name


def _normalize_tax_id(kind: str, raw: str) -> str:
    if kind == "sin":
        digits = re.sub(r"\D", "", raw or "")
        if len(digits) != 9 or not luhn_ok(digits):
            raise ValueError("Enter a valid 9-digit social insurance number")
        return digits
    if kind == "bn":
        digits = re.sub(r"\D", "", raw or "")
        if len(digits) != 9:
            raise ValueError("Enter the 9-digit business number")
        return digits
    if kind == "gst":
        return normalize_gst_hst(raw)
    compact = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if not 4 <= len(compact) <= 32:
        raise ValueError("Enter the tax identifier, 4 to 32 letters or digits")
    return compact


def _normalize_address(profile: dict) -> dict[str, str]:
    legal_name = " ".join(str(profile.get("legal_name") or "").split())
    if not 2 <= len(legal_name) <= 200:
        raise ValueError("Enter the legal name on the tax account")
    from host_metadata import profile_address_values

    address, errors = profile_address_values(profile)
    if errors:
        labels = {"address_line1": "street address", "city": "city", "postal_code": "postal code"}
        raise ValueError("Enter a valid " + ", ".join(labels.get(field, field) for field in errors))
    line1 = " ".join(str(profile.get("address_line1") or "").split())
    line2 = " ".join(str(profile.get("address_line2") or "").split())
    city = " ".join(str(profile.get("city") or "").split())
    if len(line2) > 200:
        raise ValueError("Address line 2 must be 200 characters or fewer")
    country = address["country"]
    if len(country) != 2 or not country.isalpha():
        raise ValueError("Enter a two-letter country code")
    from host_metadata import require_address_location

    location = require_address_location(country, str(profile.get("province") or ""))
    country, province = location["country"], location["province"]
    postal = address["postal_code"]
    if country == "CA":
        postal = re.sub(r"\s+", "", postal)
    kind = str(profile.get("tax_id_kind") or "").strip().lower()
    if kind not in _TAX_KINDS:
        raise ValueError("Tax identifier must be a SIN, a business number, a GST/HST number, or another tax id")
    return {
        "legal_name": legal_name,
        "address_line1": line1,
        "address_line2": line2,
        "city": city,
        "province": province,
        "postal_code": postal,
        "country": country,
        "tax_id_kind": kind,
        "gst_hst_number": normalize_gst_hst(str(profile.get("gst_hst_number") or "")),
    }


def normalize_seller_profile(profile: dict, *, require_tax_id: bool) -> dict[str, str]:
    """Validate a seller profile. The tax id is normalized, not encrypted."""
    if profile.get("accepted_terms") is not True:
        raise ValueError(
            "Confirm that you are an independent supplier and responsible for your own taxes"
        )
    cleaned = _normalize_address(profile)
    raw_tax_id = str(profile.get("tax_id") or "").strip()
    if raw_tax_id:
        cleaned["tax_id"] = _normalize_tax_id(cleaned["tax_id_kind"], raw_tax_id)
    elif require_tax_id:
        raise ValueError("Enter a SIN, business number, or GST/HST number")
    else:
        cleaned["tax_id"] = ""
    if cleaned["tax_id_kind"] == "gst" and cleaned["tax_id"]:
        cleaned["gst_hst_number"] = cleaned["tax_id"]
    return cleaned


def seller_profile_complete(row: dict | None) -> bool:
    if not row:
        return False
    try:
        _normalize_address(row)
    except ValueError:
        return False
    accepted_at = row.get("seller_terms_accepted_at")
    if isinstance(accepted_at, bool) or not isinstance(accepted_at, (int, float)):
        return False
    if not math.isfinite(accepted_at) or accepted_at <= 0:
        return False
    pattern = r"[0-9]{4}" if row.get("tax_id_kind") in {"sin", "bn", "gst"} else r"[A-Z0-9]{4}"
    return re.fullmatch(pattern, str(row.get("tax_id_last4") or "").strip()) is not None


def present_provider(provider: dict) -> dict:
    """Drop the encrypted tax identifier and say whether reporting fields are filled."""
    for column in SECRET_COLUMNS:
        provider.pop(column, None)
    provider["seller_profile_complete"] = seller_profile_complete(provider)
    return provider
