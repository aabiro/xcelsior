"""Seller identity for Canadian marketplace reporting.

Providers are independent suppliers. The platform reports what each one earned,
so onboarding stores a legal name, a mailing address, and a tax identifier.
The identifier is encrypted. API responses keep only the last four characters.
"""

from __future__ import annotations

import re

from stripe_tax import CA_PROVINCES

_GST_HST = re.compile(r"^\d{9}RT\d{4}$")
_CA_POSTAL = re.compile(r"^[A-Z]\d[A-Z]\d[A-Z]\d$")
_TAX_KINDS = frozenset({"sin", "bn", "other"})

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
    compact = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if not 4 <= len(compact) <= 32:
        raise ValueError("Enter the tax identifier, 4 to 32 letters or digits")
    return compact


def _normalize_address(profile: dict) -> dict[str, str]:
    legal_name = " ".join(str(profile.get("legal_name") or "").split())
    if not 2 <= len(legal_name) <= 200:
        raise ValueError("Enter the legal name on the tax account")
    line1 = " ".join(str(profile.get("address_line1") or "").split())
    line2 = " ".join(str(profile.get("address_line2") or "").split())
    city = " ".join(str(profile.get("city") or "").split())
    if not 4 <= len(line1) <= 200:
        raise ValueError("Enter the street address")
    if len(line2) > 200:
        raise ValueError("Address line 2 must be 200 characters or fewer")
    if not 2 <= len(city) <= 80:
        raise ValueError("Enter the city")
    country = str(profile.get("country") or "CA").strip().upper()[:2]
    if len(country) != 2 or not country.isalpha():
        raise ValueError("Enter a two-letter country code")
    province = str(profile.get("province") or "").strip().upper()
    if country == "CA":
        if province not in CA_PROVINCES:
            raise ValueError("Enter a Canadian province or territory code")
        postal = re.sub(r"\s+", "", str(profile.get("postal_code") or "")).upper()
        if not _CA_POSTAL.match(postal):
            raise ValueError("Enter a Canadian postal code")
    else:
        if not 2 <= len(province) <= 32:
            raise ValueError("Enter the state or region")
        postal = re.sub(r"\s+", "", str(profile.get("postal_code") or "")).upper()
        if not 3 <= len(postal) <= 12:
            raise ValueError("Enter the postal code")
    kind = str(profile.get("tax_id_kind") or "").strip().lower()
    if kind not in _TAX_KINDS:
        raise ValueError("Tax identifier must be a SIN, a business number, or another tax id")
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
    if not profile.get("accepted_terms"):
        raise ValueError(
            "Confirm that you are an independent supplier and responsible for your own taxes"
        )
    cleaned = _normalize_address(profile)
    raw_tax_id = str(profile.get("tax_id") or "").strip()
    if raw_tax_id:
        cleaned["tax_id"] = _normalize_tax_id(cleaned["tax_id_kind"], raw_tax_id)
    elif require_tax_id:
        raise ValueError("Enter a SIN or business number")
    else:
        cleaned["tax_id"] = ""
    return cleaned


def seller_profile_complete(row: dict | None) -> bool:
    if not row:
        return False
    required = (
        "legal_name",
        "address_line1",
        "city",
        "province",
        "postal_code",
        "country",
        "tax_id_kind",
        "tax_id_last4",
    )
    return all(str(row.get(key) or "").strip() for key in required)


def present_provider(provider: dict) -> dict:
    """Drop the encrypted tax identifier and say whether reporting fields are filled."""
    for column in SECRET_COLUMNS:
        provider.pop(column, None)
    provider["seller_profile_complete"] = seller_profile_complete(provider)
    return provider
