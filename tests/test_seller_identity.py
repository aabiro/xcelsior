"""Seller identity validation. No database and no real tax numbers.

A SIN here is nine digits that pass the Luhn check. It is generated in the
test and is not a person's number.
"""

from __future__ import annotations

import pytest

from seller_identity import (
    luhn_ok,
    normalize_business_name,
    normalize_gst_hst,
    normalize_seller_profile,
    present_provider,
    seller_profile_complete,
)


def _sin(prefix: str = "13069254") -> str:
    for digit in "0123456789":
        candidate = prefix + digit
        if luhn_ok(candidate):
            return candidate
    raise AssertionError(f"no Luhn check digit for {prefix}")


def _profile(**overrides: object) -> dict:
    base = {
        "legal_name": "Test Supplier",
        "address_line1": "1 Test Street",
        "address_line2": "",
        "city": "London",
        "province": "ON",
        "postal_code": "n6a 1a1",
        "country": "ca",
        "tax_id_kind": "sin",
        "tax_id": _sin(),
        "gst_hst_number": "",
        "accepted_terms": True,
    }
    base.update(overrides)
    return base


def test_gst_hst_number_is_compact_and_optional():
    assert normalize_gst_hst("") == ""
    assert normalize_gst_hst("  123456789 rt 0001 ") == "123456789RT0001"
    with pytest.raises(ValueError, match="123456789RT0001"):
        normalize_gst_hst("123456789")


def test_business_name_collapses_space_and_rejects_a_long_name():
    assert normalize_business_name("  North   Labs  ") == "North Labs"
    assert normalize_business_name("") == ""
    with pytest.raises(ValueError, match="200"):
        normalize_business_name("A" * 201)


def test_a_sin_must_pass_luhn_and_a_business_number_is_nine_digits():
    sin = _sin()
    cleaned = normalize_seller_profile(_profile(tax_id=sin), require_tax_id=True)
    assert cleaned["tax_id"] == sin
    assert cleaned["postal_code"] == "N6A1A1"
    assert cleaned["country"] == "CA"
    assert cleaned["province"] == "ON"

    with pytest.raises(ValueError, match="social insurance number"):
        normalize_seller_profile(_profile(tax_id="123456789"), require_tax_id=True)

    bn = normalize_seller_profile(
        _profile(tax_id_kind="bn", tax_id="123456789"),
        require_tax_id=True,
    )
    assert bn["tax_id"] == "123456789"

    other = normalize_seller_profile(
        _profile(
            tax_id_kind="other", tax_id="ab-12 cd", country="US", province="NY", postal_code="10001"
        ),
        require_tax_id=True,
    )
    assert other["tax_id"] == "AB12CD"


def test_the_supplier_confirmation_and_the_address_are_required():
    with pytest.raises(ValueError, match="independent supplier"):
        normalize_seller_profile(_profile(accepted_terms=False), require_tax_id=True)
    with pytest.raises(ValueError, match="postal code"):
        normalize_seller_profile(_profile(postal_code="12345"), require_tax_id=True)
    with pytest.raises(ValueError, match="SIN or business number"):
        normalize_seller_profile(_profile(tax_id=""), require_tax_id=True)
    kept = normalize_seller_profile(_profile(tax_id=""), require_tax_id=False)
    assert kept["tax_id"] == ""


def test_a_provider_response_drops_the_encrypted_tax_identifier():
    sin = _sin()
    row = {
        "legal_name": "Test Supplier",
        "address_line1": "1 Test Street",
        "city": "London",
        "province": "ON",
        "postal_code": "N6A1A1",
        "country": "CA",
        "tax_id_kind": "sin",
        "tax_id_last4": sin[-4:],
        "tax_id_ciphertext": "ciphertext-must-not-leave",
    }
    presented = present_provider(row)
    assert "tax_id_ciphertext" not in presented
    assert presented["seller_profile_complete"] is True
    assert seller_profile_complete({"legal_name": "Test Supplier"}) is False
