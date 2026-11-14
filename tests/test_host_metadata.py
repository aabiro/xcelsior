"""Host metadata normalization for dashboard + worker heartbeats."""

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from host_metadata import (
    CANONICAL_GPU_MODELS,
    default_vram_gb,
    enrich_host_for_api,
    infer_gpu_from_host_id,
    merge_host_update,
    normalize_host_region,
    normalize_gpu_model,
    normalize_region,
    platform_rate_cad,
)

# The Stripe catalog snapshot deploy imports; models and rates come from it.
_CATALOG = json.loads(
    (Path(__file__).resolve().parents[1] / "billing/stripe_catalog.json").read_text()
)["service_tier_catalog"]["prices"]


def _standard_rate(model: str, mode: str = "on_demand") -> float:
    rows = [
        r
        for r in _CATALOG
        if r["variant"]["gpu_model"] == model and r["tier"] == "standard" and r["mode"] == mode
    ]
    row = min(
        rows,
        key=lambda r: (
            r["variant"]["vram_gb"],
            r["variant"]["high_frequency"],
            r["variant"]["form_factor"],
        ),
    )
    return float(Decimal(row["price"]["unit_amount_decimal"]) / 100)


def test_canonical_models_are_the_stripe_catalog_models():
    assert CANONICAL_GPU_MODELS == {r["variant"]["gpu_model"] for r in _CATALOG}
    assert len(CANONICAL_GPU_MODELS) == 55


def test_default_vram_is_the_smallest_catalog_variant():
    for model in CANONICAL_GPU_MODELS:
        smallest = min(r["variant"]["vram_gb"] for r in _CATALOG if r["variant"]["gpu_model"] == model)
        assert default_vram_gb(model) == smallest, model
    assert default_vram_gb("A100") == 40
    assert default_vram_gb("NVIDIA GeForce RTX 3060") == 12
    assert default_vram_gb("not a gpu") == 0


def test_platform_rate_is_the_standard_stripe_price():
    assert platform_rate_cad("RTX 3060") == pytest.approx(_standard_rate("RTX 3060"))
    assert platform_rate_cad("NVIDIA GeForce RTX 4090") == pytest.approx(_standard_rate("RTX 4090"))
    assert platform_rate_cad("RTX 4090", pricing_mode="spot") == pytest.approx(
        _standard_rate("RTX 4090", "spot")
    )
    # No static fallback table: an unpriced GPU has no platform rate.
    assert platform_rate_cad("Tesla K80") == 0.0
    assert platform_rate_cad("") == 0.0


def test_normalize_gpu_model_legacy_dash_and_nvidia_prefix():
    assert normalize_gpu_model("RTX-3060") == "RTX 3060"
    assert normalize_gpu_model("NVIDIA GeForce RTX 3060") == "RTX 3060"
    assert normalize_gpu_model("RTX 4090") == "RTX 4090"


def test_normalize_gpu_model_ignores_spacing_and_case():
    assert normalize_gpu_model("RTX4090") == "RTX 4090"
    assert normalize_gpu_model("rtx_4090") == "RTX 4090"
    assert normalize_gpu_model("rtx 4060 ti") == "RTX 4060 Ti"


def test_merge_host_update_preserves_dashboard_metadata():
    existing = {
        "host_id": "gpu-3060-aaryn",
        "hostname": "Aaryn 3060 Box",
        "gpu_model": "RTX 3060",
        "total_vram_gb": 12,
        "cost_per_hour": 0.13,
        "owner": "user-1",
    }
    incoming = {
        "host_id": "gpu-3060-aaryn",
        "ip": "100.64.0.6",
        "gpu_model": "",
        "total_vram_gb": 0,
        "free_vram_gb": 11.5,
        "cost_per_hour": 0,
    }
    merged = merge_host_update(existing, incoming)
    assert merged["hostname"] == "Aaryn 3060 Box"
    assert merged["gpu_model"] == "RTX 3060"
    assert merged["total_vram_gb"] == 12
    assert merged["cost_per_hour"] == 0.13
    assert merged["owner"] == "user-1"


def test_infer_gpu_from_host_id():
    assert infer_gpu_from_host_id("gpu-3060-aaryn") == "RTX 3060"
    assert infer_gpu_from_host_id("gpu-2060-aaryn-local") == "RTX 2060"
    assert infer_gpu_from_host_id("my-a100-box") == "A100"


def test_enrich_host_for_api_fills_missing_fields():
    host = {
        "host_id": "gpu-3060-lab",
        "gpu_model": "RTX 3060",
        "total_vram_gb": 0,
        "cost_per_hour": 0,
    }
    out = enrich_host_for_api(host)
    assert out["hostname"] == "gpu 3060 lab"
    assert out["vram_gb"] == 12
    assert out["total_vram_gb"] == 12
    assert out["cost_per_hour"] == pytest.approx(_standard_rate("RTX 3060"))


def test_enrich_host_for_api_infers_gpu_from_host_id():
    host = {
        "host_id": "gpu-3060-lab",
        "gpu_model": "",
        "total_vram_gb": 0,
        "cost_per_hour": 0,
    }
    out = enrich_host_for_api(host)
    assert out["gpu_model"] == "RTX 3060"
    assert out["vram_gb"] == 12
    assert out["cost_per_hour"] > 0


def test_region_normalization_from_province_and_legacy_values():
    assert normalize_region("ON") == "ca-on"
    assert normalize_region("ca-ON") == "ca-on"
    assert normalize_region("", country="CA", province="QC") == "ca-qc"
    assert normalize_host_region({"country": "CA", "province": "BC"}) == "ca-bc"


def test_enrich_host_for_api_fills_region_from_province():
    out = enrich_host_for_api({"host_id": "h-region", "country": "CA", "province": "ON"})
    assert out["region"] == "ca-on"



@pytest.mark.parametrize("country,province,region", [
    ("ca", "on", "ca-on"), ("US", "NY", "us-ny"), ("GB", "ENG", "gb-eng"),
    ("DE", "BE", "de-be"), ("AU", "NSW", "au-nsw"),
])
def test_exact_location_uses_the_shared_iso_catalog(country, province, region):
    from host_metadata import require_location

    assert require_location(country, province)["region"] == region


@pytest.mark.parametrize("country,province,region", [
    ("", "", ""), ("XX", "", ""), ("CA", "", "ca-central"),
    ("CA", "", "ca-east"), ("CA", "Ontario", ""),
    ("CA", "ON", "ca-east"), ("US", "ON", ""),
])
def test_missing_ambiguous_or_conflicting_location_is_an_error(country, province, region):
    from host_metadata import require_location

    with pytest.raises(ValueError):
        require_location(country, province, region)


def test_no_region_is_invented_for_an_incomplete_record():
    assert normalize_region("") == ""
    assert normalize_host_region({}) == ""


def test_country_choices_are_complete_names_from_iso_data():
    import pycountry
    from host_metadata import location_catalog

    catalog = location_catalog("CA")
    assert {c["code"] for c in catalog["countries"]} == {c.alpha_2 for c in pycountry.countries}
    assert next(c for c in catalog["countries"] if c["code"] == "CA")["name"] == "Canada"
    assert next(s for s in catalog["subdivisions"] if s["code"] == "ON")["name"] == "Ontario"


def test_profile_completion_requires_every_required_account_field():
    from host_metadata import profile_missing_fields

    valid = {"name":"Account Owner", "email":"owner@example.test", "country":"CA", "province":"ON",
             "account_type":"individual", "address_line1":"1 Test Street",
             "city":"London", "postal_code":"N6A 1A1"}
    assert profile_missing_fields(valid) == []
    assert profile_missing_fields({**valid, "onboarding_intent": ""}) == []
    for field in valid:
        assert profile_missing_fields({**valid, field:""})


def test_address_requirements_follow_each_country_not_a_canadian_default():
    from host_metadata import address_required_fields, profile_address_values

    assert "postal_code" in address_required_fields("CA")
    assert "postal_code" not in address_required_fields("HK")
    _, errors = profile_address_values({"country":"CA", "province":"ON", "address_line1":"1 Test Street", "city":"London", "postal_code":"12345"})
    assert errors == ["postal_code"]  # Not a Canadian postal code at all.


def test_a_postal_code_from_the_next_province_warns_but_never_blocks():
    """Real addresses cross these lines: Lloydminster, Alberta uses Saskatchewan's S9V."""
    from host_metadata import profile_address_values, review_profile_address

    lloydminster = {"country":"CA", "province":"AB", "address_line1":"4915 50 St", "city":"Lloydminster", "postal_code":"S9V 0A1"}
    assert profile_address_values(lloydminster)[1] == []
    review = review_profile_address(lloydminster)
    assert review["errors"] == []
    assert "Saskatchewan, not Alberta" in review["warnings"][0]["message"]
    typo = review_profile_address({**lloydminster, "province":"BC", "postal_code":"N6B 1Z2"})
    assert "Ontario, not British Columbia" in typo["warnings"][0]["message"]
    us = review_profile_address({"country":"US", "province":"CA", "address_line1":"1 Main St", "city":"New York", "postal_code":"10001"})
    assert "ZIP code in New York, not California" in us["warnings"][0]["message"]
    # Two provinces share X (and K1A spans Ontario and Quebec): say so without guessing which.
    shared = review_profile_address({**lloydminster, "province":"NU", "postal_code":"K1A 0B1"})
    assert shared["warnings"][0]["message"].startswith("K1A 0B1 isn't usually a postal code in Nunavut.")
    assert review_profile_address({**lloydminster, "province":"SK"})["warnings"] == []


@pytest.mark.parametrize("country,typed,stored", [
    ("CA", "n6b1z2", "N6B 1Z2"), ("CA", "N6B-1Z2", "N6B 1Z2"), ("US", "100011234", "10001-1234"),
    ("US", "10001", "10001"), ("GB", "sw1a1aa", "SW1A 1AA"), ("GB", "m11ae", "M1 1AE"),
    ("NL", "1234ab", "1234 AB"), ("JP", "1008994", "100-8994"), ("CA", "12345", "12345"), ("CA", "", ""),
])
def test_postal_codes_are_stored_the_way_each_country_writes_them(country, typed, stored):
    from host_metadata import accepted_postal_code, profile_address_values, standard_postal_code

    assert standard_postal_code(country, typed) == stored
    if stored and stored != "12345":
        assert accepted_postal_code(country, typed) == stored
    else:
        assert accepted_postal_code(country, typed) == ""
    if country == "CA" and typed.startswith("n"):
        cleaned, _ = profile_address_values({"country":"CA", "province":"ON", "address_line1":"300 Dufferin Ave", "city":"London", "postal_code":typed})
        assert cleaned["postal_code"] == stored


@pytest.mark.parametrize("country,value,code", [
    ("CA", "QC", "QC"), ("CA", "Québec", "QC"), ("CA", "quebec", "QC"), ("US", "District of Columbia", "DC"),
    ("GB", "England", "ENG"), ("DE", "Bayern", "BY"), ("CA", "Atlantis", ""), ("CA", "", ""), ("", "Ontario", ""),
])
def test_a_state_name_or_code_maps_to_the_selector_or_to_nothing(country, value, code):
    from host_metadata import subdivision_code

    assert subdivision_code(country, value) == code


def test_postal_lines_follow_each_countrys_layout():
    from host_metadata import postal_lines

    assert postal_lines({"country":"US", "province":"NY", "address_line1":"1 Main St", "address_line2":"Apt 4", "city":"New York", "postal_code":"10001"}) == [
        "1 Main St", "Apt 4", "New York, NY 10001", "United States"]
    assert postal_lines({"country":"DE", "province":"BE", "address_line1":"Unter den Linden 77", "city":"Berlin", "postal_code":"10117"}) == [
        "Unter den Linden 77", "10117 Berlin", "Germany"]


def test_address_metadata_covers_every_country_we_offer():
    import pycountry
    from host_metadata import address_required_fields

    for country in pycountry.countries:
        assert isinstance(address_required_fields(country.alpha_2), list)


@pytest.mark.parametrize("country,places,code", [
    ("JP", {"state": "東京都"}, "13"),                         # Google's local name
    ("KR", {"city": "Seoul"}, "11"),                          # Google's Latin name, from the city
    ("KR", {"city": "서울특별시"}, "11"),
    ("AE", {"state": "Dubai"}, "DU"),
    ("ES", {"state": "Comunidad de Madrid", "city": "Madrid"}, "M"),  # the province, the level Spanish post uses
    ("IE", {"county": "Dublin 2", "city": "Dublin"}, "D"),
    ("DE", {"city": "Berlin"}, "BE"),                         # a city that is its own Land
    ("SE", {"county": "Stockholms län"}, "AB"),               # ISO writes "Stockholms län [SE-01]"
    ("PL", {"state": "województwo mazowieckie"}, "14"),       # without the generic word
    ("CA", {"postal_code": "N6B 1Z2"}, "ON"),                 # nothing named: the postal prefix
    ("JP", {"postal_code": "100-0005"}, "13"),
    ("CA", {"state": "Alberta", "postal_code": "S9V 0A1"}, "AB"),  # a named province beats the prefix
    ("PL", {"state": "Masovian Voivodeship", "city": "Warsaw"}, ""),  # no source knows this spelling
    ("SG", {"city": "Singapore"}, ""),
])
def test_regions_resolve_only_from_an_exact_source(country, places, code):
    from host_metadata import resolve_subdivision

    assert resolve_subdivision(country, **places) == code


def test_every_country_keeps_its_regions_for_an_address():
    """Nothing is hidden: a region is offered wherever ISO lists one, required only where the post asks."""
    from host_metadata import location_catalog

    gb = location_catalog("GB", "address")
    assert len(gb["subdivisions"]) > 200 and gb["subdivision_mode"] == "optional" and not gb["requires_subdivision"]
    assert gb["labels"]["subdivision"] == "Region"
    assert all("[" not in item["name"] for item in location_catalog("SE", "address")["subdivisions"])
    us = location_catalog("US", "address")
    assert us["requires_subdivision"] and us["labels"] == {"subdivision": "State", "city": "City", "postal_code": "ZIP code"}
    jp = location_catalog("JP", "address")
    assert jp["requires_subdivision"] and jp["labels"]["subdivision"] == "Prefecture"
    assert location_catalog("HK", "address")["subdivision_mode"] == "none"  # ISO lists no regions to pick
    assert location_catalog("GB", "host")["requires_subdivision"]  # a host's region is its scheduling region


def test_a_chosen_region_is_kept_everywhere_and_a_blank_one_is_fine_unless_required():
    from host_metadata import require_address_location

    assert require_address_location("GB", "ENG")["province"] == "ENG"
    assert require_address_location("DE", "")["province"] == ""
    assert require_address_location("IE", "D")["province"] == "D"
    with pytest.raises(ValueError):
        require_address_location("US", "")
    with pytest.raises(ValueError):
        require_address_location("GB", "XX")  # given, so it must be real


@pytest.mark.parametrize("country,code,latin,shown", [
    ("CA", "ON", True, "ON"), ("US", "NY", True, "NY"),            # the post writes the abbreviation
    ("IE", "D", True, "Co. Dublin"), ("JP", "13", True, "Tokyo"), ("JP", "13", False, "東京都"),
    ("KR", "11", True, "Seoul"), ("ES", "M", True, "Madrid"), ("SE", "AB", True, "Stockholms län"),
    ("DE", "BE", True, "Berlin"), ("GB", "", True, ""),
])
def test_regions_are_written_the_way_each_post_writes_them(country, code, latin, shown):
    from host_metadata import region_display

    assert region_display(country, code, latin=latin) == shown


def test_review_lines_use_the_script_of_the_address():
    from host_metadata import postal_lines

    jp = {"country": "JP", "province": "13", "postal_code": "100-0005"}
    assert postal_lines({**jp, "address_line1": "Marunouchi 1-9-1", "city": "Chiyoda"}) == [
        "Marunouchi 1-9-1, Chiyoda, Tokyo", "100-0005", "Japan"]
    assert postal_lines({**jp, "address_line1": "丸の内一丁目9-1", "city": "千代田区"}) == [
        "〒100-0005", "東京都", "千代田区丸の内一丁目9-1", "Japan"]
    assert postal_lines({"country": "ES", "province": "M", "address_line1": "Calle de Alcalá 42", "city": "Madrid", "postal_code": "28014"}) == [
        "Calle de Alcalá 42", "28014 Madrid", "Spain"]  # not "28014 Madrid Madrid"
    assert postal_lines({"country": "IE", "province": "D", "address_line1": "1 Grafton Street", "city": "Dublin", "postal_code": "D02 K198"}) == [
        "1 Grafton Street", "Dublin", "Co. Dublin", "D02 K198", "Ireland"]
