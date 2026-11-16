"""Server-only Places API (New) address lookup. Google responses are not cached."""

import os
import re

import httpx


class PlacesUnavailable(Exception):
    """A provider failure that must not expose its response, query or API key."""


def api_key() -> str:
    return (os.environ.get("XCELSIOR_GOOGLE_PLACES_API_KEY") or "").strip()


def _payload(response: httpx.Response) -> dict:
    try:
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        raise PlacesUnavailable(f"HTTP {response.status_code}") from None
    if not isinstance(payload, dict):
        raise PlacesUnavailable("invalid response")
    return payload


def suggestions(
    text: str, *, country: str, lang: str, visitor_country: str, session_token: str
) -> list[dict]:
    body: dict[str, object] = {"input": text, "sessionToken": session_token}
    if country:
        body["includedRegionCodes"] = [country.lower()]
    if country or visitor_country:
        body["regionCode"] = (country or visitor_country).lower()
    if re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z]{2,4})?", lang):
        body["languageCode"] = lang
    try:
        response = httpx.post(
            "https://places.googleapis.com/v1/places:autocomplete",
            json=body,
            headers={
                "X-Goog-Api-Key": api_key(),
                "X-Goog-FieldMask": "suggestions.placePrediction.placeId,suggestions.placePrediction.structuredFormat,suggestions.placePrediction.text",
            },
            timeout=4.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise PlacesUnavailable(type(exc).__name__) from None
    payload = _payload(response)
    items = payload.get("suggestions", [])  # Google returns {} when nothing matches.
    if not isinstance(items, list):
        raise PlacesUnavailable("invalid suggestions")
    rows = []
    for item in items:
        prediction = item.get("placePrediction") if isinstance(item, dict) else None
        if not isinstance(prediction, dict):
            continue
        place_id = prediction.get("placeId")
        if not isinstance(place_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,255}", place_id):
            continue
        structured = prediction.get("structuredFormat")
        structured = structured if isinstance(structured, dict) else {}

        def text(value: object) -> str:
            field = value.get("text") if isinstance(value, dict) else None
            return field if isinstance(field, str) else ""

        main = text(structured.get("mainText")) or text(prediction.get("text"))
        secondary = text(structured.get("secondaryText"))
        if isinstance(main, str) and isinstance(secondary, str) and main:
            rows.append(
                {
                    "id": f"google-{place_id}",
                    "provider": "google",
                    "place_id": place_id,
                    "text": main,
                    "description": secondary,
                }
            )
    return rows[:5]


def details(place_id: str, *, session_token: str, country: str, lang: str) -> dict[str, str]:
    from host_metadata import accepted_postal_code, resolve_subdivision, standard_postal_code
    from routes.hosts import _NUMBER_AFTER_STREET, _non_latin

    params = {"sessionToken": session_token}
    if re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z]{2,4})?", lang):
        params["languageCode"] = lang
    try:
        response = httpx.get(
            f"https://places.googleapis.com/v1/places/{place_id}",
            params=params,
            headers={
                "X-Goog-Api-Key": api_key(),
                "X-Goog-FieldMask": "addressComponents,formattedAddress",
            },
            timeout=4.0,
            follow_redirects=False,
        )
    except httpx.HTTPError as exc:
        raise PlacesUnavailable(type(exc).__name__) from None
    payload = _payload(response)
    components = payload.get("addressComponents")
    if not isinstance(components, list):
        raise PlacesUnavailable("missing address")

    def component(kind: str, *, short: bool = False) -> str:
        for value in components:
            if (
                isinstance(value, dict)
                and isinstance(value.get("types"), list)
                and kind in value["types"]
            ):
                text = value.get("shortText" if short else "longText", "")
                return " ".join(text.split()) if isinstance(text, str) else ""
        return ""

    found_country = component("country", short=True).upper()
    import pycountry

    if pycountry.countries.get(alpha_2=found_country) is None or (
        country and country != found_country
    ):
        raise PlacesUnavailable("address country mismatch")
    street, number = component("route"), component("street_number")
    city = (
        component("locality")
        or component("postal_town")
        or component("administrative_area_level_3")
        or component("sublocality_level_1")
    )
    if street:
        line1 = " ".join(
            part
            for part in (
                (street, number)
                if found_country in _NUMBER_AFTER_STREET or _non_latin(street)
                else (number, street)
            )
            if part
        )
    else:
        line1 = component("premise") or component("subpremise")
    if not line1:
        raise PlacesUnavailable("no street address")
    postal = component("postal_code")
    suffix = component("postal_code_suffix")
    if suffix and found_country == "US":
        postal = f"{postal}-{suffix}"
    formatted = standard_postal_code(found_country, postal)
    return {
        "address_line1": line1,
        "address_line2": component("subpremise") if street else "",
        "city": city,
        "postal_code": accepted_postal_code(found_country, formatted),
        "country": found_country,
        "province": resolve_subdivision(
            found_country,
            state=component("administrative_area_level_1"),
            county=component("administrative_area_level_2"),
            city=city,
            postal_code=formatted,
        ),
    }
