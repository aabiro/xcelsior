"""Lock a known catalog spot price before freezing a new admission."""

import math


class SpotAdmissionRejected(ValueError):
    pass


def lock_admission_quote(job: dict) -> None:
    from spot_pricing import compute_live_spot_quote

    model = str(job.get("gpu_model") or "").strip()
    if not model:
        raise SpotAdmissionRejected("Spot admission requires an explicit GPU model")
    # The quote itself must refuse its compatibility defaults; a preflight
    # lookup cannot protect against a later lookup losing its catalog row.
    try:
        quote = compute_live_spot_quote(model, require_catalog=True)
    except ValueError as exc:
        raise SpotAdmissionRejected(str(exc)) from exc
    if not math.isfinite(quote.rate_cad) or quote.rate_cad <= 0:
        raise SpotAdmissionRejected("Spot quote is unavailable")
    job["spot_rate_cad"] = quote.rate_cad
