"""Features that are built but not on for everyone, and whether each is on now.

Several of Xcelsior's features are finished code that is switched off by
default: serverless inference, Bitcoin and Lightning deposits, Wise payouts.
Nothing told a customer or provider which of them they could use, so a feature
that was off looked the same from outside as one that did not exist, and one
that was on for a few owners looked like it was on for everyone.

The public status page lists these with their live state. Each entry reads the
**same switch the feature itself reads**, never its own copy of an environment
variable: a re-parsed flag can disagree with the code it describes (the Bitcoin
switch accepts only ``true``, the cloud-burst switch also accepts ``1`` and
``yes``), and then the page would say "on" while the feature refused.

Every flag in ``.env.example`` that defaults off is either listed here or in
``INTERNAL_FLAGS`` with the reason it is not a customer-facing feature;
``tests/test_labs_features.py`` fails on a new one that is in neither.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

State = Literal["live", "limited", "off"]
Audience = Literal["customers", "providers"]


def _serverless_state() -> State:
    from serverless.feature import serverless_feature_status

    status = serverless_feature_status()
    if not status["global_enabled"]:
        return "off"
    return "limited" if status["allowlist_active"] else "live"


def _bitcoin_state() -> State:
    import bitcoin

    return "live" if bitcoin.BTC_ENABLED else "off"


def _lightning_state() -> State:
    import lightning

    return "live" if lightning.LN_ENABLED else "off"


def _wise_state() -> State:
    import wise_payouts

    return "live" if wise_payouts.payouts_enabled() else "off"


@dataclass(frozen=True)
class PreviewFeature:
    #: Stable identifier. The website's copy for it is ``status.feature_<key>_*``.
    key: str
    audience: Audience
    #: The environment variable an operator sets to turn it on.
    flag: str
    state: Callable[[], State]


FEATURES: tuple[PreviewFeature, ...] = (
    PreviewFeature("serverless", "customers", "XCELSIOR_SERVERLESS_ENABLED", _serverless_state),
    PreviewFeature("bitcoin", "customers", "XCELSIOR_BTC_ENABLED", _bitcoin_state),
    PreviewFeature("lightning", "customers", "XCELSIOR_LN_ENABLED", _lightning_state),
    PreviewFeature("wise_payouts", "providers", "XCELSIOR_WISE_PAYOUTS_ENABLED", _wise_state),
)

#: Off-by-default flags that are not something a customer or provider uses,
#: and why. A flag belongs here only if a person outside the company would
#: never notice it change.
INTERNAL_FLAGS: dict[str, str] = {
    "XCELSIOR_TAILSCALE_ENABLED": "joins hosts to the operator's mesh network",
    "XCELSIOR_BURST_ENABLED": "operator capacity overflow to other clouds, not a customer option",
    "XCELSIOR_ANALYTICS_EXPORT_ENABLED": "internal analytics projection",
    "XCELSIOR_RETRIEVAL_ENABLED": "internal retrieval and model service",
}


def preview_features() -> list[dict[str, str]]:
    """Each preview feature's key, audience and state, for the status page.

    A feature whose switch cannot be read (its module failed to import) is
    reported ``off``: the page must never claim a feature works because the
    code that would refuse it is missing.
    """
    out = []
    for feature in FEATURES:
        try:
            state = feature.state()
        except Exception:  # noqa: BLE001 - any failure reads as "not available"
            state = "off"
        out.append({"key": feature.key, "audience": feature.audience, "state": state})
    return out
