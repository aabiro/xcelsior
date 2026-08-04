"""Every endpoint reachable without a credential is classified, by name.

Two shapes of "open" exist in this codebase and they were being tracked
separately, which meant neither list was complete:

* **conditional scope** — `user = _get_current_user(...)` then
  `if user: _require_scope(...)`. Anonymous callers skip the check.
  `tests/test_conditional_scope_guard.py` owns that shape.
* **no auth at all** — the handler reaches no auth call on any path.
  `/api/ssh/pubkey` is the obvious member, and pinning it in an ad-hoc test
  meant the thing that tracks open endpoints no longer tracked all of them.

This file covers the second shape and is the inventory of record.

**How the list is derived.** A fixed seed of auth primitives misses indirection:
`api_terminate_instance` authorizes through `_authorize_instance_mutation` and
`api_deposit` through `_require_customer_access`, so a naive scan reported 153
open routes when the real number is 96. The scan below computes a transitive
closure — any function calling an auth-carrying function is itself
auth-carrying — and iterates to a fixpoint.

**Known limitation, stated rather than hidden.** That closure is loose in one
direction: a route handler that calls `_require_auth` becomes an auth helper, so
a second handler calling *it* reads as authenticated. The result is a floor, not
a ceiling — it can miss an open route, it will not invent one. Tightening this
means distinguishing helpers from handlers, which is worth doing if a false
negative is ever found.

Every entry needs a category and a reason. An endpoint that is open for a good
reason and one that is open by accident look identical in a route table; the
difference only exists if someone writes it down.
"""

from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTES = ROOT / "routes"

AUTH_PRIMITIVES = {
    "_require_auth", "_require_admin", "_require_user_grant", "_require_scope",
    "_get_current_user", "_require_provider_or_admin", "_require_agent_auth",
    "_require_user_or_scoped_machine", "_require_host_operator", "_require_master_token",
}

#: handler -> (category, why it is reachable without a credential)
OPEN_ENDPOINTS: dict[str, tuple[str, str]] = {
    # ── OAuth/OIDC protocol surface: unauthenticated by specification ──
    "oauth_authorization_server_metadata": ("oauth_protocol", "RFC 8414 discovery"),
    "oauth_jwks_document": ("oauth_protocol", "public verification keys"),
    "oauth_token": ("oauth_protocol", "clients authenticate in the body, not a bearer"),
    "oauth_token_compat": ("oauth_protocol", "legacy alias of the token endpoint"),
    "oauth_dynamic_client_registration": ("oauth_protocol", "RFC 7591 DCR"),
    "oauth_device_authorize": ("oauth_protocol", "RFC 8628 device flow start"),
    "oauth_device_authorize_compat": ("oauth_protocol", "legacy alias"),
    "oauth_verify_page": ("oauth_protocol", "device-code entry page"),
    "api_auth_device_code": ("oauth_protocol", "device flow start"),
    "api_auth_device_token": ("oauth_protocol", "device flow poll"),
    "api_auth_verify_page": ("oauth_protocol", "device-code entry page"),
    # ── Pre-authentication: the caller has no credential yet, by definition ──
    "api_auth_login": ("pre_auth", "issues the credential"),
    "api_auth_register": ("pre_auth", "creates the account"),
    "api_auth_password_reset": ("pre_auth", "the user cannot sign in"),
    "api_auth_password_reset_confirm": ("pre_auth", "bearer is the emailed token"),
    "api_auth_verify_email": ("pre_auth", "bearer is the emailed token"),
    "api_auth_resend_verification": ("pre_auth", "account not yet usable"),
    "api_auth_email_change_confirm": ("pre_auth", "bearer is the emailed token"),
    "api_auth_oauth_initiate": ("pre_auth", "social sign-in start"),
    "api_auth_oauth_callback": ("pre_auth", "social sign-in return"),
    "api_auth_demo_credentials": ("pre_auth", "IP-gated demo account"),
    "api_mfa_verify_login": ("pre_auth", "second factor, before the session exists"),
    "api_mfa_sms_send_login": ("pre_auth", "second factor delivery"),
    "api_mfa_passkey_authenticate_options": ("pre_auth", "WebAuthn challenge"),
    "api_mfa_passkey_authenticate_complete": ("pre_auth", "WebAuthn assertion"),
    "api_accept_team_invite": ("pre_auth", "bearer is the emailed invite token"),
    # ── Liveness and operations: must answer before auth is available ──
    "healthz": ("liveness", "container liveness probe"),
    "livez": ("liveness", "container liveness probe"),
    "readyz": ("liveness", "readiness probe"),
    "startupz": ("liveness", "startup gate report"),
    "service_status": ("liveness", "status page"),
    "metrics": ("liveness", "scrape endpoint"),
    "metrics_prometheus": ("liveness", "scrape endpoint"),
    "root": ("liveness", "index route; serves the marketing shell, no account data"),
    # ── Signature-verified: the signature is the credential, not a bearer ──
    "api_stripe_webhook": ("signed_webhook", "Stripe signature verification"),
    "api_paypal_webhook": ("signed_webhook", "PayPal signature verification"),
    "facebook_deauthorize": ("signed_webhook", "signed_request verification"),
    "facebook_delete_data": ("signed_webhook", "signed_request verification"),
    # ── Public catalogue: pricing and availability are the product's shop window ──
    "api_pricing_rates": ("public_catalog", "published price list"),
    "api_reference_pricing": ("public_catalog", "published price list"),
    "api_pricing_models": ("public_catalog", "published price list"),
    "api_reserved_plans": ("public_catalog", "published plans"),
    "api_spot_quote": ("public_catalog", "live spot quote"),
    "api_spot_prices": ("public_catalog", "live spot prices"),
    "api_spot_enabled": ("public_catalog", "feature availability"),
    "api_spot_floor_suggestion": ("public_catalog", "published floor guidance"),
    "api_estimate_cost": ("public_catalog", "quote for an unsigned-up visitor"),
    "api_crypto_enabled": ("public_catalog", "payment method availability"),
    "api_crypto_rate": ("public_catalog", "published rate"),
    "api_ln_enabled": ("public_catalog", "payment method availability"),
    "api_ln_rate": ("public_catalog", "published rate"),
    "api_paypal_enabled": ("public_catalog", "payment method availability"),
    "api_marketplace_search": ("public_catalog", "browse without an account"),
    "api_marketplace_spot_prices": ("public_catalog", "live competing rates"),
    "api_marketplace_spot_history": ("public_catalog", "price history"),
    "api_marketplace_stats_v2": ("public_catalog", "aggregate marketplace stats"),
    "api_list_tiers": ("public_catalog", "published tiers"),
    "api_image_templates": ("public_catalog", "published image list"),
    "api_sla_targets": ("public_catalog", "published SLA targets"),
    "api_get_compute_score": ("public_catalog", "published host score"),
    "api_list_compute_scores": ("public_catalog", "published host scores"),
    "api_verified_hosts": ("public_catalog", "published verification status"),
    "api_verification_status": ("public_catalog", "published verification status"),
    "api_tax_rates": ("public_catalog", "published tax rates"),
    "api_compliance_status": ("public_catalog", "published posture"),
    "api_provider_attestation": ("public_catalog", "published attestation"),
    "api_chat_suggestions": ("public_catalog", "static prompt suggestions"),
    # ── Published documents and static pages ──
    "api_llms_txt": ("static_document", "agent-facing site description"),
    "dashboard": ("static_document", "HTML shell; data is fetched with credentials"),
    "legacy_dashboard": ("static_document", "HTML shell"),
    "api_attestation_schema": ("static_document", "published JSON schema"),
    "api_scip_partner_loi": ("static_document", "published partner document"),
    "api_scip_alignment": ("static_document", "published partner document"),
    "api_h100_partner": ("static_document", "published partner document"),
    "api_platform_ops_plan": ("static_document", "published operations plan"),
    "api_v1_openapi": ("static_document", "published API description"),
    "api_get_pubkey": (
        "static_document",
        "the platform's host-access PUBLIC key, for hosts' authorized_keys. "
        "No user key material, so no enumeration surface, and locking it would "
        "break host provisioning for no security gain",
    ),
    # ── Tombstones: permanently disabled, answer 410 ──
    "api_generate_api_key": ("tombstone", "410 Gone; API keys are disabled"),
    "api_list_keys": ("tombstone", "410 Gone; nothing to list since keys are disabled"),
    "api_revoke_key": ("tombstone", "410 Gone; nothing to revoke since keys are disabled"),
    # ── Internal callbacks authenticated by their own shared secret ──
    "api_agent_versions": ("internal_secret", "agent version manifest"),
    "api_v2_token_rotate": ("internal_secret", "presents the token being rotated"),
    "api_agent_verify": ("internal_secret", "agent attestation callback"),
    "api_internal_route": ("internal_secret", "internal routing lookup"),
    "sse_stream": ("internal_secret", "ticket-authenticated event stream"),
    # ── Stripe Connect: router denies by default; these are its exemptions ──
    "handle_thin_webhook": (
        "signed_webhook",
        "requires a stripe-signature header, rejects when absent, and "
        "parse_event_notification verifies it against the endpoint secret — "
        "Stripe sends no bearer, so the signature is the credential",
    ),
    "connect_dashboard_page": (
        "static_document",
        "returns the _DASHBOARD_HTML constant; renders no account data and "
        "fetches from the JSON API with the browser's credential",
    ),
    "storefront_page": ("static_document", "static HTML shell, no account data rendered"),
    "success_page": ("static_document", "static post-checkout return page"),
}

#: Open **and not yet justified**. This is a ratchet: entries leave when they
#: are either guarded or moved above with a reason. It may only shrink.
#:
#: `routes/stripe_connect_v2.py` is unauthenticated and its router is mounted.
#: Verified by request: `GET /api/connect/accounts` returns 200 with no
#: credential, and the POSTs fail on schema validation (422) rather than
#: authorization, so a well-formed body proceeds. The module logs
#: `Stripe Connect ENABLED (mode=live)`, so it is not flagged off.
#:
#: Left unclassified deliberately. The correct guard differs per endpoint —
#: `list_connected_accounts` is operator-shaped, `create_connected_account` is
#: provider-shaped, and `create_checkout_session` may be legitimately public
#: because a buyer has no platform credential. Guessing the authorization model
#: for a live money surface is how the `hosts:read` reclassification broke
#: provider onboarding.
NEEDS_REVIEW: dict[str, str] = {}

#: Renamed from `MAX_UNCLASSIFIED`, which collided with the GT0 ratchet in
#: `test_gt0_classification_ratchet.py`. Two ceilings with one name, bounding
#: different sets, is how a number gets quoted against the wrong denominator —
#: the 39-vs-30 and 528-vs-516 confusions in this codebase were both that.
#:
#: This bounds **open endpoints lacking a justification**. GT0's bounds
#: **inventory rows lacking a classification**. They are unrelated counts.
MAX_UNJUSTIFIED_OPEN_ENDPOINTS = 0  # was 11: all of stripe_connect_v2, now router-gated
VALID_CATEGORIES = {
    "oauth_protocol", "pre_auth", "liveness", "signed_webhook",
    "public_catalog", "static_document", "tombstone", "internal_secret",
}


def _auth_carrying_functions() -> tuple[set[str], dict[tuple[str, str], set[str]], dict]:
    trees = {p: ast.parse(p.read_text(encoding="utf-8")) for p in sorted(ROUTES.glob("*.py"))}
    defs, calls = {}, {}
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = (path.name, node.name)
                defs[key] = node
                calls[key] = {
                    n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                    for n in ast.walk(node)
                    if isinstance(n, ast.Call)
                }
    auth = set(AUTH_PRIMITIVES)
    for _ in range(20):
        grew = False
        for (_, name), called in calls.items():
            if name not in auth and (called & auth):
                auth.add(name)
                grew = True
        if not grew:
            break
    return auth, calls, defs


def _router_level_guards() -> dict[str, set[str]]:
    """file -> paths its router-level dependency leaves public.

    A router declared `APIRouter(dependencies=[Depends(guard)])` authorizes
    every route on it without the handler body containing an auth call, so a
    scan of handler bodies alone reports all of them as open. That blind spot
    was created by the `stripe_connect_v2` deny-by-default gate and caught
    immediately — the registry still listed eleven endpoints as unauthenticated
    after they had been guarded.

    A guard that exempts by path is read for its exemption set, so the routes
    that really are still public stay visible rather than being hidden behind
    the router's dependency.

    **Parsed with AST, not by splitting on `frozenset({`.** The first version
    did exactly that, and it fails in the dangerous direction: written as a
    plain set literal, a tuple, or a constant imported from another module, the
    exemption set parses as empty — and an empty exemption set means every route
    on that router reads as guarded. False negatives, which is the one direction
    this scanner is supposed never to go.
    """
    guarded: dict[str, set[str]] = {}
    for path in sorted(ROUTES.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue

        # Does this module construct a router with router-level dependencies?
        has_router_dependency = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "APIRouter"
            and any(kw.arg == "dependencies" for kw in node.keywords)
            for node in ast.walk(tree)
        )
        if not has_router_dependency:
            continue

        # Every string constant that looks like a path, anywhere in a
        # module-level collection. Deliberately broad: over-collecting an
        # exemption keeps a route *visible* as open, which is the safe error.
        exempt: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                for element in node.elts:
                    if (
                        isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                        and element.value.startswith("/")
                    ):
                        exempt.add(element.value)
        guarded[path.name] = exempt
    return guarded


def open_handlers() -> dict[str, str]:
    """handler name -> file, for every route reaching no auth call."""
    auth, calls, defs = _auth_carrying_functions()
    router_guards = _router_level_guards()
    found: dict[str, str] = {}
    for (fname, name), node in defs.items():
        methods = [
            (d.func if isinstance(d, ast.Call) else d)
            for d in node.decorator_list
        ]
        if not any(
            isinstance(m, ast.Attribute)
            and m.attr.lower() in {"get", "post", "put", "patch", "delete"}
            for m in methods
        ):
            continue
        if calls[(fname, name)] & auth:
            continue
        if fname in router_guards:
            # Guarded at the router. Only the paths its dependency exempts are
            # still open, and those are matched by decorator path below.
            decorated = {
                d.args[0].value
                for d in node.decorator_list
                if isinstance(d, ast.Call) and d.args
                and isinstance(d.args[0], ast.Constant)
                and isinstance(d.args[0].value, str)
            }
            if not (decorated & router_guards[fname]):
                continue
        found[name] = fname
    return found


def test_every_open_endpoint_is_classified_or_under_review():
    """No third category. An endpoint is justified, or it is on the ratchet."""
    unaccounted = sorted(
        set(open_handlers()) - set(OPEN_ENDPOINTS) - set(NEEDS_REVIEW)
    )
    assert not unaccounted, (
        "these endpoints are reachable without a credential and appear in no "
        "list. Classify each with a category and a reason, or add it to "
        f"NEEDS_REVIEW and raise the ratchet deliberately: {unaccounted}"
    )


def test_the_unjustified_open_endpoint_count_does_not_grow():
    """The ratchet. It may only shrink.

    This asserted `len(NEEDS_REVIEW) <= MAX_UNCLASSIFIED` where the ceiling was
    `len(NEEDS_REVIEW)` — a number compared to itself, which could never fail.
    Every prior report that "the ratchet held" was worth nothing. The ceiling is
    now a recorded literal, so lowering it is what progress means.
    """
    assert len(NEEDS_REVIEW) <= MAX_UNJUSTIFIED_OPEN_ENDPOINTS, (
        f"{len(NEEDS_REVIEW)} open endpoints await justification, up from "
        f"{MAX_UNJUSTIFIED_OPEN_ENDPOINTS}. An open endpoint leaves this list "
        "by being justified or guarded, never by the ceiling rising."
    )


def test_every_category_is_a_known_one():
    """A typo would create a category nobody reviews."""
    bad = {n: c for n, (c, _) in OPEN_ENDPOINTS.items() if c not in VALID_CATEGORIES}
    assert not bad, f"unknown categories: {bad}"


def test_every_classification_carries_a_reason():
    """A category without a reason is a label, not a justification."""
    empty = sorted(n for n, (_, why) in OPEN_ENDPOINTS.items() if len(why.strip()) < 10)
    assert not empty, f"classified with no usable reason: {empty}"


def test_no_dead_entries():
    """An endpoint that gained auth, or was deleted, must leave the registry.

    A stale entry is a permission nobody is reviewing, and it slowly turns the
    list into a blanket the next author reads as approval.
    """
    live = set(open_handlers())
    dead = sorted((set(OPEN_ENDPOINTS) | set(NEEDS_REVIEW)) - live)
    assert not dead, (
        f"registry entries for endpoints that are no longer open: {dead}. "
        "Remove them — if one was guarded, that is progress worth recording."
    )


def test_the_scanner_finds_the_endpoint_that_started_this():
    """Prove the reach rather than trusting the silence.

    `/api/ssh/pubkey` is the endpoint whose ad-hoc pin revealed that the
    open-endpoint tracking was incomplete. If the scanner stops finding it,
    every assertion above passes on a smaller set and reports clean.
    """
    handlers = open_handlers()
    assert "api_get_pubkey" in handlers, (
        "the scanner no longer detects a known-open endpoint; the checks above "
        "would pass vacuously"
    )
    assert len(handlers) > 50, f"scan found only {len(handlers)} open routes"
