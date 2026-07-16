# Xcelsior Demo Account — IP-gated one-click login for the owner's networks.
#
# WHY THIS EXISTS
# --------------
# Playwright / E2E automation kept dying at the auth gate. This gives a single,
# standing demo admin account whose credentials are revealed only to the owner's
# own IPs (home public IP, both Tailscale nodes, the home LAN, and localhost).
# On the login page, a "Demo account" button appears for those IPs and one click
# fills the email + password so a human — or a browser-automation script — is one
# click from an authenticated session.
#
# SINGLE SOURCE OF TRUTH
# ----------------------
# The credentials and the whitelist live here. The login endpoint
# (GET /api/auth/demo-credentials), the seed script (scripts/seed_demo_account.py),
# and the Playwright helper (frontend/e2e/helpers/demo-auth.ts) all read from here
# (directly or via env) so nothing drifts.
#
# SECURITY POSTURE — read this before widening the whitelist
# ----------------------------------------------------------
# * This is a REAL admin account with a FIXED, known password. The IP whitelist
#   only decides whether the password is *handed out* by the endpoint and whether
#   the button renders — the account itself is reachable by anyone who knows the
#   password. Treat the password as public-ish; its protection is the whitelist.
# * The client IP comes from proxy headers (cf-connecting-ip / x-real-ip /
#   x-forwarded-for via routes/_deps._get_real_client_ip). That is only
#   trustworthy if the edge (Cloudflare / nginx) sets those headers and strips
#   client-supplied ones. Ensure that before relying on the gate in prod.
# * Kill switch: set XCELSIOR_DEMO_ENABLED=0 to make the endpoint 404 and the
#   button disappear everywhere, without removing the account.

from __future__ import annotations

import ipaddress
import os

# ── Credentials (overridable by env, but these are the standing defaults) ──────
DEMO_EMAIL = os.environ.get("XCELSIOR_DEMO_EMAIL", "demo@xcelsior.ca").strip().lower()
DEMO_PASSWORD = os.environ.get("XCELSIOR_DEMO_PASSWORD", "DemoUser123abc!")
DEMO_NAME = os.environ.get("XCELSIOR_DEMO_NAME", "Demo Account")

# ── Default whitelist — the owner's own networks (2026-07-15) ──────────────────
# Determined live on the ASUS host (aaryn-tuf-rtx2060):
#   public IP        153.68.20.238        (home / ASUS egress)
#   tailscale ASUS   100.64.0.6           (+ IPv6 fd7a:115c:a1e0::6)
#   tailscale Mac    100.64.0.3           (per tailnet map; NFS volume server)
#   home LAN         192.168.1.0/24       (covers both machines' LAN IPs)
# The Tailscale ULA /48 covers every tailnet node over IPv6; the LAN /24 covers
# both boxes' LAN addresses without pinning a single host.
_DEFAULT_WHITELIST = [
    "153.68.20.238/32",     # public IP (home / ASUS)
    "100.64.0.6/32",        # tailscale — ASUS (aaryn-tuf-rtx2060)
    "100.64.0.3/32",        # tailscale — Mac (volume server)
    "192.168.1.0/24",       # home LAN (both machines)
    "fd7a:115c:a1e0::/48",  # tailscale IPv6 (whole tailnet ULA)
    "127.0.0.1/32",         # localhost (local dev + smoke server)
    "::1/128",              # localhost IPv6
]


def _parse_networks(entries: list[str]) -> list:
    nets = []
    for e in entries:
        e = e.strip()
        if not e:
            continue
        try:
            nets.append(ipaddress.ip_network(e, strict=False))
        except ValueError:
            # A bare address (no /prefix) is fine too.
            try:
                nets.append(ipaddress.ip_network(f"{e}/32", strict=False))
            except ValueError:
                continue
    return nets


def whitelist_networks() -> list:
    """Effective whitelist. XCELSIOR_DEMO_IP_WHITELIST (comma-separated CIDRs or
    IPs) REPLACES the defaults when set; XCELSIOR_DEMO_IP_WHITELIST_EXTRA is
    always appended. Both support IPv4/IPv6 with optional /prefix."""
    override = os.environ.get("XCELSIOR_DEMO_IP_WHITELIST", "").strip()
    base = override.split(",") if override else list(_DEFAULT_WHITELIST)
    extra = os.environ.get("XCELSIOR_DEMO_IP_WHITELIST_EXTRA", "").strip()
    if extra:
        base = list(base) + extra.split(",")
    return _parse_networks(base)


def is_ip_whitelisted(ip: str | None) -> bool:
    if not ip:
        return False
    # Strip zone id (fe80::1%eth0) and any surrounding brackets.
    cleaned = ip.strip().strip("[]").split("%", 1)[0]
    try:
        addr = ipaddress.ip_address(cleaned)
    except ValueError:
        return False
    # Normalize IPv4-mapped IPv6 (::ffff:192.168.1.5) to plain IPv4.
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return any(addr in net for net in whitelist_networks())


def demo_enabled() -> bool:
    """Master kill switch. Default on; set XCELSIOR_DEMO_ENABLED=0 to disable."""
    return os.environ.get("XCELSIOR_DEMO_ENABLED", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
