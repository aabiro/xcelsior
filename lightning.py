# Xcelsior — Lightning Network Deposit Support
# Instant BTC deposits via Core Lightning (CLN) node.
# Credits the existing CAD wallet at a locked BTC/CAD spot rate.
# Uses the CLN clnrest HTTP API over Headscale.

import json
import logging
import os
import ssl
from decimal import Decimal
from psycopg.rows import dict_row
from db import pg_transaction, pg_connection
from money import cad_to_minor, minor_to_cad
import time
import urllib.error
import urllib.request
import uuid

log = logging.getLogger("xcelsior")

# ── Configuration ─────────────────────────────────────────────────────

LN_ENABLED = os.environ.get("XCELSIOR_LN_ENABLED", "false").lower() == "true"
LN_CLNREST_URL = os.environ.get("XCELSIOR_LN_CLNREST_URL", "https://127.0.0.1:3010")
LN_RUNE = os.environ.get("XCELSIOR_LN_RUNE", "")
LN_INVOICE_EXPIRY = int(os.environ.get("XCELSIOR_LN_INVOICE_EXPIRY", "600"))  # 10 min
LN_MIN_CAD = float(os.environ.get("XCELSIOR_LN_MIN_CAD", "1"))
LN_MAX_CAD = float(os.environ.get("XCELSIOR_LN_MAX_CAD", "1000"))

# Idempotency namespace for wallet credits. One deposit credits a wallet
# exactly once, no matter how many times the watcher re-observes it
# (companion §10.1: "provider polling/webhook events idempotent under
# duplicates and reordering").
LN_CREDIT_IDEMPOTENCY_PREFIX = "ln-deposit:"


def credit_idempotency_key(deposit_id: str) -> str:
    """Ledger idempotency key for one deposit's wallet credit.

    Every caller of ``process_ln_deposits`` must route its credit through
    this key. ``BillingEngine.deposit`` deduplicates on it, so a retry
    after a failed ``mark_credited`` re-posts nothing.
    """
    return f"{LN_CREDIT_IDEMPOTENCY_PREFIX}{deposit_id}"


# SSL context for clnrest. `create_default_context()` verifies the
# certificate chain and hostname by default; do not replace it with an
# unverified context (companion §2.7 — a database transaction cannot
# compensate for an unauthenticated payment endpoint).
_ssl_ctx = ssl.create_default_context()
if os.environ.get("XCELSIOR_LN_CA_CERT"):
    _ssl_ctx.load_verify_locations(os.environ.get("XCELSIOR_LN_CA_CERT"))


# ── Row helpers ───────────────────────────────────────────────────────
# The shared pool hands out `tuple_row` connections, and mutating
# `conn.row_factory` leaks dict rows to the next pool borrower. A cursor
# scoped to `dict_row` keeps the mapping local to one query.


def _fetch_one(conn, sql: str, params: tuple = ()) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        return cur.execute(sql, params).fetchone()


def _fetch_all(conn, sql: str, params: tuple = ()) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        return list(cur.execute(sql, params).fetchall())


# ── CLN clnrest HTTP API ──────────────────────────────────────────────


def _rpc_call(method: str, params: dict | list | None = None, timeout: float = 10.0) -> dict:
    """Call Core Lightning via the clnrest HTTP API."""
    url = f"{LN_CLNREST_URL}/v1/{method}"
    data = json.dumps(params or {}).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Rune": LN_RUNE,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        try:
            err = json.loads(body)
            raise RuntimeError(
                f"CLN REST error ({err.get('code', '?')}): {err.get('message', body)}"
            ) from e
        except (json.JSONDecodeError, KeyError):
            raise RuntimeError(f"CLN REST error ({e.code}): {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"CLN REST connection failed: {e.reason}") from e


def get_node_info() -> dict:
    """Get Lightning node info — alias, pubkey, channels, block height."""
    return _rpc_call("getinfo")


def create_invoice(
    amount_msat: int, label: str, description: str, expiry: int | None = None
) -> dict:
    """Create a BOLT11 invoice.

    Args:
        amount_msat: Amount in millisatoshis.
        label: Unique label for this invoice.
        description: Human-readable description.
        expiry: Seconds until invoice expires (default: LN_INVOICE_EXPIRY).

    Returns:
        dict with bolt11, payment_hash, payment_secret, expires_at.
    """
    expiry = expiry or LN_INVOICE_EXPIRY
    return _rpc_call(
        "invoice",
        {
            "amount_msat": amount_msat,
            "label": label,
            "description": description,
            "expiry": expiry,
        },
    )


def check_invoice(label: str) -> dict | None:
    """Check the status of an invoice by label.

    Returns:
        dict with status ('unpaid', 'paid', 'expired'), amount_msat,
        amount_received_msat, paid_at, payment_preimage, etc.
        None if invoice not found.
    """
    try:
        result = _rpc_call("listinvoices", {"label": label})
        invoices = result.get("invoices", [])
        if not invoices:
            return None
        return invoices[0]
    except RuntimeError:
        return None


def wait_invoice(label: str, timeout: float = 30.0) -> dict | None:
    """Block until an invoice is paid or timeout. Non-blocking alternative to polling."""
    try:
        return _rpc_call("waitinvoice", {"label": label}, timeout=timeout)
    except (RuntimeError, TimeoutError):
        return None


# ── BTC/CAD Rate (reuse from bitcoin.py) ──────────────────────────────

_ln_rate_cache: dict = {"rate": 0.0, "fetched_at": 0.0}
LN_RATE_CACHE_TTL = 300  # 5 min


def get_btc_cad_rate() -> float:
    """Fetch BTC/CAD spot rate from CoinGecko (cached 5 min)."""
    now = time.time()
    if _ln_rate_cache["rate"] > 0 and (now - _ln_rate_cache["fetched_at"]) < LN_RATE_CACHE_TTL:
        return _ln_rate_cache["rate"]

    import urllib.request

    try:
        req = urllib.request.Request(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=cad",
            headers={"Accept": "application/json", "User-Agent": "Xcelsior/1.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            rate = float(data["bitcoin"]["cad"])
            _ln_rate_cache["rate"] = rate
            _ln_rate_cache["fetched_at"] = now
            return rate
    except Exception as e:
        if _ln_rate_cache["rate"] > 0:
            return _ln_rate_cache["rate"]
        raise RuntimeError(f"Unable to fetch BTC/CAD rate: {e}") from e




# ── Deposit Lifecycle ─────────────────────────────────────────────────


def get_service_status() -> dict:
    """Check if Lightning service is available."""
    result = {
        "enabled": LN_ENABLED,
        "available": False,
        "reason": "",
        "node_alias": "",
        "node_id": "",
        "num_active_channels": 0,
        "blockheight": 0,
        "network": "",
    }

    if not LN_ENABLED:
        result["reason"] = "Lightning deposits are not enabled"
        return result

    if not LN_RUNE:
        result["reason"] = "Lightning rune not configured"
        return result

    try:
        info = get_node_info()
        result["node_alias"] = info.get("alias", "")
        result["node_id"] = info.get("id", "")
        result["num_active_channels"] = info.get("num_active_channels", 0)
        result["blockheight"] = info.get("blockheight", 0)
        result["network"] = info.get("network", "")
        result["available"] = True
    except Exception as e:
        msg = str(e).lower()
        if any(
            token in msg
            for token in (
                "connection refused",
                "connection failed",
                "timed out",
                "timeout",
                "urlopen error",
            )
        ):
            result["reason"] = "Lightning node is starting (waiting for Bitcoin node)"
        else:
            result["reason"] = f"Lightning node unavailable: {e}"

    return result


def create_deposit(customer_id: str, amount_cad: float) -> dict:
    """Create a Lightning invoice for a CAD deposit.

    Returns:
        dict with deposit_id, bolt11, amount_sats, amount_cad, btc_cad_rate,
        expires_at, payment_hash.
    """

    if not LN_ENABLED:
        raise RuntimeError("Lightning deposits are not enabled")

    if amount_cad < LN_MIN_CAD or amount_cad > LN_MAX_CAD:
        raise ValueError(f"Amount must be between ${LN_MIN_CAD} and ${LN_MAX_CAD} CAD")

    rate = get_btc_cad_rate()
    amount_btc = round(amount_cad / rate, 8)
    amount_sats = int(amount_btc * 1e8)
    amount_msat = amount_sats * 1000

    if amount_sats < 1:
        raise ValueError("Amount too small for Lightning payment")

    deposit_id = f"ln-{uuid.uuid4().hex[:12]}"
    label = f"xcelsior-{deposit_id}"
    description = f"Xcelsior deposit ${amount_cad:.2f} CAD"
    now = time.time()

    invoice = create_invoice(amount_msat, label, description)

    # Exact ledger values. The legacy float columns are written alongside
    # only so an un-upgraded replica can still read the row; they are
    # dropped at contract phase (Track B B16.2) and nothing derives money
    # from them.
    amount_cad_minor = cad_to_minor(amount_cad)
    rate_exact = Decimal(str(rate))

    with pg_transaction() as conn:
        conn.execute(
            """INSERT INTO ln_deposits
               (deposit_id, customer_id, tenant_id, currency,
                label, bolt11, payment_hash,
                amount_msat, amount_sats, amount_btc, amount_cad, btc_cad_rate,
                amount_cad_minor, btc_cad_rate_exact,
                status, created_at, expires_at, created_at_ts, expires_at_ts)
               VALUES (%s, %s, %s, 'CAD', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                       'pending', %s, %s, to_timestamp(%s), to_timestamp(%s))""",
            (
                deposit_id,
                customer_id,
                # Transitional single-user tenancy, matching migration 054's
                # rule for jobs: billing already treats the paying customer
                # as the tenant until real workspaces land.
                customer_id,
                label,
                invoice["bolt11"],
                invoice["payment_hash"],
                amount_msat,
                amount_sats,
                amount_btc,
                amount_cad,
                rate,
                amount_cad_minor,
                rate_exact,
                now,
                invoice["expires_at"],
                now,
                invoice["expires_at"],
            ),
        )

    log.info(
        "LN DEPOSIT CREATED: %s | %s | %d sats ($%.2f CAD) @ $%.0f",
        deposit_id,
        customer_id,
        amount_sats,
        amount_cad,
        rate,
    )

    return {
        "deposit_id": deposit_id,
        "bolt11": invoice["bolt11"],
        "payment_hash": invoice["payment_hash"],
        "amount_sats": amount_sats,
        "amount_btc": amount_btc,
        "amount_cad": amount_cad,
        "btc_cad_rate": rate,
        "expires_at": invoice["expires_at"],
    }


def check_deposit(deposit_id: str) -> dict | None:
    """Check status of a Lightning deposit. Updates DB if newly paid."""

    with pg_transaction() as conn:
        dep = _fetch_one(
            conn, "SELECT * FROM ln_deposits WHERE deposit_id = %s", (deposit_id,)
        )
        if not dep:
            return None

    # Already terminal
    if dep["status"] in ("credited", "expired"):
        return dep

    # Check CLN invoice status
    inv = check_invoice(dep["label"])
    if not inv:
        return dep

    cln_status = inv.get("status", "unpaid")
    now = time.time()

    if cln_status == "paid" and dep["status"] != "paid":
        with pg_transaction() as conn:
            conn.execute(
                """UPDATE ln_deposits
                   SET status = 'paid',
                       payment_preimage = %s,
                       paid_at = %s,
                       paid_at_ts = to_timestamp(%s)
                   WHERE deposit_id = %s AND status IN ('pending')""",
                (
                    inv.get("payment_preimage", ""),
                    inv.get("paid_at", now),
                    inv.get("paid_at", now),
                    deposit_id,
                ),
            )
        dep["status"] = "paid"
        dep["payment_preimage"] = inv.get("payment_preimage", "")
        dep["paid_at"] = inv.get("paid_at", now)
        log.info("LN DEPOSIT PAID: %s | %d sats", deposit_id, dep["amount_sats"])

    elif cln_status == "expired" and dep["status"] == "pending":
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE ln_deposits SET status = 'expired' WHERE deposit_id = %s AND status = 'pending'",
                (deposit_id,),
            )
        dep["status"] = "expired"

    return dep


def mark_credited(deposit_id: str, wallet_ledger_entry_id: str | None = None) -> bool:
    """Mark a paid deposit as credited (wallet balance updated).

    Compare-and-swap on ``status = 'paid'`` so only one caller can make
    the transition; a duplicate call returns False rather than re-marking.

    ``wallet_ledger_entry_id`` links the deposit to the exact ledger row
    that credited it (companion §10.1). With it, "credited but no ledger
    entry" and "ledger entry with no deposit" are both single queries
    rather than a manual audit.
    """
    with pg_transaction() as conn:
        now = time.time()
        result = conn.execute(
            """UPDATE ln_deposits
               SET status = 'credited',
                   credited_at = %s,
                   credited_at_ts = to_timestamp(%s),
                   wallet_ledger_entry_id = COALESCE(%s, wallet_ledger_entry_id)
               WHERE deposit_id = %s AND status = 'paid'""",
            (now, now, wallet_ledger_entry_id, deposit_id),
        )
        return result.rowcount > 0


def get_pending_deposits() -> list[dict]:
    """Get all pending (unpaid) Lightning deposits for confirmation checking."""
    with pg_transaction() as conn:
        return _fetch_all(conn, "SELECT * FROM ln_deposits WHERE status = 'pending'")


def get_paid_uncredited() -> list[dict]:
    """Get all paid but not yet credited deposits."""
    with pg_transaction() as conn:
        return _fetch_all(conn, "SELECT * FROM ln_deposits WHERE status = 'paid'")


def get_customer_deposits(customer_id: str, limit: int = 20) -> list[dict]:
    """Get recent Lightning deposits for a customer."""
    with pg_transaction() as conn:
        return _fetch_all(
            conn,
            "SELECT * FROM ln_deposits WHERE customer_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (customer_id, limit),
        )


# ── Background Watcher ────────────────────────────────────────────────


def process_ln_deposits(credit_callback=None):
    """Check all pending deposits and credit paid ones.

    Args:
        credit_callback: fn(customer_id, amount_cad, deposit_id) that credits
            the wallet. It **must** deduplicate on
            ``credit_idempotency_key(deposit_id)`` — see the ordering note
            below.

    Ordering and safety: credit first, then mark credited. The reverse
    order would lose a customer's money if the credit failed after the
    mark. Because the credit is idempotent on the deposit id and
    ``mark_credited`` is a compare-and-swap on ``status = 'paid'``, a crash
    or failure anywhere in this sequence replays safely: the retry re-posts
    nothing and completes the mark.

    Each deposit is isolated. A single malformed row must not stop the
    sweep for every other customer — that is how the pre-2026-07-22 version
    failed closed on the whole fleet.
    """
    if not LN_ENABLED:
        return

    # Check pending invoices for payment
    try:
        pending = get_pending_deposits()
    except Exception as e:
        log.error("LN pending-deposit query failed: %s", e)
        pending = []
    for dep in pending:
        try:
            check_deposit(dep["deposit_id"])
        except Exception as e:
            log.debug("LN deposit check failed for %s: %s", dep.get("deposit_id"), e)

    # Credit paid deposits
    try:
        paid = get_paid_uncredited()
    except Exception as e:
        log.error("LN paid-uncredited query failed: %s", e)
        return
    for dep in paid:
        deposit_id = str(dep.get("deposit_id") or "")
        if not deposit_id:
            log.error("LN paid deposit row has no deposit_id; skipping: %r", dep)
            continue
        # Credit from the exact integer-cent column, never the legacy
        # float: a stored 25.0 can be 24.999999999999996, and that error
        # would be posted to the wallet ledger verbatim.
        minor = dep.get("amount_cad_minor")
        amount_cad = (
            minor_to_cad(minor) if minor is not None else float(dep["amount_cad"])
        )
        try:
            ledger_entry_id = None
            if credit_callback:
                # A callback that returns the ledger row it wrote lets the
                # deposit point at it; one that returns None still works.
                result = credit_callback(dep["customer_id"], amount_cad, deposit_id)
                if isinstance(result, dict):
                    ledger_entry_id = result.get("tx_id")
            if mark_credited(deposit_id, wallet_ledger_entry_id=ledger_entry_id):
                log.info(
                    "LN WALLET CREDITED: %s +$%.2f CAD (%s)",
                    dep["customer_id"],
                    amount_cad,
                    deposit_id,
                )
        except Exception as e:
            # Left as 'paid' on purpose: the next sweep retries, and the
            # idempotency key stops the retry from crediting twice.
            log.error("LN credit failed for %s: %s", deposit_id, e)


def start_ln_watcher(interval: int = 5, credit_callback=None):
    """Start background thread to poll Lightning deposits.

    Lightning payments are instant so we poll frequently (every 5s).
    """
    import threading

    def _loop():
        while True:
            try:
                process_ln_deposits(credit_callback=credit_callback)
            except Exception as e:
                log.debug("LN watcher error: %s", e)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True, name="ln-watcher")
    t.start()
    log.info("Lightning deposit watcher started (interval=%ds)", interval)
    return t
