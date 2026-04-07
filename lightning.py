# Xcelsior — Lightning Network Deposit Support
# Instant BTC deposits via Core Lightning (CLN) node.
# Credits the existing CAD wallet at a locked BTC/CAD spot rate.
# Uses the CLN clnrest HTTP API over Tailscale.

import json
import logging
import os
import ssl
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager

log = logging.getLogger("xcelsior")

# ── Configuration ─────────────────────────────────────────────────────

LN_ENABLED = os.environ.get("XCELSIOR_LN_ENABLED", "false").lower() == "true"
LN_CLNREST_URL = os.environ.get("XCELSIOR_LN_CLNREST_URL", "https://127.0.0.1:3010")
LN_RUNE = os.environ.get("XCELSIOR_LN_RUNE", "")
LN_INVOICE_EXPIRY = int(os.environ.get("XCELSIOR_LN_INVOICE_EXPIRY", "600"))  # 10 min
LN_DB_PATH = os.environ.get("XCELSIOR_LN_DB_PATH", "xcelsior_ln.db")
LN_MIN_CAD = float(os.environ.get("XCELSIOR_LN_MIN_CAD", "1"))
LN_MAX_CAD = float(os.environ.get("XCELSIOR_LN_MAX_CAD", "1000"))

# SSL context for clnrest (self-signed certificate)
_ssl_ctx = ssl.create_default_context()
_ssl_ctx.check_hostname = False
_ssl_ctx.verify_mode = ssl.CERT_NONE


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


def create_invoice(amount_msat: int, label: str, description: str, expiry: int | None = None) -> dict:
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
    return _rpc_call("invoice", {
        "amount_msat": amount_msat,
        "label": label,
        "description": description,
        "expiry": expiry,
    })


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


# ── Database ──────────────────────────────────────────────────────────


def _get_ln_db_path() -> str:
    return os.environ.get("XCELSIOR_LN_DB_PATH", LN_DB_PATH)


@contextmanager
def _ln_conn():
    path = _get_ln_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_ln_tables():
    with _ln_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ln_deposits (
                deposit_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                label TEXT NOT NULL UNIQUE,
                bolt11 TEXT NOT NULL,
                payment_hash TEXT NOT NULL,
                amount_msat INTEGER NOT NULL,
                amount_sats INTEGER NOT NULL,
                amount_btc REAL NOT NULL,
                amount_cad REAL NOT NULL,
                btc_cad_rate REAL NOT NULL,
                status TEXT DEFAULT 'pending',
                payment_preimage TEXT DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                paid_at REAL DEFAULT 0,
                credited_at REAL DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ln_deposits_customer ON ln_deposits(customer_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ln_deposits_label ON ln_deposits(label)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ln_deposits_status ON ln_deposits(status)"
        )


_tables_ensured = False


def _ensure_tables_once():
    global _tables_ensured
    if not _tables_ensured:
        _ensure_ln_tables()
        _tables_ensured = True


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
        result["reason"] = f"Lightning node unavailable: {e}"

    return result


def create_deposit(customer_id: str, amount_cad: float) -> dict:
    """Create a Lightning invoice for a CAD deposit.

    Returns:
        dict with deposit_id, bolt11, amount_sats, amount_cad, btc_cad_rate,
        expires_at, payment_hash.
    """
    _ensure_tables_once()

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

    with _ln_conn() as conn:
        conn.execute(
            """INSERT INTO ln_deposits
               (deposit_id, customer_id, label, bolt11, payment_hash,
                amount_msat, amount_sats, amount_btc, amount_cad, btc_cad_rate,
                status, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (deposit_id, customer_id, label, invoice["bolt11"],
             invoice["payment_hash"], amount_msat, amount_sats, amount_btc,
             amount_cad, rate, now, invoice["expires_at"]),
        )

    log.info(
        "LN DEPOSIT CREATED: %s | %s | %d sats ($%.2f CAD) @ $%.0f",
        deposit_id, customer_id, amount_sats, amount_cad, rate,
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
    _ensure_tables_once()

    with _ln_conn() as conn:
        row = conn.execute(
            "SELECT * FROM ln_deposits WHERE deposit_id = ?", (deposit_id,)
        ).fetchone()
        if not row:
            return None
        dep = dict(row)

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
        with _ln_conn() as conn:
            conn.execute(
                """UPDATE ln_deposits
                   SET status = 'paid',
                       payment_preimage = ?,
                       paid_at = ?
                   WHERE deposit_id = ? AND status IN ('pending')""",
                (inv.get("payment_preimage", ""), inv.get("paid_at", now), deposit_id),
            )
        dep["status"] = "paid"
        dep["payment_preimage"] = inv.get("payment_preimage", "")
        dep["paid_at"] = inv.get("paid_at", now)
        log.info("LN DEPOSIT PAID: %s | %d sats", deposit_id, dep["amount_sats"])

    elif cln_status == "expired" and dep["status"] == "pending":
        with _ln_conn() as conn:
            conn.execute(
                "UPDATE ln_deposits SET status = 'expired' WHERE deposit_id = ? AND status = 'pending'",
                (deposit_id,),
            )
        dep["status"] = "expired"

    return dep


def mark_credited(deposit_id: str) -> bool:
    """Mark a paid deposit as credited (wallet balance updated)."""
    _ensure_tables_once()
    with _ln_conn() as conn:
        result = conn.execute(
            """UPDATE ln_deposits
               SET status = 'credited', credited_at = ?
               WHERE deposit_id = ? AND status = 'paid'""",
            (time.time(), deposit_id),
        )
        return result.rowcount > 0


def get_pending_deposits() -> list[dict]:
    """Get all pending (unpaid) Lightning deposits for confirmation checking."""
    _ensure_tables_once()
    with _ln_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ln_deposits WHERE status = 'pending'"
        ).fetchall()
        return [dict(r) for r in rows]


def get_paid_uncredited() -> list[dict]:
    """Get all paid but not yet credited deposits."""
    _ensure_tables_once()
    with _ln_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ln_deposits WHERE status = 'paid'"
        ).fetchall()
        return [dict(r) for r in rows]


def get_customer_deposits(customer_id: str, limit: int = 20) -> list[dict]:
    """Get recent Lightning deposits for a customer."""
    _ensure_tables_once()
    with _ln_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM ln_deposits WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
            (customer_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Background Watcher ────────────────────────────────────────────────


def process_ln_deposits(credit_callback=None):
    """Check all pending deposits and credit paid ones.

    Args:
        credit_callback: Optional fn(customer_id, amount_cad, deposit_id) to credit wallet.
    """
    if not LN_ENABLED:
        return

    # Check pending invoices for payment
    for dep in get_pending_deposits():
        try:
            check_deposit(dep["deposit_id"])
        except Exception as e:
            log.debug("LN deposit check failed for %s: %s", dep["deposit_id"], e)

    # Credit paid deposits
    for dep in get_paid_uncredited():
        try:
            if credit_callback:
                credit_callback(dep["customer_id"], dep["amount_cad"], dep["deposit_id"])
            mark_credited(dep["deposit_id"])
            log.info(
                "LN WALLET CREDITED: %s +$%.2f CAD (%s)",
                dep["customer_id"], dep["amount_cad"], dep["deposit_id"],
            )
        except Exception as e:
            log.error("LN credit failed for %s: %s", dep["deposit_id"], e)


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
