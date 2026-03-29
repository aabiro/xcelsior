# Xcelsior — Bitcoin Deposit Support
# On-chain BTC deposits via local Bitcoin Core node RPC.
# Credits the existing CAD wallet at a locked BTC/CAD spot rate.

import json
import logging
import os
import sqlite3
import time
import urllib.request
import uuid
from contextlib import contextmanager

log = logging.getLogger("xcelsior")

# ── Configuration ─────────────────────────────────────────────────────

BTC_RPC_HOST = os.environ.get("XCELSIOR_BTC_RPC_HOST", "127.0.0.1")
BTC_RPC_PORT = int(os.environ.get("XCELSIOR_BTC_RPC_PORT", "8332"))
BTC_RPC_USER = os.environ.get("XCELSIOR_BTC_RPC_USER", "")
BTC_RPC_PASS = os.environ.get("XCELSIOR_BTC_RPC_PASS", "")
BTC_CONFIRMATIONS = int(os.environ.get("XCELSIOR_BTC_CONFIRMATIONS", "3"))
BTC_DEPOSIT_EXPIRY = int(os.environ.get("XCELSIOR_BTC_DEPOSIT_EXPIRY", "1800"))  # 30 min
BTC_ENABLED = os.environ.get("XCELSIOR_BTC_ENABLED", "false").lower() == "true"
BTC_DB_PATH = os.environ.get("XCELSIOR_BTC_DB", os.path.join(os.path.dirname(__file__), "xcelsior_btc.db"))

_rate_cache: dict = {"rate": 0.0, "fetched_at": 0.0}
RATE_CACHE_TTL = 300  # 5 minutes


# ── Bitcoin Core RPC ──────────────────────────────────────────────────


def _rpc_call(method: str, params: list | None = None) -> dict:
    """Call Bitcoin Core JSON-RPC."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or [],
    }).encode()

    url = f"http://{BTC_RPC_HOST}:{BTC_RPC_PORT}"
    import base64
    auth = base64.b64encode(f"{BTC_RPC_USER}:{BTC_RPC_PASS}".encode()).decode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {auth}",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read())
        if result.get("error"):
            raise RuntimeError(f"Bitcoin RPC error: {result['error']}")
        return result["result"]


def get_new_address(label: str = "xcelsior") -> str:
    """Generate a new receiving address from the node wallet."""
    return _rpc_call("getnewaddress", [label, "bech32"])


def get_received_by_address(address: str, min_conf: int = 0) -> float:
    """Get total BTC received by an address."""
    return float(_rpc_call("getreceivedbyaddress", [address, min_conf]))


def get_transaction(txid: str) -> dict:
    """Get transaction details."""
    return _rpc_call("gettransaction", [txid])


def list_received_by_address(min_conf: int = 0, include_empty: bool = True) -> list:
    """List all receiving addresses and amounts."""
    return _rpc_call("listreceivedbyaddress", [min_conf, include_empty])


# ── BTC/CAD Rate ──────────────────────────────────────────────────────


def get_btc_cad_rate() -> float:
    """Fetch current BTC/CAD rate from CoinGecko. Cached for 5 minutes."""
    now = time.time()
    if _rate_cache["rate"] > 0 and (now - _rate_cache["fetched_at"]) < RATE_CACHE_TTL:
        return _rate_cache["rate"]

    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=cad"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            rate = float(data["bitcoin"]["cad"])
            _rate_cache["rate"] = rate
            _rate_cache["fetched_at"] = now
            log.info("BTC/CAD rate: $%.2f", rate)
            return rate
    except Exception as e:
        log.error("Failed to fetch BTC/CAD rate: %s", e)
        if _rate_cache["rate"] > 0:
            return _rate_cache["rate"]
        raise RuntimeError("Unable to fetch BTC/CAD rate") from e


# ── SQLite Store ──────────────────────────────────────────────────────


@contextmanager
def _conn():
    c = sqlite3.connect(BTC_DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def _ensure_tables():
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS crypto_deposits (
                deposit_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                btc_address TEXT NOT NULL,
                amount_btc REAL NOT NULL,
                amount_cad REAL NOT NULL,
                btc_cad_rate REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                confirmations INTEGER DEFAULT 0,
                txid TEXT DEFAULT '',
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                confirmed_at REAL DEFAULT 0,
                credited_at REAL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_crypto_status
                ON crypto_deposits(status);
            CREATE INDEX IF NOT EXISTS idx_crypto_customer
                ON crypto_deposits(customer_id);
            CREATE INDEX IF NOT EXISTS idx_crypto_address
                ON crypto_deposits(btc_address);
        """)


_tables_ready = False


def _init():
    global _tables_ready
    if not _tables_ready:
        _ensure_tables()
        _tables_ready = True


# ── Deposit API ───────────────────────────────────────────────────────


def create_deposit(customer_id: str, amount_cad: float) -> dict:
    """Create a new BTC deposit request.

    Generates a unique address, locks the BTC/CAD rate for 30 minutes.
    """
    _init()
    rate = get_btc_cad_rate()
    amount_btc = round(amount_cad / rate, 8)
    address = get_new_address(f"xcelsior-{customer_id[:8]}")
    now = time.time()
    deposit_id = f"btc-{uuid.uuid4().hex[:12]}"

    with _conn() as c:
        c.execute(
            """INSERT INTO crypto_deposits
               (deposit_id, customer_id, btc_address, amount_btc, amount_cad,
                btc_cad_rate, status, confirmations, txid, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, '', ?, ?)""",
            (deposit_id, customer_id, address, amount_btc, amount_cad,
             rate, now, now + BTC_DEPOSIT_EXPIRY),
        )

    log.info(
        "BTC deposit created: %s addr=%s amount=%.8f BTC ($%.2f CAD @ %.2f)",
        deposit_id, address, amount_btc, amount_cad, rate,
    )

    return {
        "deposit_id": deposit_id,
        "btc_address": address,
        "amount_btc": amount_btc,
        "amount_cad": amount_cad,
        "btc_cad_rate": rate,
        "expires_at": now + BTC_DEPOSIT_EXPIRY,
        "qr_data": f"bitcoin:{address}?amount={amount_btc}",
    }


def get_deposit(deposit_id: str) -> dict | None:
    """Get deposit status."""
    _init()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM crypto_deposits WHERE deposit_id = ?",
            (deposit_id,),
        ).fetchone()
        return dict(row) if row else None


def get_deposits_by_customer(customer_id: str, limit: int = 20) -> list[dict]:
    """Get deposit history for a customer."""
    _init()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM crypto_deposits WHERE customer_id = ? ORDER BY created_at DESC LIMIT ?",
            (customer_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def refresh_deposit(deposit_id: str) -> dict | None:
    """Refresh an expired deposit with a new rate. Reuses the same address."""
    _init()
    dep = get_deposit(deposit_id)
    if not dep:
        return None
    if dep["status"] not in ("pending", "expired"):
        return dep  # already confirmed/credited

    rate = get_btc_cad_rate()
    amount_btc = round(dep["amount_cad"] / rate, 8)
    now = time.time()

    with _conn() as c:
        c.execute(
            """UPDATE crypto_deposits
               SET btc_cad_rate = ?, amount_btc = ?, status = 'pending',
                   expires_at = ?, created_at = ?
               WHERE deposit_id = ?""",
            (rate, amount_btc, now + BTC_DEPOSIT_EXPIRY, now, deposit_id),
        )

    dep.update({
        "btc_cad_rate": rate,
        "amount_btc": amount_btc,
        "status": "pending",
        "expires_at": now + BTC_DEPOSIT_EXPIRY,
        "qr_data": f"bitcoin:{dep['btc_address']}?amount={amount_btc}",
    })
    log.info("BTC deposit refreshed: %s new_rate=%.2f amount=%.8f BTC", deposit_id, rate, amount_btc)
    return dep


def get_pending_deposits() -> list[dict]:
    """Get all deposits that need confirmation checking."""
    _init()
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM crypto_deposits WHERE status IN ('pending', 'confirming')",
        ).fetchall()
        return [dict(r) for r in rows]


def check_and_update_deposit(deposit: dict) -> dict:
    """Check a single deposit against the Bitcoin node and update status.

    Returns the updated deposit dict.
    """
    _init()
    deposit_id = deposit["deposit_id"]
    address = deposit["btc_address"]
    now = time.time()

    # Expire if no confirmations and past expiry
    if deposit["status"] == "pending" and now > deposit["expires_at"]:
        received = get_received_by_address(address, 0)
        if received < deposit["amount_btc"] * 0.99:  # 1% tolerance
            with _conn() as c:
                c.execute(
                    "UPDATE crypto_deposits SET status = 'expired' WHERE deposit_id = ?",
                    (deposit_id,),
                )
            deposit["status"] = "expired"
            log.info("BTC deposit expired: %s", deposit_id)
            return deposit

    # Check received amount (0-conf)
    received_0 = get_received_by_address(address, 0)
    if received_0 < deposit["amount_btc"] * 0.99:
        return deposit  # nothing received yet

    # Check confirmed amount
    received_confirmed = get_received_by_address(address, BTC_CONFIRMATIONS)
    if received_confirmed >= deposit["amount_btc"] * 0.99:
        confirmations = BTC_CONFIRMATIONS
        new_status = "confirmed"
    else:
        # Partially confirmed — check 1-conf
        received_1 = get_received_by_address(address, 1)
        if received_1 >= deposit["amount_btc"] * 0.99:
            confirmations = 1  # at least 1 but less than required
            new_status = "confirming"
        else:
            confirmations = 0
            new_status = "confirming"  # in mempool

    confirmed_at = now if new_status == "confirmed" else 0

    with _conn() as c:
        c.execute(
            """UPDATE crypto_deposits
               SET status = ?, confirmations = ?, confirmed_at = ?
               WHERE deposit_id = ? AND status IN ('pending', 'confirming')""",
            (new_status, confirmations, confirmed_at, deposit_id),
        )

    deposit["status"] = new_status
    deposit["confirmations"] = confirmations
    deposit["confirmed_at"] = confirmed_at

    log.info(
        "BTC deposit %s: status=%s confirmations=%d",
        deposit_id, new_status, confirmations,
    )
    return deposit


def mark_credited(deposit_id: str) -> None:
    """Mark a confirmed deposit as credited to the wallet."""
    _init()
    with _conn() as c:
        c.execute(
            """UPDATE crypto_deposits
               SET status = 'credited', credited_at = ?
               WHERE deposit_id = ? AND status = 'confirmed'""",
            (time.time(), deposit_id),
        )
