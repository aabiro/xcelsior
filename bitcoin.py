# Xcelsior — Bitcoin Deposit Support
# On-chain BTC deposits via local Bitcoin Core node RPC.
# Credits the existing CAD wallet at a locked BTC/CAD spot rate.

import json
import logging
import os
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager

log = logging.getLogger("xcelsior")

# ── Configuration ─────────────────────────────────────────────────────

BTC_RPC_HOST = os.environ.get("XCELSIOR_BTC_RPC_HOST", "127.0.0.1")
BTC_RPC_PORT = int(os.environ.get("XCELSIOR_BTC_RPC_PORT", "8332"))
BTC_RPC_USER = os.environ.get("XCELSIOR_BTC_RPC_USER", "")
BTC_RPC_PASS = os.environ.get("XCELSIOR_BTC_RPC_PASS", "")
BTC_RPC_WALLET = os.environ.get("XCELSIOR_BTC_RPC_WALLET", "").strip()
BTC_AUTO_WALLET = os.environ.get("XCELSIOR_BTC_AUTO_WALLET", "xcelsior").strip() or "xcelsior"
BTC_CONFIRMATIONS = int(os.environ.get("XCELSIOR_BTC_CONFIRMATIONS", "3"))
BTC_DEPOSIT_EXPIRY = int(os.environ.get("XCELSIOR_BTC_DEPOSIT_EXPIRY", "1800"))  # 30 min
BTC_ENABLED = os.environ.get("XCELSIOR_BTC_ENABLED", "false").lower() == "true"
BTC_RPC_TIMEOUT = float(os.environ.get("XCELSIOR_BTC_RPC_TIMEOUT", "3"))
BTC_STATUS_RPC_TIMEOUT = float(os.environ.get("XCELSIOR_BTC_STATUS_RPC_TIMEOUT", "1.5"))
BTC_RPC_FALLBACK_HOSTS = tuple(
    host.strip()
    for host in os.environ.get("XCELSIOR_BTC_RPC_FALLBACK_HOSTS", "").split(",")
    if host.strip()
)

_rate_cache: dict = {"rate": 0.0, "fetched_at": 0.0}
_active_wallet_name: str | None = BTC_RPC_WALLET or None
_resolved_rpc_host: str | None = None
RATE_CACHE_TTL = 300  # 5 minutes


# ── Bitcoin Core RPC ──────────────────────────────────────────────────


def _running_in_docker() -> bool:
    return os.path.exists("/.dockerenv")


def _candidate_rpc_hosts() -> list[str]:
    candidates: list[str] = []

    if _resolved_rpc_host:
        candidates.append(_resolved_rpc_host)

    configured_host = BTC_RPC_HOST.strip()
    if configured_host:
        candidates.append(configured_host)

    candidates.extend(BTC_RPC_FALLBACK_HOSTS)

    if configured_host not in {"", "127.0.0.1", "localhost"}:
        if _running_in_docker():
            candidates.extend(["host.docker.internal", "172.17.0.1"])
        candidates.extend(["127.0.0.1", "localhost"])

    unique_hosts: list[str] = []
    seen: set[str] = set()
    for host in candidates:
        if host in seen:
            continue
        unique_hosts.append(host)
        seen.add(host)
    return unique_hosts


def _is_transport_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        if isinstance(reason, (TimeoutError, OSError)):
            return True
        return True
    return isinstance(exc, OSError)


def _rpc_call_host(
    host: str,
    method: str,
    params: list | None = None,
    wallet: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Call a specific Bitcoin Core JSON-RPC host."""
    payload = json.dumps({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or [],
    }).encode()

    url = f"http://{host}:{BTC_RPC_PORT}"
    if wallet:
        quoted = urllib.parse.quote(wallet, safe="")
        url = f"{url}/wallet/{quoted}"
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
    with urllib.request.urlopen(req, timeout=timeout or BTC_RPC_TIMEOUT) as resp:
        result = json.loads(resp.read())
        if result.get("error"):
            raise RuntimeError(f"Bitcoin RPC error: {result['error']}")
        return result["result"]


def _rpc_call(
    method: str,
    params: list | None = None,
    wallet: str | None = None,
    timeout: float | None = None,
    skip_fallback: bool = False,
) -> dict:
    """Call Bitcoin Core JSON-RPC, falling back across likely local endpoints."""
    global _resolved_rpc_host

    last_exc: Exception | None = None
    first_host: str | None = None

    if skip_fallback:
        # Only try the resolved host or the configured host — no fallback cascade.
        host = _resolved_rpc_host or BTC_RPC_HOST.strip()
        if not host:
            raise RuntimeError("Bitcoin RPC host is not configured")
        return _rpc_call_host(host, method, params, wallet=wallet, timeout=timeout)

    for host in _candidate_rpc_hosts():
        if first_host is None:
            first_host = host
        try:
            result = _rpc_call_host(host, method, params, wallet=wallet, timeout=timeout)
            if host != first_host:
                log.warning(
                    "BTC RPC fallback engaged: %s via %s after %s was unreachable",
                    method,
                    host,
                    first_host,
                )
            _resolved_rpc_host = host
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_transport_error(exc):
                raise

    if last_exc is None:
        raise RuntimeError("Bitcoin RPC host is not configured")
    raise last_exc


def _wallet_has_receiving_keys(info: dict) -> bool:
    """Return True when a wallet can hand out fresh receiving addresses."""
    if not info.get("private_keys_enabled", False):
        return False
    if info.get("blank", False):
        return False
    return int(info.get("keypoolsize", 0)) > 0 or int(info.get("keypoolsize_hd_internal", 0)) > 0


def _ensure_wallet_ready(
    wallet_name: str,
    timeout: float | None = None,
    skip_fallback: bool = False,
) -> None:
    """Load or create a dedicated receiving wallet, then verify it has keys."""
    try:
        loaded = _rpc_call("listwallets", timeout=timeout, skip_fallback=skip_fallback)
    except Exception:
        loaded = []

    if wallet_name not in loaded:
        try:
            _rpc_call("loadwallet", [wallet_name], timeout=timeout, skip_fallback=skip_fallback)
        except Exception:
            # If the wallet does not exist yet, create a fresh descriptor wallet.
            _rpc_call("createwallet", [wallet_name], timeout=timeout, skip_fallback=skip_fallback)

    info = _rpc_call("getwalletinfo", wallet=wallet_name, timeout=timeout, skip_fallback=skip_fallback)
    if _wallet_has_receiving_keys(info):
        return

    try:
        _rpc_call("keypoolrefill", wallet=wallet_name, timeout=timeout, skip_fallback=skip_fallback)
    except Exception:
        pass

    info = _rpc_call("getwalletinfo", wallet=wallet_name, timeout=timeout, skip_fallback=skip_fallback)
    if not _wallet_has_receiving_keys(info):
        raise RuntimeError(
            f"Bitcoin wallet '{wallet_name}' has no receiving keys available",
        )


def _should_provision_wallet(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "no available keys" in msg
        or "wallet file not specified" in msg
        or "requested wallet does not exist" in msg
        or "method not found" in msg
    )


def _wallet_rpc_call(method: str, params: list | None = None, timeout: float | None = None):
    """Call the active wallet, provisioning a dedicated wallet if the default one is unusable."""
    global _active_wallet_name

    wallet_name = _active_wallet_name if _active_wallet_name is not None else BTC_RPC_WALLET
    try:
        return _rpc_call(method, params, wallet=wallet_name or None, timeout=timeout)
    except Exception as exc:
        if not _should_provision_wallet(exc):
            raise

        fallback_wallet = BTC_RPC_WALLET or BTC_AUTO_WALLET
        _ensure_wallet_ready(fallback_wallet, timeout=timeout)
        _active_wallet_name = fallback_wallet
        log.warning(
            "BTC wallet fallback engaged: using '%s' after address generation failed on '%s' (%s)",
            fallback_wallet,
            wallet_name or "<default>",
            exc,
        )
        return _rpc_call(method, params, wallet=fallback_wallet, timeout=timeout)


def get_new_address(label: str = "xcelsior") -> str:
    """Generate a new receiving address from the node wallet."""
    return _wallet_rpc_call("getnewaddress", [label, "bech32"])


def get_received_by_address(address: str, min_conf: int = 0) -> float:
    """Get total BTC received by an address."""
    return float(_wallet_rpc_call("getreceivedbyaddress", [address, min_conf]))


def get_transaction(txid: str) -> dict:
    """Get transaction details."""
    return _wallet_rpc_call("gettransaction", [txid])


def list_received_by_address(min_conf: int = 0, include_empty: bool = True) -> list:
    """List all receiving addresses and amounts."""
    return _wallet_rpc_call("listreceivedbyaddress", [min_conf, include_empty])


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


def describe_service_error(exc: Exception | str) -> str:
    """Normalize low-level BTC service failures into concise user-facing text."""
    message = exc.strip() if isinstance(exc, str) else str(exc).strip()
    normalized = message.lower()

    if (
        "unable to fetch btc/cad rate" in normalized
        or "failed to fetch btc/cad rate" in normalized
    ):
        return "Bitcoin pricing service is currently unavailable"

    if any(
        token in normalized
        for token in (
            "connection refused",
            "timed out",
            "timeout",
            "failed to establish a new connection",
            "name or service not known",
            "temporary failure in name resolution",
            "urlopen error",
            "connection reset by peer",
        )
    ):
        return "Bitcoin node is offline or unavailable"

    return message or "Bitcoin service is currently unavailable"


def get_service_status() -> dict:
    """Report whether new BTC deposits can currently be created."""
    wallet_name = BTC_RPC_WALLET or BTC_AUTO_WALLET
    status = {
        "enabled": BTC_ENABLED,
        "available": False,
        "reason": "Bitcoin deposits are not enabled",
        "wallet_name": wallet_name,
        "rpc_reachable": False,
        "wallet_ready": False,
    }

    if not BTC_ENABLED:
        return status

    try:
        chain_info = _rpc_call(
            "getblockchaininfo",
            timeout=BTC_STATUS_RPC_TIMEOUT,
            skip_fallback=True,
        )
        status["rpc_reachable"] = True
        status["network"] = chain_info.get("chain")
        status["blocks"] = chain_info.get("blocks")
    except Exception as exc:
        status["reason"] = describe_service_error(exc)
        return status

    try:
        _ensure_wallet_ready(wallet_name, timeout=BTC_STATUS_RPC_TIMEOUT, skip_fallback=True)
        status["wallet_ready"] = True
    except Exception as exc:
        status["reason"] = describe_service_error(exc)
        return status

    status["available"] = True
    status["reason"] = "ok"
    return status


# ── PostgreSQL Store ───────────────────────────────────────────────────


def _ensure_sqlite_tables(conn: sqlite3.Connection) -> None:
    """Ensure BTC-related tables exist for lightweight SQLite-backed tests."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS crypto_deposits (
               deposit_id TEXT PRIMARY KEY,
               customer_id TEXT NOT NULL,
               btc_address TEXT NOT NULL,
               amount_btc REAL NOT NULL,
               amount_cad REAL NOT NULL,
               btc_cad_rate REAL NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending',
               confirmations INTEGER NOT NULL DEFAULT 0,
               txid TEXT NOT NULL DEFAULT '',
               created_at REAL NOT NULL,
               expires_at REAL NOT NULL,
               confirmed_at REAL NOT NULL DEFAULT 0,
               credited_at REAL NOT NULL DEFAULT 0
           )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_crypto_deposits_status ON crypto_deposits(status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_crypto_deposits_customer ON crypto_deposits(customer_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_crypto_deposits_address ON crypto_deposits(btc_address)"
    )

    conn.execute(
        """CREATE TABLE IF NOT EXISTS fintrac_reports (
               report_id TEXT PRIMARY KEY,
               customer_id TEXT NOT NULL,
               report_type TEXT NOT NULL,
               trigger_amount_cad REAL NOT NULL,
               trigger_currency TEXT NOT NULL,
               aggregate_window_start REAL NOT NULL,
               aggregate_window_end REAL NOT NULL,
               status TEXT NOT NULL DEFAULT 'pending',
               created_at REAL NOT NULL,
               notes TEXT NOT NULL DEFAULT ''
           )"""
    )


class _SqliteCompatConnection:
    """Adapt Postgres-style `%s` placeholders to SQLite's `?` placeholders."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, query: str, params: tuple | list | None = None):
        normalized_query = query.replace("%s", "?")
        return self._conn.execute(normalized_query, params or ())

    def __getattr__(self, name: str):
        return getattr(self._conn, name)


@contextmanager
def _conn():
    backend = os.environ.get("XCELSIOR_DB_BACKEND", "postgres").lower()

    if backend != "postgres":
        from db import sqlite_transaction

        with sqlite_transaction() as c:
            _ensure_sqlite_tables(c)
            yield _SqliteCompatConnection(c)
        return

    from db import _get_pg_pool
    from psycopg.rows import dict_row

    pool = _get_pg_pool()
    with pool.connection() as c:
        c.row_factory = dict_row
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise


# ── Deposit API ───────────────────────────────────────────────────────


def create_deposit(customer_id: str, amount_cad: float) -> dict:
    """Create a new BTC deposit request.

    Generates a unique address, locks the BTC/CAD rate for 30 minutes.
    """
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
               VALUES (%s, %s, %s, %s, %s, %s, 'pending', 0, '', %s, %s)""",
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
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM crypto_deposits WHERE deposit_id = %s",
            (deposit_id,),
        ).fetchone()
        return dict(row) if row else None


def get_deposits_by_customer(customer_id: str, limit: int = 20) -> list[dict]:
    """Get deposit history for a customer."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM crypto_deposits WHERE customer_id = %s ORDER BY created_at DESC LIMIT %s",
            (customer_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def refresh_deposit(deposit_id: str) -> dict | None:
    """Refresh an expired deposit with a new rate. Reuses the same address."""
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
               SET btc_cad_rate = %s, amount_btc = %s, status = 'pending',
                   expires_at = %s, created_at = %s
               WHERE deposit_id = %s""",
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
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM crypto_deposits WHERE status IN ('pending', 'confirming')",
        ).fetchall()
        return [dict(r) for r in rows]


def check_and_update_deposit(deposit: dict) -> dict:
    """Check a single deposit against the Bitcoin node and update status.

    Returns the updated deposit dict.
    """
    deposit_id = deposit["deposit_id"]
    address = deposit["btc_address"]
    now = time.time()

    # Expire if no confirmations and past expiry
    if deposit["status"] == "pending" and now > deposit["expires_at"]:
        received = get_received_by_address(address, 0)
        if received < deposit["amount_btc"] * 0.99:  # 1% tolerance
            with _conn() as c:
                c.execute(
                    "UPDATE crypto_deposits SET status = 'expired' WHERE deposit_id = %s",
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
               SET status = %s, confirmations = %s, confirmed_at = %s
               WHERE deposit_id = %s AND status IN ('pending', 'confirming')""",
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
    with _conn() as c:
        c.execute(
            """UPDATE crypto_deposits
               SET status = 'credited', credited_at = %s
               WHERE deposit_id = %s AND status = 'confirmed'""",
            (time.time(), deposit_id),
        )


# ── FINTRAC LVCTR Reporting for Bitcoin ───────────────────────────────
# Per Proceeds of Crime (Money Laundering) and Terrorist Financing Act:
# - Large Virtual Currency Transaction Report (LVCTR): >= $10,000 CAD
# - 24-hour aggregation rule: multiple deposits by same user totaling >= $10K
# - Must be filed within 5 business days

FINTRAC_LVCTR_THRESHOLD_CAD = 10_000.0


def fintrac_check_btc_deposit(customer_id: str, amount_cad: float) -> dict | None:
    """Check if a BTC deposit triggers FINTRAC LVCTR reporting.

    Checks both single-transaction and 24-hour aggregate thresholds.

    Args:
        customer_id: The wallet owner
        amount_cad: CAD-equivalent value of the BTC deposit

    Returns:
        FINTRAC report dict if threshold triggered, None otherwise.
    """
    now = time.time()
    report = None

    # Single transaction >= $10K
    if amount_cad >= FINTRAC_LVCTR_THRESHOLD_CAD:
        report = _create_btc_fintrac_report(
            customer_id=customer_id,
            report_type="LVCTR",
            trigger_amount=amount_cad,
            notes=f"Single BTC deposit: ${amount_cad:.2f} CAD",
        )
        return report

    # 24-hour aggregate check
    window_start = now - 86400
    with _conn() as c:
        row = c.execute(
            """SELECT COALESCE(SUM(amount_cad), 0) as total_24h
               FROM crypto_deposits
               WHERE customer_id = %s
                 AND created_at >= %s
                 AND status IN ('confirmed', 'credited')""",
            (customer_id, window_start),
        ).fetchone()

    total_24h = float(row["total_24h"]) if row else 0.0

    if total_24h + amount_cad >= FINTRAC_LVCTR_THRESHOLD_CAD:
        report = _create_btc_fintrac_report(
            customer_id=customer_id,
            report_type="LVCTR",
            trigger_amount=total_24h + amount_cad,
            notes=f"24-hour BTC aggregate: ${total_24h + amount_cad:.2f} CAD "
                  f"({total_24h:.2f} prior + {amount_cad:.2f} new)",
        )

    return report


def _create_btc_fintrac_report(
    customer_id: str,
    report_type: str,
    trigger_amount: float,
    notes: str = "",
) -> dict:
    """Create a FINTRAC report record for a BTC transaction."""
    now = time.time()
    report_id = f"FIN-BTC-{int(now)}-{os.urandom(3).hex()}"

    with _conn() as c:
        c.execute(
            """INSERT INTO fintrac_reports
               (report_id, customer_id, report_type, trigger_amount_cad,
                trigger_currency, aggregate_window_start, aggregate_window_end,
                status, created_at, notes)
               VALUES (%s, %s, %s, %s, 'BTC', %s, %s, 'pending', %s, %s)""",
            (report_id, customer_id, report_type, trigger_amount,
             now - 86400, now, now, notes),
        )

    log.warning(
        "FINTRAC %s report (BTC): %s customer=%s amount=$%.2f CAD",
        report_type, report_id, customer_id, trigger_amount,
    )

    return {
        "report_id": report_id,
        "report_type": report_type,
        "customer_id": customer_id,
        "trigger_amount_cad": trigger_amount,
        "trigger_currency": "BTC",
        "status": "pending",
    }


def get_pending_fintrac_reports() -> list[dict]:
    """Retrieve all pending FINTRAC reports for BTC transactions."""
    with _conn() as c:
        rows = c.execute(
            """SELECT * FROM fintrac_reports
               WHERE trigger_currency = 'BTC' AND status = 'pending'
               ORDER BY created_at""",
        ).fetchall()
        return [dict(r) for r in rows]
