"""Tests for Bitcoin deposit module (bitcoin.py)."""

import os
import time
import json
from unittest.mock import patch, MagicMock
import pytest

# Ensure BTC is enabled for tests
os.environ["XCELSIOR_BTC_ENABLED"] = "true"
os.environ["XCELSIOR_BTC_RPC_HOST"] = "127.0.0.1"
os.environ["XCELSIOR_BTC_RPC_PORT"] = "18332"
os.environ["XCELSIOR_BTC_RPC_USER"] = "testuser"
os.environ["XCELSIOR_BTC_RPC_PASS"] = "testpass"
os.environ["XCELSIOR_BTC_CONFIRMATIONS"] = "3"
os.environ["XCELSIOR_BTC_DEPOSIT_EXPIRY"] = "1800"

import bitcoin


@pytest.fixture(autouse=True)
def _btc_clean_table():
    """Clean crypto_deposits table before each test."""
    with bitcoin._conn() as c:
        c.execute("DELETE FROM crypto_deposits")
    yield


@pytest.fixture()
def mock_rpc():
    """Mock Bitcoin Core RPC calls."""
    with patch("bitcoin._rpc_call") as m:
        yield m


@pytest.fixture()
def mock_rate():
    """Mock BTC/CAD rate to a fixed value."""
    with patch("bitcoin.get_btc_cad_rate", return_value=90000.0) as m:
        yield m


# ── Rate Fetching ─────────────────────────────────────────────────────


class TestBtcCadRate:
    def test_rate_from_coingecko(self):
        response_data = json.dumps({"bitcoin": {"cad": 91234.56}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("bitcoin._rate_cache", {"rate": 0.0, "fetched_at": 0.0}):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                rate = bitcoin.get_btc_cad_rate()
                assert rate == 91234.56

    def test_rate_cache_hits(self):
        """Cached rate should be returned within TTL."""
        with patch(
            "bitcoin._rate_cache",
            {"rate": 85000.0, "fetched_at": time.time()},
        ):
            rate = bitcoin.get_btc_cad_rate()
            assert rate == 85000.0

    def test_rate_cache_expired(self):
        """Expired cache should fetch a new rate."""
        response_data = json.dumps({"bitcoin": {"cad": 92000.0}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "bitcoin._rate_cache",
            {"rate": 85000.0, "fetched_at": time.time() - 600},
        ):
            with patch("urllib.request.urlopen", return_value=mock_resp):
                rate = bitcoin.get_btc_cad_rate()
                assert rate == 92000.0

    def test_rate_failure_uses_stale_cache(self):
        """On fetch failure, return stale cache if available."""
        with patch(
            "bitcoin._rate_cache",
            {"rate": 85000.0, "fetched_at": time.time() - 600},
        ):
            with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
                rate = bitcoin.get_btc_cad_rate()
                assert rate == 85000.0

    def test_rate_failure_no_cache_raises(self):
        """On failure with no cache, raise."""
        with patch("bitcoin._rate_cache", {"rate": 0.0, "fetched_at": 0.0}):
            with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
                with pytest.raises(RuntimeError, match="Unable to fetch"):
                    bitcoin.get_btc_cad_rate()


# ── Deposit Creation ──────────────────────────────────────────────────


class TestCreateDeposit:
    def test_creates_deposit(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qtest123address"
        result = bitcoin.create_deposit("cust-abc", 100.0)

        assert result["deposit_id"].startswith("btc-")
        assert result["btc_address"] == "bc1qtest123address"
        assert result["amount_cad"] == 100.0
        assert result["amount_btc"] == pytest.approx(100.0 / 90000.0, rel=1e-6)
        assert result["btc_cad_rate"] == 90000.0
        assert "qr_data" in result
        assert result["qr_data"].startswith("bitcoin:bc1qtest123address")

    def test_deposit_stored_in_db(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qtest456"
        result = bitcoin.create_deposit("cust-xyz", 50.0)

        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep is not None
        assert dep["customer_id"] == "cust-xyz"
        assert dep["status"] == "pending"
        assert dep["amount_cad"] == 50.0

    def test_deposit_expires_at_set(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qtest789"
        result = bitcoin.create_deposit("cust-abc", 25.0)

        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep["expires_at"] > time.time()
        assert dep["expires_at"] <= time.time() + 1801


# ── Deposit Status ────────────────────────────────────────────────────


class TestGetDeposit:
    def test_get_nonexistent(self):
        dep = bitcoin.get_deposit("btc-nonexistent")
        assert dep is None

    def test_get_existing(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qabc"
        result = bitcoin.create_deposit("cust-1", 10.0)
        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep["deposit_id"] == result["deposit_id"]

    def test_get_deposits_by_customer(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1q111"
        bitcoin.create_deposit("cust-multi", 10.0)
        mock_rpc.return_value = "bc1q222"
        bitcoin.create_deposit("cust-multi", 20.0)
        mock_rpc.return_value = "bc1q333"
        bitcoin.create_deposit("cust-other", 30.0)

        deps = bitcoin.get_deposits_by_customer("cust-multi")
        assert len(deps) == 2
        assert all(d["customer_id"] == "cust-multi" for d in deps)


# ── Refresh Deposit ───────────────────────────────────────────────────


class TestRefreshDeposit:
    def test_refresh_expired(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qrefresh"
        result = bitcoin.create_deposit("cust-r", 100.0)

        # Manually expire it
        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET status = 'expired' WHERE deposit_id = %s",
                (result["deposit_id"],),
            )

        with patch("bitcoin.get_btc_cad_rate", return_value=95000.0):
            refreshed = bitcoin.refresh_deposit(result["deposit_id"])
            assert refreshed["status"] == "pending"
            assert refreshed["btc_cad_rate"] == 95000.0
            assert refreshed["amount_btc"] == pytest.approx(100.0 / 95000.0, rel=1e-4)

    def test_refresh_nonexistent(self, mock_rpc, mock_rate):
        result = bitcoin.refresh_deposit("btc-nope")
        assert result is None

    def test_refresh_confirmed_noop(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qconf"
        result = bitcoin.create_deposit("cust-c", 50.0)

        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET status = 'confirmed' WHERE deposit_id = %s",
                (result["deposit_id"],),
            )

        refreshed = bitcoin.refresh_deposit(result["deposit_id"])
        assert refreshed["status"] == "confirmed"  # not changed


# ── Confirmation Checking ─────────────────────────────────────────────


class TestCheckAndUpdateDeposit:
    def _make_deposit(self, mock_rpc, mock_rate, customer_id="cust-check"):
        mock_rpc.return_value = "bc1qcheck123"
        return bitcoin.create_deposit(customer_id, 200.0)

    def test_no_payment_received(self, mock_rpc, mock_rate):
        result = self._make_deposit(mock_rpc, mock_rate)
        dep = bitcoin.get_deposit(result["deposit_id"])

        # No BTC received at the address
        mock_rpc.return_value = 0.0
        with patch("bitcoin.get_received_by_address", return_value=0.0):
            updated = bitcoin.check_and_update_deposit(dep)
            assert updated["status"] == "pending"

    def test_payment_in_mempool(self, mock_rpc, mock_rate):
        result = self._make_deposit(mock_rpc, mock_rate)
        dep = bitcoin.get_deposit(result["deposit_id"])
        expected_btc = dep["amount_btc"]

        def fake_received(addr, min_conf=0):
            if min_conf == 0:
                return expected_btc
            return 0.0

        with patch("bitcoin.get_received_by_address", side_effect=fake_received):
            updated = bitcoin.check_and_update_deposit(dep)
            assert updated["status"] == "confirming"
            assert updated["confirmations"] == 0

    def test_payment_1_confirmation(self, mock_rpc, mock_rate):
        result = self._make_deposit(mock_rpc, mock_rate)
        dep = bitcoin.get_deposit(result["deposit_id"])
        expected_btc = dep["amount_btc"]

        def fake_received(addr, min_conf=0):
            if min_conf <= 1:
                return expected_btc
            return 0.0

        with patch("bitcoin.get_received_by_address", side_effect=fake_received):
            updated = bitcoin.check_and_update_deposit(dep)
            # When BTC_CONFIRMATIONS=1, 1-conf matches the required threshold → confirmed
            # When BTC_CONFIRMATIONS>1, 1-conf is partial → confirming
            if bitcoin.BTC_CONFIRMATIONS <= 1:
                assert updated["status"] == "confirmed"
                assert updated["confirmations"] == bitcoin.BTC_CONFIRMATIONS
            else:
                assert updated["status"] == "confirming"
                assert updated["confirmations"] == 1

    def test_payment_fully_confirmed(self, mock_rpc, mock_rate):
        result = self._make_deposit(mock_rpc, mock_rate)
        dep = bitcoin.get_deposit(result["deposit_id"])
        expected_btc = dep["amount_btc"]

        def fake_received(addr, min_conf=0):
            return expected_btc  # any minconf returns full amount

        with patch("bitcoin.get_received_by_address", side_effect=fake_received):
            updated = bitcoin.check_and_update_deposit(dep)
            assert updated["status"] == "confirmed"
            assert updated["confirmations"] == bitcoin.BTC_CONFIRMATIONS
            assert updated["confirmed_at"] > 0

    def test_expired_no_payment(self, mock_rpc, mock_rate):
        result = self._make_deposit(mock_rpc, mock_rate)

        # Force expiry
        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET expires_at = %s WHERE deposit_id = %s",
                (time.time() - 10, result["deposit_id"]),
            )

        dep = bitcoin.get_deposit(result["deposit_id"])

        with patch("bitcoin.get_received_by_address", return_value=0.0):
            updated = bitcoin.check_and_update_deposit(dep)
            assert updated["status"] == "expired"


# ── Mark Credited ─────────────────────────────────────────────────────


class TestMarkCredited:
    def test_mark_confirmed_as_credited(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1qcredit"
        result = bitcoin.create_deposit("cust-cr", 75.0)

        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET status = 'confirmed' WHERE deposit_id = %s",
                (result["deposit_id"],),
            )

        bitcoin.mark_credited(result["deposit_id"])
        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep["status"] == "credited"
        assert dep["credited_at"] > 0

    def test_mark_pending_not_credited(self, mock_rpc, mock_rate):
        """Only confirmed deposits can be marked as credited."""
        mock_rpc.return_value = "bc1qpending"
        result = bitcoin.create_deposit("cust-pen", 50.0)

        bitcoin.mark_credited(result["deposit_id"])
        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep["status"] == "pending"  # unchanged


# ── Pending Deposits Query ────────────────────────────────────────────


class TestGetPendingDeposits:
    def test_returns_pending_and_confirming(self, mock_rpc, mock_rate):
        mock_rpc.return_value = "bc1q_p1"
        bitcoin.create_deposit("cust-1", 10.0)
        mock_rpc.return_value = "bc1q_p2"
        r2 = bitcoin.create_deposit("cust-2", 20.0)

        # Mark one as confirming
        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET status = 'confirming' WHERE deposit_id = %s",
                (r2["deposit_id"],),
            )

        # Mark a third as credited (should not appear)
        mock_rpc.return_value = "bc1q_p3"
        r3 = bitcoin.create_deposit("cust-3", 30.0)
        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET status = 'credited' WHERE deposit_id = %s",
                (r3["deposit_id"],),
            )

        pending = bitcoin.get_pending_deposits()
        assert len(pending) == 2
        statuses = {d["status"] for d in pending}
        assert statuses == {"pending", "confirming"}


# ── RPC Client ────────────────────────────────────────────────────────


class TestRpcClient:
    def test_get_new_address(self, mock_rpc):
        mock_rpc.return_value = "bc1qnew123"
        addr = bitcoin.get_new_address("test-label")
        assert addr == "bc1qnew123"
        mock_rpc.assert_called_once_with("getnewaddress", ["test-label", "bech32"])

    def test_get_received_by_address(self, mock_rpc):
        mock_rpc.return_value = 0.00123456
        amount = bitcoin.get_received_by_address("bc1qtest", 3)
        assert amount == 0.00123456
        mock_rpc.assert_called_once_with("getreceivedbyaddress", ["bc1qtest", 3])

    def test_rpc_error_raises(self):
        error_resp = json.dumps({"result": None, "error": {"code": -5, "message": "Invalid address"}}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = error_resp
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Bitcoin RPC error"):
                bitcoin._rpc_call("validateaddress", ["bad"])


# ── Idempotency ───────────────────────────────────────────────────────


class TestIdempotency:
    def test_double_credit_prevented(self, mock_rpc, mock_rate):
        """mark_credited should only work once (confirmed → credited)."""
        mock_rpc.return_value = "bc1qidem"
        result = bitcoin.create_deposit("cust-idem", 100.0)

        with bitcoin._conn() as c:
            c.execute(
                "UPDATE crypto_deposits SET status = 'confirmed' WHERE deposit_id = %s",
                (result["deposit_id"],),
            )

        bitcoin.mark_credited(result["deposit_id"])
        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep["status"] == "credited"

        # Second call should not change anything (already credited, not confirmed)
        bitcoin.mark_credited(result["deposit_id"])
        dep = bitcoin.get_deposit(result["deposit_id"])
        assert dep["status"] == "credited"
