"""Authoritative provider settlement: exact money, rails, and concurrency."""

from __future__ import annotations

import inspect
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from psycopg.rows import dict_row

import paypal_connect
import stripe_connect
from provider_settlement import (
    SettlementConflict,
    SettlementNotFound,
    claim_settlements,
    get_settlement,
    prepare_settlement,
    split_source_micros,
)


def _pool():
    from db import _get_pg_pool

    return _get_pg_pool()


def _require_080() -> None:
    with _pool().connection() as conn:
        exists = conn.execute(
            """
            SELECT 1
              FROM information_schema.columns
             WHERE table_name = 'payout_splits'
               AND column_name = 'settlement_key'
            """
        ).fetchone()
    if not exists:
        pytest.skip("migration 080 is not installed on the test database")


@pytest.fixture
def settlement_authority():
    _require_080()
    suffix = uuid.uuid4().hex[:12]
    provider_id = f"provider-settle-{suffix}"
    customer_id = f"customer-settle-{suffix}"
    host_id = f"host-settle-{suffix}"
    job_id = f"job-settle-{suffix}"
    meter_id = f"meter-settle-{suffix}"
    tx_id = f"tx-settle-{suffix}"
    source_micros = 10_005_000  # $10.005 -> one explicit rail-cent adjustment.
    now = time.time()

    with _pool().connection() as conn:
        conn.row_factory = dict_row
        conn.execute(
            """
            INSERT INTO provider_accounts (
                provider_id, provider_type, stripe_account_id, status,
                email, legal_name, country, province, created_at,
                default_currency, paypal_status, paypal_merchant_id
            )
            VALUES (%s, 'individual', %s, 'active', %s, 'Settlement Provider',
                    'CA', 'ON', %s, 'cad', 'active', %s)
            """,
            (
                provider_id,
                f"acct_{suffix}",
                f"{provider_id}@example.test",
                now,
                f"merchant-{suffix}",
            ),
        )
        conn.execute(
            """
            INSERT INTO hosts (
                host_id, status, registered_at, payload,
                provider_id, owner_id, country, province, admission_state)
            VALUES (%s, 'active', %s, %s::jsonb, %s, %s, 'CA', 'ON', 'admitted')
            """,
            (
                host_id,
                now,
                f'{{"provider_id":"{provider_id}","owner":"{provider_id}"}}',
                provider_id,
                provider_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, priority, submitted_at, host_id, payload,
                tenant_id, owner_id, desired_state, phase
            )
            VALUES (%s, 'completed', 0, %s, %s, %s::jsonb,
                    %s, %s, 'stopped', 'succeeded')
            """,
            (
                job_id,
                now - 60,
                host_id,
                f'{{"owner":"{customer_id}"}}',
                customer_id,
                customer_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO wallet_transactions (
                tx_id, customer_id, tx_type, amount_micros,
                balance_after_micros, description, job_id, created_at
            )
            VALUES (%s, %s, 'charge', %s, 0, 'settlement test charge', %s, %s)
            """,
            (tx_id, customer_id, -source_micros, job_id, now - 1),
        )
        conn.execute(
            """
            INSERT INTO usage_meters (
                meter_id, job_id, host_id, owner,
                started_at, completed_at, duration_sec, gpu_seconds,
                total_cost_micros, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, 60, 60, %s, %s)
            """,
            (
                meter_id,
                job_id,
                host_id,
                customer_id,
                now - 60,
                now,
                source_micros,
                now,
            ),
        )
        conn.commit()

    authority = {
        "provider_id": provider_id,
        "customer_id": customer_id,
        "host_id": host_id,
        "job_id": job_id,
        "meter_id": meter_id,
        "tx_id": tx_id,
        "source_micros": source_micros,
    }
    yield authority

    with _pool().connection() as conn:
        conn.execute("DELETE FROM payout_splits WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM usage_meters WHERE meter_id = %s", (meter_id,))
        conn.execute("DELETE FROM wallet_transactions WHERE tx_id = %s", (tx_id,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))
        conn.execute("DELETE FROM hosts WHERE host_id = %s", (host_id,))
        conn.execute(
            "DELETE FROM provider_accounts WHERE provider_id = %s",
            (provider_id,),
        )
        conn.commit()


def test_exact_split_has_zero_residual_and_explicit_cent_adjustment():
    split = split_source_micros(10_005_000, cut_bps=1500, tax_bps=1300)
    assert split.source_total_micros == 10_005_000
    assert split.total_micros == 10_010_000
    assert split.rounding_adjustment_micros == 5_000
    assert split.provider_share_micros == 8_510_000
    assert split.platform_share_micros == 1_500_000
    assert split.provider_share_micros + split.platform_share_micros == split.total_micros
    assert split.gst_hst_micros == 1_300_000


def test_prepare_derives_money_owner_currency_and_tax_from_postgres(
    settlement_authority,
):
    authority = settlement_authority
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        row = prepare_settlement(
            conn,
            job_id=authority["job_id"],
            provider_id=authority["provider_id"],
            rail="stripe",
            expected_customer_id=authority["customer_id"],
            cut_bps=1500,
        )
        conn.commit()

    assert row["customer_id"] == authority["customer_id"]
    assert row["currency"] == "CAD"
    assert row["source_total_micros"] == authority["source_micros"]
    assert row["total_micros"] == 10_010_000
    assert row["provider_share_micros"] + row["platform_share_micros"] == row["total_micros"]
    assert row["tax_rate_bps"] == 1300
    assert row["settlement_key"] == f"provider-job:{authority['job_id']}"


def test_customer_and_provider_authorization_are_checked_in_authority_query(
    settlement_authority,
):
    authority = settlement_authority
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        with pytest.raises(SettlementNotFound):
            prepare_settlement(
                conn,
                job_id=authority["job_id"],
                provider_id="another-provider",
                rail="stripe",
            )
        conn.rollback()
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        with pytest.raises(SettlementNotFound):
            prepare_settlement(
                conn,
                job_id=authority["job_id"],
                provider_id=authority["provider_id"],
                rail="paypal",
                expected_customer_id="another-customer",
            )
        conn.rollback()


def test_concurrent_cross_rail_prepare_creates_one_settlement(
    settlement_authority,
):
    authority = settlement_authority
    barrier = threading.Barrier(2)

    def _prepare(rail: str) -> str:
        barrier.wait(timeout=5)
        try:
            with _pool().connection() as conn:
                conn.row_factory = dict_row
                prepare_settlement(
                    conn,
                    job_id=authority["job_id"],
                    provider_id=authority["provider_id"],
                    rail=rail,
                    expected_customer_id=authority["customer_id"],
                )
                conn.commit()
            return f"created:{rail}"
        except SettlementConflict:
            return f"conflict:{rail}"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(_prepare, ("stripe", "paypal")))
    assert sum(value.startswith("created:") for value in outcomes) == 1
    assert sum(value.startswith("conflict:") for value in outcomes) == 1

    with _pool().connection() as conn:
        count = conn.execute(
            "SELECT count(*) FROM payout_splits WHERE settlement_key = %s",
            (f"provider-job:{authority['job_id']}",),
        ).fetchone()[0]
    assert count == 1


def test_concurrent_workers_claim_a_settlement_once(settlement_authority):
    authority = settlement_authority
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        prepare_settlement(
            conn,
            job_id=authority["job_id"],
            provider_id=authority["provider_id"],
            rail="stripe",
        )
        conn.commit()

    barrier = threading.Barrier(2)

    def _claim(owner: str) -> list[dict]:
        barrier.wait(timeout=5)
        with _pool().connection() as conn:
            conn.row_factory = dict_row
            rows = claim_settlements(
                conn,
                rail="stripe",
                owner=owner,
                limit=1,
                job_id=authority["job_id"],
            )
            conn.commit()
            return rows

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(_claim, ("worker-a", "worker-b")))
    assert sorted(len(rows) for rows in claims) == [0, 1]
    winner = next(rows[0] for rows in claims if rows)
    assert winner["settlement_status"] == "processing"
    assert winner["claim_token"]
    assert winner["claim_expires_at"] is not None


def test_expired_processing_lease_is_reclaimed(settlement_authority):
    authority = settlement_authority
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        prepare_settlement(
            conn,
            job_id=authority["job_id"],
            provider_id=authority["provider_id"],
            rail="stripe",
        )
        first = claim_settlements(
            conn,
            rail="stripe",
            owner="worker-before-crash",
            limit=1,
            job_id=authority["job_id"],
        )[0]
        conn.execute(
            """
            UPDATE payout_splits
               SET claim_expires_at = clock_timestamp() - interval '1 second'
             WHERE id = %s
            """,
            (first["id"],),
        )
        conn.commit()

    with _pool().connection() as conn:
        conn.row_factory = dict_row
        reclaimed = claim_settlements(
            conn,
            rail="stripe",
            owner="recovery-worker",
            limit=1,
            job_id=authority["job_id"],
        )
        conn.commit()

    assert len(reclaimed) == 1
    assert reclaimed[0]["claim_owner"] == "recovery-worker"
    assert reclaimed[0]["claim_token"] != first["claim_token"]
    assert reclaimed[0]["attempt_count"] == 2


def test_stripe_transfer_uses_exact_db_amount_and_is_idempotent(
    settlement_authority,
    monkeypatch,
):
    authority = settlement_authority
    fake_stripe = MagicMock()
    fake_stripe.Balance.retrieve.return_value = SimpleNamespace(
        available=[{"currency": "cad", "amount": 100_000}]
    )
    fake_stripe.Transfer.create.return_value = SimpleNamespace(id="tr_exact_080")
    monkeypatch.setattr(stripe_connect, "STRIPE_ENABLED", True)
    monkeypatch.setattr(stripe_connect, "stripe", fake_stripe)

    manager = stripe_connect.StripeConnectManager()
    first = manager.split_payout(authority["job_id"], authority["provider_id"])
    second = manager.split_payout(authority["job_id"], authority["provider_id"])

    assert first["settlement_status"] == "paid"
    assert first["source_total_micros"] == authority["source_micros"]
    assert first["provider_share_micros"] == 8_510_000
    assert second["stripe_transfer_id"] == "tr_exact_080"
    assert "claim_token" not in first
    assert "rail_idempotency_key" not in first
    fake_stripe.Transfer.create.assert_called_once()
    kwargs = fake_stripe.Transfer.create.call_args.kwargs
    assert kwargs["amount"] == 851
    assert kwargs["currency"] == "cad"
    assert kwargs["idempotency_key"] == f"provider-settlement:{authority['job_id']}"


def test_paypal_order_and_capture_use_persisted_exact_authority(
    settlement_authority,
    monkeypatch,
):
    authority = settlement_authority
    order_id = f"ORDER-{uuid.uuid4().hex[:10]}"
    capture_id = f"CAPTURE-{uuid.uuid4().hex[:10]}"
    posts: list[tuple[str, dict]] = []

    class Response:
        def __init__(self, payload: dict, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code
            self.text = ""

        def json(self):
            return self._payload

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http error")

    approved = {
        "status": "APPROVED",
        "purchase_units": [
            {
                "custom_id": f"{authority['provider_id']}:{authority['job_id']}",
                "amount": {"currency_code": "CAD", "value": "10.01"},
            }
        ],
    }
    completed = {
        "status": "COMPLETED",
        "purchase_units": [
            {
                "custom_id": f"{authority['provider_id']}:{authority['job_id']}",
                "amount": {"currency_code": "CAD", "value": "10.01"},
                "payments": {
                    "captures": [
                        {
                            "id": capture_id,
                            "amount": {"currency_code": "CAD", "value": "10.01"},
                        }
                    ]
                },
            }
        ],
    }

    def _post(url: str, **kwargs):
        posts.append((url, kwargs))
        if url.endswith("/capture"):
            return Response(completed)
        return Response({"id": order_id})

    monkeypatch.setattr(paypal_connect, "PAYPAL_ENABLED", True)
    monkeypatch.setattr(paypal_connect, "_access_token", lambda: "paypal-token")
    monkeypatch.setattr(paypal_connect.httpx, "post", _post)
    monkeypatch.setattr(
        paypal_connect.httpx,
        "get",
        lambda *args, **kwargs: Response(approved),
    )

    manager = paypal_connect.PayPalConnectManager()
    created = manager.create_marketplace_order(
        authority["provider_id"],
        authority["job_id"],
        expected_customer_id=authority["customer_id"],
    )
    purchase = posts[0][1]["json"]["purchase_units"][0]
    assert created["order_id"] == order_id
    assert purchase["amount"]["value"] == "10.01"
    assert purchase["payment_instruction"]["platform_fees"][0]["amount"]["value"] == "1.50"
    assert (
        posts[0][1]["headers"]["PayPal-Request-Id"] == f"provider-settlement:{authority['job_id']}"
    )

    paid = manager.capture_marketplace_order(
        authority["provider_id"],
        order_id,
        expected_customer_id=authority["customer_id"],
    )
    replay = manager.capture_marketplace_order(
        authority["provider_id"],
        order_id,
        expected_customer_id=authority["customer_id"],
    )
    assert paid["settlement_status"] == "paid"
    assert paid["paypal_capture_id"] == capture_id
    assert replay["capture_id"] == capture_id
    assert sum(url.endswith("/capture") for url, _kwargs in posts) == 1

    with _pool().connection() as conn:
        conn.row_factory = dict_row
        with pytest.raises(SettlementConflict, match="already bound"):
            prepare_settlement(
                conn,
                job_id=authority["job_id"],
                provider_id=authority["provider_id"],
                rail="stripe",
            )
        conn.rollback()


def test_payout_api_signature_has_no_caller_amount():
    from routes.billing import PayPalMarketplaceCreateRequest
    from routes.providers import api_provider_payout

    assert "total_cad" not in inspect.signature(api_provider_payout).parameters
    request = PayPalMarketplaceCreateRequest(
        customer_id="customer",
        provider_id="provider",
        job_id="job",
        amount_cad=9_999_999,
    )
    assert "amount_cad" not in request.model_dump()
