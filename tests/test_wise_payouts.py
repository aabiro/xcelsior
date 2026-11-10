"""Wise payout rail: what it must refuse, and that the ledger stays exact.

Wise is a fake HTTP transport here. Everything else is real: the settlement
rows, the constraints of migration 159, and the functions the routes call.
"""

from __future__ import annotations

import os

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import base64
import json
import time
import uuid

import httpx
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from psycopg.rows import dict_row

import wise_payouts as wise
from provider_settlement import claim_settlements, prepare_settlement

PROFILE = 30000000


def _pool():
    from db import _get_pg_pool

    return _get_pg_pool()


def _one(sql: str, params: tuple = ()) -> dict | None:
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        row = conn.execute(sql, params).fetchone()
        conn.commit()
    return row


def _require_159() -> None:
    if not _one("SELECT to_regclass('public.wise_payouts') AS t")["t"]:
        pytest.skip("migration 159 is not installed on the test database")


# ── A fake Wise ──────────────────────────────────────────────────────


class FakeWise:
    """Enough of Wise's API to pay someone, with the knobs a test needs."""

    def __init__(self):
        self.recipients: dict[int, dict] = {}
        self.transfers: dict[int, dict] = {}
        self.by_customer_tx: dict[str, int] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_next_transfer: str | None = None  # "timeout" | "422" | "500"
        self.skew_amount = False
        self.balance = "10000.00"
        self._next = 50000000

    def add_recipient(self, *, currency="CAD", profile=PROFILE, wise_hash=None, active=True):
        self._next += 1
        rid = self._next
        self.recipients[rid] = {
            "id": rid,
            "profileId": profile,
            "currency": currency,
            "type": "interac",
            "active": active,
            "hash": wise_hash or uuid.uuid4().hex * 2,
            "longAccountSummary": f"{currency} account ending in 4242",
            "details": {},
        }
        return rid

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        self.calls.append((method, path))
        assert request.headers["authorization"] == "Bearer test-wise-token"
        body = json.loads(request.content) if request.content else {}
        if method == "GET" and path.startswith("/v2/accounts/"):
            rid = int(path.rsplit("/", 1)[1])
            if rid not in self.recipients:
                return httpx.Response(404, json={"errors": [{"code": "NOT_FOUND"}]})
            return httpx.Response(200, json=self.recipients[rid])
        if method == "POST" and path == f"/v3/profiles/{PROFILE}/quotes":
            assert body["preferredPayIn"] == "BALANCE"
            return httpx.Response(
                200,
                json={
                    "id": str(uuid.uuid4()),
                    "sourceAmount": body["sourceAmount"],
                    "paymentOptions": [
                        {
                            "payIn": "BALANCE",
                            "disabled": False,
                            "price": {"total": {"value": {"amount": 0.61}}},
                        }
                    ],
                },
            )
        if method == "POST" and path == "/v1/transfers":
            failure, self.fail_next_transfer = self.fail_next_transfer, None
            if failure == "timeout":
                raise httpx.ReadTimeout("no answer", request=request)
            if failure == "500":
                return httpx.Response(500, json={})
            if failure == "422":
                return httpx.Response(
                    422, json={"errors": [{"code": "NOT_VALID", "message": "missing approval"}]}
                )
            ctid = body["customerTransactionId"]
            if ctid in self.by_customer_tx:  # idempotent, as Wise documents
                return httpx.Response(200, json=self.transfers[self.by_customer_tx[ctid]])
            self._next += 1
            tid = self._next
            recipient = self.recipients[body["targetAccount"]]
            self.transfers[tid] = {
                "id": tid,
                "targetAccount": body["targetAccount"],
                "sourceCurrency": "CAD",
                "targetCurrency": recipient["currency"],
                "sourceValue": 999.99 if self.skew_amount else self._quoted,
                "targetValue": self._quoted,
                "status": "incoming_payment_waiting",
                "reference": body["details"]["reference"],
                "customerTransactionId": ctid,
            }
            self.by_customer_tx[ctid] = tid
            return httpx.Response(200, json=self.transfers[tid])
        if method == "GET" and path.startswith("/v1/transfers/"):
            return httpx.Response(200, json=self.transfers[int(path.rsplit("/", 1)[1])])
        if method == "PUT" and path.endswith("/cancel"):
            tid = int(path.split("/")[3])
            self.transfers[tid]["status"] = "cancelled"
            return httpx.Response(200, json=self.transfers[tid])
        if method == "GET" and path == f"/v4/profiles/{PROFILE}/balances":
            return httpx.Response(
                200,
                json=[{"currency": "CAD", "amount": {"value": self.balance, "currency": "CAD"}}],
            )
        if method == "POST" and path.endswith("/payments"):
            tid = int(path.split("/")[5])
            self.transfers[tid]["status"] = "processing"
            return httpx.Response(201, json={"status": "COMPLETED", "type": "BALANCE"})
        return httpx.Response(404, json={})

    _quoted = 0.0

    def quoted(self, amount: float) -> "FakeWise":
        self._quoted = amount
        return self


@pytest.fixture
def wise_env(monkeypatch):
    _require_159()
    monkeypatch.setenv("XCELSIOR_WISE_API_TOKEN", "test-wise-token")
    monkeypatch.setenv("XCELSIOR_WISE_PROFILE_ID", str(PROFILE))
    monkeypatch.setenv("XCELSIOR_WISE_ENV", "sandbox")
    monkeypatch.setenv("XCELSIOR_WISE_PAYOUTS_ENABLED", "1")
    monkeypatch.setenv("XCELSIOR_WISE_MIN_PAYOUT_CAD", "5")
    monkeypatch.setenv("XCELSIOR_WISE_MAX_PAYOUT_CAD", "2000")
    monkeypatch.setenv("XCELSIOR_WISE_DAILY_CAP_CAD", "100000")
    monkeypatch.setenv("XCELSIOR_WISE_RECIPIENT_COOLOFF_HOURS", "72")
    monkeypatch.delenv("XCELSIOR_WISE_AUTO_FUND", raising=False)
    monkeypatch.delenv("XCELSIOR_WISE_EMAIL_RECIPIENTS", raising=False)
    fake = FakeWise()
    client = wise.WiseClient(transport=fake.transport())
    yield fake, client
    client.close()


def _make_job(provider_id: str, suffix: str, source_micros: int) -> str:
    """One completed, charged, metered job on this provider's host."""

    customer_id = f"customer-wise-{suffix}"
    host_id = f"host-wise-{suffix}"
    job_id = f"job-wise-{suffix}"
    now = time.time()
    with _pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO hosts (
                host_id, status, registered_at, payload,
                provider_id, owner_id, country, province, admission_state,
                administrative_state, availability_state)
            VALUES (%s, 'active', %s, %s::jsonb, %s, %s, 'CA', 'ON', 'admitted', 'admitted', 'ready')
            """,
            (host_id, now, json.dumps({"provider_id": provider_id}), provider_id, provider_id),
        )
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, priority, submitted_at, host_id, payload,
                tenant_id, owner_id, desired_state, phase, effective_priority, queued_at)
            VALUES (%s, 'completed', 0, %s, %s, %s::jsonb, %s, %s, 'stopped', 'succeeded', 0, NULL)
            """,
            (
                job_id,
                now - 60,
                host_id,
                json.dumps({"owner": customer_id}),
                customer_id,
                customer_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO wallet_transactions (
                tx_id, customer_id, tx_type, amount_micros,
                balance_after_micros, description, job_id, created_at)
            VALUES (%s, %s, 'charge', %s, 0, 'wise test charge', %s, %s)
            """,
            (f"tx-wise-{suffix}", customer_id, -source_micros, job_id, now - 1),
        )
        conn.execute(
            """
            INSERT INTO usage_meters (
                meter_id, job_id, host_id, owner, started_at, completed_at,
                duration_sec, gpu_seconds, total_cost_micros, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, 60, 60, %s, %s)
            """,
            (
                f"meter-wise-{suffix}",
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
    return job_id


@pytest.fixture
def provider():
    """A provider with two settled jobs queued on the Wise rail: $85.00 + $17.00."""

    _require_159()
    tag = uuid.uuid4().hex[:10]
    provider_id = f"provider-wise-{tag}"
    with _pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO provider_accounts (
                provider_id, provider_type, stripe_account_id, status, email,
                legal_name, country, province, created_at, default_currency)
            VALUES (%s, 'individual', '', 'active', %s, 'Wise Provider', 'CA', 'ON', %s, 'cad')
            """,
            (provider_id, f"{provider_id}@example.test", time.time()),
        )
        conn.commit()
    jobs = [
        _make_job(provider_id, f"{tag}-a", 100_000_000),
        _make_job(provider_id, f"{tag}-b", 20_000_000),
    ]
    for job_id in jobs:
        with _pool().connection() as conn:
            conn.row_factory = dict_row
            prepare_settlement(
                conn, job_id=job_id, provider_id=provider_id, rail="wise", cut_bps=1500
            )
            conn.commit()
    yield {"provider_id": provider_id, "jobs": jobs, "tag": tag}

    with _pool().connection() as conn:
        conn.execute("DELETE FROM payout_splits WHERE provider_id = %s", (provider_id,))
        conn.execute("DELETE FROM wise_payouts WHERE provider_id = %s", (provider_id,))
        conn.execute("DELETE FROM provider_wise_recipients WHERE provider_id = %s", (provider_id,))
        conn.execute("DELETE FROM usage_meters WHERE meter_id LIKE %s", (f"meter-wise-{tag}-%",))
        conn.execute("DELETE FROM wallet_transactions WHERE tx_id LIKE %s", (f"tx-wise-{tag}-%",))
        conn.execute("DELETE FROM jobs WHERE job_id LIKE %s", (f"job-wise-{tag}-%",))
        conn.execute("DELETE FROM hosts WHERE host_id LIKE %s", (f"host-wise-{tag}-%",))
        conn.execute("DELETE FROM provider_accounts WHERE provider_id = %s", (provider_id,))
        conn.commit()


def _end_cooloff(provider_id: str) -> None:
    _one(
        "UPDATE provider_wise_recipients SET payouts_enabled_after = clock_timestamp() "
        "- interval '1 second' WHERE provider_id = %s RETURNING provider_id",
        (provider_id,),
    )


def _linked(provider, fake, client, **kw) -> int:
    rid = fake.add_recipient(**kw)
    wise.link_recipient(
        provider_id=provider["provider_id"], recipient_id=rid, actor="admin:test", client=client
    )
    return rid


def _approved_payout(provider, fake, client) -> dict:
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    created = wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"]
    assert len(created) == 1
    return wise.approve_payout(payout_id=created[0], approver="admin-1")


def _splits(provider_id: str) -> list[dict]:
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        return conn.execute(
            "SELECT settlement_status, wise_payout_id, provider_share_micros, settled_at "
            "FROM payout_splits WHERE provider_id = %s ORDER BY id",
            (provider_id,),
        ).fetchall()


# ── Money and signatures ─────────────────────────────────────────────


def test_amounts_are_whole_cents_or_refused():
    assert wise.micros_to_amount(102_000_000) == "102.00"
    assert wise.micros_to_amount(50_000) == "0.05"
    for bad in (0, -10_000, 12_345):
        with pytest.raises(wise.WiseError):
            wise.micros_to_amount(bad)


def test_pinned_webhook_keys_are_rsa_public_keys():
    for pem in wise._WEBHOOK_KEYS.values():
        key = serialization.load_pem_public_key(pem.encode())
        assert isinstance(key, rsa.RSAPublicKey) and key.key_size >= 2048


@pytest.fixture
def signing_key(monkeypatch):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    monkeypatch.setenv("XCELSIOR_WISE_WEBHOOK_PUBLIC_KEY", pem.decode())

    def sign(body: bytes) -> str:
        return base64.b64encode(key.sign(body, padding.PKCS1v15(), hashes.SHA256())).decode()

    return sign


def test_webhook_signature_must_cover_the_exact_body(signing_key, monkeypatch):
    body = b'{"event_type":"transfers#state-change"}'
    good = signing_key(body)
    assert wise.verify_webhook_signature(body, good)
    assert not wise.verify_webhook_signature(body + b" ", good)
    assert not wise.verify_webhook_signature(body, "")
    assert not wise.verify_webhook_signature(body, "not base64!")
    # Signed by a key that is not the pinned one: the real Wise key is pinned
    # once the override is gone, and this signature was not made by Wise.
    monkeypatch.delenv("XCELSIOR_WISE_WEBHOOK_PUBLIC_KEY")
    monkeypatch.setenv("XCELSIOR_WISE_ENV", "sandbox")
    assert not wise.verify_webhook_signature(body, good)


def test_production_cooloff_cannot_be_switched_off(monkeypatch):
    monkeypatch.setenv("XCELSIOR_WISE_ENV", "production")
    monkeypatch.setenv("XCELSIOR_WISE_RECIPIENT_COOLOFF_HOURS", "0")
    assert wise.cooloff_hours() == 24


def test_nothing_is_enabled_by_default(monkeypatch):
    for name in (
        "XCELSIOR_WISE_API_TOKEN",
        "XCELSIOR_WISE_PROFILE_ID",
        "XCELSIOR_WISE_PAYOUTS_ENABLED",
        "XCELSIOR_WISE_AUTO_FUND",
        "XCELSIOR_WISE_EMAIL_RECIPIENTS",
        "XCELSIOR_WISE_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    assert not wise.wise_configured()
    assert not wise.payouts_enabled()
    assert not wise.auto_fund_enabled()
    assert not wise.email_recipients_enabled()
    assert wise.wise_env() == "sandbox"
    # A token alone does not turn payouts on.
    monkeypatch.setenv("XCELSIOR_WISE_API_TOKEN", "t")
    monkeypatch.setenv("XCELSIOR_WISE_PROFILE_ID", "1")
    assert wise.wise_configured() and not wise.payouts_enabled()


# ── Destination ──────────────────────────────────────────────────────


def test_a_recipient_from_another_wise_account_is_refused(wise_env, provider):
    fake, client = wise_env
    foreign = fake.add_recipient(profile=PROFILE + 1)
    with pytest.raises(wise.WiseError) as exc:
        wise.link_recipient(
            provider_id=provider["provider_id"], recipient_id=foreign, actor="a", client=client
        )
    assert exc.value.reason == "foreign_recipient"


def test_one_recipient_cannot_serve_two_providers(wise_env, provider):
    fake, client = wise_env
    rid = _linked(provider, fake, client)
    with pytest.raises(wise.WiseError) as exc:
        wise.link_recipient(
            provider_id="some-other-provider", recipient_id=rid, actor="a", client=client
        )
    assert exc.value.reason == "recipient_in_use"


def test_self_service_email_is_off_until_enabled(wise_env, provider):
    _fake, client = wise_env
    with pytest.raises(wise.WiseError) as exc:
        wise.set_email_recipient(
            provider_id=provider["provider_id"],
            email="p@example.test",
            currency="CAD",
            holder_name="Pat Provider",
            actor="user:p",
            client=client,
        )
    assert exc.value.reason == "email_recipients_disabled"


def test_a_new_destination_cannot_be_paid_during_the_cooloff(wise_env, provider):
    fake, client = wise_env
    _linked(provider, fake, client)
    result = wise.batch_queued_settlements(provider_id=provider["provider_id"])
    assert result == {
        "created": [],
        "skipped": {provider["provider_id"]: "destination_cooling_off"},
    }
    assert all(s["settlement_status"] == "queued" for s in _splits(provider["provider_id"]))
    status = wise.get_recipient_status(provider["provider_id"])
    assert status["recipient"]["payable"] is False
    assert status["recipient"]["display_hint"] == "CAD account ending in 4242"
    assert status["queued_micros"] == 102_000_000


def test_relinking_the_same_destination_does_not_restart_the_cooloff(wise_env, provider):
    fake, client = wise_env
    rid = _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    again = wise.link_recipient(
        provider_id=provider["provider_id"], recipient_id=rid, actor="a", client=client
    )
    assert again["changed"] is False
    assert wise.get_recipient_status(provider["provider_id"])["recipient"]["payable"] is True


def test_changing_the_destination_dissolves_an_unapproved_payout(wise_env, provider):
    fake, client = wise_env
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    payout_id = wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"][0]
    _linked(provider, fake, client)  # a different recipient
    assert (
        _one("SELECT status FROM wise_payouts WHERE id = %s", (payout_id,))["status"] == "rejected"
    )
    splits = _splits(provider["provider_id"])
    assert all(s["settlement_status"] == "queued" and s["wise_payout_id"] is None for s in splits)
    # And the new destination is cooling off again.
    assert wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"] == []


def test_the_destination_is_frozen_once_a_payout_is_approved(wise_env, provider):
    fake, client = wise_env
    _approved_payout(provider, fake, client)
    other = fake.add_recipient()
    with pytest.raises(wise.WiseError) as exc:
        wise.link_recipient(
            provider_id=provider["provider_id"], recipient_id=other, actor="a", client=client
        )
    assert exc.value.reason == "payout_in_progress"


# ── Batching ─────────────────────────────────────────────────────────


def test_batch_adds_up_the_ledger_and_takes_rows_out_of_reach(wise_env, provider):
    fake, client = wise_env
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    payout_id = wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"][0]
    payout = _one("SELECT * FROM wise_payouts WHERE id = %s", (payout_id,))
    # 85% of $100.00 and of $20.00.
    assert payout["amount_micros"] == 85_000_000 + 17_000_000
    assert payout["split_count"] == 2
    assert payout["status"] == "pending_approval"
    splits = _splits(provider["provider_id"])
    assert {s["settlement_status"] for s in splits} == {"in_transit"}
    assert {s["wise_payout_id"] for s in splits} == {payout_id}
    # A second pass does not make a second payout.
    again = wise.batch_queued_settlements(provider_id=provider["provider_id"])
    assert again["created"] == []
    assert fake.calls.count(("POST", "/v1/transfers")) == 0


def test_wise_rows_cannot_be_claimed_by_the_per_job_claimers(wise_env, provider):
    with _pool().connection() as conn:
        conn.row_factory = dict_row
        with pytest.raises(ValueError):
            claim_settlements(conn, rail="wise", owner="x")
        for rail in ("stripe", "paypal"):
            assert claim_settlements(conn, rail=rail, owner="x", job_id=provider["jobs"][0]) == []
        conn.rollback()


def test_below_the_minimum_stays_queued(wise_env, provider, monkeypatch):
    fake, client = wise_env
    monkeypatch.setenv("XCELSIOR_WISE_MIN_PAYOUT_CAD", "500")
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    result = wise.batch_queued_settlements(provider_id=provider["provider_id"])
    assert result["skipped"] == {provider["provider_id"]: "below_minimum"}


def test_the_per_payout_cap_bounds_a_batch(wise_env, provider, monkeypatch):
    fake, client = wise_env
    monkeypatch.setenv("XCELSIOR_WISE_MAX_PAYOUT_CAD", "90")
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    payout_id = wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"][0]
    payout = _one("SELECT amount_micros, split_count FROM wise_payouts WHERE id = %s", (payout_id,))
    assert (payout["amount_micros"], payout["split_count"]) == (85_000_000, 1)
    assert sorted(s["settlement_status"] for s in _splits(provider["provider_id"])) == [
        "in_transit",
        "queued",
    ]


def test_a_split_cannot_be_in_transit_without_a_payout(wise_env, provider):
    import psycopg

    with _pool().connection() as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                "UPDATE payout_splits SET settlement_status = 'in_transit' WHERE provider_id = %s",
                (provider["provider_id"],),
            )
        conn.rollback()


# ── Approval ─────────────────────────────────────────────────────────


def _pending(provider, fake, client) -> int:
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    return wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"][0]


def test_approval_is_refused_while_the_kill_switch_is_off(wise_env, provider, monkeypatch):
    fake, client = wise_env
    payout_id = _pending(provider, fake, client)
    monkeypatch.setenv("XCELSIOR_WISE_PAYOUTS_ENABLED", "0")
    with pytest.raises(wise.WiseError) as exc:
        wise.approve_payout(payout_id=payout_id, approver="admin-1")
    assert exc.value.reason == "payouts_disabled"
    assert wise.execute_approved_payouts(client=client)["reason"] == "payouts_disabled"


def test_approval_is_refused_over_the_daily_cap(wise_env, provider, monkeypatch):
    fake, client = wise_env
    payout_id = _pending(provider, fake, client)
    monkeypatch.setenv("XCELSIOR_WISE_DAILY_CAP_CAD", "50")
    with pytest.raises(wise.WiseError) as exc:
        wise.approve_payout(payout_id=payout_id, approver="admin-1")
    assert exc.value.reason == "daily_cap"


def test_approval_is_refused_if_the_ledger_moved_under_the_batch(wise_env, provider):
    fake, client = wise_env
    payout_id = _pending(provider, fake, client)
    _one(
        "UPDATE payout_splits SET settlement_status = 'manual_review' WHERE id = "
        "(SELECT min(id) FROM payout_splits WHERE wise_payout_id = %s) RETURNING id",
        (payout_id,),
    )
    with pytest.raises(wise.WiseError) as exc:
        wise.approve_payout(payout_id=payout_id, approver="admin-1")
    assert exc.value.reason == "ledger_mismatch"


def test_a_payout_is_approved_once_and_needs_an_approver(wise_env, provider):
    fake, client = wise_env
    payout_id = _pending(provider, fake, client)
    with pytest.raises(wise.WiseError):
        wise.approve_payout(payout_id=payout_id, approver="  ")
    approved = wise.approve_payout(payout_id=payout_id, approver="admin-1")
    assert approved["status"] == "approved" and approved["approved_by"] == "admin-1"
    assert "customer_transaction_id" not in approved and "recipient_fingerprint" not in approved
    with pytest.raises(wise.WiseError) as exc:
        wise.approve_payout(payout_id=payout_id, approver="admin-2")
    assert exc.value.reason == "not_pending"


def test_rejecting_returns_the_rows_to_the_queue(wise_env, provider):
    fake, client = wise_env
    payout_id = _pending(provider, fake, client)
    wise.reject_payout(payout_id=payout_id, actor="admin-1", reason="wrong week")
    assert all(
        s["settlement_status"] == "queued" and s["wise_payout_id"] is None
        for s in _splits(provider["provider_id"])
    )


# ── Execution ────────────────────────────────────────────────────────


def test_an_unapproved_payout_is_never_sent(wise_env, provider):
    fake, client = wise_env
    _pending(provider, fake, client)
    assert wise.execute_approved_payouts(client=client) == {"created": 0, "unknown": 0, "review": 0}
    assert ("POST", "/v1/transfers") not in fake.calls


def test_an_approved_payout_becomes_one_unfunded_transfer(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    assert wise.execute_approved_payouts(client=client)["created"] == 1
    row = _one("SELECT * FROM wise_payouts WHERE id = %s", (payout["id"],))
    assert row["status"] == "awaiting_funding"
    assert row["fee_micros"] == 610_000
    transfer = fake.transfers[row["wise_transfer_id"]]
    assert transfer["customerTransactionId"] == str(row["customer_transaction_id"])
    assert transfer["reference"] == f"XCL-{payout['id']}"
    # Manual funding is the default: no payment was made, and no balance read.
    assert not any(path.endswith("/payments") for _m, path in fake.calls)
    # Running again creates nothing more.
    assert wise.execute_approved_payouts(client=client) == {"created": 0, "unknown": 0, "review": 0}
    assert len(fake.transfers) == 1
    # The rows are still not paid: nothing has been sent yet.
    assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"in_transit"}


def test_a_timeout_is_retried_with_the_same_id_and_never_pays_twice(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    fake.fail_next_transfer = "timeout"
    assert wise.execute_approved_payouts(client=client)["unknown"] == 1
    assert (
        _one("SELECT status FROM wise_payouts WHERE id = %s", (payout["id"],))["status"]
        == "unknown"
    )
    # Suppose Wise did create it before the connection dropped.
    row = _one("SELECT * FROM wise_payouts WHERE id = %s", (payout["id"],))
    fake._next += 1
    fake.transfers[fake._next] = {
        "id": fake._next,
        "targetAccount": row["wise_recipient_id"],
        "sourceCurrency": "CAD",
        "targetCurrency": "CAD",
        "sourceValue": 102.0,
        "targetValue": 102.0,
        "status": "incoming_payment_waiting",
        "reference": f"XCL-{row['id']}",
        "customerTransactionId": str(row["customer_transaction_id"]),
    }
    fake.by_customer_tx[str(row["customer_transaction_id"])] = fake._next
    _one(
        "UPDATE wise_payouts SET updated_at = clock_timestamp() - interval '2 hours' "
        "WHERE id = %s RETURNING id",
        (payout["id"],),
    )
    assert wise.execute_approved_payouts(client=client)["created"] == 1
    assert len(fake.transfers) == 1
    after = _one("SELECT status, wise_transfer_id FROM wise_payouts WHERE id = %s", (payout["id"],))
    assert (after["status"], after["wise_transfer_id"]) == ("awaiting_funding", fake._next)


def test_a_definite_refusal_stops_for_review_and_is_not_retried(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    fake.fail_next_transfer = "422"
    assert wise.execute_approved_payouts(client=client)["review"] == 1
    row = _one("SELECT status, error FROM wise_payouts WHERE id = %s", (payout["id"],))
    assert row["status"] == "needs_review" and "missing approval" in row["error"]
    assert "test-wise-token" not in row["error"]
    # Releasing a payout with no transfer on record needs an explicit statement.
    with pytest.raises(wise.WiseError) as exc:
        wise.release_payout(payout_id=payout["id"], actor="admin-1", client=client)
    assert exc.value.reason == "confirm_no_transfer"
    wise.release_payout(
        payout_id=payout["id"], actor="admin-1", confirm_no_transfer=True, client=client
    )
    assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"queued"}


def test_a_destination_edited_at_wise_after_approval_is_not_paid(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    rid = _one("SELECT wise_recipient_id FROM wise_payouts WHERE id = %s", (payout["id"],))
    fake.recipients[rid["wise_recipient_id"]]["hash"] = "f" * 64  # details changed at Wise
    assert wise.execute_approved_payouts(client=client)["review"] == 1
    assert ("POST", "/v1/transfers") not in fake.calls
    assert (
        _one("SELECT status FROM wise_payouts WHERE id = %s", (payout["id"],))["status"]
        == "needs_review"
    )


def test_a_transfer_for_the_wrong_amount_is_cancelled_and_held(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    fake.skew_amount = True
    payout = _approved_payout(provider, fake, client)
    assert wise.execute_approved_payouts(client=client)["review"] == 1
    row = _one(
        "SELECT status, error, wise_transfer_id FROM wise_payouts WHERE id = %s", (payout["id"],)
    )
    assert row["status"] == "needs_review" and "amount" in row["error"]
    assert fake.transfers[row["wise_transfer_id"]]["status"] == "cancelled"


def test_auto_fund_pays_from_the_balance_only_when_it_covers_the_payout(
    wise_env, provider, monkeypatch
):
    fake, client = wise_env
    fake.quoted(102.0)
    monkeypatch.setenv("XCELSIOR_WISE_AUTO_FUND", "1")
    fake.balance = "50.00"
    payout = _approved_payout(provider, fake, client)
    wise.execute_approved_payouts(client=client)
    row = _one("SELECT status, error FROM wise_payouts WHERE id = %s", (payout["id"],))
    assert row["status"] == "awaiting_funding" and "balance" in row["error"]
    assert not any(path.endswith("/payments") for _m, path in fake.calls)


# ── State sync, webhooks, release ────────────────────────────────────


def _sent_payout(provider, fake, client) -> dict:
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    wise.execute_approved_payouts(client=client)
    row = _one("SELECT * FROM wise_payouts WHERE id = %s", (payout["id"],))
    fake.transfers[row["wise_transfer_id"]]["status"] = "outgoing_payment_sent"
    wise.sync_payout(payout_id=row["id"], client=client)
    return row


def test_rows_are_paid_only_when_wise_says_the_money_was_sent(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    wise.execute_approved_payouts(client=client)
    row = _one("SELECT * FROM wise_payouts WHERE id = %s", (payout["id"],))
    fake.transfers[row["wise_transfer_id"]]["status"] = "processing"
    assert wise.sync_payout(payout_id=row["id"], client=client)["status"] == "funded"
    assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"in_transit"}
    fake.transfers[row["wise_transfer_id"]]["status"] = "outgoing_payment_sent"
    assert wise.sync_payout(payout_id=row["id"], client=client)["status"] == "sent"
    splits = _splits(provider["provider_id"])
    assert {s["settlement_status"] for s in splits} == {"paid"}
    assert all(s["settled_at"] is not None for s in splits)
    # A late read of an earlier state does not un-send it.
    fake.transfers[row["wise_transfer_id"]]["status"] = "funds_converted"
    assert wise.sync_payout(payout_id=row["id"], client=client)["status"] == "sent"


def test_a_bounce_unpays_the_rows_and_release_waits_for_the_refund(wise_env, provider):
    fake, client = wise_env
    row = _sent_payout(provider, fake, client)
    fake.transfers[row["wise_transfer_id"]]["status"] = "bounced_back"
    assert wise.reconcile_open_payouts(client=client)["checked"] == 1
    assert (
        _one("SELECT status FROM wise_payouts WHERE id = %s", (row["id"],))["status"]
        == "needs_review"
    )
    assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"manual_review"}
    # Not refunded yet: releasing now could pay twice.
    with pytest.raises(wise.WiseError) as exc:
        wise.release_payout(payout_id=row["id"], actor="admin-1", client=client)
    assert exc.value.reason == "money_not_returned"
    # No new payout can be batched for this provider while it is in review.
    assert wise.batch_queued_settlements(provider_id=provider["provider_id"])["created"] == []
    fake.transfers[row["wise_transfer_id"]]["status"] = "funds_refunded"
    wise.release_payout(payout_id=row["id"], actor="admin-1", client=client)
    splits = _splits(provider["provider_id"])
    assert all(s["settlement_status"] == "queued" and s["wise_payout_id"] is None for s in splits)


def test_a_bounce_that_wise_redelivers_is_paid_again_without_a_second_transfer(wise_env, provider):
    fake, client = wise_env
    row = _sent_payout(provider, fake, client)
    fake.transfers[row["wise_transfer_id"]]["status"] = "bounced_back"
    wise.sync_payout(payout_id=row["id"], client=client)
    assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"manual_review"}
    fake.transfers[row["wise_transfer_id"]]["status"] = "outgoing_payment_sent"
    assert wise.sync_payout(payout_id=row["id"], client=client)["status"] == "sent"
    assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"paid"}
    assert len(fake.transfers) == 1


def _event(transfer_id: int, state: str) -> bytes:
    return json.dumps(
        {
            "data": {
                "resource": {"type": "transfer", "id": transfer_id, "profile_id": PROFILE},
                "current_state": state,
                "previous_state": "processing",
                "occurred_at": "2026-11-10T12:00:00Z",
            },
            "subscription_id": str(uuid.uuid4()),
            "event_type": "transfers#state-change",
            "schema_version": "4.0.0",
            "sent_at": "2026-11-10T12:00:01Z",
        }
    ).encode()


def test_an_unsigned_or_forged_webhook_is_refused_and_not_recorded(wise_env, signing_key):
    body = _event(123, "outgoing_payment_sent")
    delivery = f"forged-{uuid.uuid4()}"
    for signature in ("", base64.b64encode(b"x" * 256).decode()):
        with pytest.raises(wise.WiseError) as exc:
            wise.handle_webhook(body=body, signature=signature, delivery_id=delivery)
        assert exc.value.status_code == 400
    assert _one("SELECT 1 AS x FROM wise_webhook_inbox WHERE delivery_id = %s", (delivery,)) is None


def test_a_webhook_is_recorded_once_and_its_claim_is_not_believed(wise_env, provider, signing_key):
    fake, client = wise_env
    fake.quoted(102.0)
    payout = _approved_payout(provider, fake, client)
    wise.execute_approved_payouts(client=client)
    row = _one("SELECT * FROM wise_payouts WHERE id = %s", (payout["id"],))
    # The delivery says "sent". Wise's API says the transfer is only processing.
    fake.transfers[row["wise_transfer_id"]]["status"] = "processing"
    body = _event(row["wise_transfer_id"], "outgoing_payment_sent")
    delivery = f"d-{uuid.uuid4()}"
    calls_before = len(fake.calls)
    first = wise.handle_webhook(body=body, signature=signing_key(body), delivery_id=delivery)
    assert first == {"ok": True, "outcome": "pending"}
    assert len(fake.calls) == calls_before  # recording a delivery calls nothing
    again = wise.handle_webhook(body=body, signature=signing_key(body), delivery_id=delivery)
    assert again["outcome"] == "duplicate"
    try:
        assert wise.drain_webhook_inbox(client=client)["applied"] >= 1
        assert (
            _one("SELECT status FROM wise_payouts WHERE id = %s", (row["id"],))["status"]
            == "funded"
        )
        assert {s["settlement_status"] for s in _splits(provider["provider_id"])} == {"in_transit"}
        stored = _one(
            "SELECT outcome, payload FROM wise_webhook_inbox WHERE delivery_id = %s", (delivery,)
        )
        assert stored["outcome"] == "applied"
        assert "subscription_id" not in stored["payload"]
    finally:
        _one("DELETE FROM wise_webhook_inbox WHERE delivery_id = %s RETURNING 1", (delivery,))


def test_a_webhook_for_someone_elses_transfer_is_ignored(wise_env, signing_key):
    _fake, client = wise_env
    body = _event(987654321, "outgoing_payment_sent")
    delivery = f"d-{uuid.uuid4()}"
    wise.handle_webhook(body=body, signature=signing_key(body), delivery_id=delivery)
    try:
        wise.drain_webhook_inbox(client=client)
        assert (
            _one("SELECT outcome FROM wise_webhook_inbox WHERE delivery_id = %s", (delivery,))[
                "outcome"
            ]
            == "ignored"
        )
    finally:
        _one("DELETE FROM wise_webhook_inbox WHERE delivery_id = %s RETURNING 1", (delivery,))


# ── Who may act ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "user, allowed",
    [
        ({"is_admin": True, "auth_type": "session", "session_type": "browser"}, True),
        ({"is_admin": True, "auth_type": "master_token"}, False),
        ({"is_admin": True, "auth_type": "api_key"}, False),
        ({"is_admin": True, "auth_type": "agent_api_key"}, False),
        ({"is_admin": True, "grant_type": "client_credentials"}, False),
        (
            {"is_admin": True, "auth_type": "oauth_access_token", "client_id": "some-connector"},
            False,
        ),
        ({"is_admin": False, "role": "provider", "auth_type": "session"}, False),
    ],
)
def test_only_a_signed_in_admin_may_direct_or_approve_payouts(monkeypatch, user, allowed):
    from fastapi import HTTPException

    import routes.wise as routes_wise

    monkeypatch.setattr(routes_wise, "_require_auth", lambda _request: user)
    if allowed:
        assert routes_wise._require_wise_admin(object()) is user
    else:
        with pytest.raises(HTTPException) as exc:
            routes_wise._require_wise_admin(object())
        assert exc.value.status_code == 403


def test_no_wise_route_accepts_an_amount():
    import routes.wise as routes_wise

    seen: set[str] = set()
    for route in routes_wise.router.routes:
        names = {p.name for p in route.dependant.query_params + route.dependant.path_params}
        for body in route.dependant.body_params:
            names |= set(getattr(body.field_info.annotation, "model_fields", {}) or {})
        assert not {n for n in names if "amount" in n or "micros" in n or n.endswith("_cad")}, (
            route.path
        )
        seen.update(names)
    # The check above is only worth something if it saw the request bodies.
    assert {"recipient_id", "account_holder_name", "confirm_no_transfer"} <= seen


# ── Funding position and the top-up notice ───────────────────────────


@pytest.fixture
def notices(monkeypatch):
    import scheduler

    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        scheduler,
        "send_email",
        lambda subject, body, to_email=None: sent.append((subject, body)) or True,
    )
    monkeypatch.setattr(wise, "_last_topup_notice", (0.0, 0))
    return sent


def test_funding_position_counts_what_is_approved_against_the_balance(wise_env, provider):
    fake, client = wise_env
    fake.balance = "0.00"
    before = wise.funding_position(client=client)
    _approved_payout(provider, fake, client)
    after = wise.funding_position(client=client)
    assert after["committed_micros"] - before["committed_micros"] == 102_000_000
    assert after["shortfall_micros"] == after["committed_micros"]
    fake.balance = "1000000.00"
    assert wise.funding_position(client=client)["shortfall_micros"] == 0


def test_a_payout_waiting_for_approval_is_not_a_shortfall_yet(wise_env, provider):
    fake, client = wise_env
    fake.balance = "0.00"
    before = wise.funding_position(client=client)
    _linked(provider, fake, client)
    _end_cooloff(provider["provider_id"])
    wise.batch_queued_settlements(provider_id=provider["provider_id"])
    after = wise.funding_position(client=client)
    assert after["committed_micros"] == before["committed_micros"]
    assert after["pending_approval_micros"] - before["pending_approval_micros"] == 102_000_000
    assert (
        after["shortfall_if_all_approved_micros"] - before["shortfall_if_all_approved_micros"]
        == 102_000_000
    )


def test_the_operator_is_told_once_how_much_to_add(wise_env, provider, notices):
    fake, client = wise_env
    fake.balance = "2.00"
    _approved_payout(provider, fake, client)
    first = wise.notify_topup_needed(client=client, now=1000.0)
    assert first["notified"] is True
    again = wise.notify_topup_needed(client=client, now=1100.0)
    assert again["notified"] is False
    assert len(notices) == 1
    subject, body = notices[0]
    assert wise.micros_to_amount(first["shortfall_micros"]) in subject
    # An amount, and nothing that names a provider or an account.
    assert provider["provider_id"] not in body and "recipient" not in body.lower()
    # Still short six hours later: said again.
    assert wise.notify_topup_needed(client=client, now=1000.0 + 6 * 3600 + 1)["notified"] is True


def test_a_larger_shortfall_is_reported_without_waiting(wise_env, provider, notices):
    fake, client = wise_env
    fake.balance = "50.00"
    _approved_payout(provider, fake, client)
    wise.notify_topup_needed(client=client, now=1000.0)
    fake.balance = "10.00"
    assert wise.notify_topup_needed(client=client, now=1010.0)["notified"] is True
    assert len(notices) == 2


def test_a_covered_balance_sends_nothing_and_resets_the_notice(wise_env, provider, notices):
    fake, client = wise_env
    fake.balance = "2.00"
    _approved_payout(provider, fake, client)
    wise.notify_topup_needed(client=client, now=1000.0)
    fake.balance = "1000000.00"
    assert wise.notify_topup_needed(client=client, now=1010.0)["notified"] is False
    fake.balance = "2.00"
    assert wise.notify_topup_needed(client=client, now=1020.0)["notified"] is True
    assert len(notices) == 2


def test_admin_status_shows_the_amount_to_add(wise_env, provider):
    fake, client = wise_env
    fake.balance = "2.00"
    _approved_payout(provider, fake, client)
    status = wise.rail_status(client=client)
    assert status["balance_cad"] == 2.0
    assert status["top_up_needed_cad"] == status["committed_cad"] - 2.0
    assert status["top_up_needed_cad"] >= 100.0


def test_an_unfunded_transfer_is_funded_after_the_balance_is_topped_up(
    wise_env, provider, monkeypatch
):
    fake, client = wise_env
    fake.quoted(102.0)
    monkeypatch.setenv("XCELSIOR_WISE_AUTO_FUND", "1")
    fake.balance = "50.00"
    payout = _approved_payout(provider, fake, client)
    wise.execute_approved_payouts(client=client)

    wise.reconcile_open_payouts(client=client)
    assert not any(path.endswith("/payments") for _m, path in fake.calls)

    fake.balance = "500.00"
    wise.reconcile_open_payouts(client=client)
    assert sum(path.endswith("/payments") for _m, path in fake.calls) == 1
    row = _one("SELECT status, error FROM wise_payouts WHERE id = %s", (payout["id"],))
    assert row["status"] == "funded" and not row["error"]


def test_manual_funding_never_pays_from_the_balance_by_itself(wise_env, provider):
    fake, client = wise_env
    fake.quoted(102.0)
    fake.balance = "500.00"
    payout = _approved_payout(provider, fake, client)
    wise.execute_approved_payouts(client=client)
    wise.reconcile_open_payouts(client=client)
    assert not any(path.endswith("/payments") for _m, path in fake.calls)
    assert (
        _one("SELECT status FROM wise_payouts WHERE id = %s", (payout["id"],))["status"]
        == "awaiting_funding"
    )
