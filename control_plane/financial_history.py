"""Qualified duplicate-volume credits; source cycles and debits remain intact."""

from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
import time

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


def evidence_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def qualify_cluster(conn, job_id, period_start):
    """Only identical-start, near-identical-end charges with matching debits qualify.

    Timing alone never maps an individual cycle to a debit. Qualification is
    at the whole cluster: same owner/rate/amount, exact cardinality, contiguous
    ledger balances and the same rounded union price. Ambiguity refuses it.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cycles = cur.execute(
            """SELECT cycle_id,job_id,customer_id,resource_type,period_start,period_end,
                      duration_seconds,rate_per_hour,amount_micros,status,created_at
                 FROM billing_cycles WHERE job_id=%s AND resource_type='volume'
                  AND period_start=%s ORDER BY period_end,cycle_id""",
            (job_id, period_start),
        ).fetchall()
        charged = [row for row in cycles if row["status"] == "charged"]
        if not 2 <= len(charged) <= 20 or len(cycles) > 40:
            raise ValueError("a bounded duplicate charged cluster is required")
        owner = charged[0]["customer_id"]
        amount = charged[0]["amount_micros"]
        rate = charged[0]["rate_per_hour"]
        ends = [row["period_end"] for row in charged]
        if not owner or type(amount) is not int or amount <= 0:
            raise ValueError("recorded financial identity and positive integer amount required")
        if any(
            not math.isfinite(float(value))
            for row in cycles
            for value in (
                row["period_start"],
                row["period_end"],
                row["duration_seconds"],
                row["rate_per_hour"],
                row["created_at"],
            )
        ):
            raise ValueError("nonfinite historical interval")
        if (
            max(ends) - min(ends) > 1
            or max(row["created_at"] for row in cycles) - min(row["created_at"] for row in cycles)
            > 2
        ):
            raise ValueError("intervals do not form one duplicate billing tick")
        for row in cycles:
            if (
                row["customer_id"] != owner
                or row["amount_micros"] != amount
                or row["rate_per_hour"] != rate
                or row["status"] not in {"charged", "failed"}
                or row["period_end"] <= period_start
                or abs(row["duration_seconds"] - (row["period_end"] - period_start)) > 0.001
                or abs(row["created_at"] - row["period_end"]) > 2
            ):
                raise ValueError("inconsistent recorded cluster evidence")
        union_amount = int(
            (
                (Decimal(str(max(ends))) - Decimal(str(period_start)))
                * Decimal(str(rate))
                / Decimal(3600)
            ).quantize(Decimal(".0001"), rounding=ROUND_HALF_EVEN)
            * 1_000_000
        )
        if union_amount != amount:
            raise ValueError("rounded union price differs from the recorded charge")
        transactions = cur.execute(
            """SELECT tx_id,job_id,customer_id,tx_type,amount_micros,balance_after_micros,created_at
                 FROM wallet_transactions WHERE job_id=%s AND created_at BETWEEN %s AND %s
                 ORDER BY created_at,tx_id""",
            (
                job_id,
                min(row["created_at"] for row in cycles) - 2,
                max(row["created_at"] for row in cycles) + 2,
            ),
        ).fetchall()
        if len(transactions) != len(charged) or any(
            row["tx_type"] != "charge"
            or row["customer_id"] != owner
            or row["amount_micros"] != -amount
            for row in transactions
        ):
            raise ValueError("charged cycles do not have an unambiguous matching debit cluster")
        if any(
            later["balance_after_micros"] != earlier["balance_after_micros"] - amount
            for earlier, later in zip(transactions, transactions[1:])
        ):
            raise ValueError("debit cluster contains intervening wallet changes")
        if cur.execute(
            "SELECT 1 FROM wallet_transactions WHERE job_id=%s AND tx_type IN ('refund','credit') AND tx_id NOT LIKE 'TX-history-%%' LIMIT 1",
            (job_id,),
        ).fetchone():
            raise ValueError("existing historical job credits require separate disposition")
        evidence = {
            "policy_version": "volume_overlap/v1",
            "job_id": job_id,
            "customer_id": owner,
            "period_start": period_start,
            "cycles": cycles,
            "transactions": transactions,
            "union_amount_micros": union_amount,
        }
        digest = evidence_hash(evidence)
        key = "volume_overlap:" + evidence_hash({"job_id": job_id, "period_start": period_start})
        return {
            "disposition_key": key,
            "evidence_sha256": digest,
            "credit_micros": (len(charged) - 1) * amount,
            "evidence": evidence,
        }


def apply_cluster(conn, plan, *, max_credit_micros):
    """Caller owns commit; ledger, balance and immutable receipt are all-or-none."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (plan["disposition_key"],)
        )
        prior = cur.execute(
            "SELECT evidence_sha256,credit_tx_id,credit_micros FROM billing_history_dispositions WHERE disposition_key=%s",
            (plan["disposition_key"],),
        ).fetchone()
        if prior:
            if prior["evidence_sha256"] != plan["evidence_sha256"]:
                raise ValueError("disposition replay has different evidence")
            return {**prior, "replayed": True}
        owner = plan["evidence"]["customer_id"]
        wallet = cur.execute(
            "SELECT balance_micros FROM wallets WHERE customer_id=%s FOR UPDATE", (owner,)
        ).fetchone()
        if wallet is None:
            raise ValueError("recorded wallet is missing")
        # Lock the exact evidence rows after the wallet, matching normal debit
        # order. Any change between preview and application invalidates the plan.
        cur.execute(
            "SELECT cycle_id FROM billing_cycles WHERE cycle_id=ANY(%s) ORDER BY cycle_id FOR UPDATE",
            ([row["cycle_id"] for row in plan["evidence"]["cycles"]],),
        ).fetchall()
        cur.execute(
            "SELECT tx_id FROM wallet_transactions WHERE tx_id=ANY(%s) ORDER BY tx_id FOR UPDATE",
            ([row["tx_id"] for row in plan["evidence"]["transactions"]],),
        ).fetchall()
        qualified = qualify_cluster(
            conn, plan["evidence"]["job_id"], plan["evidence"]["period_start"]
        )
        if qualified != plan:
            raise ValueError("financial evidence changed since qualification")
        credit = qualified["credit_micros"]
        if not 0 < credit <= max_credit_micros:
            raise ValueError("credit exceeds the explicit adjustment budget")
        ledger = cur.execute(
            "SELECT COALESCE(sum(amount_micros),0) AS balance FROM wallet_transactions WHERE customer_id=%s",
            (owner,),
        ).fetchone()["balance"]
        if ledger != wallet["balance_micros"]:
            raise ValueError("wallet does not reconcile with its signed ledger")
        tx_id = "TX-history-" + evidence_hash(plan["disposition_key"])
        balance = wallet["balance_micros"] + credit
        cur.execute(
            "UPDATE wallets SET balance_micros=%s,total_refunded_micros=COALESCE(total_refunded_micros,0)+%s,updated_at=%s WHERE customer_id=%s",
            (balance, credit, time.time(), owner),
        )
        cur.execute(
            """INSERT INTO wallet_transactions(tx_id,customer_id,tx_type,amount_micros,balance_after_micros,description,job_id,created_at)
            VALUES(%s,%s,'refund',%s,%s,'Qualified duplicate historical volume interval',%s,%s)""",
            (tx_id, owner, credit, balance, plan["evidence"]["job_id"], time.time()),
        )
        cur.execute(
            """INSERT INTO billing_history_dispositions(disposition_key,tenant_id,job_id,policy_version,disposition,evidence,evidence_sha256,credit_micros,credit_tx_id,created_by)
            VALUES(%s,%s,%s,'volume_overlap/v1','credited_duplicate_interval',%s,%s,%s,%s,current_user)""",
            (
                plan["disposition_key"],
                owner,
                plan["evidence"]["job_id"],
                Jsonb(plan["evidence"]),
                plan["evidence_sha256"],
                credit,
                tx_id,
            ),
        )
        return {"credit_tx_id": tx_id, "credit_micros": credit, "replayed": False}
