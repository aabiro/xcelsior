"""Serverless endpoint spend limit (Track B B3.2).

Blueprint §4.2 / §15.3: an inference call must never require a human approval
click. Instead each endpoint carries a server-enforced spend ceiling; when the
endpoint's accrued cost reaches it, invocations are denied with a typed problem
rather than approved case by case or allowed to run up an unbounded bill.

`spend_limit_cad` is nullable — NULL means "no endpoint cap" (the wallet
preflight and client budgets still apply). Expand-only and additive; nothing
enforces it until the invocation path lands (B3.2 enforcement).

Revision ID: 071
Revises: 070
Create Date: 2026-07-23
"""

from typing import Sequence, Union

from alembic import op

revision: str = "071"
down_revision: Union[str, None] = "070"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE serverless_endpoints ADD COLUMN IF NOT EXISTS spend_limit_cad double precision"
    )
    # A non-negative ceiling when set; NULL = uncapped. NOT VALID → VALIDATE is
    # trivial (every existing row is NULL) and keeps the add lock-light.
    op.execute(
        """
        ALTER TABLE serverless_endpoints
            ADD CONSTRAINT ck_serverless_endpoints_spend_limit
            CHECK (spend_limit_cad IS NULL OR spend_limit_cad >= 0)
            NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE serverless_endpoints VALIDATE CONSTRAINT ck_serverless_endpoints_spend_limit"
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE serverless_endpoints DROP CONSTRAINT IF EXISTS ck_serverless_endpoints_spend_limit"
    )
    op.execute("ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS spend_limit_cad")
