"""Bind serverless endpoint creation to one durable action plan.

Revision ID: 076
Revises: 075
"""

from typing import Sequence, Union
from alembic import op

revision: str = "076"
down_revision: Union[str, None] = "075"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE serverless_endpoints "
        "ADD COLUMN IF NOT EXISTS action_plan_id UUID REFERENCES action_plans(plan_id)"
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_serverless_endpoint_action_plan "
        "ON serverless_endpoints(action_plan_id) WHERE action_plan_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_serverless_endpoint_action_plan")
    op.execute("ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS action_plan_id")
