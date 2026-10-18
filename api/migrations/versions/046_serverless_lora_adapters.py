"""Add multi-LoRA adapter config to serverless endpoints.

Lets a preset (vLLM) endpoint register one or more LoRA adapters alongside its
base model. vLLM natively multiplexes LoRA adapters on a single running
engine (--enable-lora --lora-modules name=source ...); requests select an
adapter by name in the OpenAI-compatible "model" field.

Revision ID: 046
"""

from alembic import op

revision = "046"
down_revision = "045"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE serverless_endpoints "
        "ADD COLUMN IF NOT EXISTS lora_adapters JSONB NOT NULL DEFAULT '[]'::jsonb"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE serverless_endpoints DROP COLUMN IF EXISTS lora_adapters")
