"""AI Assistant: conversations, messages, confirmations, and document chunks

Adds tables for the Xcel AI assistant feature:
- ai_conversations: per-user conversation sessions with token accounting
- ai_messages: messages with tool call/result tracking
- ai_confirmations: pending write-action approvals
- ai_docs: BM25 full-text search over documentation chunks

Revision ID: 006
Revises: 005
Create Date: 2026-03-31
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── AI Conversations ─────────────────────────────────────────────
    op.create_table(
        "ai_conversations",
        sa.Column("conversation_id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), server_default=""),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.Float(), nullable=False),
        sa.Column("message_count", sa.Integer(), server_default="0"),
        sa.Column("total_input_tokens", sa.Integer(), server_default="0"),
        sa.Column("total_output_tokens", sa.Integer(), server_default="0"),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), server_default="{}"),
    )
    op.create_index(
        "idx_ai_conv_user",
        "ai_conversations",
        ["user_id", sa.text("updated_at DESC")],
    )

    # ── AI Messages ──────────────────────────────────────────────────
    op.create_table(
        "ai_messages",
        sa.Column("message_id", sa.Text(), primary_key=True),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), server_default=""),
        sa.Column("tool_name", sa.Text(), server_default=""),
        sa.Column("tool_input", sa.dialects.postgresql.JSONB(), server_default="{}"),
        sa.Column("tool_output", sa.dialects.postgresql.JSONB(), server_default="{}"),
        sa.Column("tokens_in", sa.Integer(), server_default="0"),
        sa.Column("tokens_out", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.Float(), nullable=False),
    )
    op.create_index(
        "idx_ai_msg_conv",
        "ai_messages",
        ["conversation_id", "created_at"],
    )
    op.create_foreign_key(
        "fk_ai_msg_conv",
        "ai_messages",
        "ai_conversations",
        ["conversation_id"],
        ["conversation_id"],
        ondelete="CASCADE",
    )

    # ── AI Confirmations (pending write-action approvals) ────────────
    op.create_table(
        "ai_confirmations",
        sa.Column("confirmation_id", sa.Text(), primary_key=True),
        sa.Column("conversation_id", sa.Text(), nullable=False),
        sa.Column("user_id", sa.Text(), nullable=False),
        sa.Column("tool_name", sa.Text(), nullable=False),
        sa.Column("tool_args", sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), server_default="pending"),
        sa.Column("created_at", sa.Float(), nullable=False),
        sa.Column("resolved_at", sa.Float(), server_default="0"),
    )

    # ── AI Docs (BM25 full-text search) ──────────────────────────────
    op.execute("""
        CREATE TABLE ai_docs (
            doc_id SERIAL PRIMARY KEY,
            source TEXT NOT NULL DEFAULT '',
            chunk TEXT NOT NULL DEFAULT '',
            tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', chunk)) STORED
        )
    """)
    op.execute("CREATE INDEX idx_ai_docs_tsv ON ai_docs USING GIN (tsv)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS ai_docs CASCADE")
    op.drop_table("ai_confirmations")
    op.drop_table("ai_messages")
    op.drop_table("ai_conversations")
