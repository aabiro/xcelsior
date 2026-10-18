"""Normalize GPU model names in hosts/jobs payloads + index sessions for filtered query.

Two concerns in one migration:

1. GPU name normalization
   The worker agent canonicalizes nvidia-smi names via _NVIDIA_SMI_NAME_MAP, but
   older hosts (or manually-registered entries) may carry dirty strings like
   "L4OS", "NVIDIA L40S", "NVIDIA GeForce RTX 4090", etc. in their JSONB payload.
   This migration normalizes the most common variants in place so catalog queries,
   pricing lookups, and the marketing fleet page all see consistent names.

2. Sessions composite index
   list_user_sessions() now filters by session_type IN ('browser','legacy') and
   client_id to exclude machine principals. The existing idx_sessions_session_type
   only covers session_type; a composite index on (email, session_type) makes the
   filtered query a single index seek instead of a full scan.

Revision ID: 042
Revises: 041
Create Date: 2026-06-22
"""

from alembic import op

revision = "042"
down_revision = "041"
branch_labels = None
depends_on = None

# ── GPU model aliases: (dirty_value, canonical_value) ─────────────────────────
# Ordered from most-specific to least-specific so substring replacements don't
# collide. Only strings actually observed from nvidia-smi or manual entry are
# listed; canonical names (e.g. "RTX 4090") are left untouched.
_GPU_ALIASES = [
    # L40S mislabels — must come before plain "L4" entries
    ("L4OS",          "L40S"),
    ("l4os",          "L40S"),
    ("L4 OS",         "L40S"),
    ("NVIDIA L40S",   "L40S"),
    ("NVIDIA L40",    "L40"),
    ("NVIDIA L4",     "L4"),
    # Full nvidia-smi strings that slip through (host_metadata edge cases)
    ("NVIDIA H200",                "H200"),
    ("NVIDIA H100 80GB HBM3",      "H100"),
    ("NVIDIA H100 PCIe",           "H100"),
    ("NVIDIA H100 NVL",            "H100 NVL"),
    ("NVIDIA H100",                "H100"),
    ("NVIDIA A100-SXM4-80GB",      "A100"),
    ("NVIDIA A100-SXM4-40GB",      "A100"),
    ("NVIDIA A100-PCIE-40GB",      "A100"),
    ("NVIDIA A100 80GB PCIe",      "A100"),
    ("NVIDIA A40",                 "A40"),
    ("NVIDIA A30",                 "A30"),
    ("NVIDIA A10",                 "A10"),
    ("NVIDIA A16",                 "A16"),
    ("Tesla T4",                   "T4"),
    ("Tesla V100-SXM2-32GB",       "V100"),
    ("Tesla V100-PCIE-32GB",       "V100"),
    ("Tesla V100-SXM2-16GB",       "V100"),
    ("Tesla V100-PCIE-16GB",       "V100"),
    ("NVIDIA GeForce RTX 5090",    "RTX 5090"),
    ("NVIDIA GeForce RTX 5080",    "RTX 5080"),
    ("NVIDIA GeForce RTX 5070 Ti", "RTX 5070 Ti"),
    ("NVIDIA GeForce RTX 5070",    "RTX 5070"),
    ("NVIDIA GeForce RTX 5060 Ti", "RTX 5060 Ti"),
    ("NVIDIA GeForce RTX 5060",    "RTX 5060"),
    ("NVIDIA GeForce RTX 4090",    "RTX 4090"),
    ("NVIDIA GeForce RTX 4080 SUPER", "RTX 4080 Super"),
    ("NVIDIA GeForce RTX 4080",    "RTX 4080"),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", "RTX 4070 Ti Super"),
    ("NVIDIA GeForce RTX 4070 Ti", "RTX 4070 Ti"),
    ("NVIDIA GeForce RTX 4070 SUPER", "RTX 4070 Super"),
    ("NVIDIA GeForce RTX 4070",    "RTX 4070"),
    ("NVIDIA GeForce RTX 4060 Ti", "RTX 4060 Ti"),
    ("NVIDIA GeForce RTX 4060",    "RTX 4060"),
    ("NVIDIA GeForce RTX 3090 Ti", "RTX 3090 Ti"),
    ("NVIDIA GeForce RTX 3090",    "RTX 3090"),
    ("NVIDIA GeForce RTX 3080 Ti", "RTX 3080 Ti"),
    ("NVIDIA GeForce RTX 3080",    "RTX 3080"),
    ("NVIDIA GeForce RTX 3070 Ti", "RTX 3070 Ti"),
    ("NVIDIA GeForce RTX 3070",    "RTX 3070"),
    ("NVIDIA GeForce RTX 3060 Ti", "RTX 3060 Ti"),
    ("NVIDIA GeForce RTX 3060",    "RTX 3060"),
    ("NVIDIA GeForce RTX 2080 Ti", "RTX 2080 Ti"),
    ("NVIDIA GeForce RTX 2080 SUPER", "RTX 2080 Super"),
    ("NVIDIA GeForce RTX 2080",    "RTX 2080"),
    ("NVIDIA GeForce RTX 2070 SUPER", "RTX 2070 Super"),
    ("NVIDIA GeForce RTX 2070",    "RTX 2070"),
    ("NVIDIA GeForce RTX 2060 SUPER", "RTX 2060 Super"),
    ("NVIDIA GeForce RTX 2060",    "RTX 2060"),
    ("NVIDIA RTX 6000 Ada Generation", "RTX 6000 Ada"),
    ("NVIDIA RTX 5000 Ada Generation", "RTX 5000 Ada"),
    ("NVIDIA RTX 4000 Ada Generation", "RTX 4000 Ada"),
    ("NVIDIA RTX A6000",           "RTX A6000"),
    ("NVIDIA RTX A5000",           "RTX A5000"),
    ("NVIDIA RTX A4000",           "RTX A4000"),
    ("AMD Instinct MI300X",        "MI300X"),
    ("AMD Instinct MI250X",        "MI250X"),
    ("AMD Instinct MI210",         "MI210"),
    ("AMD Radeon RX 7900 XTX",     "RX 7900 XTX"),
    ("AMD Radeon RX 7900 XT",      "RX 7900 XT"),
]


def upgrade() -> None:
    # ── 1. Normalize gpu_model inside hosts.payload JSONB ────────────────────
    for dirty, canonical in _GPU_ALIASES:
        op.execute(f"""
            UPDATE hosts
            SET payload = jsonb_set(payload, '{{gpu_model}}', to_jsonb('{canonical}'::text))
            WHERE payload->>'gpu_model' = '{dirty}'
        """)

    # Also normalize gpu_model in individual gpu_specs objects if present
    for dirty, canonical in _GPU_ALIASES:
        op.execute(f"""
            UPDATE hosts
            SET payload = jsonb_set(
                    payload,
                    '{{gpu_specs,model}}',
                    to_jsonb('{canonical}'::text)
                )
            WHERE payload->'gpu_specs'->>'model' = '{dirty}'
        """)

    # ── 2. Normalize gpu_model inside jobs.payload JSONB ────────────────────
    for col in ("gpu_model", "gpu_type", "host_gpu_model"):
        for dirty, canonical in _GPU_ALIASES:
            op.execute(f"""
                UPDATE jobs
                SET payload = jsonb_set(payload, '{{{col}}}', to_jsonb('{canonical}'::text))
                WHERE payload->>'{col}' = '{dirty}'
            """)

    # ── 3. Composite index for filtered list_user_sessions query ─────────────
    # Replaces the single-column idx_sessions_session_type with a covering
    # index that satisfies: WHERE email = ? AND session_type IN (...)
    # No partial predicate: EXTRACT(EPOCH FROM NOW()) is not immutable and
    # would cause index creation to fail in Postgres.
    op.execute("DROP INDEX IF EXISTS idx_sessions_session_type")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_email_type "
        "ON sessions (email, session_type, last_active DESC)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_sessions_email_type")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_session_type ON sessions (session_type)"
    )
    # GPU name normalization is not reversible (original dirty strings are gone).
    # A downgrade here would require a backup; we leave it as a no-op.
