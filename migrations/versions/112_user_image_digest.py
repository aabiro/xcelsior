"""The manifest digest of a snapshot. Gate P7's sweep is unprovable without it.

Gate P7: *"a sweep of N nodes from one snapshot is **byte-identical** in
environment"*.

`_build_image_ref` returns `{registry}/{slug}/{name}:{tag}` — a **mutable tag**.
Launching N containers from a tag establishes that N containers were asked for
the same *name*, not that they received the same bytes. A tag can be re-pushed
between the first launch and the last, and nothing in the resulting sweep would
show it. So the clause is not merely unasserted from a tag; it is unprovable in
principle.

`image_digest` holds `repo@sha256:…`, captured by the worker with `docker
inspect` immediately after the push that produced it — the only moment anything
knows the digest belongs to *those* bytes. Resolving the tag from the registry
later answers "what does this tag point at now", which is a different question
and the wrong one.

## Nullable, no backfill, no default

A snapshot that predates this genuinely has no recorded digest, and so does one
whose push succeeded while the inspect failed. `NULL` means unknown and every
consumer must keep treating it that way: a sweep that cannot pin a digest should
refuse to claim byte-identity rather than fall back to the tag, because falling
back is exactly the substitution that makes the claim empty.
"""

from alembic import op

revision = "112"
down_revision = "111"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE user_images ADD COLUMN IF NOT EXISTS image_digest TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE user_images DROP COLUMN IF EXISTS image_digest")
