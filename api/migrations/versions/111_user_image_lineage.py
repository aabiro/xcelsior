"""What a snapshot was built from. Half of Gate P7's lineage clause.

Gate P7: *"a sweep of N nodes from one snapshot is byte-identical in
environment; **a snapshot records its lineage**"* — and the plan spells lineage
out as *"what it was built from, when, by which run"*.

Three of those four were already recorded on `user_images`: `created_at` is the
when, `source_job_id` is the run, and `host_id` is where it ran. **What it was
built from was not.** A snapshot is `docker commit` over a running container,
so the resulting image is a *diff* on top of whatever base the job launched
with — and without that base recorded, the row says which run produced the
image but not what the image actually contains beneath the commit.

That is the half of provenance an audit needs. "Which run built this" answers a
question about history; "what was it built from" answers a question about
contents, and only the second tells you whether a CVE in a base image is inside
a snapshot someone is still launching from.

## Nullable, with no backfill

Every `user_images` row that exists today was created without this, and the
honest value for those is unknown. A backfill would have to infer the base from
the job — which is exactly the inference this column exists to stop people
making — so old rows keep `NULL` and mean it.
"""

from alembic import op

revision = "111"
down_revision = "110"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE user_images ADD COLUMN IF NOT EXISTS base_image_ref TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE user_images DROP COLUMN IF EXISTS base_image_ref")
