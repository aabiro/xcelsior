"""A business account's display name, website, tagline and logo.

`users.business_name` (144) is the legal name on invoices. A business also has
a name and a mark it wants to be seen as: these are that, shown as the
account's badge in the dashboard.

`users` gains `business_display_name`, `business_website` and
`business_tagline`, all text and blank by default, so every existing row is
valid and a release that does not know the columns keeps working.

`business_logos` holds one logo per account, beside `user_avatars` (051) and
for the same reason: the bytes must survive a redeploy. The image stored is the
one the API re-encoded (a 512 px square, metadata removed), never the upload.
`tile` records which background the logo needs behind it: `light` or `dark`
for transparent artwork, `none` for an opaque image.

Each unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py).
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "160"
down_revision = "159"
branch_labels = None
depends_on = None


# Written out in full, not built from a list: no identifier is interpolated
# into DDL (tests/test_sql_injection_guard.py). A constant default makes each
# ADD COLUMN a catalogue-only change, so `users` is not rewritten.
def _user_columns(conn):
    conn.execute(
        text(
            """
            ALTER TABLE users
              ADD COLUMN IF NOT EXISTS business_display_name TEXT NOT NULL DEFAULT '',
              ADD COLUMN IF NOT EXISTS business_website TEXT NOT NULL DEFAULT '',
              ADD COLUMN IF NOT EXISTS business_tagline TEXT NOT NULL DEFAULT ''
            """
        )
    )


def _logos(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS business_logos (
                user_id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                data BYTEA NOT NULL,
                tile TEXT NOT NULL DEFAULT 'none',
                updated_at DOUBLE PRECISION NOT NULL DEFAULT 0,
                CONSTRAINT business_logos_tile_check CHECK (tile IN ('none', 'light', 'dark'))
            )
            """
        )
    )


def _drop_logos(conn):
    conn.execute(text("DROP TABLE IF EXISTS business_logos"))


def _drop_user_columns(conn):
    conn.execute(
        text(
            """
            ALTER TABLE users
              DROP COLUMN IF EXISTS business_tagline,
              DROP COLUMN IF EXISTS business_website,
              DROP COLUMN IF EXISTS business_display_name
            """
        )
    )


def upgrade():
    apply_in_own_transactions(
        [
            ("users business profile columns", _user_columns),
            ("business_logos", _logos),
        ],
        tables=("users", "business_logos"),
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("drop business_logos", _drop_logos),
            ("drop users business profile columns", _drop_user_columns),
        ],
        tables=("users", "business_logos"),
    )
