"""Clear region labels that defaults stamped instead of a person choosing them.

Serverless endpoints, inference endpoints and volumes were created with
``ca-east`` whenever the request named no region (model defaults, the
``DEFAULT 'ca-east'`` columns from 010 and 037, and picker defaults), and GPU
offers inherited the hosts' default labels. None of these say where anything
is, and placement matches region exactly, so a stamped label pins work to a
place no located host will ever report. An empty region means "any region".

Only the broad pre-ISO labels are cleared. A region a person picked from real
inventory (an ISO subdivision such as ``ca-on``) is kept. Hosts are left alone:
their location is confirmed by the owner, and a live older release restamps a
host's label on its next heartbeat during the blue-green window.

``downgrade`` restores the ``ca-east`` column defaults. It cannot restore the
cleared labels, which carried no information.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "151"
down_revision = "150"
branch_labels = None
depends_on = None

#: Every label the retired region pickers and defaults could produce.
LEGACY_REGIONS = (
    "ca-east",
    "ca-central",
    "ca-west",
    "us-east",
    "us-west",
    "us-central",
    "eu-west",
    "eu-central",
    "eu-uk",
    "eu-nordics",
    "ap-southeast",
    "ap-northeast",
    "ap-south",
)
#: Tables whose column default stamped ``ca-east``.
DEFAULTED = ("serverless_endpoints", "inference_endpoints")
TABLES = ("serverless_endpoints", "inference_endpoints", "volumes", "gpu_offers")


def _clear(table):
    def run(conn):
        if table in DEFAULTED:
            conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN region SET DEFAULT ''"))
        conn.execute(
            text(f"UPDATE {table} SET region = '' WHERE lower(region) = ANY(:legacy)"),
            {"legacy": list(LEGACY_REGIONS)},
        )

    return run


def _restore_default(table):
    def run(conn):
        conn.execute(text(f"ALTER TABLE {table} ALTER COLUMN region SET DEFAULT 'ca-east'"))

    return run


def upgrade():
    apply_in_own_transactions([(table, _clear(table)) for table in TABLES], tables=TABLES)


def downgrade():
    apply_in_own_transactions(
        [(table, _restore_default(table)) for table in DEFAULTED], tables=DEFAULTED
    )
