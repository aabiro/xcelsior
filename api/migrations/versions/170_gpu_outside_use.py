"""GPU outside use: compute processes a host runs on a GPU outside Xcelsior.

* `host_gpu_devices.outside_processes` / `outside_vram_mb` — what the host's
  agent (2.4.0+) last observed on the device: how many compute processes it
  did not start, and how much memory they hold. Written on every observation
  by `control_plane.physical_inventory.sync_physical_inventory`; zero when
  there are none, or when the agent does not report it.

A device with outside processes is counted but never free: the snapshot,
the reservation and preemption planning all skip it
(`control_plane.scheduler.filters.device_outside_use`). Before this the
tower's RTX 3060 was offered as a free 12 GB GPU while the owner's own work
held 2.9 GB of it, and its benchmark measured what was left and de-verified it.

Expand only: two NOT NULL DEFAULT 0 columns, which PostgreSQL adds without a
table rewrite, and which the previous release ignores. Placement domain. One
idempotent unit through `lock_safe`; `downgrade` drops both columns.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "170"
down_revision = "169"
branch_labels = None
depends_on = None


def _add(conn):
    conn.execute(
        text(
            "ALTER TABLE host_gpu_devices"
            " ADD COLUMN IF NOT EXISTS outside_processes INTEGER NOT NULL DEFAULT 0"
            " CHECK (outside_processes >= 0),"
            " ADD COLUMN IF NOT EXISTS outside_vram_mb INTEGER NOT NULL DEFAULT 0"
            " CHECK (outside_vram_mb >= 0)"
        )
    )


def _drop(conn):
    conn.execute(
        text(
            "ALTER TABLE host_gpu_devices"
            " DROP COLUMN IF EXISTS outside_processes,"
            " DROP COLUMN IF EXISTS outside_vram_mb"
        )
    )


def upgrade():
    apply_in_own_transactions(
        [("host_gpu_devices outside use", _add)],
        tables=("host_gpu_devices",),
    )


def downgrade():
    apply_in_own_transactions(
        [("host_gpu_devices outside use", _drop)],
        tables=("host_gpu_devices",),
    )
