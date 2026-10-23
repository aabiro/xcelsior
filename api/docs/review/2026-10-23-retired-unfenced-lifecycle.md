# Retire unfenced lifecycle execution

Host `start_container`, `pause_container` and `reset_container` directives,
the historical-container stop classifier, their queue guards and Docker
handlers, and the periodic unfenced stopped-job redelivery loop are removed.
Old queued directives receive a permanent unsupported-command NACK. Instance
reset requests a native fenced restart; the next attempt creates its workspace
from the admitted workload and preserves named volumes. API lifecycle docs now
describe fresh-attempt behavior and return conflict responses for missing
native admission/attempt authority.

Migration 136 archives only `reconcile_paused_stopped`, checks the exact record
and refuses a live task claim, then removes that schedule. A constraint prevents
old writers from recreating it. Downgrade restores its exact recorded row and
retains the archive. Runtime roles can read, never rewrite, that evidence.
Deploy the maintenance writer without the handler before applying 136.

Stopped-job JSON observations no longer authorize storage charges. Normalized
volume billing remains the storage path. Suspended-wallet and failed-topup
target selection use admitted identity, and the lifecycle service validates the
hash again. Depletion estimates use the attempt's captured billing quote;
missing prices are explicitly incomplete rather than fabricated from current
host observations. No historical money rows are deleted or repriced.

The native idle-termination guard checks `gpu_device_allocations`, the native
device authority. A regression with an orphaned active device allocation proves
that removing a job pointer and settling its lease/attempt does not bypass it.

`native-host-wallet-qualified`: **309 passed**, no errors, failures or skips.
It includes real PostgreSQL admission, lifecycle, orphaned allocation, wallet
drift, reset, retirement/rollback, archive permissions and existing worker
directive checks. All twelve worker distribution signatures verify. See the
adjacent JSON for the disposable-cluster manifest and protected-file hashes.
Earlier failed runs remain diagnostic evidence; they are not passing receipts.

The operator-only stale-container cleanup primitive remains separate from
instance execution. Its repair policies stay `report_only`; removing instance
compatibility does not enable remediation or authorize a host cleanup.

Source is qualified by the checks above; final schema/inventory and combined
regression, production image handoff, dedicated-role handoff and real worker
acceptance remain. Production is still schema 125. No deployment is claimed.
