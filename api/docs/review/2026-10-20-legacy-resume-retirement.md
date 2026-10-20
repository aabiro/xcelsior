# Legacy resume safeguards and retirement inventory — 2026-10-20

The twelve production stopped jobs are not twelve saved containers awaiting a
simple resume. Read-only checks found no matching containers on their assigned
Docker hosts. One job still references an NFS volume whose metadata says
`available`, while its expected storage paths were absent. No job was restarted,
requeued, terminated, deleted or relabeled during this audit.

## Physical and metadata inventory

The workstation identity was verified through its running worker's host-ID
setting and Docker daemon name. Tower's mesh identity and daemon name were
verified independently. The inventory checked each expected container name plus
all container names and Xcelsior job labels, including differently named attempt
containers. Both hosts returned complete observations; daemon/permission errors
were not treated as absence.

| Stopped job | Assigned host | Matching container | Managed volume hint |
| --- | --- | --- | --- |
| `07b3353b` | aaryn-tuf-rtx2060 | absent | `vol-36592ad0362e` |
| `177a3447` | aaryn-tuf-rtx2060 | absent | none |
| `19ee0dee` | aaryn-tuf-rtx2060 | absent | none |
| `1c6eb4c0` | aaryn-tuf-rtx2060 | absent | none |
| `2b9e67b8` | aaryn-tuf-rtx2060 | absent | none |
| `3dadeca7` | aaryn-tuf-rtx2060 | absent | none |
| `4f8a1562` | aaryn-tuf-rtx2060 | absent | none |
| `68298d8b` | tower-rtx3060 | absent | none |
| `86843a78` | aaryn-tuf-rtx2060 | absent | none |
| `9c116a30` | tower-rtx3060 | absent | none |
| `cbe0a410` | aaryn-tuf-rtx2060 | absent | none |
| `f004d4f9` | aaryn-tuf-rtx2060 | absent | none |

All twelve lack fenced-attempt history. Their custom-volume hint counts were
zero, and no unexpired legacy start/pause commands were queued at the metadata
snapshot. This is a point-in-time inventory, not a zero-use retention window.

The referenced volume is a 1 GiB unencrypted NFS volume, registered `available`,
with no attachment or snapshot rows. On the configured storage host the export
base existed, but these expected paths did not:

- `/exports/volumes/vol-36592ad0362e`
- `/exports/volumes/vol-36592ad0362e.img`
- `/exports/volumes/_snapshots/vol-36592ad0362e`

This is a recovery/metadata discrepancy. Backups, image exports and other
recovery sources were not proven or exhausted, so the evidence does not establish
irrecoverable data loss. No replacement empty directory was created and the
volume metadata was not changed to imply a restore.

## Safeguards implemented

Old `start_container` commands bypassed the runner's legacy lease guard by
calling Docker directly. Queue insertion now locks the job row, verifies its
current host and canonical container name, and rejects any fenced history.
Scheduler active mode refuses old starts. State-preserving pause commands for
genuine legacy work may still drain it during cutover. Both are measured by
`xcelsior_legacy_execution_total` under `command_start` / `command_pause`;
`used` means the directive was queued, not that Docker completed it.

Before execution the worker rechecks confirmed negotiation, start permission,
local attempt ownership and physical Docker job/attempt labels. Unknown inspect
results fail closed. Refused commands are not counted as dispatched and emit a
degraded report. The old destructive command drain itself remains a separate
B16 retirement gate; this change does not convert it to durable claim/ACK.

The legacy interactive monitor also removed cleanly stopped containers even
though stop/resume promises to retain them. It now preserves clean/signal exit
codes `0`, `130`, `137`, `143`. This defect is not evidence of what caused the
twelve historical absences. A new real-container acceptance writes a marker to
the writable layer, stops naturally, runs the real worker monitor and HTTP status
path, then copies the retained marker out and checks its exact contents. That
test exposed a missing host ID on legacy status reports using host tokens; the
shared report funnel now supplies that attribution.

## Validation and evidence

- Nine lifecycle-boundary regressions failed against the previous code before
  implementation, after correcting the test fixture's attempt status.
- **523 selected tests passed**, with zero failures/errors/skips. The separate
  opt-in Docker Compose smoke class remains explicitly deselected.
- The real CUDA/HTTP/PostgreSQL gate passed its five fenced journeys and the
  additional legacy writable-layer preservation case. The legacy job becomes
  `stopped`, its lease becomes `released`, and its container/marker survive
  until the harness cleans up its own synthetic container. This is not a
  production restore or full public legacy-resume journey.
- All 12 distribution signatures verified. Targeted Ruff, Pyright and whitespace
  checks passed.
- `api/scripts/audit_legacy_containers.py` is a standalone read-only tool. It
  requires an independently verified daemon name, handles renamed containers,
  excludes environment/command contents, and exits 2 when observation is
  incomplete. Tests cover wrong-host, unavailable, malformed and renamed state.

The [machine-readable evidence](2026-10-20-legacy-resume-evidence.json) contains
the selected source hashes, test/gate results, CUDA and preservation journey,
curated production metadata, host inventories and original artifact checksums.
Full local logs and manifests are under `artifacts/legacy-resume-evidence`.
Production hosts were only inspected; no service rollout or repair-policy
change was performed.

## Remaining gates

1. Deploy and verify the accumulated API/worker safeguards and metrics, keeping
   repairs `report_only`. Confirm deployed source/signature identity.
2. Investigate image exports/backups and reconcile the referenced NFS volume's
   registered state with storage. Restore into an isolated target and verify
   contents before claiming recovered user data.
3. Choose an explicit recovery or retirement disposition for each stopped row.
   A fresh fenced attempt is reprovisioning, not restoration of a missing
   writable layer; no automatic requeue is authorized by this inventory.
4. Finish the remaining B16 instrumentation, active cutover and continuous
   zero-use evidence before schema contraction and deleting compatibility
   handlers. These safeguards and observations do not close B16.
