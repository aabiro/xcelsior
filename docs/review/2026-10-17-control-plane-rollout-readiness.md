# Control-plane rollout readiness — 2026-10-17

**Update:** The user subsequently explicitly approved the release. Deployment
completed successfully, including migration 123 and service handoffs, with
repairs kept report-only. See the [production rollout record](2026-10-17-control-plane-production-rollout.md)
for execution evidence and the newer production state. The record below
describes the preparation and approval boundary before execution.

Application candidate: **`da4442cf084037e3c4eea3091813541f715468d3`**.
The candidate is committed on local `main`; it has not been pushed or deployed.
CI cockpit and unrelated applications are outside this release.

## Decision

The release is prepared, but production activation awaits explicit approval.
Automatic approval review rejected execution of `scripts/deploy.sh --docker`
because it would synchronize code, apply database migration 123, rebuild images
and hand over live services without explicit production authorization. The
deployment command did not execute. No alternative deployment path was used.

The prepared scope is the six previously committed control-plane/security/
dependency changes through `3d0f5ca8`, plus the deployment fixes in `da4442cf`.
The concurrent, uncommitted security-audit and GX0 changes and editor settings
remain separate and untouched. A clean worktree isolates the candidate.

## Deployment fixes and verification

The script previously recorded desired fingerprints immediately after building
images. If migration, standby readiness or a later service handoff failed, the
next attempt could consider those inputs deployed and skip activation. It now
records them only after the deployment function succeeds. Docker's build cache
still avoids redundant work, and the memory check still controls parallel builds.

Five regressions reproduced the original failure at migration, standby readiness,
scheduler restart, frontend restart and final API readiness. Each now proves
that the previous fingerprints survive failure and are replaced after a
successful retry. Deployment archives include `.env`, so backup creation now
restricts an existing backup directory to mode 0700 and creates mode-0600 archives.
A filesystem test exercises this with synthetic configuration.

The migration-test helper now honors Unix-socket DSNs, including user and port
query parameters, and preserves their URL form. Creation and cleanup use only
the configured cluster; they no longer fall back to the machine's default
PostgreSQL cluster through sudo.

Final gate: **148 passed, zero failed, zero skipped** on Python 3.12, with a fresh
local Unix-socket-only PostgreSQL cluster and no production credentials. This
includes populated migration upgrade/downgrade and runtime-trigger rehearsals,
deployment retry/readiness/environment preservation, nginx installation and
startup validation. Ruff, shell syntax and diff checks passed.

Evidence directory on the operator workstation:
`/mnt/storage/projects/xcelsior/artifacts/release-evidence/deploy-final/`.
It contains the manifest, JUnit results and phase logs. Earlier evidence records
five intentionally failing retry regressions, and a run with five migration
skips; neither is counted as the final pass. The migration skips were eliminated
by the Unix-socket helper fix.

The prior [active-meter repair](2026-10-17-active-meter-repair.md) and
[real CUDA worker-to-invoice acceptance](2026-10-17-worker-billing-acceptance.md)
remain separate evidence for the application changes. This increment does not
claim a fresh full backend/frontend suite or a built-image acceptance pass.

## Production inspection and rollback preparation

Read-only checks on October 17 confirmed:

- All six application containers were healthy. API `/healthz` and `/readyz`
  returned HTTP 200. Production schema remained **122**.
- The deployment marker was `155f0be69252d51d7e244f048386b468075546b0`.
  Source inspection confirmed `control_plane/active_meter.py` was absent.
- A comparison covering 1,710 tracked source paths found no missing prior-release
  files and no frontend differences. The only existing file that matched neither
  the old marker nor the candidate was `public_http.py`; review confirmed that
  the candidate preserves that live hotfix and adds the committed resolver,
  timeout-validation and refusal-metric changes.
- Scheduler mode remained `canary`. The new repair flags and strict database-role
  requirement were unset. The initial release keeps repair actions at their
  `report_only` defaults and leaves strict role cutover disabled.
- Local and production frontend public configuration agreed. The deployment
  guard will preserve production `.env`; local secret/configuration differences
  will not be uploaded.

Rollback material was prepared on the production server at
`/opt/xcelsior-backups/control-plane-20261017-da4442cf/`: a custom-format database
dump whose archive list was checked, nginx configuration, deployment state/input
markers, an environment checksum and the current image IDs. The database archive
was 41,919,297 bytes with mode 0600. The backup directory is mode 0700. Existing
API, scheduler, maintenance, SSH gateway, MCP and frontend images were retained
with `xcelsior-rollout-backup:20261017-da4442cf-<service>` tags. No database dump or
production secrets were copied to the workstation. This is backup preparation,
not a restore-drill pass. The normal deploy also makes its source/config backup
before synchronization.

## Next authorized production step

After explicit approval, deploy the isolated candidate using the repository
script. Disable local worker-mode switching and mirror sync for this release;
force serial API/frontend builds to limit host pressure. Preserve production
configuration and keep the new repair actions report-only.

Require the migration to reach 123, standby and final readiness to pass, and
the MCP protocol/canonical-route checks to pass. Then independently verify live
image/source identities, service health, effective repair flags and unchanged
production configuration. Measure reconciliation findings before considering
individual enforcement flags.

Strict database-role cutover, full public launch provisioning, lost-terminal-
report recovery, wallet debit/provider settlement and remaining Track B/live
journey gates are still open. This preparation does not close B2.9 or the wider
control-plane programme.
