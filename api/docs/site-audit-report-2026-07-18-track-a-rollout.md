# Xcelsior — Site Audit Companion: Track A Rollout Verification (2026-07-18)

Companion to the 2026-06 site-audit series (see
[`site-audit-report-2026-06-04-reaudit.md`](./site-audit-report-2026-06-04-reaudit.md)).
Scope: production state of the Track A transactional control plane after
the phased rollout completed on 2026-07-18. Evidence gathered live from
the public API, the enrolled worker host (`aaryn-tuf-rtx2060`, local
journal access), and the production database review recorded in
`track-a-implementation-checklist.md`.

## Deployment state

| Check | Result |
|---|---|
| Deployed checkpoint | `c349c3b` (feat: harden fenced worker lifecycle) |
| Public health | `GET /healthz` → `{"ok": true}` |
| Scheduler mode | `shadow` (deliberate posture: canary re-enable is the next operational step, after the hardened-agent soak) |
| DB schema | migrations 054–059 applied (deploy runs `alembic upgrade head`); full 059→053→059 cycle verified clean |
| Worker agent | `xcelsior-worker` active, **v2.2.0 signed build** |
| Protocol negotiation | `GET /agent/v2/negotiate/aaryn-tuf-rtx2060` → `protocol: 2`, features `commands.claim_ack`, `leases.fenced`, `attempts.fenced_status` |

## Track A end-to-end proofs (production traffic)

- **Shadow comparison (P3.3, signed off):** 142 comparisons over a
  24-hour window — 140 `match_queue`, 2 `legacy_placed_shadow_queued`
  (both explained by comparison-window timing around an occupied
  single-GPU host), zero unexplained mismatches. 98.6% literal match.
- **Legacy path unharmed:** instance `lucid-storm` (`3dadeca7`,
  submitted `gpu_model=any`) placed and ran through the legacy
  scheduler with the canary partition correctly declining ownership.
- **v2 fenced path proven:** instance `ultra-pulse` (`2b9e67b8`,
  submitted `gpu_model=RTX 2060`) exercised the canary window:
  queued → assigned → lease claim → container start → running, with the
  worker enrolled in the fenced protocol.
- **Rollback drill exercised for real:** production was moved
  canary → shadow and redeployed without stranding any state — the
  kill-switch/rollback posture in the runbooks is not theoretical.

## Control-plane surface now live (dark or active per env flags)

| Layer | Flag | Prod state |
|---|---|---|
| Shadow comparison runner | `XCELSIOR_SCHEDULER_MODE=shadow` | **active** |
| Canary transactional placement | `XCELSIOR_SCHEDULER_MODE=canary` + `CANARY_GPU_MODELS`/`_HOSTS` | dark (sign-off posture) |
| Fenced worker protocol `/agent/v2` | `XCELSIOR_AGENT_V2_HOSTS` | **active** for `aaryn-tuf-rtx2060` |
| Outbox dispatcher → SSE | `XCELSIOR_OUTBOX_DISPATCHER` | **active** (scheduler-worker thread) |
| Observations + reconciler | worker v2 + service sweeps | **active**, report-only findings |
| Claims kill switch | `XCELSIOR_SCHEDULER_CLAIMS_ENABLED` | armed (true) |

## Residual items (tracked in `track-a-implementation-checklist.md`)

- P4.4b: re-enable canary (mode + scope envs), widen → active, retire
  inline `process_queue()` calls.
- P5.5 tail: fence diagnostic telemetry; fresh-attempt resume and
  terminate/cancel lifecycle controller.
- P6.3: enable reconciler actions per finding type; domain controllers.
- A4.4 tail: migrate legacy SSE/webhook/audit/billing producers.
- A1.6: extract legacy runtime DDL from `db.py` startup (contract phase).

Regenerated evidence commands: `curl https://xcelsior.ca/healthz`,
`curl -H "Authorization: Bearer $TOKEN" https://xcelsior.ca/agent/v2/negotiate/<host>`,
`journalctl -u xcelsior-worker | grep -i negotiat`.
