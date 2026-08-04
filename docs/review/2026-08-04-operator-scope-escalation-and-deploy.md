# Incident review — operator-scope escalation, and the deploy that could not ship it

**Status:** closed. The escalation is patched and **verified in production** at both
write paths. Production is at Alembic `098`.

Written after deployment, deliberately. The reproduction below was held out of the
repository while the fix was merged but unshipped, because publishing a working
escalation against an unpatched target — in a public repository — hands it to
anyone reading. That constraint expired when the fix reached production on
2026-08-04.

**Scope of this document.** One escalation, and the five days of latent defects the
attempt to deploy it exposed. It is not a review of `feat/mcp-p0-scopes` (see
[`p0-review.md`](p0-review.md)) and it is not evidence about the MCP surface —
§7 states plainly what remains unproven.

---

## 1. The escalation

`oauth_clients.scopes` was writable by any authenticated user, and
`control_plane_v1._require_host_operator` authorises a machine principal **on
scope alone**, with no role check. Anything able to write that column could
therefore obtain platform-operator authority.

Two routes wrote it:

| # | Path | Guard before the fix |
|---|---|---|
| 1 | `POST /api/oauth/clients` | none |
| 2 | `PATCH /api/oauth/clients/{client_id}` | none |

The full chain, in one request:

```
POST /api/oauth/clients
  { "client_name": "...", "grant_types": ["client_credentials"],
    "scopes": ["hosts:evict"] }
→ POST /oauth/token  grant_type=client_credentials
→ drain or evict any host on the platform
```

Or in two, which is how the second writer was missed: register a client with a
benign scope, then `PATCH` it to add the operator scope. Guarding registration
alone leaves that path open, and the first fix guarded registration alone.

**Fixed** by `_refuse_undelegatable_scopes` (`cc11b6f`, PR #19), called on both
routes. It returns early for `_is_platform_admin(user)` — an admin may grant
operator scopes; that is the point of the check being about delegation rather
than about scopes.

**Not fixed, and structural:** the guard is on the door, not the lock. Any future
path that writes `oauth_clients.scopes` — admin tooling, a seeding script, a
migration backfill, a service-provisioning flow — reintroduces this with every
test still green. Tracked as **#16**; the remedy is to move the check into the
function that persists scopes so every caller inherits it, plus set-equality over
the call sites that write the column.

**Audit before patching.** `scripts/incident/audit_operator_clients.py` ran
against production before the fix shipped: **18 clients, zero operator-scoped.**
No credential was minted through this path. That audit had to run first, because
patching changes what is observable — afterwards a row created earlier still
exists but you can no longer distinguish "never happened" from "happened and was
cleaned up" by re-testing the route.

---

## 2. Timeline, 2026-08-04 (UTC)

| Time | Event |
|---|---|
| 2026-08-03 17:42 | API container starts. This image serves for the next 15 hours. |
| 03:52 | Deploy attempt. `backup_current` tars `/opt/xcelsior`. |
| 04:04:52 | **API killed**, `Exited (137)`, `OOMKilled=false`. |
| 04:06:30 | Migrations begin. `095` dies: `deadlock detected` on `ALTER TABLE jobs ADD COLUMN spot_rate_micros`. Deploy aborts. Whole run rolls back. |
| 04:05 – 08:31 | No API. `/` serves the frontend container, so the site looks up; `/healthz` and every API path 404. |
| 08:31 | Production confirmed at Alembic **`079`** — not half-migrated. |
| 09:00:22 – 09:00:43 | `079 → 098` applies. `095` completes in ~20s under the per-table fix. |
| 09:02:20 | `api-blue` **refuses to boot**: `StartupValidationError`, three findings. |
| 09:17:53 | Deploy completes after the wiring fix. API healthy on 9501 (blue). |
| ~09:50 | Operator-client audit: clean. |
| ~10:20 | Live refusal probe: exit 0 at both write paths. **Escalation closed.** |
| 11:31:49 | Third deploy. API healthy on 9500 (green). PR #29 live. |
| 11:40 | Pre-incident tarball moved out of the backup rotation. |

---

## 3. Why the deploy could not ship the fix

### 3.1 Migration `095` deadlocked against live traffic

Deploys here are blue-green by construction: `scripts/deploy.sh` runs
`alembic upgrade head` while the API, scheduler and background workers keep
serving. `095` took `ACCESS EXCLUSIVE` on fifteen tables inside one transaction
and then backfilled twenty-six columns. A concurrent transaction holding a read
lock on one of those tables and reaching for another closes the cycle.

`migrations/README.md` rule 5 already required `lock_timeout` set. It was prose
with nothing that failed when it was violated.

Two things made it worse than a single lost migration. `migrations/env.py`
wrapped the entire upgrade in **one** transaction, so the sixteen migrations that
had already applied rolled back with the failure — meaning each retry replayed all
sixteen and re-widened the window. And Alembic logs each step as it *starts* it,
so sixteen `Running upgrade` lines with zero applied migrations is exactly what a
clean rollback looks like. That output was initially read as a half-migrated
database and sent one reviewer looking for destructive-migration damage that did
not exist. **Querying `alembic_version` is the only way to know; the log cannot
tell you.**

**Fixed** in PR #20: `transaction_per_migration=True` plus a session
`lock_timeout`; `migrations/lock_safe.py` runs one transaction per table on its
own connection with retry on `40P01`/`55P03`, and checks its own precondition
against `pg_locks` rather than documenting it. Gated by
`tests/test_migration_lock_discipline.py`, driven both ways against real
PostgreSQL — one arm reproduces the deadlock in the old shape, the other survives
identical contention with neither side killed.

**Consequence accepted:** a migration using `lock_safe` is resumable rather than
atomic. Sound only because every statement in it is idempotent, which is a
condition on the caller and is stated in both migration docstrings.

### 3.2 The build killed the API — and that was the outage

Not the migration. The API died at 04:04:52, ninety-eight seconds *before*
migrations began. `deploy.sh` builds the API image set and the Next.js frontend
image **concurrently, on the deploy target, while the existing stack serves**. On a
memory-constrained host the global OOM killer takes the largest resident process.
It reports as `Exited (137)` with `OOMKilled=false`, because the cgroup limit did
not trigger — so it does not look like memory in `docker inspect`.

Compounding: `persist_deploy_inputs` runs *after* the migration step, and
`deploy_docker` aborts on migration failure. A deploy that fails at migrations
never records its build hashes, so the next attempt rebuilds both images and
reproduces the same concurrent build. Every retry.

Two symptoms arrived together with a single cause under one of them, which cost
real diagnostic time. Tracked as **#26**, unfixed. Workaround used for the
successful deploys: quiesce the workers, pre-build the API image set serially so
the parallel step finds it cached.

### 3.3 Two production secrets were set and never reached the container

With the schema finally at `098`, the API refused to boot:

```
StartupValidationError: production startup validation failed —
  compat_session_secret_missing: XCELSIOR_COMPAT_SESSION_SECRET is unset — host
    compatibility sessions would derive submit tokens from a public development
    constant, making provider evidence forgeable
  audit_signing_key_default: Neither XCELSIOR_AUDIT_SIGNING_KEYS nor
    XCELSIOR_AUDIT_SIGNING_KEY is set — audit checkpoints would be signed with
    the public development key and the audit trail would be forgeable
  host_token_coverage_incomplete: XCELSIOR_AGENT_HOST_TOKENS=require but 4
    host(s) have no live token
```

**Both secrets were in `.env`, with high-entropy values, the whole time.**

`docker-compose.yml` declares no `env_file:` on any service. A variable reaches a
container only by being named in the `x-api-environment` anchor; `.env` is
otherwise consumed for `${...}` interpolation alone. So a correctly configured
secret that nothing maps is **present on the host and absent in the process that
reads it**, and nothing reports a problem: the file looks right, the check is
right, and the two never meet.

The startup gate did its job — it refused to serve traffic with a forgeable audit
trail rather than degrading to a development key.

**Fixed** in PR #21, with `tests/test_startup_env_is_wired.py` making the rule
total: if `control_plane/startup_validation.py` reads a variable, every service on
the anchor receives it. No exemption for variables whose default happens to be
safe — an exemption list here would be a list of variables permitted to lie about
being configured. The failing arm is the anchor as it stood before the fix.

The third finding was not a wiring bug but a half-finished rotation:
`XCELSIOR_AGENT_HOST_TOKENS=require` had been set while four hosts held no live
token, and `require` locks a tokenless host out. Issuing tokens needs the API
running, so `require` before issuance is a deadlock rather than a stricter
setting. `.env` returned to `allow` with the restore condition recorded beside it.
Note that this condition **is** already enforced —
`startup_validation._check_host_token_coverage` is what produced the finding — so
the flip cannot be made accidentally, only deliberately.

### 3.4 The audit that had to run first could not run

`audit_operator_clients.py` selected `registration_source`, a column migration
`091` adds. Production was at `079`. It raised `UndefinedColumn` and exited **1** —
the code its own contract reserves for *"at least one client holds an operator
scope — read the output as an incident, not a finding"*.

The audit designed to run *before* the patch would have reported an incident by
crashing, and the exit code could not tell you which had happened.

Fixed before deployment: it selects only columns that exist, exits 2 on any schema
failure, and prints which columns were absent so a degraded audit cannot read as a
complete one. On its first real run it reported
`columns absent: registration_source` — the fix earning its keep immediately.

Root cause of the class: both incident scripts were written and tested against the
dev database, which was nineteen migrations ahead of production.

---

## 4. The verification nearly passed for the wrong reason

The live probe's first run against production returned **403 on its positive
control**:

```
✗ token cannot register even a benign client (HTTP 403).
  Response: {'_raw': 'error code: 1010\n'}
```

`error code: 1010` is Cloudflare's browser-integrity check rejecting
`Python-urllib`, answered **before the request reaches the origin**. By status code
it is indistinguishable from an authorization refusal.

A refusal-only probe would have seen 403 at both write paths and reported the fix
live — on an edge rule that never touched the code. This is the deepest instance
of the pattern running through every finding here, and it was caught by the one
design decision that exists for it: **each refusal is paired with a benign
operation that must succeed, on the same token, through the same path.** The
control failed, so the probe refused to conclude.

Two hardenings followed, both in `tests/live/test_scope_refusals_live.py`:

- **`assert_refusal_came_from_the_origin`.** The registration case asserts the
  body *names* the refused scope, so an edge page cannot pass. The update path
  answers with a bare detail message, so it needed the check explicitly — without
  it, an edge 403 on that one request would have read as "the second writer is
  guarded", reintroducing the same false pass one test later.
- **An identity assertion.** `_refuse_undelegatable_scopes` returns early for an
  admin *by design*. Run the gate with an admin token and every probe is allowed,
  the gate reports the deployment vulnerable, and it leaves a real `hosts:evict`
  client owned by that admin. The gate now calls `/api/auth/me` and refuses to run
  rather than trusting whoever set the secret.

The edge detector is itself driven both ways, with no credentials required, so it
runs in ordinary CI rather than only during a live dispatch.

---

## 5. What the live gate proves, exactly

```
[1/3] baseline: benign client registers          ok
[2/3] registration with 'hosts:evict'            refused 403, naming the scope
[3/3] PATCH amending a client to 'hosts:evict'   refused 403
      cleanup: benign client removed
exit=0
```

A non-admin user session, from outside, against `https://xcelsior.ca`. Six cases
pass with credentials; three of them (the edge detector) pass without.

This is the **first live-credential evidence in the sequence**. Everything
asserting these refusals previously ran against a `TestClient`, and P0's gate
exists precisely because a mock is what passed while production did not.

It proves three things on one deployment at one moment. See §7 for what it does
not.

---

## 6. Attribution: what production was actually running

`deploy.sh` tars `/opt/xcelsior` before every sync (`backup_current`, keeping
five). Because a deploy overwrites the tree, the pre-incident state exists **only**
in that tarball — and the tarball is a better artifact than the live files ever
were.

`scripts/incident/identify_files_in_backup.sh` hashes files out of the archive and
searches history for a matching blob:

| File | Blob matches |
|---|---|
| `routes/terminal.py` | `4e9fb32` (2026-07-28) |
| `api.py` | `4e9fb32` |
| `routes/auth.py` | `4e9fb32` |
| `security.py` | `47d5fc8` (2026-07-13, unchanged since) |

**The deployed tree was not dirty.** Every file is attributable to a commit — the
opposite of what was feared, and worth having established rather than assumed. A
`NO MATCH` would have been the finding, in the negative.

It also settles the startup-validation swallow-path question the hard way round.
`a75251b` (2026-07-21) **is** an ancestor of `4e9fb32`, so the deployed tree did
contain the swallow path. An earlier check on the running container found the code
absent — because the container was running an image built *before* that commit,
not the tree on disk. **Two artifacts, two answers, and the difference is the
finding:** hashing the tree tells you what was deployed, not what was executing.

The pre-incident archive has been moved out of the rotation to
`/opt/xcelsior-forensics/preincident-20260803T235214-tree-production-ran-15h.tar.gz`.
It was one deploy away from being overwritten.

---

## 7. What is **not** proven

Stated plainly, because §5 is easy to over-read.

- **Nothing about MCP compliance.** The MCP live gates — the scope-refusal matrix
  driven by `XCELSIOR_MCP_TOKEN`/`XCELSIOR_NARROWED_TOKEN`, and the tool-selection
  eval baseline — **have never run.** Both need a staging tenant and an MCP token
  bound to a per-tenant audience. `XCELSIOR_MCP_RESOURCE_AUDIENCE` is unset, so
  staging and production would share an audience and a staging token would be a
  production token. The eval remains `BLOCKED(env)`.
- **One operator scope of seven.** The gate probes `hosts:evict`. The other six are
  asserted in pytest only.
- **Two scope-writing paths of five.** Registration and update are covered live.
  The DCR path, the quick-connect path, and `ensure_default_oauth_clients` (which
  reaches `OAuthStore.create_client` directly, bypassing the route guard entirely)
  are not. They are safe by their contents, not by a check — and contents go stale.
  This is #16 restated.
- **One moment on one deployment.** Add a sixth writer of `oauth_clients.scopes`
  and the gate still passes while the escalation reopens.
- **Nobody has read the `feat/mcp-p0-scopes` diffs.** Unchanged from
  [`p0-review.md`](p0-review.md): that review was written from commit messages.

---

## 8. The pattern

Nine findings in one night, and they are one shape in three costumes.

**A rule that exists only as prose.** `migrations/README.md` rule 5 required
`lock_timeout`; nothing failed when it was absent. Every fix here therefore ships
with a gate, and every gate is driven both ways — the failing arm is the actual
prior state, not a synthetic one, so it has been observed to fail on the real
defect.

**A value that is configured but not effective.** Two secrets in `.env` that no
service mapped (#21). `XCELSIOR_EMAIL_REPLY_TO`, still unmapped (#24).
`ssh:read`/`ssh:write` enforced by routes but absent from the grantable scope map,
so no machine credential can hold them (#27). In each case the configuration is
correct, the reader is correct, and they never meet.

**A check that reports success without effect.** `POST /api/auth/logout` and
`DELETE /api/auth/sessions/{prefix}` both return `ok` and leave the presented
bearer valid (#28) — the control you reach for *after* a leak. The verification
email swallowed at `log.debug` with file logging disabled, so a failed send leaves
no trace anywhere (#23). Cloudflare's 403 reading as an authorization refusal
(§4). And `exit=1` from a crashed audit reading as "a client holds an operator
scope" (§3.4).

The common defect is not the bug. It is that **the failure is indistinguishable
from success at the layer where someone looks.** Every gate added here is aimed at
that: name the scope in the refusal body, assert the origin produced the 403,
exit 2 rather than 1 when a check cannot run, log a failed send at a level someone
reads.

---

## 9. Ledger

**Fixed, with gates**

| PR | Fix | Gate |
|---|---|---|
| #19 | operator-scope refusal on both client-write routes | `test_oauth_operator_scope_refusal.py` |
| #20 | per-table locks for `095`/`097`; schema-tolerant audit | `test_migration_lock_discipline.py` |
| #21 | startup secrets wired into the container | `test_startup_env_is_wired.py` |
| #22 | live scope-refusal gate; backup attribution script | the gate itself, plus its edge detector |
| #29 | an agent key is an API key; `DELETE /api/ssh/keys` scoped | `test_agent_key_grant_and_ssh_scopes.py` |

**Open**

| # | Finding |
|---|---|
| #16 | the guard is on the door, not the lock — move it to the store |
| #17 | two booleans encode three environment states |
| #23 | mail failures leave no trace: `log.debug` swallow, file logging off |
| #24 | `EMAIL_REPLY_TO`, `PUBLIC_URL` set in `.env`, never reach the container |
| #25 | OAuth leaves `email_verified=0` on an existing account and never checks it |
| #26 | deploy builds two images concurrently on the target while the stack serves |
| #27 | `ssh:read`/`ssh:write` enforced but absent from the grantable scope map |
| #28 | logout and session revocation return `ok` and leave the bearer valid |

**Process notes, recorded because they cost something**

- PR #29 shipped two fixes that should have been separated. Adding a scope check
  to three SSH routes is hotfix-shaped; correcting an inert `auth_type == "api_key"`
  comparison changes behaviour at **29** call sites — the measured figure, not the
  ~40 estimated in review — including password change, MFA and account deletion, and deserved its own change with probes. It happened to be
  safe — the MCP surface avoids `/api/auth/me` deliberately
  (`mcp/src/auth/bearer.ts:36`), and production logged **zero 403s** in the 25
  minutes after deployment — but "happened to be safe" is not the same as "was
  separated correctly".
- The first port of the live gate mis-read the registration response shape
  (`{"ok": true, "client": {...}}`, not flat), so its fixture failed before
  teardown and left six benign OAuth clients on production. Found, deleted by name
  prefix, zero remaining, then re-run clean.
- A `git stash pop` consumed the previous session's pair-2 work-in-progress and
  left it uncommitted on `main` — where `deploy.sh` would have rsynced it into
  production. Preserved as `wip/pair2-startup-gate-recovered` and as a file copy
  outside the repository; `main` clean.
- Commit messages on the migration branch briefly published production's schema
  version and the fact that the fix was merged but undeployed. Reworded and
  force-pushed within about ten minutes — an unmerged branch with no PR and nothing
  citing the hashes, which is the case where rewriting costs nothing. Distinct from
  `8100f27`, whose message is public, immutable, and cited by two issues.

---

## 10. Next

1. Probes across the account-security routes for the `_require_user_grant` change
   — the assertion should exist independently of "no 403s tonight".
2. **#16**, the store-level guard. It is what makes §5's result durable instead of
   momentary.
3. `#18` with pair 2 rebased from `wip/pair2-startup-gate-recovered` onto
   `feat/mcp-p0-scopes`, and pair 3 already landed via #29.
4. **#26**, so a deploy stops being able to kill the thing it is deploying.
5. The MCP live gates, once a staging tenant and a per-tenant audience exist. Until
   then §7 stands unchanged.
