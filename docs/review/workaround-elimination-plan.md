# Workaround elimination plan

**Written 2026-08-04**, after a day in which four latent defects surfaced only
because a deploy failed. The question that produced this document was whether
workarounds are accumulating. They are — but not the ones that were visible, and
the largest is not a workaround at all. It is a **whole configuration mechanism
that reports success and does nothing**.

---

## 0. What "foolproof" can and cannot mean

It cannot mean resolve. Every compromise in §1 was accepted by someone competent
who intended to come back to it, and the record of intent is what failed.

So the only definition worth planning against: **each item ends in something that
fails when it regresses, and the plan's own completion is measurable rather than
believed.** Where a defect is too large to fix in one pass, it ends in a
*ratchet* — a counter that may only decrease — rather than a promise. This repo
already uses that idiom (`MAX_LEGACY_FLOAT_CAD_COLUMNS`), and it is the only
mechanism here that has demonstrably worked.

Anything in this plan that ends in "remember to" is not done.

---

## 1. The measured inventory

Numbers are counted, not estimated. Reproduce with the commands in §6.

| # | Compromise | Measured | Ends in |
|---|---|---|---|
| 1 | **Env vars set in `.env` that reach no container** | **115** server-side (117 total, 2 worker-only) | ratchet at 115, may only fall |
| 2 | CI has not executed since ~2026-07-21 (billing) | every gate local-only | billing restored, or gates run on a self-hosted runner |
| 3 | Filesystem-walking gates unprotected against macOS sidecars | **7 of 10** | one shared iterator, no per-file skips |
| 4 | `XCELSIOR_AGENT_HOST_TOKENS=allow`, reverted from `require` | 4 hosts tokenless | coverage `ready=true`, then `require` |
| 5 | Deploy builds two images concurrently on the live target | reproduces per retry (#26) | serialized or off-host build |
| 6 | `.env` on a workstation *is* the production secret store | 306 values | secret manager, or an explicit decision recorded |
| 7 | Every merge bypasses the ruleset with `--admin` | 100% of merges | review requirement made satisfiable, or removed |
| 8 | Guard on the door, not the lock (#16) | 5 write paths, 2 guarded | check moved into the store |
| 9 | `ssh:read`/`ssh:write` enforced but ungrantable (#27) | 2 scopes | in the scope map, or renamed |
| 10 | Revocation reports success without effect (#28) | 2 endpoints | revoke, then assert the same token is refused |
| 11 | Mail failures invisible (#23) | `log.debug` + unwritable log path | failure logged at a level someone reads |
| 12 | `wip(...)` commit inside #18's history | 1 commit | squashed before merge |

### 1.1 Item 1 is the one that matters

`docker-compose.yml` declares **no `env_file:`** on any service. A variable
reaches a container only by being named in the `x-api-environment` anchor; `.env`
is otherwise consumed for `${...}` interpolation alone. So a variable set with a
real value, read by application code, and absent from the anchor is **present on
the host and absent in the process that reads it** — with nothing reporting a
problem.

On 2026-08-04 this refused a production boot for two secrets. Those were not
instances of a small problem. They were the two that happened to be checked at
startup. The rest fail silently, on defaults.

Confirmed consequences in the current deployment:

- **`XCELSIOR_STRIPE_THIN_WEBHOOK_SECRET` and `XCELSIOR_STRIPE_CONNECT_WEBHOOK_SECRET`
  are absent from the container.** `/api/connect/webhooks` therefore answers
  **503** to every Stripe delivery. It fails *closed*, which is the right
  direction — the endpoint's exemption from authentication is justified on the
  signature being the credential, and without the secret it refuses rather than
  accepting unverified. But webhook-driven flows have been dead for as long as
  this has been deployed, and the only signal is a `log.warning` at import into a
  log file the container cannot write.
- **Every terminal and websocket limit runs on code defaults**, not the operator's
  values: session lifetime, concurrent sessions per user, input frame bytes,
  malformed-frame tolerance, resize caps, WS connect rate limits. These are the
  containment limits earlier sessions deliberately built.
- **`XCELSIOR_INPUT_TOKEN_PRICE` / `XCELSIOR_OUTPUT_TOKEN_PRICE` do not arrive.**
  If the configured prices differ from the code defaults, inference has been
  billed at the wrong rate. **Verify before anything else in this plan** — it is
  the only item here that may have moved money.
- `XCELSIOR_PII_SCRUB`, `XCELSIOR_READYZ_SCHEMA_CHECK`, the volume quotas
  (`MAX_VOLUME_GB`, `MAX_VOLUMES_PER_OWNER`, `MAX_TOTAL_STORAGE_GB`), the billing
  payment rate limits, and the Postgres pool sizing all likewise do nothing.

**The root cause is not the 115 missing lines.** It is that the mapping is
hand-maintained with no gate over the whole set, so it was always going to drift,
and drift silently. `tests/test_startup_env_is_wired.py` (added 2026-08-04) gates
only the ~11 variables `startup_validation` reads — which is why 115 survived it.

---

## 2. The three shapes, and a fourth

§8 of the incident review named three. The inventory adds one.

1. **A rule that exists only as prose.** Items 4, 6, 12.
2. **A value configured but not effective.** Items 1, 9.
3. **A check that reports success without effect.** Items 10, 11.
4. **Verification that does not run.** Items 2, 3, 7 — and this one is
   load-bearing for the other three. With CI locked, every gate is only as good
   as someone remembering to run it locally; with `--admin` on every merge, the
   ruleset asserts a review that never happens. **Fixing 2 and 7 first is what
   makes the rest of this plan trustworthy rather than aspirational.**

---

## 3. The eliminations

Ordered by what makes other work verifiable, not by severity.

### Phase A — make verification real (items 2, 7, 3)

**A1. Restore CI, or stop pretending.** Either settle the Actions billing, or
stand up a self-hosted runner on the ASUS box and point `gates.yml` at it. Until
one of those, delete nothing and claim nothing: a green local run is evidence
about one machine at one moment.
*Ends in:* `gh run list --workflow=Gates` showing a completed run whose jobs took
longer than 10s. The 3-second failures are the current tell.

**A2. Make the review requirement satisfiable, or remove it.** A solo maintainer
cannot approve their own PR, so `required_approving_review_count: 1` guarantees
`--admin` on every merge. A rule bypassed 100% of the time trains everyone to
bypass. Either add a second reviewer (human or a bot whose approval means
something), or drop the requirement and keep the checks.
*Ends in:* a merge that succeeds without `--admin`.

**A3. One source-file iterator for every static gate.** 7 of 10 walkers will fail
again the next time macOS touches the tree. Replace the per-file `._` skips with
a single `tests/_source_tree.py::iter_source_files()` and convert all ten.
*Ends in:* a test asserting no test module calls `rglob("*.py")` or `os.walk`
directly.

### Phase B — the configuration mechanism (item 1)

This is the largest and it does not get fixed by adding 115 lines.

**B1. Measure the damage first.** Compare configured against effective for the
subset that can have moved money or weakened containment: the two token prices,
the volume quotas, the terminal/WS limits, the billing rate limits. Anything
where the operator's value differs from the code default is a live incident, not
a config gap.
*Ends in:* a written diff of configured-vs-effective for those, and a decision per
variable.

**B2. Declare the environment once, generate the mapping.** A single manifest —
variable, which services need it, whether it is a secret — from which the compose
`environment:` blocks are generated. Hand-maintained mapping is what drifted;
generating it removes the failure mode rather than patching this instance.
Follow the existing pattern: `scripts/generate_*.py` plus a byte-identical
regeneration gate, as `docs/generated/endpoint-inventory.md` already has.
*Ends in:* `test_compose_env_matches_manifest` — a fresh generation equals the
committed file.

**B3. Ratchet the gap to zero.** Extend `test_startup_env_is_wired.py` from the
startup-validation subset to **every** `XCELSIOR_*` the application reads, with a
declared exemption list that carries a reason per entry (worker-agent-only,
build-time-only, deliberately defaulted). Seed the ratchet at the measured **115**
and let it only fall.
*Ends in:* `MAX_UNWIRED_ENV_VARS = 115` in the test, decreasing per commit, with
zero as the terminal state and no wholesale exemption permitted.

**Explicitly rejected: `env_file: .env` on every service.** It would wire all 306
in one line and pass B3 instantly — while handing the complete production secret
set to every container, including services with no business holding it. The
existing `test_startup_env_is_wired.py` already forbids it, and that test should
stay.

### Phase C — the standing security compromises (items 4, 8, 9, 10, 11)

**C1. Item 8 (#16) first, because it makes the rest durable.** Move
`assert_scopes_delegable` into the function that persists client scopes, so all
five write paths inherit it instead of two calling it. Add set-equality over the
call sites that write the column.
*Ends in:* a test asserting the store refuses an undelegatable scope with **no
route involved**, plus set-equality over writers.

**C2. Item 4** — issue the four host tokens, confirm coverage `ready=true`, flip
`require`. Already gated by `startup_validation._check_host_token_coverage`, which
is why it cannot be flipped accidentally; the work is the issuance.
*Ends in:* boot succeeds with `require` set.

**C3. Item 9** — decide `ssh:manage` versus `ssh:read`/`ssh:write`, put it in the
scope map with consent text, record the decision. Registering a public key grants
shell access; the consent string matters.
*Ends in:* a test asserting every scope any route enforces is grantable — which
generalises past this instance.

**C4. Item 10** — invalidate the auth cache on logout and on session revocation.
*Ends in:* revoke, then assert **the same token** is refused. Both endpoints
currently assert their success and neither asserts its effect.

**C5. Item 11** — `log.warning` on send failure, log at warning when the mailer
returns early unconfigured, `try/except` inside the daemon thread, and a writable
log path.
*Ends in:* a test asserting a failed send emits at `warning` or above.

### Phase D — infrastructure (items 5, 6, 12)

**D1. Item 5** — persist build hashes *before* migrations, and serialize or
offload the image build. Currently a migration failure guarantees the next
attempt reproduces the build that killed the API.

**D2. Item 6** — the `compat_session_secret` remediation says "store it in the
production secret manager." There is no secret manager; `.env` on a workstation is
it, rsynced by `deploy.sh`. Either introduce one or record the decision not to,
with the blast radius stated. Right now the remediation text describes a control
that does not exist.

**D3. Item 12** — squash the `wip(...)` commit before #18 merges, or it is
permanent in `main`'s history labelled "not reviewed, not verified".

---

## 4. What is deliberately **not** on this list

A plan that churns sound design in the name of tidiness is worse than none.

- **`migrations/lock_safe.py`'s second connection.** Not a workaround. A migration
  running against live traffic must not hold locks across many tables, and the
  resumable-not-atomic trade is stated in both docstrings and gated both ways.
- **The two credential sets** (`_MACHINE_AUTH_TYPES`, `_API_KEY_AUTH_TYPES`). They
  answer different questions and the file now says so. Merging them would either
  refuse `client_credentials` from fourteen routes or stop enforcing scopes on the
  legacy key class.
- **The `User-Agent` in the live gate.** The edge rejects default Python agents;
  identifying the client is not a bypass, and the edge-detector asserts any 403
  came from the origin. *One improvement worth making:* run the gate against the
  origin **and** through the edge, so a failure names which layer refused.
- **`p0-commit-messages.md` un-gated.** Deliberate: it derives from a commit range
  and would fail on every commit, which is how a gate becomes noise. The
  self-describing header is the weaker but honest alternative.

---

## 5. Order, and why

```
A1 A2 A3   →   B1   →   C1   →   B2 B3   →   C2 C4 C5   →   D1 D2   →   C3 D3
verification    money    durable   the whole    the standing   infra    decisions
is real         first    guard     mechanism    reports-ok     
```

**A before everything**: every item below it is verified by gates that currently
run only when someone remembers.
**B1 before B2/B3**: if a token price differs from its default, that is an
incident and it outranks the mechanism that caused it.
**C1 early**: #16 is what stops the operator-scope fix being one refactor away
from reopening — and that fix is the only thing this week that was verified in
production.

---

## 6. Reproducing the numbers

Every figure in §1 comes from a command, so the inventory can be re-measured
rather than trusted:

```bash
# item 1 — configured but not effective (expects 115 server-side)
python3 - <<'PY'
import re, pathlib, yaml
read=set()
for p in list(pathlib.Path('.').glob('*.py'))+list(pathlib.Path('routes').glob('*.py'))+list(pathlib.Path('control_plane').glob('*.py')):
    read |= set(re.findall(r'environ(?:\.get)?\(?\[?["\'](XCELSIOR_[A-Z0-9_]+)["\']', p.read_text()))
env={m.group(1) for l in pathlib.Path('.env').read_text().splitlines()
     if (m:=re.match(r'^(XCELSIOR_[A-Z0-9_]+)=(.+)$', l.strip())) and m.group(2).strip().strip('"').strip("'")}
d=yaml.safe_load(open('docker-compose.yml')); mapped=set()
for s in ('api','api-blue','scheduler-worker','bg-worker'): mapped |= set(d['services'][s].get('environment') or {})
print(len((env & read) - mapped))
PY

# item 3 — walkers protected against macOS sidecars (expects 3 of 10)
grep -ln "rglob\|os.walk" tests/*.py | wc -l
grep -ln 'startswith("._")' tests/*.py | wc -l

# item 1, the deployed half — what the container actually sees
ssh -o ControlPath=none -i ~/.ssh/xcelsior linuxuser@149.28.121.61 \
  'cd /opt/xcelsior && docker compose run --rm -T api env | grep -c ^XCELSIOR_'
```

---

## 7. The single sentence

Nothing on this list was introduced carelessly. Each was a reasonable trade whose
record of intent was prose, and prose does not fail. **Every item above therefore
ends in a gate or a ratchet, and the items that make gates run at all come
first** — because a plan verified by checks nobody executes is the same defect,
one level up.
