# Review — `feat/mcp-p0-scopes` (13 commits, 8100f27 → 24db86b)

*Reviewed at 13. The branch is now 15: `50f522a` ("all") was split into
`ae22bb1` — the Connect deny-by-default gate discussed in §3.3 — and `5d1a41c`,
the open-endpoint scanner learning to read router-level dependencies. Nothing
below was rewritten to match; the two commits arrived after this review and are
the part of the branch it has not been through.*

Reviewed from `docs/review/p0-commit-messages.md` and the conversation record.
**Not reviewed: the diffs.** Every finding below is about design and composition,
derived from what the commits claim. Someone still has to read the code, and the
two items in §1 are where that reading should start.

Verdict: land it, with §1.1 and §1.2 addressed first. The work is unusually well
evidenced — every guard verified by planting the defect it exists to catch, every
superseded test rewritten with its supersession stated rather than deleted.

---

## 1. Blocking

### 1.1 The escalation guard is on the door, not the lock

8100f27's own reasoning is the problem with 8100f27's own fix:

> Every individual check behaves exactly as written; the composition is what fails.

`_require_host_operator` still authorizes a machine principal on scope alone.
That remains sound only while no non-admin can obtain such a credential, and the
fix places `assert_scopes_delegable` at `POST /api/oauth/clients`. The invariant
now depends on that being the *only* path that writes scopes onto a client row.

Any future minting path — an admin tool, a migration backfill, a seeding script,
a team-invite flow that provisions a service client — reintroduces the escalation
and **no test in this branch fails.**

Move the guard from the endpoint to the write. `assert_scopes_delegable` should
be called inside whatever function persists client scopes, so every caller
inherits it. Then add the set-equality guard already used for the three SSH
routes: enumerate every call site that writes the client scopes column, assert
that set by name, and require each member to route through the delegation check.
A fourth writer then fails the build instead of silently widening the surface.

### 1.2 Two booleans over three states

`is_relaxed_env()` and `is_production()` are deliberately not complements, and
staging is neither. That distinction is correct and audited. It is also a trap:

```python
if is_production():
    strict()
else:
    relaxed()      # staging lands here
```

That shape is fail-open for staging, which holds real data — the same defect
class 5a2b819 exists to close, arriving through the new API rather than through
`os.environ.get`. Tests won't catch it because they encode one reading of a
docstring, exactly as flagged.

Two options, in order of preference:

1. **Make the resolver return an enum** (`RELAXED | STAGING | PRODUCTION`) and
   have call sites handle it exhaustively, so adding or missing a state is a
   failure rather than a default.
2. **Guard the shape**: an AST check forbidding `is_production()` as the test of
   an `if`/`else` where the `else` branch relaxes anything. Cheaper, weaker,
   still closes the observed case.

Whichever, `env_config`'s 25-decision inventory should record which of the three
states each decision distinguishes — several will turn out to distinguish only
two, and those are the candidates.

---

## 2. Raise now, outside the branch

### 2.1 The terminal host-key finding is buried

5a2b819 mentions in passing:

> `routes/terminal.py` left **SSH host-key pinning off** unless the value was
> exactly `prod` or `production`, so staging or a typo disabled MITM protection
> on the terminal.

That is the most severe individual finding in the set and it has one line in a
commit body. It needs an answer to a question no commit can answer: **did any
non-production deployment ever carry real terminal sessions?** If staging did,
there is a window during which terminal traffic was unpinned, and that is an
incident with a disclosure question attached, not a code change.

Recommend: determine the window from deploy history, record the answer in
`docs/review/`, and treat the outcome as a decision rather than leaving it
implicit in a paragraph about environment variables.

### 2.2 The compromised constants still have an unresolved limit

`xcelsior-dev-jwt-secret` and the deterministic Fernet key are in the source tree
and in git history. The dev-pool check found zero rows in the three consuming
tables, and the limit was stated honestly: production and CI/staging volumes were
not queried.

The definitive check does not depend on trusting configuration: attempt
decryption of production rows in those three tables using the repo constant. Any
success is a confirmed hit; zero across all rows is an answer rather than an
inference from `.env`.

---

## 3. Endorsed, with one extension

### 3.1 GT0's `gap` set is the phase plan read back against the API

a537866 classifies `volumes`, `artifacts`, `ssh`, `spot`, `reputation` and `sla`
as `gap` because each is something a later phase needs and no tool exposes. That
is a stronger property than an audit label, and it is worth making mechanical:

**When P3 lands, `volumes` and `artifacts` must move `gap → covered`, and a test
should assert that transition.** Same for P2 with `ssh`, P5 with `spot`,
`reputation` and `sla`. GT0 then stops being a one-time audit and becomes the
progress meter for the phase plan — a phase that claims completion while its
modules are still `gap` fails.

### 3.2 The per-module granularity argument is right

> Eighteen identical per-row reasons would be that same claim wearing a disguise.

Correct, and worth preserving as the rule when the remaining 358 are done: the
classification belongs at the granularity the judgement is actually true at. The
mixed modules — `serverless`, `auth`, `billing`, `health`, `instances` — are
genuinely per-row, and leaving them at 358 rather than pattern-matching them into
a green gate was the right call.

### 3.3 Reversing the Stripe exposure before designing the model

Deny-by-default at the router, with named exemptions and a rule forbidding any
exempt path from carrying a mutating verb, is the correct order of operations.
The per-endpoint model can now be designed without racing an open money surface.

---

## 4. The composition risk in the branch itself

Thirteen commits, one author, no second reader, touching `security.py`,
`oauth_service.py`, `routes/auth.py`, `routes/_deps.py` and a new `env_config.py`.
Each change is individually well evidenced. 8100f27's lesson applies to the set:

Before merge, do one pass that reads only the **interactions** — the env resolver
against `AUTH_REQUIRED`, `AUTH_REQUIRED` against the startup gate, the startup
gate against the three SSH routes, and the router-level deny against the
handler-body scanner. Each of those pairs was changed independently by a process
optimised for proving individual claims. The failure mode this branch is most
exposed to is not a wrong check; it is two right checks that no longer compose.

---

## 5. CI

`.github/workflows/gates.yml` runs every gate that does not need a tenant:
the suite with auth enforced, the fail-closed resolver asserted directly, the
generated-artifact equality checks, the mcp surface snapshot, and migrations from
empty. One aggregating `gates` job so branch protection has a single stable name.

`.github/workflows/live-gates.yml` is manual dispatch and **refuses to run
without its secrets** rather than skipping — a live gate that silently no-ops is
worse than one that is visibly absent. Its preflight fails if staging advertises
production's audience, which is the current state.

Neither has been executed. The account has no billing, so these are unverified
until it does — the plan's "a green push is unverified" still holds, and the
first real run should be treated as the first run, not as a formality.
