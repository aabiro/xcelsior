# Scope vocabulary audit

*Directive Priority 7. Built 2026-08-08.*

The `marketplace:write` case — one scope covering both a customer's reservation
and an internal allocation — prompted the question of whether other scopes hide
the same shape. They do, and the worst one is not a naming collision at all.

## Method

Every `_require_scope(user, …)` call in `routes/` mapped to the route that
contains it, then each scope read for two failure shapes:

1. **Vocabulary collision** — one scope covering operations with different
   audiences (customer vs operator vs internal).
2. **Risk-tier collision** — one scope covering operations of materially
   different consequence, so consenting to the mild one grants the severe one.

The second turned out to matter more.

## Finding 1 — `mfa:write` removes the account's second factor

**Severity: the highest thing on this page.**

`mfa:write` guards eleven routes. Ten are setup and verification. The eleventh is
`DELETE /api/auth/mfa/all` ([routes/mfa.py:933](routes/mfa.py#L933)), which
deletes every MFA method **and** every backup code:

```sql
DELETE FROM mfa_methods WHERE email = %s
DELETE FROM mfa_backup_codes WHERE email = %s
```

Its complete authorization is: authenticated, and holds `mfa:write`. **No
password re-entry, no step-up challenge, no MFA confirmation to remove MFA.**
`POST /api/auth/mfa/backup-codes/regenerate` is on the same footing and is
arguably worse, because it invalidates the codes the user has saved *and* returns
the new ones to the caller.

`mfa:write` is not in `OPERATOR_SCOPES`, so **any non-admin can delegate it to a
third-party OAuth client** with no extra ceremony — it sits in the same tier as
`instances:read`.

Set against the rest of the platform, the asymmetry is stark. As of today,
raising an auto-top-up amount requires a server-authored approved plan that a
machine principal cannot approve. Removing every second factor on the account
requires one scope and one call.

**Closed today, and worth recording because it was live until this afternoon.**
`_require_scope` was a no-op for `oauth_access_token` until the connector cutover
(`929b3cc`). On that surface the consequence was not "a narrowed token behaves
like a full one" — it was that **any third-party connector token could call
`DELETE /api/auth/mfa/all` whether or not it had ever been granted `mfa:write`**,
because the check did not run for that credential class at all. The cutover
closed it. Nothing else on this page is as sharp as that was.

**Recommendation (owner decision).** Two options, not mutually exclusive:

- *Step-up.* Destructive MFA routes require a fresh authentication — the standard
  pattern, and the one users already expect from every other platform.
- *Tier the scope.* Split `mfa:write` into setup (`mfa:write`) and removal, and
  put removal behind `_is_interactive_human`, exactly as widening auto-top-up now
  is. An agent should be able to help you *add* a factor and never remove one.

The second is cheap — the predicate already exists and is already tested.

## Finding 2 — `marketplace:write` covers two audiences

The known one. It guards both customer reservations and internal
allocate/release. The owner's ruling stands: reservations move to
`billing:write`, matching `/api/pricing/reserve` which is the canonical handler;
allocate/release move behind an internal guard rather than a user scope, since no
user should hold them at all.

Not yet implemented. It is entangled with the `release_allocation` no-op below.

## Finding 3 — `POST /release/{allocation_id}` is a no-op reporting success

Not a scope issue, but it lives in the same surface and blocks Finding 2. The
route passes an allocation id to `release_allocation(job_id)`, which queries
`WHERE job_id = %s`, matches nothing, and returns `ok: true`.

It cannot simply be fixed by correcting the lookup: `gpu_allocations` has **no
owner column**, so a working `release_allocation(allocation_id)` would release
any tenant's allocation by id. The no-op is currently the only thing preventing a
cross-tenant capability, which is why this needs a schema change and not a
one-line fix, and why it gets its own commit.

## Finding 4 — `transparency:*` is operator-only and reads like a customer scope

`transparency:read` and `transparency:write` are in `OPERATOR_SCOPES`, so a
non-admin cannot delegate them. The guard is correct. The *name* is the problem:
nothing about "transparency" suggests platform authority, and a future reader
adding a customer-facing transparency endpoint would reach for the existing scope
and silently make it admin-only — or, worse, notice it is admin-only and remove
it from `OPERATOR_SCOPES` to make their endpoint work.

Recorded rather than changed. Renaming a scope in `OPERATOR_SCOPES` invalidates
issued tokens, and this is a legibility risk rather than a live hole.

## What was checked and found clean

- **`billing:write`** — fifteen routes, all customer-facing money movement, one
  audience. Widening auto-top-up now needs an approved plan on top of the scope,
  so the highest-consequence member is separately gated.
- **`hosts:evict` / `hosts:operate` / `hosts:fleet`** — all in `OPERATOR_SCOPES`,
  refused to non-admins, and covered by the live refusal gate.
- **`ssh:read` / `ssh:write`** — correctly split. The plan originally specified a
  single `ssh:manage`; the split is what lets an agent list keys without being
  able to add one, and adding a key is what grants shell access.
- **`inference:read` / `inference:write`** — one audience, and now enforced on
  both the `/api/v2` surface and the `/v1` family after the resolver inversion
  was closed.
- **`reputation:write`** — annotated `(operator)` at one point and it is not; it
  guards a provider claiming milestones they earned. The annotation was
  corrected rather than the operator set widened to match it. Still correct.

## Ranking

1. **`mfa:write` removal without step-up** — implement Finding 1. The predicate
   exists; this is a small change with a large consequence.
2. **`release_allocation` no-op** — a route reporting success for nothing done,
   and a schema gap behind it.
3. **`marketplace:write` split** — decided, unimplemented, blocked on (2).
4. **`transparency:*` naming** — record only.
