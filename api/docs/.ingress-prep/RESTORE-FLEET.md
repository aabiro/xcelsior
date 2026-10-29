# Restore worker connectivity — install note

Mechanical. No decisions in here. Applies to **4 prod hosts + the local worker**.

The workers currently get `410 agent_ingress_retired` on every heartbeat and
every command drain, because prod runs `XCELSIOR_AGENT_PUBLIC_INGRESS=deny` and
the replacement gateway does not exist yet. A worker that presents the gateway
shared secret is let through that gate. This does **not** need a client CA, a
server certificate, an nginx change or a DNS change — the check is
hostname-independent, so workers keep using the origin they already use.

> This is a stopgap. The secret is acting as a bearer credential on public
> ingress rather than as gateway attestation: whoever holds it is effectively at
> `allow`. The mTLS cutover in `README.md` here is still the destination.

## What this does *not* do — read before budgeting an evening for it

An earlier version of this note said it made the blocked gates testable. **It
does not.** The live gates run against **staging**, not prod:
`scripts/run_live_gates.sh` sets `BASE="http://127.0.0.1:${PORT}"`, the local
compose stack, which has its own database. Prod's fleet is invisible to it.
Staging's `hosts` table holds one row, `admission_state='pending'`, and nothing
heartbeats to it.

So the 9 fleet-dependent gates skip for a reason this note does not address, and
restoring five hosts will not change a single one of them. Restore the fleet
because prod's telemetry, SLA rows and verification stamps depend on it and a
worker is crash-looping — all of which are worth doing. Do not restore it
expecting the gates to move.

To move the gates you need a host admitted **in staging's database**: point a
worker at the staging port, or run the gates against an origin that has a fleet
by setting `XCELSIOR_LIVE_BASE_URL` (the runner honours it now). The second is a
deliberate act against production and is not mechanical, which is why it is not
in this note.

## What you need

The value already in prod's `/opt/xcelsior/.env` as `XCELSIOR_AGENT_GATEWAY_SECRET`.
Copy it from there. **Do not paste it into a ticket, a commit, or a chat.**

## Per host

**1. Add the variable to the worker's env file.**

| Host | File |
|---|---|
| local worker (`aaryn-tuf-rtx2060`) | `/home/aaryn/.xcelsior/worker.env` |
| each prod-registered host | that host's worker env file — same file its `XCELSIOR_SCHEDULER_URL` is in |

Append one line:

    XCELSIOR_AGENT_GATEWAY_SECRET=<value from prod .env>

Same variable name the server reads. No second name, no translation.

**2. Restart the worker.**

    sudo systemctl restart xcelsior-worker.service

**3. Confirm within ~30s** (one heartbeat interval):

    journalctl -u xcelsior-worker.service --since "2 min ago" | grep -iE "heartbeat|410"

Expect the `410 Client Error: Gone` lines to stop. No news is the success case.

## Verify from the control plane

After all hosts are done, hosts should leave `dead`:

    SELECT host_id, status FROM hosts ORDER BY host_id;

And the reverification sweep's backlog should start draining — it has been
enqueuing ~2/hour into a queue nothing could collect:

    SELECT status, count(*) FROM agent_commands WHERE command = 'reverify' GROUP BY 1;

## Rollback

Remove the line, restart the worker. The host returns to exactly its current
state (410 on every worker path). Nothing else changes, and no other host is
affected — enrolment is per host.

## What this does not do

- Does not touch nginx, DNS, or certificates.
- Does not change `XCELSIOR_AGENT_PUBLIC_INGRESS`. Setting that back to `allow`
  would remove the authentication rather than restore the fleet, which is the
  one option worth avoiding.
- Does not affect PixelEnhance on the shared box.
