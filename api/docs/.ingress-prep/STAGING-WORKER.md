# Give the staging worker its gateway secret — install note

Mechanical. Applies to **one host: this machine**, `aaryn-tuf-rtx2060`, against
**staging only**. Mirrors `RESTORE-FLEET.md` in shape; the values are different
and that difference is the point.

The host is already registered (`591a5295-442`) and **admitted** in staging's
database, through `decide_admission` with real hardware evidence — RTX 2060,
6144 MiB, driver 580.173.02, compute 7.5, GPU verified visible inside a
container. What is missing is a heartbeat, and the heartbeat is missing because
staging runs `XCELSIOR_AGENT_PUBLIC_INGRESS=deny` and the worker has no gateway
secret to get through it.

## Read this before copying anything

**Staging's own secret. Never production's.** If the two boxes share a value, a
staging compromise is production worker ingress — the check is
hostname-independent, so whoever holds the secret is effectively at `allow`
against *any* origin that accepts it.

Confirm they differ without either value leaving its box. Same command on both;
compare the two 16-character digests, never the secrets:

```sh
# staging (this machine)
docker exec xcelsior-staging-api-1 sh -c \
  'printf "%s" "$XCELSIOR_AGENT_GATEWAY_SECRET" | sha256sum | cut -c1-16'
# -> 42fe23a76635f57f   (as of 2026-08-11)

# production
ssh <prod> 'grep -m1 ^XCELSIOR_AGENT_GATEWAY_SECRET= /opt/xcelsior/.env \
  | cut -d= -f2- | tr -d "\n" | sha256sum | cut -c1-16'
```

**If those digests match, stop.** Rotate staging's to something of its own
first; distributing a shared value is the outcome this note exists to prevent.
Both are 43 characters, so length proves nothing either way.

## Steps

**1. Copy the value out of staging's own environment.**

```sh
docker exec xcelsior-staging-api-1 sh -c \
  'printf "XCELSIOR_AGENT_GATEWAY_SECRET=%s\n" "$XCELSIOR_AGENT_GATEWAY_SECRET"' \
  >> ~/.xcelsior/staging-worker.env
chmod 600 ~/.xcelsior/staging-worker.env
```

Redirected, never printed. **Do not paste it into a ticket, a commit, or a chat.**

**2. Point the worker at staging.** Same file:

```
XCELSIOR_HOST_ID=591a5295-442
XCELSIOR_SCHEDULER_URL=http://127.0.0.1:9600
XCELSIOR_API_URL=http://127.0.0.1:9600
```

Loopback, because staging now binds `127.0.0.1` only. That was a separate fix:
with `network_mode: host` the previous `0.0.0.0` default put the worker
protocol on the LAN and the tailnet from a developer machine.

**3. Run it as a second service**, not by editing the existing one.
`xcelsior-worker.service` is the *production* worker and must keep pointing at
production. Copy the unit, give it the env file above, name it
`xcelsior-worker-staging.service`, start it.

**4. Watch the 410s stop.**

```sh
journalctl -u xcelsior-worker-staging.service -f
```

`Heartbeat failed: 410 ... agent_ingress_retired` disappearing is the signal.
Then:

```sh
docker exec xcelsior-staging-api-1 python -c "
import os, psycopg, time
with psycopg.connect(os.environ['DATABASE_URL']) as c:
    for r in c.execute(\"select host_id, status, admission_state, payload->>'last_seen' from hosts\"):
        print(r)
    print('now:', time.time())"
```

`last_seen` within 300 s of now is what the scheduler's freshness check wants.

**5. Then, and only then, run the fleet gates.**

```sh
XCELSIOR_LIVE_FLEET=1 ./scripts/run_live_gates.sh
```

The 9 that currently skip will attempt real launches.

## Rollback

Delete the line, restart the unit. Nothing else changes — no nginx, no DNS, no
certificates, and no change to the production worker.

## Two things to expect, neither of them failure

**A lag before placements succeed.** The host is admitted and will heartbeat,
but its verification stamps refresh on the sweep's own cadence. A
`require_verified` placement can keep refusing for up to an hour after the
worker is up, and that interval looks exactly like "the secret did not work".

**Log noise from the production worker.** It adopts any `xcl-*` container on
this machine and starts forwarding its logs to production, where they go
nowhere. Adoption is non-destructive when the lease claim fails — it starts no
renewal loop, and production cannot fence a job it does not have — so the
containers are safe. It is noise, not a fault. If it becomes annoying, stop the
production worker while running the gates; it is 410-ing on every heartbeat
anyway.
