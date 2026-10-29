# Agent-ingress cutover — prepared, **not applied**

Nothing here has been applied to any server. No symlink, no reload, no secret
distributed. `149.28.121.61` carries the whole PixelEnhance estate as well as
xcelsior's, and the same symlink lever has already caused damage on that box.

## What was found

**The extraction is already done.** `nginx/agent-xcelsior.conf` is a standalone
vhost in this repo — it does not need carving out of the 17 KB
`sites-available/xcelsior`. It is a real mTLS terminator:
`ssl_verify_client on`, a client CA, per-client-cert rate limiting, and
`$ssl_client_s_dn` mapped to an agent host id.

**It has never been installed.** It is referenced by no deploy step, it is not in
`/etc/nginx/sites-available/` on the box, and it is not enabled.

**Its prerequisites do not exist on the box.** All three are missing:

    /etc/nginx/certs/agent.xcelsior.ca/fullchain.pem
    /etc/nginx/certs/agent.xcelsior.ca/privkey.pem
    /etc/nginx/certs/agent-client-ca.pem

So the cutover is not one symlink away. It needs a server cert for
`agent.xcelsior.ca` and, more substantially, **a client CA that does not yet
exist** — every worker needs a cert issued from it before `ssl_verify_client on`
will admit anything. Enabling the vhost before that would take the hostname over
and refuse every agent, which is worse than the current 410.

## Baseline

`nginx-T.baseline.txt` — full `nginx -T` from the box, 3248 lines, 28 distinct
`server_name` lines. Diff a post-change `nginx -T` against it before reloading:
`nginx -t` proves the config parses, only the `-T` diff proves no other hostname
was silently re-homed. The current fallthrough serving the `docs.xcelsior.ca`
certificate for `agent.xcelsior.ca` is evidence that default-vhost resolution on
that box already does something nobody chose.

## The ordering hazard, and the small fix that removes it

**No block on that box declares `default_server`.** nginx falls back to the
first block in config order, the includes are globs, so the default is decided
by **filename**. `agent-xcelsior.conf` sorts ahead of every vhost in this repo
and every PixelEnhance hostname on the box.

Symlink it as-is and it becomes the implicit default. With
`ssl_verify_client on`, it would then refuse **every unmatched hostname across
both estates** at the TLS handshake — not misroute one site, refuse all of them.

`nginx/unmatched-hosts.conf` claims the default explicitly and returns 421.
After that, filename order stops deciding anything, for both estates and for
every hostname added later — and it fixes the live bug where
`agent.xcelsior.ca` is served the `docs.xcelsior.ca` certificate. It is one
directive on one block; do it first and independently of the cutover.

`hostname-map.sh` extracts `server_name -> (certificate) [DEFAULT]` from an
`nginx -T` dump so the before/after diff is on the mapping that silently moves
rather than on 3,248 lines of shifted context. Baseline in
`hostname-map.baseline.txt`: 44 hostnames, **zero** marked DEFAULT.

## Order, if it goes ahead

1. Issue the client CA and a worker cert; server cert for `agent.xcelsior.ca`.
2. Copy `nginx/agent-xcelsior.conf` to `sites-available/agent-xcelsior`.
3. `nginx -t`, then capture `nginx -T` and diff against the baseline here.
4. Symlink, reload, verify one worker end to end before repointing the fleet.

Not done, not scheduled, and not mine to schedule.
