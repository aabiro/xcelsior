# Agent mTLS cutover — runbook

**Nothing here has been applied.** Do `RESTORE-FLEET.md` first; that brings the
fleet back today with no certificates and no nginx change. This is the
destination, not the unblock, and it can be scheduled properly.

`149.28.121.61` carries the whole PixelEnhance estate as well as xcelsior's.
**No server block on that box declares `default_server`**, so nginx falls back to
the *first* block in config order for any unmatched hostname — which is why
`agent.xcelsior.ca` is currently served the `docs.xcelsior.ca` certificate.
Include order is load-bearing and invisible. That is what the `nginx -T` diff in
step 5 is for; `nginx -t` cannot see it.

## Step 0 — claim the default, before anything else

**Smaller than the cutover and it removes the failure mode that makes the
cutover risky.** One directive on one block, and it fixes today's live bug as a
side effect: `agent.xcelsior.ca` is served the `docs.xcelsior.ca` certificate
because nothing owns unmatched traffic.

No block on that box declares `default_server`, so nginx falls back to the first
in config order — and the includes are globs, so that means **filename order**.
`agent-xcelsior.conf` sorts ahead of every vhost in this repo and every
PixelEnhance hostname on the box. Symlinking it without this step makes an
`ssl_verify_client on` block the implicit default, which refuses *every*
unmatched hostname across both estates.

1. Copy `nginx/unmatched-hosts.conf` → `/etc/nginx/sites-available/unmatched-hosts`,
   symlink it into `sites-enabled/`.
2. `sudo nginx -t`.
3. `sudo nginx -T > /tmp/after.txt`, then
   `./hostname-map.sh /tmp/after.txt > /tmp/after-map.txt` and
   `diff hostname-map.baseline.txt /tmp/after-map.txt`.
   **Expected: exactly one added line, `_ -> …/docs.xcelsior.ca/fullchain.pem [DEFAULT]`.**
   Any *existing* hostname whose certificate or block changed means it has been
   re-homed — stop and roll back.
4. `sudo systemctl reload nginx`.
5. Confirm: `curl -sk -o /dev/null -w '%{http_code}\n' https://agent.xcelsior.ca/`
   should now return **421**, not a docs page.

Rollback: remove the symlink, `nginx -t`, reload. Ordering returns to what it is
today.

Once this is done, adding the agent vhost cannot change what any other hostname
resolves to, and the rest of this runbook is an ordinary vhost install.

## Before the window

1. **Prove the mechanism locally.** `./mtls-dryrun.sh` — three cases in a
   throwaway container: valid cert accepted, missing refused, wrong-CA refused.
   Each must fail independently. Touches nothing remote. *(Verified passing.)*
2. **Issue the CA and worker certs.** `OUT=/secure/path ./issue-agent-certs.sh ca`,
   then `... worker <host_id>` for each of the 4 prod hosts and the local worker.
   The CA key never leaves `OUT`. Only `<host>.crt` / `<host>.key` go to a
   worker; only `agent-client-ca.pem` goes to nginx.
3. **Server certificate** for `agent.xcelsior.ca` into
   `/etc/nginx/certs/agent.xcelsior.ca/`. All three paths the vhost references
   are currently absent — confirmed.

## In the window

4. Copy `nginx/agent-xcelsior.conf` → `/etc/nginx/sites-available/agent-xcelsior`.
   **Do not symlink yet.**
5. `sudo nginx -t` — proves it parses.
6. `sudo nginx -T > /tmp/after.txt`, then compare the **mapping**, not the raw
   config: `./hostname-map.sh /tmp/after.txt | diff hostname-map.baseline.txt -`.
   Diffing `nginx -T` directly is the wrong comparison — 3,248 lines, and the
   new block shifts context everywhere, drowning the thing that matters. What
   matters is whether any *existing* hostname now resolves to a different block
   or certificate. **Expected: one added line for `agent.xcelsior.ca`.**
7. Symlink into `sites-enabled/`, `sudo nginx -t` again, `sudo systemctl reload nginx`.
8. **Verify one worker before the fleet.** Point a single host at
   `https://agent.xcelsior.ca` with its client cert; confirm heartbeats stop
   returning 410. Only then repoint the rest.
9. Remove `XCELSIOR_AGENT_GATEWAY_SECRET` from the workers once mTLS carries
   them — the stopgap should not outlive its need, since it is a bearer
   credential on public ingress.

## Rollback

Remove the symlink, `nginx -t`, reload. Workers keep the shared-secret path and
stay reachable throughout, which is the point of doing `RESTORE-FLEET.md` first:
this cutover is then never load-bearing for availability.

## Checks that must hold afterwards

- `tests/test_agent_vhost_cannot_take_the_default.py` — the vhost must not claim
  `default_server`, or on this box it captures every unmatched hostname and,
  with `ssl_verify_client on`, refuses all of them.
- The PixelEnhance hostnames in the baseline still resolve to their own blocks.
