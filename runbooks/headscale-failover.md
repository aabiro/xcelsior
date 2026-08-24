# Headscale control-plane failover

When the VPS that serves `https://hs.xcelsior.ca` is down, the Xcelsior mesh
has no coordination server. Peers that already have a live WireGuard session
keep talking; any client that reboots cannot fetch the Noise control key and
falls into `NoState`. That is the "there is no Tailscale" failure.

This runbook keeps an off-box copy of Headscale's sqlite database and Noise
private key, promotes a standby that presents the **same** `hs.xcelsior.ca`
identity (same `100.64.0.0/10` node IPs), and copies that state back before
DNS is pointed at the VPS again.

## What must never happen

- Do **not** promote the empty Headscale that ships on a laptop (`users=0`,
  `nodes=0`). That mints a new tailnet. When the VPS returns the IPs will not
  match.
- Do **not** `tailscale logout` or `tailscale up --force-reauth` during an
  outage. Re-auth against a different Headscale identity creates new node IDs.
- Do **not** flip `hs.xcelsior.ca` DNS back to the VPS before the replica
  sqlite + `noise_private.key` have been copied there. Failback order is
  copy, start Headscale on the VPS, then DNS.
- Do **not** orange-cloud `hs.xcelsior.ca`. WebSockets and Noise break behind
  the Cloudflare proxy.

## Topology

| Role | Address | Notes |
|---|---|---|
| Headscale primary | `45.76.3.128` (`hs.xcelsior.ca`, DNS-only, TTL 60–120) | Vultr, SSH as `root` |
| API VPS | `149.28.121.61` | Must be reachable **without** ProxyJump through the Headscale VPS. Use `ssh xcelsior-api`. |
| Standby | the host running `headscale-failover` (this laptop unless moved) | Replica lives in `/var/backups/headscale` |
| Public DERP | Tailscale's published map | NAT relay must not depend on the VPS |

`scripts/headscale_failover.py` is the operator CLI, installed as
`/usr/local/sbin/headscale-failover`.

## During an outage (now)

1. Check whether a replica exists:
   ```bash
   sudo headscale-failover status --json
   ```
   `replica.promotable` must be true. If it is false (`users=0` / `nodes=0`),
   there is **no** identity to restore until the VPS is back and `replicate`
   succeeds once. Do not promote.

2. If the replica is promotable:
   ```bash
   sudo headscale-failover promote
   ```
   That installs sqlite + Noise key into `/var/lib/headscale`, points the
   Cloudflare A record at the standby public IP (still grey-cloud), and
   restarts Headscale. Clients reconnect to the same node IDs.

3. Home-LAN clients that cannot wait for DNS: add
   `<standby-lan-ip> hs.xcelsior.ca` to `/etc/hosts`, then
   `sudo tailscale --socket=/var/run/tailscale-xcelsior.sock up`.

4. Watchdog (`headscale-failover-watchdog.timer`) will promote automatically
   after 3 failed health checks **only** when the replica is promotable. It
   will **not** fail back automatically — that is a split-brain risk.

## Rebuilding identity by hand (no replica was ever captured)

> **Done on 2026-08-24.** `asus-pc` is back on `100.64.0.6`,
> `aaryns-macbook-pro` on `100.64.0.3`, and `100.64.0.1` is left free for the
> VPS. Both peers hold a direct LAN session and
> `tests/test_anchor_workloads.py` passes against the hardcoded
> `aaryn@100.64.0.3` with no override. The corrected database is copied to
> `/var/lib/headscale/db.sqlite.restored-20260824-163220` — **that copy is the
> only record of this addressing**, so keep it until a real replica exists.
>
> **Read this before the VPS comes back.** Both nodes are now pointed at
> `--login-server=http://192.168.1.127:8080` — the LAN control plane, not
> `hs.xcelsior.ca`. This is a *different tailnet* from the VPS's, sharing only
> the addressing. When `45.76.3.128` returns it will still hold its own
> authoritative database, and these two nodes will **not** move back on their
> own: each needs `tailscale up --login-server=https://hs.xcelsior.ca --reset`.
> Decide deliberately which database wins before doing that — the VPS's is the
> older, larger one; this one is two nodes rebuilt by hand.
>
> It also means the mesh is LAN-only until then. That is fine for the anchor
> workloads, which run between two boxes on the same switch, and not fine for
> anything expecting to reach these nodes from outside.



The case this runbook did not cover, and the one we are in. `replicate` never
succeeded before `45.76.3.128` went dark on 2026-08-19 —
`headscale-failover status --json` shows `promotable: false`, "no replicated
sqlite database", `last_replicate_ok_at: ""`, and thousands of `refuse_promote`
decisions. The watchdog is correct to refuse: there is nothing to restore.

So the addresses have to be reassigned by hand, and **the old assignments are
the specification**, recovered from what the code already depends on:

| Address | Node | Where it is written down |
|---|---|---|
| `100.64.0.1` | **the VPS** | `XCELSIOR_NFS_SERVER`, worker `~/.xcelsior/worker.env` |
| `100.64.0.3` | the Mac | `XCELSIOR_MAC_HOST=aaryn@100.64.0.3` |
| `100.64.0.6` | ASUS RTX 2060 (this box) | `XCELSIOR_MAC_INFERENCE_API_HOST`, `--worker-host` |

A freshly promoted Headscale allocates sequentially from `100.64.0.0/10`, so the
first node to register takes **`.1` — the VPS's address**. That is what happened:
`asus-pc` holds `.1` today. Left alone it collides the moment the VPS returns.

Headscale v0.28 has no `nodes set-ip`; `backfillips` only fills empties. The
assignment lives in `nodes.ipv4` / `nodes.ipv6`, and Headscale caches node state,
so the edit must happen while it is stopped.

```bash
# 1. Back up first, always.
sudo cp -a /var/lib/headscale/db.sqlite \
           /var/lib/headscale/db.sqlite.bak-$(date +%Y%m%d-%H%M%S)
sudo systemctl stop headscale

# 2. Put this box back on .6 and leave .1 free for the VPS.
sudo sqlite3 /var/lib/headscale/db.sqlite \
  "UPDATE nodes SET ipv4='100.64.0.6', ipv6='fd7a:115c:a1e0::6' WHERE hostname='asus-pc';"

sudo systemctl start headscale
sudo systemctl restart tailscaled-xcelsior   # or the unit running the custom socket
tailscale --socket=/var/run/tailscale-xcelsior.sock status
```

Then bring the Mac on. It runs the **App Store** build
(`~/Library/Containers/io.tailscale.ipn.macos`), whose CLI lives inside the
bundle, and it is currently in `NoState` — logged out, still trying to fetch the
control key from the dead VPS.

```bash
# On this box: mint a key.
sudo headscale preauthkeys create --user 1 --expiration 1h

# On the Mac (ssh aaryn@192.168.1.87):
TS=/Applications/Tailscale.app/Contents/MacOS/Tailscale
"$TS" logout
"$TS" up --login-server=http://192.168.1.127:8080 --authkey=<key> --accept-routes

# Back here: pin it to .3, the address the code expects.
sudo systemctl stop headscale
sudo sqlite3 /var/lib/headscale/db.sqlite \
  "UPDATE nodes SET ipv4='100.64.0.3', ipv6='fd7a:115c:a1e0::3' WHERE hostname LIKE '%acbook%';"
sudo systemctl start headscale
```

Verify with the thing that actually depends on it:

```bash
.venv/bin/python -m pytest tests/test_anchor_workloads.py -q
```

That test skips when the Mac is unreachable and runs when it is not, so a pass
means the tailnet genuinely carries the address. It has been confirmed to pass
over the LAN today with `XCELSIOR_MAC_HOST=aaryn@192.168.1.87` — the Mac, the
SSH path and the remote builders are all fine, and the address is the only thing
missing.

**`server_url` is the one thing to think about before doing this.** The live
config uses `http://192.168.1.127:8080`, which works only while both boxes are
on the same LAN. `infra/headscale/config.yaml` in this repo uses
`https://hs.xcelsior.ca` behind the standby nginx, which is what a real promotion
should use — and needs the DNS move that `headscale-failover promote` performs.
The LAN URL is fine for restoring local reachability; it is not a promotion.

**Capture a replica as soon as the VPS is reachable again.** The whole reason
this section exists is that no replica existed when it was needed:

```bash
sudo systemctl start headscale-replicate.timer
sudo headscale-failover replicate
```

## When the VPS returns

```bash
sudo headscale-failover failback
```

Failback:

1. Snapshots the standby database (captures any nodes seen during the outage).
2. Refuses if the VPS is still unreachable or the replica is empty.
3. Stops Headscale on the VPS, copies sqlite + Noise key, starts Headscale.
4. Points `hs.xcelsior.ca` back at `45.76.3.128`.
5. Leaves the standby in `role=standby` and resumes replica refresh.

Node IPs are the ones in sqlite. They do not change.

## Routine (VPS healthy)

- `headscale-replicate.timer` — every 15 minutes, sqlite `.backup` + Noise
  key off the VPS into `/var/backups/headscale`.
- `headscale-failover-watchdog.timer` — every 2 minutes, health check.
- Confirm: `sudo headscale-failover status --json` shows `promotable: true`
  and a recent `replicated_at`.

## First install on a standby host

```bash
sudo install -m 0755 scripts/headscale_failover.py /usr/local/sbin/headscale-failover
sudo mkdir -p /var/backups/headscale
sudo chmod 700 /var/backups/headscale
sudo cp infra/systemd/headscale-replicate.* infra/systemd/headscale-failover-watchdog.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now headscale-replicate.timer headscale-failover-watchdog.timer
sudo headscale-failover replicate   # requires the VPS to be up
sudo headscale-failover status --json
```

SSH to the Headscale VPS uses `root@45.76.3.128` and `~/.ssh/id_ed25519`
(`XCELSIOR_HEADSCALE_HOST` / `XCELSIOR_HEADSCALE_SSH_KEY` override).

Cloudflare and Telegram credentials are read as **data** from the project
`.env` (`CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ZONE_ID`, `XCELSIOR_TG_TOKEN`,
`XCELSIOR_TG_CHAT_ID`). The file is not executed.

## Client hardening

`tailscaled-xcelsior` must keep `/var/lib/tailscale-xcelsior/tailscaled.state`.
Never pass `--reset`. After a control-plane outage the daemon reports
`NoState` until `hs.xcelsior.ca` answers again; once Headscale is back (primary
or promoted replica with the same Noise key) a plain `tailscale up` rebinds
the existing node. `--force-reauth` is the last resort, and only against the
same login server.

## DERP

`infra/headscale/config.yaml` sets `derp.server.enabled: false` and uses
`https://controlplane.tailscale.com/derpmap/default`. Mesh NAT traversal then
survives the VPS even before DNS failover, as long as clients still have a
netmap. Apply that config on the primary when it is next reachable.

## Limits

- The standby host has to be running. A laptop that is asleep cannot promote.
- Home NAT needs port 443 forwarded to the standby, **or** LAN `/etc/hosts`,
  for clients off the LAN to reach a promoted Headscale.
- There is no Headscale replica until `replicate` has succeeded at least once
  against the live VPS. An outage before that first copy cannot preserve IPs.
