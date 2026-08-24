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
