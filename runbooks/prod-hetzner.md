# prod-hetzner — the production host since 2026-10-07

One Hetzner box, `prod-hetzner` (46.225.20.97), replaced both Vultr hosts on
2026-10-07: the API VPS `149.28.121.61` and the `hs.xcelsior.ca` front
`45.76.3.128`. Neither Vultr address is in service. The box also serves sites
that are not Xcelsior's, through the same nginx, which is why several of the
notes below exist.

Everything here is what the move changed or what tripped it. The host-level
record (other tenants, backups, what was archived where) is kept on the box at
`/root/PROD-HETZNER.md`, not in this public repository.

## Access

| Who | How | Notes |
|---|---|---|
| Deploys | `linuxuser@46.225.20.97`, `~/.ssh/xcelsior` | `scripts/deploy.sh` default. |
| Operator / failover standby | `root@46.225.20.97`, `~/.ssh/id_ed25519` | Root lists `~/.ssh/xcelsior` too, but only `from=` the box itself: the API container uses that key for volume management. From anywhere else sshd logs *"correct key but not from a permitted host"*. |

Reach the box directly, never through the tailnet. Snippet:
`infra/ssh/xcelsior-vps.conf` (alias `prod-hetzner`).

## nginx

- **The live binary is `/usr/sbin/nginx`.** A custom build also sits in
  `/usr/local/sbin`, and root's `PATH` finds it first, so a bare `nginx -t` or
  `nginx -T` reads a config the running server does not use. Always call
  `/usr/sbin/nginx`. The `systemd` unit `nginx.service` runs the right one.
- **Deploys install five vhosts**: `xcelsior`, `headscale`, `headscale-http`,
  `docs-xcelsior`, `downloads-xcelsior` (`scripts/install_nginx_vhosts.sh`).
  `agent-xcelsior` and `mcp-xcelsior` were placed by hand.
- **Unmatched hostnames** are owned by a `default_server` the other tenant's
  config declares, not by `nginx/unmatched-hosts.conf` (not installed here).
  Port 80 answers 404. Port 443 uses `ssl_reject_handshake on`, so a name the
  box does not serve gets no certificate at all and learns nothing about which
  other sites share the host. Before 2026-10-08 it presented another tenant's
  certificate to every unmatched name, including `*.xcelsior.ca`.
- **Port-80 traffic for a name without its own port-80 block hits that 404.**
  That is how `agent.xcelsior.ca` stopped being able to renew: its HTTP-01
  challenge now goes through the `xcelsior.ca` port-80 block in
  `nginx/xcelsior.conf`.

## Certificates

- `certbot.timer` renews every certificate on the box, twice a day, with nginx
  running. HTTP-01 lineages use webroots (`/var/www/certbot` for Xcelsior);
  `hs.xcelsior.ca` uses `dns-cloudflare`.
- `/etc/letsencrypt/renewal-hooks/deploy/reload-nginx.sh` reloads nginx after a
  renewal.
- **Do not re-enable a renewal job that stops nginx** (`--pre-hook "systemctl
  stop nginx"`). On a shared host that takes every site down and fails its own
  webroot challenges. One such timer came over in the move and is disabled.
- `certbot renew --dry-run` needs exactly one staging account. The move brought
  two, and certbot then stops to ask which, which reads as *"Missing command
  line flag or config entry for this setting"*. One was archived.
- Verified 2026-10-08: all eleven lineages pass `certbot renew --dry-run`.

## Headscale

Headscale runs here and answers `hs.xcelsior.ca` itself; see
`runbooks/headscale-failover.md`. The standby reads its host and key from
`/etc/default/headscale-failover` (host `46.225.20.97`, key `id_ed25519`), for
the timers and for a manual `sudo headscale-failover ...` alike.

## Logs

logrotate refuses a whole file when two files name the same log, and the move
brought both hosts' rules. Each log now has one owner: the stock `rsyslog`,
`btmp` and `nginx` rules, with `maxsize` caps of 100M, 50M and 200M. The
duplicate rules were archived, not deleted.

## Vultr decommissioned (2026-10-10)

1. Hetzner backups enabled on `prod-hetzner` (2026-10-08; daily, seven kept).
2. One restore actually checked (2026-10-10): the 2026-10-09 backup booted as a
   separate server with SSH in from the operator only and all egress blocked.
   Every database, table and key file matched production; the drill server was
   then deleted.
3. Both Vultr servers deleted (2026-10-10). Their full-disk copies, including
   the final pre-cutover dumps, are on the operator machine's storage, not in
   any cloud.

Restoring this host: Hetzner console → server → Backups → create a server from
the backup. Give it a firewall that blocks egress before it boots if production
is still running; otherwise it rejoins the tailnet and Lightning RPC as a second
copy.
