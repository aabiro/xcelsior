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

## Docker maintenance

Apply the [daemon configuration](../infra/docker/daemon.json) during an agreed
maintenance window for the shared host.

1. Record the running containers and their health. Check `systemctl cat docker`
   for daemon flags and note any host-specific keys in `/etc/docker/daemon.json`.
   Copy the existing file to an operator-only location if it exists.
2. Merge the log-driver, log-options and builder settings into a candidate that
   retains the host-specific keys. Preserve the current DNS configuration if
   the host already has one. Review the daemon flags for duplicate options.
   Run `sudo dockerd --validate --config-file=/path/to/daemon.candidate.json`
   before installing the candidate at `/etc/docker/daemon.json`.
3. Restart with `sudo systemctl restart docker`. Confirm that Docker is active
   and that the affected services recover, including the public API and MCP.
   If startup fails or services regress, restore the saved config (or remove
   the candidate if there was no prior file), restart Docker, and recheck health.
4. Verify `docker info --format '{{.LoggingDriver}}'` returns `local`. The
   default applies when containers are created. Recreate running services
   through their normal rollout and verify the driver and caps on each new
   container with `docker inspect --format '{{json .HostConfig.LogConfig}}' CONTAINER`.
   [Docker logging configuration](https://docs.docker.com/engine/logging/configure/)

## Workstation image release

Run the build on the ASUS workstation with Buildx and Compose. The API image
also supplies the workers and SSH gateway. The frontend and MCP each have
their own image. A local build works with `python3 api/scripts/release_images.py build`.

For a release, use a clean checkout of the approved commit on `origin/main`.
Set up GHCR push access for the workstation and keep the previous release
manifest for rollback. Put the public frontend settings matching the target
host in a separate dotenv file and write the manifest outside the checkout.

```sh
python3 api/scripts/release_images.py build --push \
  --frontend-env /path/to/frontend-public.env --manifest /path/to/release.json
XCELSIOR_DEPLOY_IMAGE_MANIFEST=/path/to/release.json \
  XCELSIOR_DEPLOY_GHCR_AUTH=github-cli bash api/scripts/deploy.sh
```

The manifest binds the three image digests to the checked-out commit. The
GitHub CLI mode uses the workstation's GitHub session to request a separate
pull token for each package. It streams each token to the host for the
corresponding Docker pull. The workstation keeps its GitHub credential.
Deployments with a dedicated GHCR read credential already configured on the
host can use the default `XCELSIOR_DEPLOY_GHCR_AUTH=host`.

The deploy script checks the frontend build settings against the production
Compose settings, pulls those digests, and verifies their source revision
before migrations. Migrations and blue-green activation use the staged
Compose overlay with builds disabled for those services. Keep supplying the
manifest on future releases to use this path.

For a manual Compose start after a release, use the same base files and the
versioned overlay in `/opt/xcelsior-backups/.deploy-state/`. For rollback, use
the corresponding prior checkout and manifest after reviewing schema
compatibility. The normal deploy script guards the active services during
blue-green swaps.

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
