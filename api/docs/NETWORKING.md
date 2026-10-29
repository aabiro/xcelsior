# Xcelsior Networking & Mesh Infrastructure

> Last updated: 2026-02-13

## Overview

Xcelsior uses a self-hosted **Headscale** control server (open-source Tailscale coordination) to create a private WireGuard mesh network connecting GPU workers and the central API server. This runs alongside (not replacing) any existing personal Tailscale network.

Headscale is a single control plane. If the VPS that serves `hs.xcelsior.ca` is down, clients that reboot cannot fetch the Noise control key and the mesh disappears (`NoState`). Off-box sqlite + Noise-key replication, standby promotion, and failback that restores the **same** `100.64.0.0/10` node IPs are documented in [`runbooks/headscale-failover.md`](../runbooks/headscale-failover.md). Operator CLI: `scripts/headscale_failover.py`.

---

## Domain & DNS

| Record | Type | Value | Proxy |
|---|---|---|---|
| `xcelsior.ca` | A | `149.28.121.61` | Cloudflare (proxied) |
| `www.xcelsior.ca` | CNAME | `xcelsior.ca` | Cloudflare (proxied) |
| `hs.xcelsior.ca` | A | `45.76.3.128` | DNS only (TTL 60–120) |
| `*.xcelsior.ca` | CNAME | `xcelsior.ca` | DNS only |

**Cloudflare nameservers:** `frida.ns.cloudflare.com`, `harvey.ns.cloudflare.com`

**Anti-spoofing records:**
- SPF: `v=spf1 -all`
- DMARC: `v=DMARC1; p=reject; sp=reject; adkim=s; aspf=s;`

> `hs.xcelsior.ca` must be **DNS only** (grey cloud) — Headscale uses WebSockets and long-lived connections that don't work well behind Cloudflare's proxy.

---

## Network Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Headscale Control Server                │
│     https://hs.xcelsior.ca → 45.76.3.128 (Vultr)         │
│         nginx → 127.0.0.1:8080 (headscale)               │
│     sqlite + Noise key replicated off-box every 15 min   │
└──────────┬──────────────────────────┬────────────────────┘
           │                          │
     WireGuard mesh              WireGuard mesh
     (port 41641)                (port 41642)
           │                          │
┌──────────▼──────────┐   ┌──────────▼──────────────────┐
│   xcelsior-vps      │   │   aaryn-gpu-laptop           │
│   100.64.0.1        │   │   100.64.0.2                 │
│   tailscale0        │   │   tailscale1 (Xcelsior)      │
│   149.28.121.61     │   │   tailscale0 (Personal)      │
│                     │   │   100.85.179.112              │
└─────────────────────┘   └─────────────────────────────┘
```

---

## Headscale Server (VPS)

- **Host:** `45.76.3.128` (Vultr, SSH `root@45.76.3.128`, aliases `aarynfans-root` / `vps-root`)
- **API VPS (not Headscale):** `149.28.121.61` — SSH `xcelsior-api` (direct, no ProxyJump)
- **Version:** Headscale v0.28.0
- **Config:** `/etc/headscale/config.yaml` (repo copy: `infra/headscale/config.yaml`)
- **ACL:** `/etc/headscale/acl.json`
- **Database:** `/var/lib/headscale/db.sqlite`
- **Off-box replica:** `/var/backups/headscale` on the standby host
- **User:** `xcelsior` (ID: 1)
- **Health check:** `curl -s https://hs.xcelsior.ca/health` → `{"status":"pass"}`
- **DERP:** public Tailscale DERP map (embedded VPS DERP stays off so NAT relay survives a VPS outage)

### Nginx Reverse Proxy

Headscale listens on `127.0.0.1:8443` (TLS disabled internally). Nginx handles TLS termination:

- **Config:** `/etc/nginx/sites-available/headscale`
- **Cert:** `/etc/letsencrypt/live/hs.xcelsior.ca/` (auto-renewed by certbot)
- **HTTP redirect:** `/etc/nginx/sites-available/headscale-http`

### Key Headscale Commands (run on VPS)

```bash
# Health check
curl -s https://hs.xcelsior.ca/health

# List nodes
headscale nodes list

# Create a new pre-auth key (reusable, 90-day expiry)
headscale preauthkeys create --user xcelsior --reusable --expiration 90d

# List pre-auth keys
headscale preauthkeys list --user xcelsior

# Remove a node
headscale nodes delete -i <NODE_ID>

# Rename a node
headscale nodes rename -i <NODE_ID> <new-name>
```

---

## Mesh Nodes

| Node | Mesh IP | OS | Role | TUN Device |
|---|---|---|---|---|
| `xcelsior-vps` | `100.64.0.1` | Ubuntu 24.04 | API server, scheduler | `tailscale0` |
| `aaryn-gpu-laptop` | `100.64.0.2` | Ubuntu (ASUS TUF A15) | GPU worker | `tailscale1` |

---

## Dual Tailscale Setup (Laptop)

The GPU laptop runs **two independent tailscaled instances** simultaneously:

| Property | Personal Tailscale | Xcelsior Headscale |
|---|---|---|
| **Service** | `tailscaled.service` (default) | `tailscaled-xcelsior.service` |
| **TUN device** | `tailscale0` | `tailscale1` |
| **Socket** | `/var/run/tailscale/tailscaled.sock` | `/var/run/tailscale-xcelsior.sock` |
| **State dir** | `/var/lib/tailscale/` | `/var/lib/tailscale-xcelsior/` |
| **WireGuard port** | 41641 | 41642 |
| **Mesh IP** | `100.85.179.112` | `100.64.0.2` |
| **Control server** | `login.tailscale.com` | `https://hs.xcelsior.ca` |
| **Peers** | MacBook Pro, iPhone 14 Pro | xcelsior-vps |

### Systemd Service File

`/etc/systemd/system/tailscaled-xcelsior.service`:

```ini
[Unit]
Description=Tailscale daemon (Xcelsior Headscale mesh)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/sbin/tailscaled --state=/var/lib/tailscale-xcelsior/tailscaled.state --socket=/var/run/tailscale-xcelsior.sock --port=41642 --tun=tailscale1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### CLI Usage

All commands for the Xcelsior network require the `--socket` flag:

```bash
# Check Xcelsior mesh status
sudo tailscale --socket=/var/run/tailscale-xcelsior.sock status

# Ping a mesh peer
sudo tailscale --socket=/var/run/tailscale-xcelsior.sock ping 100.64.0.1

# Check IP
sudo tailscale --socket=/var/run/tailscale-xcelsior.sock ip

# Disconnect (without removing)
sudo tailscale --socket=/var/run/tailscale-xcelsior.sock down

# Reconnect
sudo tailscale --socket=/var/run/tailscale-xcelsior.sock up
```

Personal Tailscale uses the default `tailscale` command with no extra flags.

---

## Adding a New GPU Worker

1. **Install Tailscale** on the new machine ([tailscale.com/download](https://tailscale.com/download))

2. **Get a pre-auth key** from the VPS:
   ```bash
   ssh linuxuser@149.28.121.61
   headscale preauthkeys create --user xcelsior --reusable --expiration 90d
   ```

3. **Join the mesh:**
   ```bash
   sudo tailscale up \
     --login-server https://hs.xcelsior.ca \
     --authkey <PREAUTH_KEY> \
     --hostname <worker-name>
   ```

4. **Verify:**
   ```bash
   tailscale status
   tailscale ping 100.64.0.1   # ping VPS
   ```

> If the machine already runs a personal Tailscale, follow the [Dual Tailscale Setup](#dual-tailscale-setup-laptop) pattern above — create a second `tailscaled` instance with separate state, socket, port, and TUN device.

---

## Environment Variables

Set in `.env` on any machine running the Xcelsior worker or API:

```bash
XCELSIOR_TAILSCALE_ENABLED=true
XCELSIOR_HEADSCALE_URL=https://hs.xcelsior.ca
XCELSIOR_TAILSCALE_AUTHKEY=<preauth-key>
```

---

## Troubleshooting

### Second tailscaled won't start / socket missing

```bash
sudo journalctl -u tailscaled-xcelsior --no-pager -n 30
```

**Common cause:** TUN device conflict — both instances trying to use `tailscale0`. Ensure `--tun=tailscale1` (or `tailscale2`, etc.) is in the `ExecStart` line.

### "device or resource busy" on TUN

The first tailscaled already owns `tailscale0`. Each instance needs a unique `--tun` name.

### Pre-auth key expired

Keys expire after 90 days. Generate a new one on the VPS and re-authenticate:

```bash
# On VPS
headscale preauthkeys create --user xcelsior --reusable --expiration 90d

# On worker
sudo tailscale --socket=/var/run/tailscale-xcelsior.sock up \
  --login-server https://hs.xcelsior.ca \
  --authkey <NEW_KEY> \
  --force-reauth
```

### Headscale health check failing

```bash
# On VPS
sudo systemctl status headscale
sudo journalctl -u headscale --no-pager -n 20
sudo nginx -t && sudo systemctl reload nginx
curl -s https://hs.xcelsior.ca/health
```

### Cloudflare proxy interfering with Headscale

`hs.xcelsior.ca` must be **DNS only** (grey cloud icon). Cloudflare's HTTP proxy breaks WebSocket/long-poll connections that Headscale requires.

### VPS down / white mesh / `NoState`

Clients report `fetch control key: ... hs.xcelsior.ca` and `BackendState: NoState`. Do not `tailscale logout` or `--force-reauth`. Follow [`runbooks/headscale-failover.md`](../runbooks/headscale-failover.md):

```bash
sudo headscale-failover status --json    # replica.promotable must be true
sudo headscale-failover promote          # only then
# when the VPS is back:
sudo headscale-failover failback         # copies sqlite+Noise to the VPS, then DNS
```

An empty local Headscale (`users=0`, `nodes=0`) is not a replica. Promoting it would issue new `100.64.0.x` addresses.

SSH to the API VPS during a Headscale-VPS outage: `ssh xcelsior-api` (direct to `149.28.121.61`). Do not ProxyJump through `45.76.3.128`.
