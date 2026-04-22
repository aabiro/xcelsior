# Snapshot Registry Configuration

The snapshot / user_images feature lets users capture their running
pods as reusable docker images. When `XCELSIOR_REGISTRY_URL` is set,
snapshots are **pushed to that registry** so any future GPU host can
pull the image back for a resume or a fresh launch.

This doc covers: deploying a registry, configuring hosts to push +
pull against it, and troubleshooting common failures.

---

## 1. Overview

Without a registry (`XCELSIOR_REGISTRY_URL` unset), snapshots remain
**local-only** on the host that took them. The user_images row is
created with `status=ready` and the image works for restarts on that
host, but the pod cannot be migrated to a different GPU host — a
known limitation for single-host deployments.

With a registry configured, the worker's `snapshot_container` handler:

1. runs `docker commit` locally,
2. runs `docker push` to the registry,
3. on push failure, runs `docker rmi` to reclaim local disk (P3/B6),
4. reports `ready` or `failed` via the API callback.

The `image_ref` format is:

```
{XCELSIOR_REGISTRY_URL}/xcelsior/{owner_slug}/{name}:{tag}
```

where `owner_slug = sanitize(owner_id)-{sha256(owner_id)[:16]}`
(P3/B5 — 16 hex chars of entropy so slugs cannot collide across
users).

---

## 2. Deploying a Registry

Any OCI-compliant registry works. Tested options:

### 2a. Harbor (recommended for self-hosted)

Harbor is the default self-hosted option because it bundles RBAC,
image scanning, and garbage collection.

```bash
# On your registry VPS:
curl -sSL https://github.com/goharbor/harbor/releases/download/v2.10.0/harbor-online-installer-v2.10.0.tgz | tar xz
cd harbor
cp harbor.yml.tmpl harbor.yml
# Edit harbor.yml: set `hostname`, enable https with a real TLS cert,
# set a strong `harbor_admin_password`.
sudo ./prepare
sudo ./install.sh --with-trivy
```

Then set on your Xcelsior API + every GPU host:

```
XCELSIOR_REGISTRY_URL=registry.xcelsior.ca/xcelsior
```

### 2b. GitHub Container Registry (ghcr.io)

For small deployments or teams already on GitHub. Create a classic PAT
with `write:packages` + `read:packages` scopes, then:

```
XCELSIOR_REGISTRY_URL=ghcr.io/your-org
```

### 2c. AWS ECR

Each host needs an IAM role (or access key) with `ecr:BatchCheckLayerAvailability`,
`ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`,
`ecr:CompleteLayerUpload`, and `ecr:GetAuthorizationToken`. Configure
via:

```
XCELSIOR_REGISTRY_URL=123456789012.dkr.ecr.us-east-1.amazonaws.com
```

ECR tokens expire every 12 hours, so on each host install a cron that
refreshes `docker login` credentials.

---

## 3. Configuring Each GPU Host

The worker runs `docker push` as the user under which `xcelsior-worker.service`
runs (default: root). Credentials must be available to that user.

### 3a. Persistent `docker login`

```bash
# On each GPU host, as root (matches the systemd service user):
sudo docker login registry.xcelsior.ca
# Enter username + password (Harbor) or PAT (ghcr.io).
```

This writes `/root/.docker/config.json` which persists across reboots.

### 3b. Systemd environment drop-in

Give the worker the registry URL without editing the unit file:

```bash
sudo mkdir -p /etc/systemd/system/xcelsior-worker.service.d
sudo tee /etc/systemd/system/xcelsior-worker.service.d/override.conf <<EOF
[Service]
Environment=XCELSIOR_REGISTRY_URL=registry.xcelsior.ca/xcelsior
EOF
sudo systemctl daemon-reload
sudo systemctl restart xcelsior-worker
```

Verify the env var is set:

```bash
sudo systemctl show xcelsior-worker --property=Environment | tr ' ' '\n' | grep REGISTRY
```

---

## 4. TLS Requirements

Docker **rejects unencrypted registry endpoints by default**. Two paths:

### 4a. Valid public cert (recommended)

Terminate TLS on the registry with a certificate from Let's Encrypt,
your internal PKI, or your cloud provider. Docker will trust it
automatically.

### 4b. Self-signed / internal CA

Either install the CA bundle system-wide:

```bash
sudo cp internal-ca.pem /usr/local/share/ca-certificates/xcelsior-ca.crt
sudo update-ca-certificates
sudo systemctl restart docker
```

or, as a last resort, mark the registry insecure:

```bash
# /etc/docker/daemon.json on each host
{
  "insecure-registries": ["registry.internal:5000"]
}
sudo systemctl restart docker
```

> **Security note:** `insecure-registries` disables TLS verification.
> Only use for isolated networks (e.g., Tailnet-only registry); never
> over the public internet.

---

## 5. Troubleshooting

### push failed: denied: requested access to the resource is denied

Auth problem. Check:

- `sudo docker login <registry>` succeeds on the host
- `/root/.docker/config.json` contains the registry entry
- The account has **push** permission on the target namespace
  (Harbor: project member with Developer role or higher)

### push failed: unauthorized: authentication required (only some hosts)

A host forgot to run `docker login`. Re-run step 3a on the affected host.

### push failed: net/http: TLS handshake timeout

Network / firewall issue. From the host:

```bash
curl -v https://registry.xcelsior.ca/v2/
```

Should return an HTTP 200 or 401. Any other failure → fix routing
before retrying snapshots.

### user_images rows stuck at status=pending for hours

The worker callback is not reaching the API. Check:

```bash
journalctl -u xcelsior-worker -n 100 --no-pager | grep -i snapshot
```

Look for `commit failed: ...` (docker issue), `push failed: ...` (registry
auth/TLS), or the B7 distinct error messages. Then look at the API:

```bash
curl -s "$API/admin/user-images?status=pending" -H "Authorization: Bearer $ADMIN_TOKEN"
```

The P3/A5 `user_images_pending_sweeper` bg_worker task marks rows
older than 2 hours as `failed` automatically, so the stuck state is
self-healing.

### Disk fills up on a GPU host

Each committed layer occupies disk until pushed *or* `docker rmi`'d.
P3/B6 removes the local tag on push failure, but orphaned dangling
layers can accumulate from crashes. Run periodically on each host:

```bash
docker image prune -a -f --filter "until=168h"  # > 7 days old
```

---

## 6. Verifying End-to-End

After configuring a host, trigger a real snapshot as a test user:

```bash
# Grab an instance id of a running container:
curl -s "$API/instances" -H "Authorization: Bearer $USER_TOKEN" | jq '.[] | select(.status == "running") | .job_id'

# Snapshot it:
curl -X POST "$API/instances/$JOB_ID/snapshot" \
  -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"registry-probe","tag":"v1","description":"registry smoke test"}'

# Wait ~30s, then confirm it's ready:
curl -s "$API/user-images" -H "Authorization: Bearer $USER_TOKEN" | jq '.[] | select(.name == "registry-probe")'
```

If `status=ready` with `size_bytes > 0`, the registry is fully wired.
