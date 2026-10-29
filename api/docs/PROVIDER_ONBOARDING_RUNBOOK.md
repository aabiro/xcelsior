# Provider Onboarding Runbook

*Last updated: 2026-10-11*

## Quick paths

| Goal | Command |
|------|---------|
| Full wizard | `curl -fsSL https://xcelsior.ca/install.sh \| bash` |
| Agent only (no Node.js) | `curl -fsSL https://xcelsior.ca/install.sh \| bash -s -- --agent-only` |
| Update agent | `bash install.sh --update` |
| Uninstall units | `bash install.sh --uninstall` |

## What the installer creates

| Path | Contents |
|------|----------|
| `~/.xcelsior/worker.env` | Host ID, scheduler URL, credentials (mode 600) |
| `/opt/xcelsior/` (owner `xcelsior`) | Signed agent modules + venv (was `~/.xcelsior/` before issue #65) |
| `xcelsior-privd.socket`, `/usr/local/libexec/xcelsior/` | Root helper for mounts/LUKS/exclusive-GPU signals; limits in `/etc/xcelsior/privileged.conf` |
| `/etc/systemd/system/xcelsior-worker.service` | The service (`systemctl cat xcelsior-worker`) |
| `/var/lib/xcelsior/` | Worker state (attempt journal); systemd `StateDirectory=`, passed as `$STATE_DIRECTORY` |

## Install fails

1. **Signature/hash error** — API may be serving an unsigned agent. Ensure `scripts/sign-static-agent.py` ran in CI and signatures are deployed.
2. **Missing openssl/python3/docker** — install prerequisites per `install.sh` platform guard.
3. **Wizard npm failure** — installer falls back to `--agent-only` automatically (B5).

## Agent crash-loop

```bash
journalctl -u xcelsior-worker -n 100 --no-pager
```

Common causes:
- **Missing venv deps (B1)** — `~/.xcelsior/venv` should exist; reinstall with `bash install.sh --update`.
- **Missing `security.py`** — `PYTHONPATH` must include `~/.xcelsior/lib`.
- **`ModuleNotFoundError: env_config` / `criu_hosts`** — agent installed before 2026-10-11 (PR #63); run `bash install.sh --update`.
- **`status=226/NAMESPACE`** — unit from before PR #63 lists a missing `/var/lib/xcelsior` in `ReadWritePaths`; `bash install.sh --update` rewrites it with `StateDirectory=xcelsior`.
- **Bad credentials** — check `~/.xcelsior/worker.env` (mode 600).

## Host not appearing in dashboard

1. Confirm `XCELSIOR_HOST_ID` in `worker.env` matches dashboard.
2. Run post-install check: `curl -s "$XCELSIOR_API_URL/host/$HOST_ID"`.
3. Verify heartbeat: `journalctl -u xcelsior-worker -f` should show successful `PUT /host`.

## Snapshot push fails

1. Check registry login: `journalctl -u xcelsior-registry-login`.
2. Ensure `XCELSIOR_REGISTRY_URL`, `_USERNAME`, `_PASSWORD` in `worker.env`.

## SDK quickstarts

### TypeScript (`@xcelsior-gpu/sdk`)

```bash
npm install @xcelsior-gpu/sdk
```

```typescript
import { XcelsiorApiClient } from "@xcelsior-gpu/sdk";
const client = new XcelsiorApiClient({ environment: "https://xcelsior.ca" });
```

Streaming: use `frontend/sdk/streaming.ts` companion for SSE chat/serverless streams.

### Worker SDK (custom inference containers)

```python
from serverless.worker_sdk import handler, run_worker

@handler
def infer(job, cancel=None):
    return {"result": job["input"], "usage": {"input_tokens": 10, "output_tokens": 20}}

if __name__ == "__main__":
    run_worker(max_concurrency=4)
```

### Wizard (`@xcelsior-gpu/wizard`)

```bash
npx @xcelsior-gpu/wizard@latest
```

Progress is checkpointed to `~/.xcelsior/wizard-state.json` for resume-after-crash.
## Worker can't mount volumes / "privileged helper not installed"
The worker runs as `xcelsior` and asks `xcelsior-privd` (root) for mounts. Check
`systemctl status xcelsior-privd.socket` and `journalctl -u 'xcelsior-privd@*' -n 50`
(every request and refusal is logged). Refusals name the rule, e.g. an NFS server
that is neither on the mesh (100.64.0.0/10) nor in `NFS_SERVERS` in
`/etc/xcelsior/privileged.conf`. "protocol N, helper speaks M" means the root copy
is older than the agent: `bash install.sh --update`.
