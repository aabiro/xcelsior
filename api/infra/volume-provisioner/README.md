# Volume provisioner (Phase 10 privilege split)

The FastAPI API container **must not** hold `SYS_ADMIN` or host export
device mounts for LUKS. Privileged volume work belongs here:

- Consumes durable volume commands from the control plane
- Has only required devices/capabilities/mounts
- Validates tenant / volume / action IDs
- Performs LUKS/NFS operations idempotently (or SSH to the storage host)
- ACKs results through the command protocol
- Inaccessible from public ingress

## Current production path

Encrypted volume create/destroy already prefer **host SSH loopback** for
`cryptsetup` (`volumes.py`), so the API process itself does not need
`SYS_ADMIN` in compose. This service is the long-term isolated home for
any remaining privileged local device work.

## Compose

Enable with profile `volume-provisioner` when a dedicated privileged
worker is required. Default stacks run API unprivileged.
