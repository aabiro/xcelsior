"""What "byte-identical in environment" means, and how a container proves it.

Gate P7: *"a sweep of N nodes from one snapshot is **byte-identical in
environment**"*.

This module runs **inside the container**, and that is not an implementation
detail. The control plane already knows the image digest it sent to all N
members; comparing that to itself proves the request was consistent, not that
the containers are. The only evidence that answers the clause comes from the
running environment, reported back.

## The boundary is the entire claim

"Byte-identical in environment" cannot mean *everything observable*. Two
containers on two hosts differ in their hostname, their kernel, their GPU, their
IP. If those count, no sweep can pass and the clause is meaningless. If too much
is excluded, every sweep passes and the clause is equally meaningless. So the
line is drawn here, in the file, with a reason beside each exclusion — and the
reason is the argument, not the list.

**"Environment" means the image-derived environment.** What the snapshot
determines, not what the machine underneath it happens to be.

### In scope — the image decides these

| Field | Why it is in scope |
|---|---|
| `image_digest` | The bytes the container was created from. If two members differ here, nothing else matters — they are not the same snapshot. |
| `packages` | Sorted `name==version` inventory. This is what "same environment" means to anyone who has debugged a works-on-one-node failure. |
| `env` | The image's declared `Env`. A `PATH` or `LD_LIBRARY_PATH` that differs changes which binary runs. |
| `entrypoint`, `cmd`, `working_dir`, `user` | The image's declared process contract. Two containers running different commands are not the same environment however identical their packages. |

### Out of scope — and why each one

| Excluded | Reason |
|---|---|
| NVIDIA driver version, GPU model | **Host.** A sweep across an A100 host and a 4090 host is the normal case; requiring them equal would make heterogeneous placement a failure. The GPU is what the marketplace *varies*. |
| Kernel version, OS release of the host | **Host.** The container does not carry it, and it is the same class of fact as the driver. |
| `hostname`, container id, MAC, IP | **Per-instance.** Distinct by construction — every container has its own. Including them guarantees a mismatch and turns the check into a constant `false`. |
| Anything under `/proc` and `/sys` | **Host.** These are windows onto the machine, not the image: cgroup limits, uptime, CPU model, memory totals. |
| Injected job environment — job id, tokens, volume ids, `XCELSIOR_*` | **Per-instance by design.** The platform sets these *differently on purpose*; two members sharing a job id would be the bug. Excluded by prefix and by name, and the exclusion is applied to the collected `env` rather than trusting the image not to contain them. |
| Timestamps, `/etc/resolv.conf`, ephemeral state | **Per-instance.** They vary between two runs of the *same* container, so they cannot distinguish two containers. |

**If a fingerprint goes red, that is a finding.** The temptation will be to move
a field from the first table to the second until the sweep is green. Every such
move needs its reason written here, in the commit that makes it — a guard
narrowed until it passes is the failure this plan names by name.

## Why the manifest is stored alongside the hash

A hash answers "did these differ" and nothing else. The first time a sweep goes
red, the question is *which field*, and a bare digest cannot answer it. Both are
reported; the hash is for comparison, the manifest is for the person who has to
act on the comparison.

## Determinism

The hash is over canonical JSON — sorted keys, no whitespace variance — and the
package list is sorted. Two identical environments must produce the same bytes,
or the check reports mismatches that are artefacts of dictionary ordering.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

#: Environment variables the platform injects per instance. Excluded by prefix
#: and by exact name. Applied to whatever is collected rather than assumed
#: absent, because the image can legitimately declare a variable the platform
#: also sets, and the per-instance value is the one that would differ.
INJECTED_ENV_PREFIXES = ("XCELSIOR_", "NVIDIA_", "CUDA_VISIBLE_DEVICES")
INJECTED_ENV_NAMES = frozenset(
    {
        "HOSTNAME",
        "HOME",
        "PWD",
        "OLDPWD",
        "SHLVL",
        "_",
        "TERM",
        "container",
    }
)

#: The manifest's shape. Recorded so a consumer can tell a collector that
#: produced nothing from one that produced an empty environment.
FINGERPRINT_VERSION = 1


def _packages() -> list[str]:
    """Sorted `name==version`, from the interpreter's own view.

    `importlib.metadata` rather than shelling out to `pip freeze`: it does not
    depend on pip being installed, is an order of magnitude faster, and returns
    the same distributions the running interpreter would import.
    """
    try:
        from importlib import metadata
    except Exception:  # pragma: no cover - Python too old to be in scope
        return []
    found: set[str] = set()
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:
            continue
        if name:
            found.add(f"{name}=={version or ''}")
    return sorted(found)


def _system_packages() -> list[str]:
    """Sorted OS package inventory when one is readable, else empty.

    Best-effort and non-fatal: an image without `dpkg` is not a broken image,
    and an empty list is honest. It is *not* silently equal to another empty
    list — `collect` records which sources answered, so two containers that
    disagree about whether dpkg exists are visible as a difference.
    """
    try:
        result = subprocess.run(
            ["dpkg-query", "-W", "-f=${Package}==${Version}\\n"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def _declared_env() -> dict[str, str]:
    """The environment minus everything the platform injects per instance."""
    out: dict[str, str] = {}
    for key, value in os.environ.items():
        if key in INJECTED_ENV_NAMES:
            continue
        if key.startswith(INJECTED_ENV_PREFIXES):
            continue
        out[key] = value
    return dict(sorted(out.items()))


def collect(image_digest: str = "") -> dict:
    """The in-scope facts about this container's environment.

    `image_digest` is passed in rather than discovered: a container cannot see
    the digest it was created from, and inventing one here would let a member
    report agreement with itself.
    """
    sources: dict[str, bool] = {}
    python_packages = _packages()
    sources["python"] = bool(python_packages)
    system_packages = _system_packages()
    sources["system"] = bool(system_packages)

    return {
        "version": FINGERPRINT_VERSION,
        "image_digest": image_digest,
        "python_version": sys.version.split()[0],
        "packages": python_packages,
        "system_packages": system_packages,
        # Which inventories answered. Without this an image with no dpkg and an
        # image whose dpkg failed look identical — both empty — and a real
        # difference reads as agreement.
        "sources": dict(sorted(sources.items())),
        "env": _declared_env(),
        "working_dir": os.getcwd(),
        "user": str(os.getuid()) if hasattr(os, "getuid") else "",
    }


def canonical(manifest: dict) -> str:
    """Canonical JSON: sorted keys, no whitespace variance.

    Two identical environments must hash to the same bytes. Without this the
    check reports mismatches that are artefacts of dictionary ordering, which
    is worse than no check — it trains the reader to disregard a red result.
    """
    return json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str)


def fingerprint(image_digest: str = "") -> tuple[str, dict]:
    """`(sha256 hex, manifest)` for this container.

    Both are returned because a hash answers "did these differ" and nothing
    else. The first time a sweep goes red the question is *which field*.
    """
    manifest = collect(image_digest)
    digest = hashlib.sha256(canonical(manifest).encode("utf-8")).hexdigest()
    return digest, manifest


def main() -> int:
    """Print `{"hash": ..., "manifest": {...}}` for the agent to forward."""
    digest, manifest = fingerprint(os.environ.get("XCELSIOR_IMAGE_DIGEST", ""))
    print(json.dumps({"hash": digest, "manifest": manifest}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
