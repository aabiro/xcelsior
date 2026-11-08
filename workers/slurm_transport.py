"""SSH to a Slurm cluster, built from its registry row (plan item 1.2).

A ``slurm_clusters`` row with an ``ssh_host`` says where to connect and how
both ends prove themselves: ``ssh_host``/``ssh_port``/``remote_user``, the
cluster's host key pinned at registration (``known_hosts_line``), and a
``credential_ref`` naming where the private key is kept, never the key.
``cluster_runner`` turns the row into a ``SlurmRunner`` whose ``-F`` config is
generated, so nothing about the connection comes from ``~/.ssh``:

- The only known host is the pinned key, with ``StrictHostKeyChecking yes``.
  A changed host key is a refused connection (``host_key_mismatch``), never a
  prompt and never trust on first use.
- One ControlMaster per cluster. A pass's squeue, sacct, tails and cancels
  share one authenticated connection, and so do passes within
  ``CONTROL_PERSIST_SEC`` of each other, instead of a TCP and SSH handshake
  per command.
- A key held in the environment (``env:NAME``) is written to a private
  directory on tmpfs only while the runner is in use. An established master
  doesn't need it; the next connection writes it again.

Everything lives under ``runtime_dir()``: a 0700 directory on ``/dev/shm``
(or ``XCELSIOR_SLURM_RUNTIME_DIR``). Each use gets its own subdirectory for
its config, known_hosts and key, removed when the use ends; the master's
socket is shared, which is what lets uses reuse a master. The socket's name
carries a hash of the pinned key and the credential: a mux client skips host
key checks, so after a re-pin or re-key an old master must not be reused,
and with a new name it can't be. The new pin is checked on a fresh
connection.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import os
import re
import shutil
import stat
import subprocess
import tempfile

import slurm_adapter as sa

#: How long a cluster's master connection outlives its last command. Longer
#: than the shortest poll interval (30 s), so consecutive passes share it.
CONTROL_PERSIST_SEC = 120

_KEY_TYPES = frozenset(
    {
        "ssh-ed25519",
        "ecdsa-sha2-nistp256",
        "ecdsa-sha2-nistp384",
        "ecdsa-sha2-nistp521",
        "rsa-sha2-256",
        "rsa-sha2-512",
        "ssh-rsa",
        "sk-ssh-ed25519@openssh.com",
        "sk-ecdsa-sha2-nistp256@openssh.com",
    }
)
_CLUSTER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
# A hostname, an IPv4 address or a bare IPv6 address: nothing ssh_config
# would read as more than one word.
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,252}")
_USER = re.compile(r"[a-z_][a-z0-9_-]{0,31}")
_PATH = re.compile(r"/[^\s]{1,4095}")
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,127}")


def runtime_dir() -> str:
    """The private directory for configs, keys and master sockets."""
    base = os.environ.get("XCELSIOR_SLURM_RUNTIME_DIR") or (
        "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    )
    path = os.path.join(base, f"xcelsior-slurm-{os.getuid()}")
    if not _PATH.fullmatch(path):  # it is written into the ssh config
        raise sa.SlurmAdapterError("unsafe_runtime_dir", f"{path!r} must be a plain absolute path")
    os.makedirs(path, mode=0o700, exist_ok=True)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
        raise sa.SlurmAdapterError(
            "unsafe_runtime_dir", f"{path} must be a directory owned by this user, mode 0700"
        )
    return path


def pinned_key(line: str) -> tuple[str, str]:
    """(key type, base64 key) from a known_hosts or ssh-keyscan line.

    Takes the key and ignores the line's host field: the generated config
    pins it under the cluster's own alias (HostKeyAlias), so the same line
    works whichever name or address it was scanned under. Marker lines
    (``@cert-authority``, ``@revoked``) are refused: they don't pin a key.
    """
    text = (line or "").strip()
    if text.startswith("@") or "\n" in text:
        raise sa.SlurmAdapterError(
            "invalid_host_key", "known_hosts_line must be one plain key line"
        )
    parts = text.split()
    for i, part in enumerate(parts[:-1]):
        if part in _KEY_TYPES:
            key = parts[i + 1]
            try:
                base64.b64decode(key, validate=True)
            except (binascii.Error, ValueError):
                break
            return part, key
    raise sa.SlurmAdapterError("invalid_host_key", "known_hosts_line holds no recognisable key")


def _identity(credential_ref: str, workdir: str) -> str:
    """The IdentityFile for this use, writing an env-held key into workdir."""
    kind, _, ref = (credential_ref or "").partition(":")
    if kind == "file" and _PATH.fullmatch(ref):
        return ref
    if kind == "env" and _ENV_NAME.fullmatch(ref):
        material = os.environ.get(ref, "")
        if "\n" not in material:
            material = material.replace("\\n", "\n")  # stored with escaped newlines
        if "PRIVATE KEY" not in material:
            raise sa.SlurmAdapterError("credential_unavailable", f"{ref} holds no private key")
        path = os.path.join(workdir, "key")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(material if material.endswith("\n") else material + "\n")
        return path
    raise sa.SlurmAdapterError(
        "invalid_credential_ref", "credential_ref must be env:NAME or file:/absolute/path"
    )


def _alias(cluster_id: str) -> str:
    if not _CLUSTER_ID.fullmatch(cluster_id or ""):
        raise sa.SlurmAdapterError("invalid_cluster", f"invalid cluster id {cluster_id!r}")
    return f"xcl-{cluster_id}"


def _config(cluster: dict, alias: str, known_hosts: str, identity: str, control_path: str) -> str:
    host = cluster["ssh_host"]
    port = int(cluster.get("ssh_port") or 22)
    user = cluster.get("remote_user")
    if not _HOST.fullmatch(host or ""):
        raise sa.SlurmAdapterError("invalid_host", f"invalid ssh_host {host!r}")
    if user and not _USER.fullmatch(user):
        raise sa.SlurmAdapterError("invalid_user", f"invalid remote_user {user!r}")
    if not 1 <= port <= 65535:
        raise sa.SlurmAdapterError("invalid_port", f"invalid ssh_port {port}")
    lines = [
        "# Generated from the Slurm cluster registry (workers/slurm_transport.py).",
        f"Host {alias}",
        f"    HostName {host}",
        f"    Port {port}",
        *([f"    User {user}"] if user else []),
        f"    HostKeyAlias {alias}",
        f"    UserKnownHostsFile {known_hosts}",
        "    GlobalKnownHostsFile /dev/null",
        "    StrictHostKeyChecking yes",
        "    UpdateHostKeys no",
        "    CheckHostIP no",
        f"    IdentityFile {identity}",
        "    IdentitiesOnly yes",
        "    IdentityAgent none",
        "    PasswordAuthentication no",
        "    KbdInteractiveAuthentication no",
        "    RequestTTY no",
        "    BatchMode yes",
        "    ControlMaster auto",
        f"    ControlPath {control_path}",
        f"    ControlPersist {CONTROL_PERSIST_SEC}",
        "    ServerAliveInterval 15",
        "    ServerAliveCountMax 2",
        "",
    ]
    return "\n".join(lines)


@contextlib.contextmanager
def cluster_runner(cluster: dict):
    """A SlurmRunner for a registered cluster that has an ``ssh_host``."""
    if not cluster.get("ssh_host"):
        raise sa.SlurmAdapterError(
            "no_registry_transport", f"cluster {cluster.get('cluster_id')!r} has no ssh_host"
        )
    alias = _alias(cluster["cluster_id"])
    ktype, key = pinned_key(cluster.get("known_hosts_line") or "")
    base = runtime_dir()
    workdir = tempfile.mkdtemp(prefix=f"{cluster['cluster_id']}-", dir=base)
    try:
        known_hosts = os.path.join(workdir, "known_hosts")
        with open(known_hosts, "w") as f:
            f.write(f"{alias} {ktype} {key}\n")
        identity = _identity(cluster.get("credential_ref") or "", workdir)
        trust = hashlib.sha256(
            f"{ktype} {key}\0{cluster.get('credential_ref')}".encode()
        ).hexdigest()[:12]
        conf = os.path.join(workdir, "ssh_config")
        with open(conf, "w") as f:
            f.write(_config(cluster, alias, known_hosts, identity, f"{base}/cm-{trust}-%C"))
        yield sa.SlurmRunner(ssh_host=alias, ssh_options=("-F", conf))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _control(runner: sa.SlurmRunner, op: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", *runner.ssh_options, "-O", op, runner.ssh_host or ""],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


def master_running(runner: sa.SlurmRunner) -> bool:
    """Whether this cluster's master connection is up (``ssh -O check``)."""
    return _control(runner, "check").returncode == 0


def close_master(runner: sa.SlurmRunner) -> None:
    """Close the master now, e.g. after re-keying, rather than at its timeout."""
    _control(runner, "exit")
