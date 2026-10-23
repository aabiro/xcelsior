"""Privileged operations for the worker agent, performed by a root helper.

The worker runs as the unprivileged ``xcelsior`` user. Everything that needs root
is a named operation here: mounting volumes, the LUKS lifecycle, attaching a
volume inside a running job's mount namespace, and pausing other users' GPU
processes for an exclusive-GPU job.

``xcelsior-privd.socket`` (root; ``/run/xcelsior-privd.sock``, group ``xcelsior``)
starts one instance of this file per connection. The instance reads one JSON
request, checks the caller and every argument, acts, and writes one JSON reply.
It runs outside the worker's sandbox, so its mounts land in the host mount
namespace where docker can see them (mounts made inside the worker's
``ProtectSystem=strict`` namespace never propagate out). The worker sends ids
and enumerated values only, never commands, mount options or free-form paths.

A worker still running as root (installs from before 2026-10-11) runs the same
code in a single-threaded child: ``setns`` refuses multi-threaded callers.

Standard library only: root runs this with the system Python, from a
root-owned copy that ``install.sh`` installs in /usr/local/libexec/xcelsior.

Issue #65 maps every operation to the syscall and capability it needs.
"""

from __future__ import annotations

import ctypes
import ipaddress
import json
import os
import pwd
import re
import socket
import struct
import subprocess
import sys
from contextlib import suppress

PROTOCOL = 1
SOCKET_PATH = os.environ.get("XCELSIOR_PRIVD_SOCKET", "/run/xcelsior-privd.sock")
CONFIG_PATH = "/etc/xcelsior/privileged.conf"
SERVICE_USER = "xcelsior"

NFS_SHARED_MOUNT = "/mnt/xcelsior-nfs"
NFS_VOLUME_ROOT = "/mnt/xcelsior-volumes"
DEFAULT_NFS_OPTS = (
    "hard,timeo=600,retrans=3,rsize=1048576,wsize=1048576,noatime,nosuid,nodev,_netdev,tcp"
)
_MESH = ipaddress.ip_network("100.64.0.0/10")  # tailnet (headscale) addresses

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_CONTAINER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}")
_NFS_OPT = re.compile(r"[a-z_]+(=[A-Za-z0-9.]+)?")
# Never mounted over inside a job: the container's own runtime state.
_DENIED_DST = (
    "/proc",
    "/sys",
    "/dev",
    "/etc",
    "/run",
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/boot",
)


class Refused(Exception):
    """A request that fails validation; the message goes back to the worker."""


# ── configuration (root-owned, written by install.sh) ─────────────────


def load_config(path: str = CONFIG_PATH) -> dict[str, str]:
    conf = {
        "NFS_SERVERS": "",
        "NFS_EXPORT_ROOTS": "/exports",
        "NFS_MOUNT_OPTS": DEFAULT_NFS_OPTS,
        "VOLUME_DIR": "/mnt/xcelsior/volumes",
        "VOLUME_KEY_DIR": "/etc/xcelsior/volume-keys",
    }
    with suppress(FileNotFoundError):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    conf[key.strip()] = value.strip().strip('"')
    return conf


# ── validation ─────────────────────────────────────────────────────────


def _id(value: object, what: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise Refused(f"invalid {what}: {value!r}")
    return value


def _nfs_server(value: object, conf: dict[str, str]) -> str:
    allowed = {s.strip() for s in conf["NFS_SERVERS"].split(",") if s.strip()}
    if not isinstance(value, str) or not _HOST.fullmatch(value):
        raise Refused(f"invalid NFS server: {value!r}")
    if value in allowed:
        return value
    with suppress(ValueError):
        if ipaddress.ip_address(value) in _MESH:
            return value
    raise Refused(f"NFS server {value} is neither on the mesh nor in NFS_SERVERS")


def _under(path: object, roots: list[str], what: str) -> str:
    if not isinstance(path, str) or not path.startswith("/") or "\0" in path:
        raise Refused(f"invalid {what}: {path!r}")
    norm = os.path.normpath(path)
    if any(norm == r or norm.startswith(r.rstrip("/") + "/") for r in roots):
        return norm
    raise Refused(f"{what} {path} is outside {', '.join(roots)}")


def _nfs_opts(conf: dict[str, str]) -> str:
    opts = [o for o in conf["NFS_MOUNT_OPTS"].split(",") if o]
    if not all(_NFS_OPT.fullmatch(o) for o in opts):
        raise Refused("NFS_MOUNT_OPTS in privileged.conf is malformed")
    for required in ("nosuid", "nodev"):  # a share must never carry setuid binaries or devices
        if required not in opts:
            opts.append(required)
    return ",".join(opts)


def _container_dst(value: object) -> str:
    if not isinstance(value, str) or not value.startswith("/") or "\0" in value or len(value) > 512:
        raise Refused(f"invalid container path: {value!r}")
    if ".." in value.split("/"):
        raise Refused(f"container path may not contain '..': {value}")
    norm = os.path.normpath(value)
    if norm == "/" or any(norm == d or norm.startswith(d + "/") for d in _DENIED_DST):
        raise Refused(f"container path {norm} is reserved")
    return norm


def _managed_container_pid(name: object) -> int:
    """PID 1 of a running job container. Looked up here, never taken from the caller."""
    if not isinstance(name, str) or not _CONTAINER.fullmatch(name):
        raise Refused(f"invalid container: {name!r}")
    out = subprocess.run(
        [
            "docker",
            "inspect",
            "-f",
            '{{.State.Running}} {{.State.Pid}} {{index .Config.Labels "ai.xcelsior.managed"}}',
            name,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    fields = out.stdout.split()
    if out.returncode != 0 or len(fields) != 3 or fields[0] != "true" or fields[2] != "true":
        raise Refused(f"{name} is not a running xcelsior job container")
    return int(fields[1])


def _run(cmd: list[str], timeout: int = 60, stdin: bytes | None = None) -> None:
    r = subprocess.run(cmd, input=stdin, capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(
            f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} failed: {r.stderr.decode(errors='replace').strip()[:300]}"
        )


def _is_mountpoint(path: str) -> bool:
    return (
        subprocess.run(["mountpoint", "-q", path], capture_output=True, timeout=5).returncode == 0
    )


# ── NFS ─────────────────────────────────────────────────────────────────


def op_nfs_mount(args: dict, conf: dict[str, str]) -> dict:
    """Mount a share: a managed volume (``volume_id``) or the shared NFS mount."""
    server = _nfs_server(args.get("server"), conf)
    roots = [r.strip() for r in conf["NFS_EXPORT_ROOTS"].split(",") if r.strip()]
    export = _under(args.get("export"), roots, "NFS export")
    if args.get("volume_id") is not None:
        mount_point = os.path.join(NFS_VOLUME_ROOT, _id(args["volume_id"], "volume id"))
    else:
        mount_point = NFS_SHARED_MOUNT
    os.makedirs(mount_point, exist_ok=True)
    if _is_mountpoint(mount_point):
        return {"mount_point": mount_point, "already": True}
    opts = _nfs_opts(conf)
    fstype = "nfs4" if "nfsvers=4" in opts else "nfs"
    _run(["mount", "-t", fstype, "-o", opts, f"{server}:{export}", mount_point], timeout=30)
    return {"mount_point": mount_point}


def op_umount(args: dict, conf: dict[str, str]) -> dict:
    path = _under(
        args.get("path"), [NFS_SHARED_MOUNT, NFS_VOLUME_ROOT, conf["VOLUME_DIR"]], "mount point"
    )
    if path in (NFS_VOLUME_ROOT, conf["VOLUME_DIR"]):
        raise Refused(f"refusing to unmount the volume root {path}")
    if _is_mountpoint(path):
        _run(["umount", "-l", path] if args.get("lazy") else ["umount", path], timeout=30)
    return {}


# ── LUKS volumes (keys never leave root) ───────────────────────────────


def _luks_paths(volume_id: object, conf: dict[str, str]) -> tuple[str, str, str, str, str]:
    vid = _id(volume_id, "volume id")
    base, keys = conf["VOLUME_DIR"], conf["VOLUME_KEY_DIR"]
    mapper = f"xcelsior-vol-{vid[:12]}"
    return (
        os.path.join(base, f"{vid}.img"),
        os.path.join(keys, f"{vid}.key"),
        mapper,
        f"/dev/mapper/{mapper}",
        os.path.join(base, vid),
    )


def op_luks_provision(args: dict, conf: dict[str, str]) -> dict:
    backing, key, mapper, dm, mount_point = _luks_paths(args.get("volume_id"), conf)
    size_gb = args.get("size_gb")
    if not isinstance(size_gb, int) or not 1 <= size_gb <= 16384:
        raise Refused(f"invalid size_gb: {size_gb!r}")
    os.makedirs(conf["VOLUME_DIR"], exist_ok=True)
    os.makedirs(conf["VOLUME_KEY_DIR"], mode=0o700, exist_ok=True)
    if os.path.exists(backing) or os.path.exists(key):
        raise Refused(f"volume {args['volume_id']} already exists")
    try:
        _run(["truncate", "-s", f"{size_gb}G", backing], timeout=30)
        fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(os.urandom(32))  # per-volume 256-bit key: destroying it erases the volume
        _run(
            [
                "cryptsetup",
                "luksFormat",
                "--batch-mode",
                "--type",
                "luks2",
                "--key-file",
                key,
                "--cipher",
                "aes-xts-plain64",
                "--key-size",
                "512",
                "--hash",
                "sha256",
                backing,
            ]
        )
        _run(["cryptsetup", "luksOpen", "--key-file", key, backing, mapper], timeout=30)
        _run(["mkfs.ext4", "-q", "-L", f"vol-{args['volume_id'][:8]}", dm], timeout=120)
        os.makedirs(mount_point, exist_ok=True)
        _run(["mount", dm, mount_point], timeout=30)
    except Exception:
        _luks_cleanup(args["volume_id"], conf)
        raise
    return {"mount_point": mount_point}


def op_luks_attach(args: dict, conf: dict[str, str]) -> dict:
    backing, key, mapper, dm, mount_point = _luks_paths(args.get("volume_id"), conf)
    if not (os.path.exists(backing) and os.path.exists(key)):
        raise Refused(f"volume {args['volume_id']}: backing file or key not found")
    if not os.path.exists(dm):
        _run(["cryptsetup", "luksOpen", "--key-file", key, backing, mapper], timeout=30)
    os.makedirs(mount_point, exist_ok=True)
    if not _is_mountpoint(mount_point):
        _run(["mount", dm, mount_point], timeout=30)
    return {"mount_point": mount_point}


def op_luks_detach(args: dict, conf: dict[str, str]) -> dict:
    _, _, mapper, dm, mount_point = _luks_paths(args.get("volume_id"), conf)
    if _is_mountpoint(mount_point):
        _run(["umount", mount_point], timeout=30)
    if os.path.exists(dm):
        _run(["cryptsetup", "luksClose", mapper], timeout=30)
    return {}


def op_luks_destroy(args: dict, conf: dict[str, str]) -> dict:
    """Cryptographic erasure: shred the key, then remove the backing file."""
    backing, key, _, _, mount_point = _luks_paths(args.get("volume_id"), conf)
    op_luks_detach(args, conf)
    if os.path.exists(key):
        _run(["shred", "-u", "-z", "-n", "3", key], timeout=30)
    with suppress(FileNotFoundError):
        os.remove(backing)
    with suppress(FileNotFoundError, OSError):
        os.rmdir(mount_point)
    return {}


def _luks_cleanup(volume_id: str, conf: dict[str, str]) -> None:
    backing, key, mapper, dm, mount_point = _luks_paths(volume_id, conf)
    with suppress(Exception):
        subprocess.run(["umount", mount_point], capture_output=True, timeout=10)
    with suppress(Exception):
        if os.path.exists(dm):
            subprocess.run(["cryptsetup", "luksClose", mapper], capture_output=True, timeout=10)
    with suppress(Exception):
        if os.path.exists(key):
            subprocess.run(["shred", "-u", "-z", "-n", "3", key], capture_output=True, timeout=30)
    with suppress(Exception):
        os.remove(backing)
    with suppress(Exception):
        os.rmdir(mount_point)


def op_luks_cleanup(args: dict, conf: dict[str, str]) -> dict:
    _luks_cleanup(_id(args.get("volume_id"), "volume id"), conf)
    return {}


# ── attach a host mount inside a running job ───────────────────────────
# open_tree(OPEN_TREE_CLONE) takes a detached copy of the source in the host
# namespace; after setns() into the job's namespace, move_mount() places it.
# A bind mount run *inside* the job's namespace would resolve the source path
# in the container's filesystem instead of the host's.

_libc = ctypes.CDLL(None, use_errno=True)
_SYS_OPEN_TREE, _SYS_MOVE_MOUNT, _SYS_MOUNT_SETATTR = 428, 429, 442  # x86_64 and aarch64
_OPEN_TREE_CLONE, _OPEN_TREE_CLOEXEC, _AT_RECURSIVE = 1, 0o2000000, 0x8000
_MOVE_MOUNT_F_EMPTY_PATH, _AT_FDCWD = 0x4, -100
_MOUNT_ATTR_RDONLY, _MNT_DETACH = 0x1, 0x2
_CLONE_NEWNS = 0x00020000


class _MountAttr(ctypes.Structure):
    _fields_ = [
        ("attr_set", ctypes.c_uint64),
        ("attr_clr", ctypes.c_uint64),
        ("propagation", ctypes.c_uint64),
        ("userns_fd", ctypes.c_uint64),
    ]


def _check(ret: int, what: str) -> int:
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{what}: {os.strerror(err)}")
    return ret


def _enter_mount_ns(pid: int) -> None:
    """Join the job's mount namespace; root and cwd move to its root (os.setns is 3.12+)."""
    ns = os.open(f"/proc/{pid}/ns/mnt", os.O_RDONLY | os.O_CLOEXEC)
    try:
        _check(_libc.setns(ns, _CLONE_NEWNS), "setns")
    finally:
        os.close(ns)


def op_container_attach(args: dict, conf: dict[str, str]) -> dict:
    pid = _managed_container_pid(args.get("container"))
    source = _under(
        args.get("source"), [NFS_SHARED_MOUNT, NFS_VOLUME_ROOT, conf["VOLUME_DIR"]], "source"
    )
    if os.path.realpath(source) != source:
        raise Refused(f"source {source} resolves elsewhere")
    dst = _container_dst(args.get("dst"))
    mode = args.get("mode", "rw")
    if mode not in ("rw", "ro"):
        raise Refused(f"invalid mode: {mode!r}")
    tree = _check(
        _libc.syscall(
            _SYS_OPEN_TREE,
            _AT_FDCWD,
            source.encode(),
            _OPEN_TREE_CLONE | _OPEN_TREE_CLOEXEC | _AT_RECURSIVE,
        ),
        "open_tree",
    )
    try:
        if mode == "ro":
            attr = _MountAttr(_MOUNT_ATTR_RDONLY, 0, 0, 0)
            _check(
                _libc.syscall(
                    _SYS_MOUNT_SETATTR,
                    tree,
                    b"",
                    0x1000 | _AT_RECURSIVE,  # AT_EMPTY_PATH
                    ctypes.byref(attr),
                    ctypes.sizeof(attr),
                ),
                "mount_setattr",
            )
        _enter_mount_ns(pid)
        os.makedirs(dst, exist_ok=True)
        _check(
            _libc.syscall(
                _SYS_MOVE_MOUNT, tree, b"", _AT_FDCWD, dst.encode(), _MOVE_MOUNT_F_EMPTY_PATH
            ),
            "move_mount",
        )
    finally:
        os.close(tree)
    return {"dst": dst}


def op_container_detach(args: dict, conf: dict[str, str]) -> dict:
    pid = _managed_container_pid(args.get("container"))
    dst = _container_dst(args.get("dst"))
    _enter_mount_ns(pid)
    _check(_libc.umount2(dst.encode(), _MNT_DETACH), "umount2")
    return {}


# ── exclusive GPU: pause other users' GPU processes ────────────────────


def _pids(match) -> list[int]:
    pids = []
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            with suppress(OSError):
                if match(int(entry)):
                    pids.append(int(entry))
    return pids


def _is_ollama(pid: int) -> bool:
    return os.path.basename(os.readlink(f"/proc/{pid}/exe")) == "ollama"


def _is_tier3(pid: int) -> bool:
    with open(f"/proc/{pid}/cmdline", "rb") as fh:
        cmdline = fh.read()
    return b"tier3-" in cmdline and os.stat(f"/proc/{pid}").st_uid != 0


def op_gpu_tenants(args: dict, conf: dict[str, str]) -> dict:
    """``pause``: SIGSTOP ollama, SIGTERM tier3 jobs (they checkpoint). ``resume``: SIGCONT."""
    action = args.get("action")
    if action == "inspect":
        return {"stopped": _pids(_is_ollama)}
    if action == "pause":
        planned = args.get("pids")
        if planned is not None and (
            not isinstance(planned, list) or not all(type(p) is int and p > 0 for p in planned)
        ):
            raise Refused("pids must be a list of positive integers")
        stopped = _pids(_is_ollama) if planned is None else [p for p in planned if _is_ollama(p)]
        for pid in stopped:
            with suppress(ProcessLookupError):
                os.kill(pid, 19)  # SIGSTOP
        for pid in _pids(_is_tier3):
            with suppress(ProcessLookupError):
                os.kill(pid, 15)  # SIGTERM
        return {"stopped": stopped}
    if action == "resume":
        pids = args.get("pids") or []
        if not isinstance(pids, list) or not all(isinstance(p, int) for p in pids):
            raise Refused("pids must be a list of integers")
        for pid in pids:
            with suppress(OSError):
                if _is_ollama(pid):  # only ever resume what pause stopped
                    os.kill(pid, 18)  # SIGCONT
        return {}
    raise Refused(f"invalid action: {action!r}")


OPS = {
    "nfs_mount": op_nfs_mount,
    "umount": op_umount,
    "luks_provision": op_luks_provision,
    "luks_attach": op_luks_attach,
    "luks_detach": op_luks_detach,
    "luks_destroy": op_luks_destroy,
    "luks_cleanup": op_luks_cleanup,
    "container_attach": op_container_attach,
    "container_detach": op_container_detach,
    "gpu_tenants": op_gpu_tenants,
}


def handle(request: dict, conf: dict[str, str] | None = None) -> dict:
    """Run one request; failures come back as ``{"ok": False, "error": ...}``."""
    try:
        if request.get("v") != PROTOCOL:
            raise Refused(
                f"protocol {request.get('v')!r}, helper speaks {PROTOCOL}: run install.sh --update"
            )
        op = OPS.get(str(request.get("op") or ""))
        if op is None:
            raise Refused(f"unknown operation {request.get('op')!r}")
        args = request.get("args") or {}
        if not isinstance(args, dict):
            raise Refused("args must be an object")
        return {"ok": True, **op(args, conf or load_config())}
    except Refused as e:
        return {"ok": False, "refused": True, "error": str(e)}
    except Exception as e:  # report, never crash the connection
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── worker side ────────────────────────────────────────────────────────


def call(op: str, timeout: int = 180, **args) -> dict:
    """Ask the root helper to perform ``op``. Returns its reply dict."""
    payload = json.dumps({"v": PROTOCOL, "op": op, "args": args}).encode() + b"\n"
    if not os.path.exists(SOCKET_PATH):
        if os.geteuid() != 0:
            return {
                "ok": False,
                "error": f"{SOCKET_PATH} missing: privileged helper not installed (run install.sh --update)",
            }
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__)],
            input=payload,
            capture_output=True,
            timeout=timeout,
        )
        return json.loads(r.stdout or b'{"ok": false, "error": "helper produced no reply"}')
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(SOCKET_PATH)
        sock.sendall(payload)
        sock.shutdown(socket.SHUT_WR)
        reply = b""
        while chunk := sock.recv(65536):
            reply += chunk
    return json.loads(reply or b'{"ok": false, "error": "helper closed the connection"}')


# ── helper side ────────────────────────────────────────────────────────


def _peer_allowed(fd: int) -> bool:
    """Root, or the service user, may connect (the socket's group/mode already limit this)."""
    try:
        conn = socket.socket(fileno=os.dup(fd))
    except OSError:
        return os.geteuid() == 0  # stdin is a pipe: a root worker ran us as a child
    with conn:
        try:
            creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        except OSError:
            return False
    _, uid, _ = struct.unpack("3i", creds)
    with suppress(KeyError):
        if uid == pwd.getpwnam(SERVICE_USER).pw_uid:
            return True
    return uid == 0


def main() -> int:
    if not _peer_allowed(0):
        reply = {"ok": False, "refused": True, "error": "caller is not the xcelsior service user"}
    else:
        try:
            request = json.loads(sys.stdin.buffer.readline() or b"{}")
        except json.JSONDecodeError as e:
            request = {"v": None, "_error": str(e)}
        reply = (
            handle(request)
            if isinstance(request, dict)
            else {"ok": False, "error": "request must be an object"}
        )
        print(
            json.dumps(
                {
                    "op": request.get("op") if isinstance(request, dict) else None,
                    "ok": reply.get("ok"),
                    "error": reply.get("error"),
                }
            ),
            file=sys.stderr,
        )  # journal
    sys.stdout.write(json.dumps(reply) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
