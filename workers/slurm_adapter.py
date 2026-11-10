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

import datetime as _dt
import hashlib
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
#   partition_gpu/_cpu       partition for GPU jobs / CPU-only jobs; None
#                            omits --partition (Alliance routes jobs itself)
#   gpu_type                 GPU model ("h100"); None requests untyped gpu:N
#   gpu_request              "gres" (--gres=gpu:T:N) or "gpus"
#                            (--gpus-per-node=T:N, --gpus=T:N across nodes)
#   cpus_per_gpu             emit --cpus-per-gpu (site bundle size)
#   account_required         refuse to submit without an account
#   retired                  refuse new submissions
#   gpus_per_node            for multi-node packing
#   max_walltime             client-side ceiling, checked before sbatch
#   default_walltime         used when the job gives none (else a heuristic)
#   qos                      True: emit --qos from priority (default False;
#                            most clusters reject QoS names they don't define)
#   accounting_time/timezone how sacct times are read (see _PROFILE_DEFAULTS)
#   output_reader            "tail" or "cat" for get_slurm_job_output
#   output_size_check        ask `ls -l` before reading output, and read only
#                            when it changed and is under a cap (default: on
#                            when output_reader is "cat")
#   mem_base_gb, mem_per_gpu_gb  memory request = base + per_gpu * gpus
#   modules, container_runtime ("apptainer" | "docker" | "none")
#   scratch_dir, project_dir, output_dir ({user}/{account} templates)
#   ssh_host, ssh_options    run Slurm commands on this host over SSH
#   remote_user              cluster account name, when ssh_host is an alias
#   research_only            an academic allocation (Alliance): research jobs
#                            under its PI only, never marketplace capacity or
#                            agent traffic (slurm_clusters enforces it)
#   marketplace, agent_access  may sell capacity / take agent-originated work
#   poll_interval_seconds    how often the reconciler polls it (30-3600)

CLUSTER_PROFILES = {
    # Alliance profiles verified against docs.alliancecan.ca on 2026-11-07
    # (Nibi, Narval, Using GPUs with Slurm, Job scheduling policies,
    # Allocations and compute scheduling, Available software, Automation in
    # the context of MFA). Facts that shaped them:
    #   - partitions are assigned by the scheduler from the request (by-node /
    #     by-core, walltime tiers); jobs must not name one
    #   - GPUs: --gpus-per-node=<model>:N (or --gpus=<model>:N spread
    #     anywhere); "--gres=gpu:... may not be supported in the future"
    #   - --account=def-<pi> and --time are required; no user QoS
    #   - per-GPU bundles: Nibi H100 14 cores / 250 GB, Narval A100 12 / 124 GB
    #   - automation goes through robot.<cluster>.alliancecan.ca with a CCDB key
    #     restricted by restrict,from=,command=; Alliance's slurm_commands.sh
    #     runs $SSH_ORIGINAL_COMMAND unquoted (word-split, no env prefix), so
    #     these profiles parse local-time accounting and read output with cat
    #   - at most 1000 pending+running jobs per user (MaxSubmit)
    "nibi": {
        "name": "SHARCNET Nibi",
        # Alliance Terms of Service 3.1: academic research under a PI only.
        "research_only": True,
        # Alliance asks scripts not to poll "at high frequency".
        "poll_interval_seconds": 120,
        # 36 GPU nodes x 8 H100 SXM 80GB (NVLink); Nokia Ethernet, 200 Gb/s
        # non-blocking between GPU nodes. Automation node:
        # robot.nibi.alliancecan.ca (by request; not for AI-agent use).
        "partition_gpu": None,
        "partition_cpu": None,
        "account_env": "SLURM_ACCOUNT",
        "account_required": True,
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "gpu_request": "gpus",
        "cpus_per_gpu": 14,
        "max_walltime": "7-00:00:00",  # 7 days
        "qos": False,
        "mem_base_gb": 4,
        "mem_per_gpu_gb": 240,  # bundle is 250 GB per GPU
        "modules": ["cuda/12.2", "python/3.11", "apptainer"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
        "accounting_time": "iso_local",
        "timezone": "America/Toronto",
        "output_reader": "cat",
        "output_size_check": True,
    },
    "graham": {
        # Retired 1 September 2025 ("service now ended"); Nibi replaced it. Kept
        # so stored mappings that name it still resolve for status reads;
        # new submissions are refused.
        "name": "Compute Canada Graham (retired)",
        "retired": True,
        "research_only": True,
        "partition_gpu": "gpu",
        "partition_cpu": "compute",
        "account_env": "SLURM_ACCOUNT",
        "gpu_type": "v100",
        "gpus_per_node": 2,
        "max_walltime": "3-00:00:00",
        "qos": False,
        "modules": ["cuda/11.8", "python/3.10"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
    },
    "narval": {
        "name": "Calcul Québec Narval",
        "research_only": True,
        "poll_interval_seconds": 120,
        # 4 x A100 SXM4 40GB per GPU node (NVLink); InfiniBand HDR.
        # Automation node: robot.narval.alliancecan.ca.
        "partition_gpu": None,
        "partition_cpu": None,
        "account_env": "SLURM_ACCOUNT",
        "account_required": True,
        "gpu_type": "a100",
        "gpus_per_node": 4,
        "gpu_request": "gpus",
        "cpus_per_gpu": 12,
        "max_walltime": "7-00:00:00",
        "qos": False,
        "mem_base_gb": 4,
        "mem_per_gpu_gb": 118,  # bundle is 124 GB per GPU
        "modules": ["cuda/12.2", "python/3.11", "apptainer"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
        "accounting_time": "iso_local",
        "timezone": "America/Montreal",
        "output_reader": "cat",
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
        "qos": False,
        "modules": [],
        "container_runtime": "docker",
        "scratch_dir": "/tmp",
        "project_dir": "/home/{user}",
    },
}

_PROFILE_DEFAULTS = {
    "account_env": "SLURM_ACCOUNT",
    "account_required": False,
    "gpu_type": None,
    "gpus_per_node": 1,
    "gpu_request": "gres",
    "cpus_per_gpu": None,
    "max_walltime": "1-00:00:00",
    "default_walltime": None,
    # Decision D6 (hpc/IMPLEMENTATION_PLAN.md): QoS is opt-in per cluster.
    "qos": False,
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
    "retired": False,
    "research_only": False,
    "marketplace": False,
    "agent_access": False,
    "poll_interval_seconds": 60,
    # "epoch_env": SLURM_TIME_FORMAT=%s via an env prefix (exact).
    # "iso_local": parse local-time ISO stamps in `timezone`, cross-checked
    # against ElapsedRaw — for wrappers that refuse an env prefix.
    "accounting_time": "epoch_env",
    "timezone": "UTC",
    "output_reader": "tail",  # or "cat" where the site wrapper lacks tail
}

_REQUIRED_PROFILE_KEYS = ("name", "partition_gpu", "partition_cpu")
_GPU_REQUEST_STYLES = ("gres", "gpus")

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


_REMOTE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_%|,.:=/+@-]+$")


@dataclass(frozen=True)
class SlurmRunner:
    """Runs Slurm client commands locally, or on a submit host over SSH.

    The SSH form sends one argv per connection as plain words, with the
    batch script on stdin. That is the shape a forced-command wrapper on a
    restricted key can validate (hpc/slurm-local/scripts/xcl-slurm-gate),
    which is how Alliance "robot" automation nodes admit automated access.

    A remote run must name its own ssh config file (``-F FILE``). Without
    one, ssh reads ~/.ssh/config, where IdentityFile entries accumulate
    across Host blocks: a ``Host *`` key is offered before the restricted
    one, and the cluster accepts it unrestricted (found on the real tower).
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
            tokens = [*prefix, *argv]
            # Remote commands are plain words joined by spaces, never quoted:
            # site wrappers such as Alliance's slurm_commands.sh run
            # $SSH_ORIGINAL_COMMAND unquoted, so quotes would arrive as
            # literal characters. A token that a shell would split, glob or
            # interpret is refused instead of sent.
            bad = [t for t in tokens if not _REMOTE_TOKEN_RE.match(str(t))]
            if bad:
                raise SlurmAdapterError(
                    "unsafe_remote_argument", f"refusing to send {bad!r} over SSH"
                )
            if "-F" not in self.ssh_options:
                raise SlurmAdapterError(
                    "insecure_ssh_config",
                    "remote Slurm runs need their own ssh config (-F FILE in ssh_options); "
                    "without it ~/.ssh/config can offer an unrestricted key first",
                )
            remote_cmd = " ".join(str(t) for t in tokens)
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
                cmd,
                input=input_text,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        full_env = {**os.environ, **env} if env else None
        # errors="replace": a job's output is whatever the job printed, and
        # `tail -c` usually starts inside a multi-byte character. Strict
        # decoding raised UnicodeDecodeError out of the read.
        return subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout,
            env=full_env,
        )


SSH_OVERRIDE_PROFILE_ENV = "XCELSIOR_SLURM_SSH_PROFILE"


def get_runner(profile=None) -> SlurmRunner:
    """Pick the transport from the profile; default is local.

    XCELSIOR_SLURM_SSH_HOST / XCELSIOR_SLURM_SSH_OPTIONS override it only for
    the profile named in XCELSIOR_SLURM_SSH_PROFILE. They used to apply to
    every profile, so a host meant for the lab cluster would also have
    received Nibi submissions. Env options are appended to the profile's,
    never substituted: replacing them dropped the profile's ``-F``.
    """
    profile = profile or {}
    host = profile.get("ssh_host")
    opts = list(profile.get("ssh_options") or [])
    env_host = os.environ.get("XCELSIOR_SLURM_SSH_HOST", "").strip()
    env_opts = os.environ.get("XCELSIOR_SLURM_SSH_OPTIONS", "").strip()
    if env_host or env_opts:
        target = os.environ.get(SSH_OVERRIDE_PROFILE_ENV, "").strip()
        if not target:
            raise SlurmAdapterError(
                "ambiguous_ssh_override",
                f"XCELSIOR_SLURM_SSH_HOST/_OPTIONS are set; name the profile they "
                f"override in {SSH_OVERRIDE_PROFILE_ENV}",
            )
        if target == profile.get("_key"):
            host = env_host or host
            opts += shlex.split(env_opts)
    # "~/" in an option (the -F file, a key) is the adapter's own home, so a
    # committed profile needn't name an account.
    opts = [os.path.expanduser(o) if str(o).startswith("~/") else o for o in opts]
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
        stderr = r.stderr.strip()
        if "Host key verification failed" in stderr or "IDENTIFICATION HAS CHANGED" in stderr:
            # The cluster's key isn't the pinned one: a re-keyed host, or not
            # the host at all. Never retried into acceptance.
            return _error("host_key_mismatch", f"ssh refused the submit host's key: {stderr}")
        return _error("transport_error", f"ssh to submit host failed: {stderr}")
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


_PING_RE = re.compile(r"Slurmctld\((?P<role>[^)]+)\) at (?P<host>\S+) is (?P<state>UP|DOWN)")


def controller_health(*, profile_name=None, runner=None):
    """Whether the cluster's slurmctld answers, from ``scontrol ping``.

    ``sinfo --version`` (is_slurm_available) proves only that SSH works and
    the client is installed; it never contacts the controller. Slurm 25.11
    prints one line per controller: ``Slurmctld(primary) at tower is UP``.

    Returns {"controller": "up"|"down", "controllers": [...]} or an error.
    """
    try:
        _profile, runner = _resolve(profile_name, runner)
        r = runner.run(["scontrol", "ping"], timeout=15)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "scontrol ping timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available on this system")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    found = [m.groupdict() for m in _PING_RE.finditer(r.stdout or "")]
    if not found:
        detail = (r.stdout or r.stderr or "").strip()[:200]
        return _error("unparseable_ping", f"scontrol ping printed {detail!r}")
    return {
        "controller": "up" if any(c["state"] == "UP" for c in found) else "down",
        "controllers": [
            {"role": c["role"], "host": c["host"], "state": c["state"].lower()} for c in found
        ],
    }


#: What a cluster's ``scontrol show config`` is kept for: the settings the
#: reconciler and the scheduler depend on. Captured at registration and on
#: every connection check, into ``slurm_clusters.facts['config']``.
CONFIG_FACTS = (
    "SLURM_VERSION",
    "ClusterName",
    "MinJobAge",
    "FirstJobId",
    "MaxJobId",
    "JobRequeue",
    "PreemptMode",
    "PreemptType",
    "AccountingStorageType",
    "AccountingStoreFlags",
    "TrackWCKey",
    "PrivateData",
    "SelectType",
    "SelectTypeParameters",
    "GresTypes",
    "ProctrackType",
    "TaskPlugin",
    "SlurmctldParameters",
    "MessageTimeout",
)


def cluster_config_facts(*, profile_name=None, runner=None) -> dict:
    """The CONFIG_FACTS settings from ``scontrol show config``.

    Returns {"facts": {name: value}, "missing": [names not printed]} or an
    error. Values are kept as Slurm prints them (``(null)`` included).
    """
    try:
        _profile, runner = _resolve(profile_name, runner)
        r = runner.run(["scontrol", "show", "config"], timeout=30)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "scontrol show config timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available on this system")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        msg = (r.stderr or "").strip()
        return _error(classify_slurm_error(msg, default="scontrol_failed"), msg)
    printed = {}
    for line in (r.stdout or "").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            printed.setdefault(key.strip(), value.strip())
    facts = {name: printed[name] for name in CONFIG_FACTS if name in printed}
    return {"facts": facts, "missing": [n for n in CONFIG_FACTS if n not in printed]}


def get_cluster_info(runner=None):
    """Get basic info about the Slurm cluster."""
    runner = runner or SlurmRunner()
    if not is_slurm_available(runner):
        return None

    info = {}
    try:
        r = runner.run(["sinfo", "-N", "--noheader", "-o", "%N|%P|%G|%m"], timeout=15)
        if r.returncode == 0:
            nodes = []
            for line in r.stdout.strip().split("\n"):
                parts = line.split("|")
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
    argv = ["sinfo", "-N", "--noheader", "-o", "%N|%P|%c|%m|%G|%T"]
    if part:
        argv[3:3] = ["-p", part]
    try:
        r = runner.run(argv, timeout=20)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
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


def duration_seconds(spec) -> int | None:
    """Seconds in a squeue or sacct duration, or None for UNLIMITED and the like."""
    try:
        return _walltime_seconds(spec)
    except SlurmAdapterError:
        return None


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


def xcelsior_job_to_sbatch(
    job, profile_name=None, extra_env=None, runner=None, submission_name=None
):
    """Translate an Xcelsior job dict into an sbatch script string.

    Args:
        submission_name: the job's name in Slurm (``xcl-<submission_name>``)
            and in its output files; a scheduler attempt passes
            ``<job id>-a<n>`` so each attempt is its own submission (D7).
            Defaults to the job id, as every job already submitted is named.
        job: Xcelsior job dict. Keys: job_id, name, vram_needed_gb,
            priority, tier, image, and optionally:
              num_gpus   0 → CPU-only job on the profile's CPU partition
              command    payload run with ``bash -c`` (inside the image
                         when one is given)
              nodelist   pin to node(s), e.g. "asus"
              partition  override the profile's partition
              cpus       --cpus-per-task
              mem_gb     override the memory formula (per node, as --mem is)
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
    name = str(submission_name or job_id)
    _require(_JOB_ID_RE, name, "submission_name")
    job_name = f"xcl-{name}"

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
    if partition is not None:
        _require(_SLURM_TOKEN_RE, partition, "partition")
    gpu_request = profile.get("gpu_request") or "gres"
    if gpu_request not in _GPU_REQUEST_STYLES:
        raise SlurmAdapterError(
            "invalid_profile", f"gpu_request must be one of {_GPU_REQUEST_STYLES}"
        )

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

    # GPU and memory requests are per node in Slurm (--gres, --gpus-per-node,
    # --mem), so a multi-node job asks for an even share on every node. The
    # old code put the job's *total* on each node: 16 GPUs on Nibi asked
    # for --gres=gpu:16 and 3,844 GB per node, which no node has.
    gpus_per_node_cap = int(profile["gpus_per_node"])
    if num_gpus > 0:
        nodes_needed = max(1, -(-num_gpus // gpus_per_node_cap))
        if num_gpus % nodes_needed:
            raise SlurmAdapterError(
                "invalid_job_spec",
                f"{num_gpus} GPUs don't split evenly over {nodes_needed} nodes of "
                f"{gpus_per_node_cap}; request a multiple of {nodes_needed}",
            )
        gpus_per_node = num_gpus // nodes_needed
    else:
        nodes_needed = 1
        gpus_per_node = 0
    # One task per node; srun runs the payload on every node of a multi-node
    # job. Without it the payload ran on the first node only.
    launch = "srun " if nodes_needed > 1 else ""

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
    ]
    if partition is not None:
        lines.append(f"#SBATCH --partition={partition}")
    if num_gpus > 0:
        spec = f"{gpu_type}:{gpus_per_node}" if gpu_type else str(gpus_per_node)
        if gpu_request == "gpus":
            lines.append(f"#SBATCH --gpus-per-node={spec}")  # Alliance form
        else:
            lines.append(f"#SBATCH --gres=gpu:{spec}")
        if profile.get("cpus_per_gpu"):
            lines.append(f"#SBATCH --cpus-per-gpu={int(profile['cpus_per_gpu'])}")
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

    # Decision D6: only where the profile says the site defines these QoS
    # names. The old fallback here was True, the opposite of the default.
    if profile["qos"] and qos != "default":
        lines.append(f"#SBATCH --qos={qos}")

    # Nodes and tasks
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

    # Memory per node (default: 4GB base + 8GB per GPU on the node; profiles
    # can scale it). A job's mem_gb is per node too, as --mem is.
    if job.get("mem_gb") is not None:
        mem_gb = job["mem_gb"]
        if isinstance(mem_gb, bool) or not isinstance(mem_gb, int) or mem_gb < 1:
            raise SlurmAdapterError("invalid_job_spec", f"invalid mem_gb: {mem_gb!r}")
    else:
        mem_gb = int(profile["mem_base_gb"]) + int(profile["mem_per_gpu_gb"]) * gpus_per_node
    lines.append(f"#SBATCH --mem={mem_gb}G")

    # Email notifications (optional). A profile can turn them off: the xcl-local
    # gate refuses --mail-user (the kit has no mail), and any environment that
    # sets XCELSIOR_SLURM_EMAIL otherwise got every submission rejected there.
    email = os.environ.get("XCELSIOR_SLURM_EMAIL", "")
    if (
        email
        and profile.get("mail_notifications", True)
        and re.match(r"^[^\s@]+@[^\s@]+$", email)
    ):
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
    # Where the job may report its own progress (PROGRESS_CONTRACT): a small
    # JSON object it rewrites atomically, by writing a temporary file and
    # renaming it over this one.
    lines.append(f'export XCELSIOR_PROGRESS_FILE="{out_dir}/{job_name}-${{SLURM_JOB_ID}}.progress"')

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
            lines.append(f"{launch}apptainer exec --nv {sif_path} {payload}")
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
        lines.append(f"{launch}docker pull {shlex.quote(image)}")
        gpu_flag = (
            '--gpus "\\"device=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}\\"" '
            if num_gpus > 0
            else ""
        )
        lines.append(
            f"{launch}docker run --rm {gpu_flag}{shlex.quote(image)}{' ' + payload if payload else ''}"
        )
    elif payload:
        lines.append("# ── Payload ──")
        lines.append(f"{launch}{payload}")
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
        "submission_name": name,
        "slurm_job_name": job_name,
        "profile": profile["_key"],
        "cluster_id": profile["cluster_id"],
        "partition": partition,
        "gpu_request": gpu_request,
        "num_gpus": num_gpus,
        "gpu_type": gpu_type,
        "walltime": walltime,
        "qos": qos,
        "nodes": nodes_needed,
        "gpus_per_node": gpus_per_node,
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
    (
        re.compile(r"MaxSubmitJob|job submit limit|QOSMaxSubmitJobPerUserLimit", re.I),
        "submit_limit_reached",
    ),
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


def submit_to_slurm(
    job, profile_name=None, extra_env=None, dry_run=False, runner=None, submission_name=None
):
    """Submit an Xcelsior job to a Slurm cluster.

    Args:
        job: Xcelsior job dict (see xcelsior_job_to_sbatch)
        profile_name: Cluster profile name
        extra_env: Additional environment variables
        dry_run: If True, return script without submitting
        runner: transport override (default: from env / profile)
        submission_name: the name this submission carries in Slurm (see
            xcelsior_job_to_sbatch); the scheduler's submitter passes the
            attempt's. Returned with the result, beside the script's sha256.

    Returns:
        dict with slurm_job_id and metadata, or error + error_code.
        Rejections come back from sbatch itself, synchronously; with
        EnforcePartLimits=ALL a request no node or partition can satisfy
        is refused at submit time instead of pending forever.
    """
    try:
        profile, runner = _resolve(profile_name, runner)
        if profile.get("retired"):
            raise SlurmAdapterError(
                "cluster_retired", f"{profile['name']} is retired; choose another cluster"
            )
        script, metadata = xcelsior_job_to_sbatch(
            job, profile_name, extra_env, runner=runner, submission_name=submission_name
        )
    except SlurmAdapterError as e:
        return _error(e.code, str(e))

    if dry_run:
        return {"script": script, "metadata": metadata, "dry_run": True}
    script_sha256 = hashlib.sha256(script.encode()).hexdigest()

    if profile.get("account_required") and not os.environ.get(profile["account_env"], ""):
        return _error(
            "missing_account",
            f"{profile['name']} requires an allocation account; set {profile['account_env']}",
        )

    if not _slurm_reachable(runner):
        return _error("slurm_unavailable", "Slurm not available on this system")

    try:
        # The script goes over stdin (no temp file left behind on the
        # submit host). --parsable prints "<jobid>[;<cluster>]".
        result = runner.run(["sbatch", "--parsable"], input_text=script, timeout=30)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
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
        "submission_name": metadata["submission_name"],
        "cluster_id": metadata["cluster_id"],
        "script_sha256": script_sha256,
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

# Xcelsior job state → slurm_job_mappings.observed_state. Writing "queued"
# there raised CheckViolation and aborted the whole sync pass the first time
# a real job was pending. Migration 152 added preempted, held and suspended.
OBSERVED_STATE_MAP = {
    "queued": "pending",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "preempted": "preempted",
    "unknown": "unknown",
}

# Slurm job state → slurm_job_mappings.observed_state, for the reconciler.
# Finer than SLURM_STATE_MAP's job vocabulary: a hold (REQUEUE_HOLD,
# SPECIAL_EXIT) waits for a release and is not a failure, and SUSPENDED
# keeps its memory and GPUs, so it is not "pending". Whether a job is
# finished is decided by the reconciler (it has left the controller and
# accounting agrees), never by one of these states alone: a requeued
# NODE_FAIL or PREEMPTED job comes back with the same id.
SLURM_OBSERVED_STATE = {
    "PENDING": "pending",
    "REQUEUED": "pending",
    "REQUEUE_FED": "pending",
    "CONFIGURING": "running",
    "RUNNING": "running",
    "COMPLETING": "running",
    "RESIZING": "running",
    "SIGNALING": "running",
    "STAGE_OUT": "running",
    "REQUEUE_HOLD": "held",
    "SPECIAL_EXIT": "held",
    "RESV_DEL_HOLD": "held",
    "SUSPENDED": "suspended",
    "STOPPED": "suspended",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "NODE_FAIL": "failed",
    "OUT_OF_MEMORY": "failed",
    "TIMEOUT": "failed",
    "CANCELLED": "cancelled",
    "REVOKED": "cancelled",
    "PREEMPTED": "preempted",
}
FINISHED_OBSERVED_STATES = ("completed", "failed", "cancelled", "preempted")


def success_verdict(observed_state, exit_code, derived_exit_code, settled) -> str | None:
    """Whether a settled job succeeded: None while it is open.

    ``succeeded`` needs COMPLETED with ExitCode 0:0 *and* DerivedExitCode
    0:0. DerivedExitCode is the highest exit code of the job's steps, so it
    catches a script that exits 0 after a step failed. A COMPLETED job with
    no DerivedExitCode on record (accounting that predates it) is
    ``unknown``, not a guess.
    """
    if not settled:
        return None
    if observed_state == "completed":
        if exit_code != "0:0":
            return "failed"
        if derived_exit_code is None:
            return "unknown"
        return "succeeded" if derived_exit_code == "0:0" else "failed"
    if observed_state in ("failed", "preempted"):
        return "failed"
    if observed_state == "cancelled":
        return "cancelled"
    return "unknown"


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
        # "|"-delimited: no spaces in the argument (site wrappers word-split
        # the command), and %R — which can contain spaces — stays intact.
        r = runner.run(
            ["squeue", "-j", str(slurm_job_id), "--noheader", "-o", "%i|%T|%M|%l|%D|%R"],
            timeout=10,
        )
        failure = _transport_failure(r, runner)
        if failure:
            return failure

        if r.returncode == 0 and r.stdout.strip():
            # Format is "%i|%T|%M|%l|%D|%R"; %R (reason / nodelist) can
            # contain spaces — e.g. "(launch failed requeued held)" — so cap
            # the split at 5. squeue -j for a single id returns one line.
            parts = r.stdout.strip().splitlines()[0].split("|", 5)
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

        # Absence is not a verdict: accounting may lag (slurmdbd down or
        # backlogged), or the id may belong to another cluster. Never
        # infer a job's fate from silence.
        return {
            "slurm_job_id": slurm_job_id,
            "slurm_state": "UNKNOWN",
            "xcelsior_state": "unknown",
            "error": "Job not found in squeue or sacct",
            "error_code": "not_found",
        }

    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "Status poll timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available")
    except Exception as e:
        return _error("adapter_exception", str(e))


# ── Batched polling (the reconciler's command set) ───────────────────
# One squeue per cluster per pass, not one per job. With more than one id,
# `squeue -j a,b,c` makes squeue load the whole job table and filter it
# client-side (Slurm 25.11 source); `--me` asks only for this account's jobs,
# and `-t all` keeps the ones that finished within MinJobAge. sacct runs
# only for jobs that have left the controller, always bounded by -S: with
# -j and no -S it searches from epoch 0, and after job ids wrap the record
# it returns can belong to another job.

_SQUEUE_FIELDS = ("job_id", "name", "state", "reason", "elapsed", "time_limit", "nodes", "nodelist")
# "|"-delimited, no spaces: site wrappers word-split the command line.
_SQUEUE_FORMAT = "%i|%j|%T|%r|%M|%l|%D|%N"
_WINDOW_FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "Submit",
    "Start",
    "End",
    "ElapsedRaw",
    "ExitCode",
    "AllocTRES",
    "NodeList",
    "DerivedExitCode",
    # Billing (workers/slurm_billing.py): ElapsedRaw excludes suspended time,
    # so Suspended is kept beside it (D8); TimelimitRaw caps a TIMEOUT run.
    "Suspended",
    "TimelimitRaw",
)
#: Ids per sacct call; the gate admits at most this many.
MAX_IDS_PER_SACCT = 200
_WINDOW_ARGV = ("sacct", "-D", "-X", "-n", "--parsable2", "--noconvert")
_STAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"


def poll_cluster_jobs(*, profile_name=None, runner=None) -> dict:
    """Every job of this account the controller still holds, in one call.

    Returns {"jobs": {slurm_job_id: {name, state, reason, elapsed,
    time_limit, nodes, nodelist}}} with ``state`` normalised, or an error.
    A job missing here has left the controller (MinJobAge) or never
    belonged to this account; only accounting can say which.
    """
    try:
        _profile, runner = _resolve(profile_name, runner)
        r = runner.run(["squeue", "--me", "-h", "-t", "all", "-o", _SQUEUE_FORMAT], timeout=30)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "squeue timed out")
    except FileNotFoundError:
        return _error("slurm_unavailable", "Slurm not available on this system")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        msg = r.stderr.strip()
        return _error(classify_slurm_error(msg, default="squeue_failed"), msg)
    jobs = {}
    for line in r.stdout.splitlines():
        parts = line.split("|", len(_SQUEUE_FIELDS) - 1)
        if len(parts) != len(_SQUEUE_FIELDS) or not _SLURM_JOB_ID_RE.match(parts[0]):
            continue
        row = dict(zip(_SQUEUE_FIELDS, parts))
        row["state"] = normalize_slurm_state(row["state"])
        jobs[row.pop("job_id")] = row
    return {"jobs": jobs}


def _run_sacct_window(argv, *, profile_name, runner, timeout=30):
    """One ``sacct -D -X`` call; (profile, result) or an error dict."""
    try:
        profile, runner = _resolve(profile_name, runner)
    except SlurmAdapterError as e:
        return None, _error(e.code, str(e))
    epoch_env = profile.get("accounting_time", "epoch_env") != "iso_local"
    try:
        r = runner.run(
            argv, timeout=timeout, env={"SLURM_TIME_FORMAT": "%s"} if epoch_env else None
        )
    except SlurmAdapterError as e:
        return None, _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return None, _error("timeout", "sacct timed out")
    except FileNotFoundError:
        return None, _error("slurm_unavailable", "Slurm not available")
    failure = _transport_failure(r, runner)
    if failure:
        return None, failure
    if r.returncode != 0:
        msg = r.stderr.strip()
        code = (
            "accounting_disabled"
            if "accounting storage is disabled" in msg.lower()
            else "sacct_failed"
        )
        return None, _error(code, msg)
    return profile, r


def _sacct_records(argv, *, profile_name, runner, timeout=30, keep) -> dict:
    """One window call as {"records": ...}, or the error dict the call returned."""
    profile, ran = _run_sacct_window(
        argv, profile_name=profile_name, runner=runner, timeout=timeout
    )
    if not isinstance(ran, subprocess.CompletedProcess) or profile is None:
        if isinstance(ran, dict):
            return ran
        return _error("sacct_failed", "sacct returned no result")
    return {"records": _window_records(ran.stdout, profile, keep)}


def _window_records(stdout: str, profile: dict, keep) -> dict:
    """Parse ``_WINDOW_FIELDS`` lines into {slurm_job_id: [run, ...]}, runs
    in submit order (latest last); ``keep(row)`` selects the rows wanted."""
    epoch_env = profile.get("accounting_time", "epoch_env") != "iso_local"
    tz_name = str(profile.get("timezone") or "UTC")
    records: dict = {}
    for line in stdout.splitlines():
        parts = line.split("|")
        if len(parts) != len(_WINDOW_FIELDS):
            continue
        row = dict(zip(_WINDOW_FIELDS, parts))
        if not _SLURM_JOB_ID_RE.match(row["JobIDRaw"]) or not keep(row):
            continue
        state = normalize_slurm_state(row["State"])
        observed = SLURM_OBSERVED_STATE.get(state, "unknown")
        elapsed = int(row["ElapsedRaw"]) if row["ElapsedRaw"].isdigit() else 0
        finished = observed in FINISHED_OBSERVED_STATES
        if epoch_env:
            submitted = _epoch_or_none(row["Submit"])
            started = _epoch_or_none(row["Start"])
            end_if_over = _epoch_or_none(row["End"])
            ended = end_if_over if finished else None
        else:
            submitted, started, ended = _local_times(row, tz_name, elapsed, finished)
            end_if_over = ended if finished else _local_times(row, tz_name, elapsed, True)[2]
        limit = row["TimelimitRaw"].strip()
        records.setdefault(row["JobIDRaw"], []).append(
            {
                "name": row["JobName"],
                "state": state,
                "raw_state": row["State"],
                "observed_state": observed,
                "submitted_at": submitted,
                "started_at": started,
                "ended_at": ended,
                "elapsed_sec": elapsed,
                # Seconds the run spent suspended; not in ElapsedRaw (D8).
                "suspended_sec": duration_seconds(row["Suspended"]) or 0,
                # TimelimitRaw is minutes; UNLIMITED and Partition_Limit → None.
                "time_limit_sec": int(limit) * 60 if limit.isdigit() else None,
                "exit_code": row["ExitCode"],
                "derived_exit_code": row["DerivedExitCode"],
                "alloc_tres": row["AllocTRES"],
                "node_list": row["NodeList"],
                "_end_if_over": end_if_over,
            }
        )
    for runs in records.values():
        runs.sort(key=lambda rec: rec["submitted_at"] or 0.0)
        # A run with a later run under the same id is over, whatever its state
        # says: REQUEUED (or a requeued PREEMPTED/NODE_FAIL) names the job's
        # fate, not the run's. Without its End the ledger stored the run with no
        # ended_at, and the daily drift, which selects runs by end time, skipped
        # every requeued run on both sides (found in the 2.3 shadow week). The
        # latest run keeps the rule above; settling reads only that one.
        for earlier in runs[:-1]:
            if earlier["ended_at"] is None:
                earlier["ended_at"] = earlier["_end_if_over"]
        for run in runs:
            run.pop("_end_if_over", None)
    return records


def accounting_window(job_ids, *, since_minutes: int, profile_name=None, runner=None) -> dict:
    """Accounting for these jobs, every run of each, in one sacct call.

    ``-D`` returns one allocation record per run: a requeued or preempted
    job has several under the same id, and without -D only the latest shows.
    ``since_minutes`` bounds the search (-S); pass the age of the oldest
    open mapping plus a margin.

    Returns {"records": {slurm_job_id: [record, ...]}} with each list in
    submit order (latest last), or an error. A job with no record is absent
    from the dict: not yet in accounting, outside the window, or unknown.
    """
    ids = [str(j) for j in job_ids]
    if not ids:
        return {"records": {}}
    if len(ids) > MAX_IDS_PER_SACCT:
        return _error("invalid_request", f"at most {MAX_IDS_PER_SACCT} ids per sacct call")
    bad = [j for j in ids if not _SLURM_JOB_ID_RE.match(j)]
    if bad:
        return _error("invalid_job_id", f"invalid Slurm job ids: {bad!r}")
    since = max(1, int(since_minutes))
    argv = [
        *_WINDOW_ARGV,
        "-S",
        f"now-{since}minutes",
        "-j",
        ",".join(ids),
        "-o",
        ",".join(_WINDOW_FIELDS),
    ]
    wanted = set(ids)
    return _sacct_records(
        argv,
        profile_name=profile_name,
        runner=runner,
        keep=lambda row: row["JobIDRaw"] in wanted,
    )


def accounting_by_name(names, *, since_minutes: int, profile_name=None, runner=None) -> dict:
    """Accounting for the jobs carrying these submission names, in one call.

    Resolves a submission whose ``sbatch`` outcome is unknown (a timeout, a
    dropped connection, a crash before the mapping was written): the job's
    name, ``xcl-<submission_name>``, is the one tag Slurm always stores and
    filters on (``--comment`` reaches accounting only with
    ``AccountingStoreFlags=job_comment``). Same result shape as
    ``accounting_window``; a name with no record is absent.
    """
    wanted = [str(n) for n in names]
    if not wanted:
        return {"records": {}}
    if len(wanted) > MAX_IDS_PER_SACCT:
        return _error("invalid_request", f"at most {MAX_IDS_PER_SACCT} names per sacct call")
    bad = [n for n in wanted if not _JOB_ID_RE.match(n)]
    if bad:
        return _error("invalid_job_id", f"invalid submission names: {bad!r}")
    since = max(1, int(since_minutes))
    job_names = {f"xcl-{n}" for n in wanted}
    argv = [
        *_WINDOW_ARGV,
        "-S",
        f"now-{since}minutes",
        "--name",
        ",".join(sorted(job_names)),
        "-o",
        ",".join(_WINDOW_FIELDS),
    ]
    return _sacct_records(
        argv,
        profile_name=profile_name,
        runner=runner,
        keep=lambda row: row["JobName"] in job_names,
    )


def accounting_between(start_epoch, end_epoch, *, profile_name=None, runner=None) -> dict:
    """Every run of every job of this account that overlaps [start, end).

    The daily drift report's one call (workers/slurm_billing.py): the window
    is given to sacct in the cluster's local time; the caller decides
    membership by each run's End. Same result shape as ``accounting_window``.
    """
    try:
        profile = _get_profile(profile_name)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(str(profile.get("timezone") or "UTC"))
    start = _dt.datetime.fromtimestamp(float(start_epoch), tz).strftime(_STAMP_FORMAT)
    end = _dt.datetime.fromtimestamp(float(end_epoch), tz).strftime(_STAMP_FORMAT)
    if end <= start:
        return _error("invalid_request", "accounting window must end after it starts")
    argv = [*_WINDOW_ARGV, "-S", start, "-E", end, "-o", ",".join(_WINDOW_FIELDS)]
    return _sacct_records(
        argv, profile_name=profile_name, runner=runner, timeout=60, keep=lambda row: True
    )


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


def _iso_local_to_epoch(value: str, tz_name: str, fold: int = 0) -> float | None:
    """'2026-11-01T01:30:00' in the cluster's zone → epoch; None if unset."""
    v = (value or "").strip()
    if not v or v in ("Unknown", "None") or "T" not in v:
        return None
    from datetime import datetime
    from zoneinfo import ZoneInfo

    try:
        naive = datetime.fromisoformat(v)
    except ValueError:
        return None
    return naive.replace(tzinfo=ZoneInfo(tz_name), fold=fold).timestamp()


def _local_times(row: dict, tz_name: str, elapsed: int, terminal: bool) -> tuple:
    """Submit/Start/End from local-time sacct output, DST-safe.

    In the hour a fall-back transition repeats, a local stamp names two
    instants. ElapsedRaw is exact, so pick the readings of Start and End
    whose difference matches it; failing that, derive End from Start.
    """
    submitted = _iso_local_to_epoch(row["Submit"], tz_name)
    s0 = _iso_local_to_epoch(row["Start"], tz_name, 0)
    if s0 is None:
        return submitted, None, (_iso_local_to_epoch(row["End"], tz_name) if terminal else None)
    if not terminal:
        return submitted, s0, None
    best = None
    for fs in (0, 1):
        for fe in (0, 1):
            st = _iso_local_to_epoch(row["Start"], tz_name, fs)
            en = _iso_local_to_epoch(row["End"], tz_name, fe)
            if st is not None and en is not None and abs((en - st) - elapsed) <= 1:
                best = (st, en)
                break
        if best:
            break
    if best is None:
        best = (s0, s0 + elapsed)
    return submitted, best[0], best[1]


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
        profile, runner = _resolve(profile_name, runner)
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
    epoch_env = profile.get("accounting_time", "epoch_env") != "iso_local"
    try:
        r = runner.run(argv, timeout=15, env={"SLURM_TIME_FORMAT": "%s"} if epoch_env else None)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
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
    elapsed = int(row["ElapsedRaw"]) if row["ElapsedRaw"].isdigit() else 0
    if epoch_env:
        submitted = _epoch_or_none(row["Submit"])
        started = _epoch_or_none(row["Start"])
        ended = _epoch_or_none(row["End"]) if terminal else None
    else:
        submitted, started, ended = _local_times(
            row, str(profile.get("timezone") or "UTC"), elapsed, terminal
        )
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

#: The files the reconciler may read for a job, by stream name.
OUTPUT_STREAMS = ("out", "err", "progress")

#: With output_reader "cat" the whole file crosses the wire on every read
#: (Alliance's wrapper allows cat, not tail), so past this size it isn't read.
OUTPUT_CAT_MAX_BYTES = 2 * 1024 * 1024


def output_size_check(profile: dict) -> bool:
    """Whether to ask ``ls -l`` for a file's size before reading it."""
    return bool(profile.get("output_size_check", profile.get("output_reader") == "cat"))


def _output_path(profile, runner, xcelsior_job_id, slurm_job_id, stream) -> str:
    _require(_JOB_ID_RE, str(xcelsior_job_id), "job_id")
    account = os.environ.get(profile["account_env"], "")
    base = _output_dir(profile, _remote_user(profile, runner), account)
    return f"{base}/xcl-{xcelsior_job_id}-{slurm_job_id}.{stream}"


def get_slurm_job_output_size(
    slurm_job_id, xcelsior_job_id, *, profile_name=None, runner=None, stream="out"
):
    """A job's output file size from ``ls -l``: {"size_bytes": n} or an error.

    For clusters where reading means ``cat``: comparing sizes lets the
    reconciler skip a read when nothing was written, and refuse one past
    OUTPUT_CAT_MAX_BYTES.
    """
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    if stream not in OUTPUT_STREAMS:
        return _error("invalid_request", "stream must be 'out', 'err' or 'progress'")
    try:
        profile, runner = _resolve(profile_name, runner)
        path = _output_path(profile, runner, xcelsior_job_id, slurm_job_id, stream)
        r = runner.run(["ls", "-l", "--", path], timeout=30)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "ls timed out")
    except FileNotFoundError:
        return _error("io_error", "ls not available")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        return _error("output_unavailable", r.stderr.strip() or f"cannot stat {path}", path=path)
    fields = (r.stdout or "").split()
    if len(fields) < 5 or not fields[4].isdigit() or not fields[0].startswith("-"):
        # Not a regular file (a link, a directory) or not ls output at all.
        return _error("output_unavailable", f"ls printed no size for {path}", path=path)
    return {"slurm_job_id": str(slurm_job_id), "path": path, "size_bytes": int(fields[4])}


#: How a job reports its own progress. Slurm has no notion of it, and
#: ``sstat`` costs a controller RPC, so the job writes a small file and the
#: reconciler reads it the way it reads output, at most once per interval.
PROGRESS_CONTRACT = """
The batch script exports XCELSIOR_PROGRESS_FILE. A job that wants to report
progress writes one JSON object there, at most 4 KiB, replacing the file
atomically (write a temporary file in the same directory, then rename it):

    {"fraction": 0.42, "message": "epoch 3 of 10", "updated_at": 1794000000}

fraction    optional, 0 to 1
message     optional, at most 200 characters
updated_at  required, Unix seconds when the job wrote it

Readers show it as reported by the job, and as stale once updated_at is more
than PROGRESS_STALE_SEC old while the job runs.
"""

#: Job-reported progress older than this, on a running job, is stale.
PROGRESS_STALE_SEC = 600


def get_slurm_job_output(
    slurm_job_id,
    xcelsior_job_id,
    *,
    profile_name=None,
    runner=None,
    stream="out",
    max_bytes=65536,
):
    """Read the tail of a job's stdout, stderr or progress file.

    ``stream`` is ``out``, ``err`` or ``progress`` (PROGRESS_CONTRACT). Works
    where the profile's output_dir is visible from the submit host: a shared
    filesystem on real clusters (/scratch), or the submit node itself on a
    cluster without one.
    """
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    if stream not in OUTPUT_STREAMS:
        return _error("invalid_request", "stream must be 'out', 'err' or 'progress'")
    try:
        profile, runner = _resolve(profile_name, runner)
        path = _output_path(profile, runner, xcelsior_job_id, slurm_job_id, stream)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    if profile.get("output_reader") == "cat":
        argv = ["cat", path]  # site wrapper allows cat, not tail; trimmed below
    else:
        argv = ["tail", "-c", str(int(max_bytes)), "--", path]
    try:
        r = runner.run(argv, timeout=30)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "output read timed out")
    except FileNotFoundError:
        return _error("io_error", "tail not available")
    failure = _transport_failure(r, runner)
    if failure:
        return failure
    if r.returncode != 0:
        return _error("output_unavailable", r.stderr.strip() or f"cannot read {path}", path=path)
    content = r.stdout[-int(max_bytes) :] if max_bytes else r.stdout
    return {"slurm_job_id": str(slurm_job_id), "path": path, "stream": stream, "content": content}


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
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", "scancel timed out")
    except Exception as e:
        return _error("adapter_exception", str(e))


_SCONTROL_JOB_VERBS = frozenset({"suspend", "resume", "requeue"})


def _scontrol_one_job(verb, slurm_job_id, *, profile_name=None, runner=None):
    """One authorized ``scontrol`` verb on one job id.

    ``suspend`` and ``requeue`` are how a shadow run produces the suspended
    and requeued accounting records. ``resume`` is the matching release, so
    a suspended job can finish and give the node back. ``update``, ``hold``,
    and a list of ids are not this function.
    """
    if verb not in _SCONTROL_JOB_VERBS:
        return _error("invalid_request", f"scontrol {verb} is not authorized")
    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    try:
        _profile, runner = _resolve(profile_name, runner)
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    if not _slurm_reachable(runner):
        return _error("slurm_unavailable", "Slurm not available")
    try:
        r = runner.run(["scontrol", verb, str(slurm_job_id)], timeout=15)
        failure = _transport_failure(r, runner)
        if failure:
            return failure
        if r.returncode == 0:
            return {"ok": True, "verb": verb, "slurm_job_id": str(slurm_job_id)}
        detail = ((r.stderr or "") + (r.stdout or "")).strip()
        if r.returncode == 126 or "refused" in detail:
            return _error("scontrol_rejected", detail or f"scontrol {verb} was refused")
        if "Invalid job id" in detail:
            return _error("not_found", detail)
        return _error("scontrol_failed", detail or f"scontrol {verb} failed")
    except SlurmAdapterError as e:
        return _error(e.code, str(e))
    except subprocess.TimeoutExpired:
        return _error("timeout", f"scontrol {verb} timed out")
    except Exception as e:
        return _error("adapter_exception", str(e))


def suspend_slurm_job(slurm_job_id, *, profile_name=None, runner=None):
    """Suspend one job. The allocation stays; resume it to let it finish."""
    return _scontrol_one_job("suspend", slurm_job_id, profile_name=profile_name, runner=runner)


def resume_slurm_job(slurm_job_id, *, profile_name=None, runner=None):
    """Resume one suspended job so it can finish and release its node."""
    return _scontrol_one_job("resume", slurm_job_id, profile_name=profile_name, runner=runner)


def requeue_slurm_job(slurm_job_id, *, profile_name=None, runner=None):
    """Requeue one job. Slurm keeps the earlier run as its own accounting record."""
    return _scontrol_one_job("requeue", slurm_job_id, profile_name=profile_name, runner=runner)


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


def list_slurm_mappings(limit: int = 200) -> list[dict]:
    """Recent mappings, open and terminal, newest first (for the admin UI).

    Unlike _load_slurm_map this includes terminal rows, which are kept for
    audit, and returns the state columns the UI shows.
    """
    from db import pg_connection

    limit = max(1, min(int(limit), 1000))
    with pg_connection() as conn:
        rows = conn.execute(
            """SELECT xcelsior_job_id, slurm_job_id, cluster_id, observed_state,
                      desired_state,
                      EXTRACT(EPOCH FROM submitted_at),
                      EXTRACT(EPOCH FROM last_observed_at),
                      EXTRACT(EPOCH FROM terminal_at),
                      slurm_state, state_reason, exit_code,
                      mapping_id::text, submission_name, xcelsior_attempt_id
                 FROM slurm_job_mappings
                ORDER BY submitted_at DESC NULLS LAST
                LIMIT %s""",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "job_id": r[0],
                "slurm_job_id": r[1],
                "cluster_id": r[2],
                "observed_state": r[3],
                "desired_state": r[4],
                "submitted_at": float(r[5]) if r[5] is not None else None,
                "last_observed_at": float(r[6]) if r[6] is not None else None,
                "terminal_at": float(r[7]) if r[7] is not None else None,
                "slurm_state": r[8],
                "state_reason": r[9],
                "exit_code": r[10],
                "mapping_id": r[11],
                "submission_name": r[12],
                "attempt_id": r[13],
            }
        )
    return out


def read_observed_status(slurm_job_id, *, cluster_id=None) -> dict:
    """A job's state as the reconciler last observed it; never polls Slurm.

    The reconciler (slurm_reconciler) is the only code that polls a cluster;
    the admin dashboard's refresh reads this instead, so refreshing costs the
    cluster nothing. A Slurm id can exist on several clusters, so without
    ``cluster_id`` an id mapped on more than one is an error, not a guess.
    On one cluster an id can come back once its first job settles; the open
    mapping is the job now holding it, and failing that, the latest.
    """
    from db import pg_connection

    if not _SLURM_JOB_ID_RE.match(str(slurm_job_id)):
        return _error("invalid_job_id", f"invalid Slurm job id: {slurm_job_id!r}")
    query = (
        "SELECT xcelsior_job_id, cluster_id, observed_state, slurm_state, state_reason, "
        "exit_code, derived_exit_code, terminal_at IS NOT NULL, EXTRACT(EPOCH FROM last_observed_at), "
        "EXTRACT(EPOCH FROM clock_timestamp() - last_observed_at) "
        "FROM slurm_job_mappings WHERE slurm_job_id = %s"
    )
    params: list = [str(slurm_job_id)]
    if cluster_id:
        query += " AND cluster_id = %s"
        params.append(cluster_id)
    query += " ORDER BY terminal_at IS NULL DESC, submitted_at DESC NULLS LAST"
    with pg_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    if not rows:
        return _error("not_found", f"Slurm job {slurm_job_id} is not tracked by Xcelsior")
    clusters = {r[1] for r in rows}
    if len(clusters) > 1:
        return _error(
            "ambiguous_job_id",
            f"Slurm job {slurm_job_id} is mapped on {len(clusters)} clusters; pass cluster_id",
        )
    xid, cid, observed, slurm_state, reason, exit_code, derived, settled, seen_at, age = rows[0]
    return {
        "slurm_job_id": str(slurm_job_id),
        "cluster_id": cid,
        "xcelsior_job_id": xid,
        "slurm_state": slurm_state or "UNKNOWN",
        # The mapping vocabulary (pending, held, suspended, ...), which the
        # dashboard labels directly.
        "xcelsior_state": observed or "unknown",
        "observed_state": observed or "unknown",
        "reason": reason,
        "exit_code": exit_code,
        "derived_exit_code": derived,
        "verdict": success_verdict(observed, exit_code, derived, bool(settled)),
        "settled": bool(settled),
        "last_observed_at": float(seen_at) if seen_at is not None else None,
        "observed_age_seconds": round(float(age), 1) if age is not None else None,
        "source": "reconciler",
    }


#: How much of each stream the output cache keeps, from the end of the file.
OUTPUT_TAIL_BYTES = 65536
#: The progress file is small by contract; anything longer isn't read.
PROGRESS_MAX_BYTES = 4096


def read_cached_output(xcelsior_job_id, stream="out") -> dict:
    """A job's output tail from the cache (slurm_job_outputs); never polls Slurm.

    Readers never reach the cluster: the reconciler fetches output on its
    pass. A read that finds the copy older than the cluster's poll interval
    files a refresh request, and the next pass serves it, so the cluster
    sees at most one fetch per job and stream per interval however many
    readers there are. Only the first read after a fetch writes the request.
    Once the job has settled and its output was fetched, the copy is final
    and is never fetched again.

    ``refresh_pending`` tells the caller a fetch is queued: read again after
    ``poll_interval_seconds``.
    """
    from db import pg_transaction

    if stream not in OUTPUT_STREAMS:
        return _error("invalid_request", "stream must be 'out', 'err' or 'progress'")
    with pg_transaction() as conn:
        # The job's current submission: its open one, else its latest. Each
        # attempt keeps its own copy (migration 157).
        mapping = conn.execute(
            "SELECT m.mapping_id, m.slurm_job_id, m.cluster_id, m.terminal_at IS NOT NULL, "
            "COALESCE(c.poll_interval_seconds, 60) "
            "FROM slurm_job_mappings m LEFT JOIN slurm_clusters c ON c.cluster_id = m.cluster_id "
            "WHERE m.xcelsior_job_id = %s "
            "ORDER BY m.terminal_at IS NULL DESC, m.submitted_at DESC NULLS LAST LIMIT 1",
            (str(xcelsior_job_id),),
        ).fetchone()
        if mapping is None:
            return _error("not_found", f"job {xcelsior_job_id} has no Slurm mapping")
        mapping_id, slurm_job_id, cluster_id, settled, interval = mapping
        # A request is filed only when none is outstanding, the copy is older
        # than one poll interval, and it isn't final.
        conn.execute(
            """
            INSERT INTO slurm_job_outputs (mapping_id, stream, slurm_job_id, requested_at)
            VALUES (%s, %s, %s, clock_timestamp())
            ON CONFLICT (mapping_id, stream) DO UPDATE
               SET requested_at = clock_timestamp()
             WHERE NOT slurm_job_outputs.final
               AND COALESCE(slurm_job_outputs.fetched_at, '-infinity'::timestamptz)
                   < clock_timestamp() - make_interval(secs => %s)
               AND COALESCE(slurm_job_outputs.requested_at, '-infinity'::timestamptz)
                   <= COALESCE(slurm_job_outputs.attempted_at, '-infinity'::timestamptz)
            """,
            (mapping_id, stream, slurm_job_id, int(interval)),
        )
        row = conn.execute(
            "SELECT content, truncated, final, error_code, error, "
            "EXTRACT(EPOCH FROM fetched_at), "
            "EXTRACT(EPOCH FROM clock_timestamp() - fetched_at), "
            "requested_at > COALESCE(attempted_at, '-infinity'::timestamptz) "
            "FROM slurm_job_outputs WHERE mapping_id = %s AND stream = %s",
            (mapping_id, stream),
        ).fetchone()
    content, truncated, final, error_code, error, fetched_at, age, pending = row
    return {
        "xcelsior_job_id": str(xcelsior_job_id),
        "slurm_job_id": slurm_job_id,
        "cluster_id": cluster_id,
        "stream": stream,
        "content": content,
        "truncated": bool(truncated),
        "final": bool(final),
        "settled": bool(settled),
        "fetched_at": float(fetched_at) if fetched_at is not None else None,
        "fetched_age_seconds": round(float(age), 1) if age is not None else None,
        "refresh_pending": bool(pending),
        "poll_interval_seconds": int(interval),
        "fetch_error_code": error_code,
        "fetch_error": error,
        "source": "reconciler",
    }


def read_reported_progress(xcelsior_job_id) -> dict | None:
    """The job's own progress report, per PROGRESS_CONTRACT, from the cache.

    Files a refresh request like any cached read (never polls Slurm).
    Returns None when the job has never written a report the reconciler has
    fetched; otherwise the parsed report, or ``{"valid": False, ...}`` when
    the file breaks the contract.
    """
    cached = read_cached_output(xcelsior_job_id, "progress")
    if "error" in cached or cached["fetched_at"] is None or not cached["content"].strip():
        return None
    report = {
        "source": "job",
        "fetched_at": cached["fetched_at"],
        "refresh_pending": cached["refresh_pending"],
    }
    try:
        data = json.loads(cached["content"])
        updated = float(data["updated_at"])
        fraction = data.get("fraction")
        if fraction is not None:
            fraction = float(fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("fraction must be between 0 and 1")
        message = data.get("message")
        if message is not None and not isinstance(message, str):
            raise ValueError("message must be a string")
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return {**report, "valid": False, "error": f"progress file breaks the contract: {e}"}
    age = time.time() - updated
    return {
        **report,
        "valid": True,
        "fraction": fraction,
        "message": (message or "")[:200] or None,
        "updated_at": updated,
        "stale": (not cached["settled"]) and age > PROGRESS_STALE_SEC,
    }


def register_slurm_job(
    xcelsior_job_id,
    slurm_job_id,
    *,
    cluster_id: str = DEFAULT_CLUSTER_ID,
    tenant_id: str | None = None,
    attempt_id: str | None = None,
    submit_idempotency_key: str | None = None,
    just_submitted: bool = False,
    submission_name: str | None = None,
):
    """Register a mapping between an Xcelsior job and an external Slurm job.

    One row per *submission* (migration 157, decision D7).
    ``submission_name`` is what the submission is called in Slurm
    (``xcl-<submission_name>``) and in its output file names, and what the
    reconciler matches a job's name against. It defaults to the job id; a
    scheduler attempt passes ``<job id>-a<attempt number>``, so each attempt
    keeps its own mapping, its own accounting and its own output.

    Registering a submission again with the same Slurm job is a no-op. With
    a different one it raises ``mapping_conflict``: a mapping is never
    re-pointed. The old ``ON CONFLICT (xcelsior_job_id) DO UPDATE SET
    slurm_job_id`` did that, last writer wins, losing the first job's record
    and the run it would bill. A job has at most one open mapping, as it
    has at most one active attempt, so a second submission while the first
    is open is refused too (a unique violation).

    Unique on ``(cluster_id, slurm_job_id)`` among open mappings, so two
    Xcelsior jobs can't act on one Slurm job (companion §10.1). Pass the
    profile's ``cluster_id``: Slurm job ids are only unique per cluster. A
    settled mapping does not hold its id: a rebuilt controller numbers jobs
    from the start again (migration 153).

    An *open* mapping for another submission under the same id is refused
    (a unique violation). The one exception is ``just_submitted=True``,
    which the submit paths pass because ``slurm_job_id`` is the id
    ``sbatch`` has only just returned. Slurm never hands out an id its
    controller still holds, so that open mapping belongs to a job the
    controller has lost (its state was wiped while accounting had no
    outcome). That job can never come back under this id, so its mapping is
    closed as ``unknown``, with the reason and its successor recorded,
    rather than refusing the job Slurm is already running. Returns the job
    ids of any mappings closed that way.
    """
    from db import pg_transaction

    name = str(submission_name or xcelsior_job_id)
    _require(_JOB_ID_RE, name, "submission_name")
    now = time.time()
    key = submit_idempotency_key or f"submit:{name}"
    cluster = cluster_id or DEFAULT_CLUSTER_ID
    with pg_transaction() as conn:
        superseded = (
            []
            if not just_submitted
            else [
                r[0]
                for r in conn.execute(
                    """
                UPDATE slurm_job_mappings
                   SET observed_state = 'unknown',
                       state_reason = 'slurm_job_id_reused',
                       terminal_at = clock_timestamp(),
                       last_changed_at = clock_timestamp(),
                       metadata = metadata || jsonb_build_object(
                           'superseded_by', %s::text, 'superseded_at', clock_timestamp()),
                       version = version + 1
                 WHERE cluster_id = %s AND slurm_job_id = %s
                   AND terminal_at IS NULL AND submission_name <> %s
             RETURNING xcelsior_job_id
                """,
                    (name, cluster, str(slurm_job_id), name),
                ).fetchall()
            ]
        )
        inserted = conn.execute(
            """
            INSERT INTO slurm_job_mappings
                (xcelsior_job_id, submission_name, slurm_job_id, created_at, cluster_id,
                 tenant_id, xcelsior_attempt_id, submit_idempotency_key,
                 desired_state, observed_state, submitted_at, version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'running', 'pending',
                    to_timestamp(%s), 1)
            ON CONFLICT (submission_name) DO NOTHING
            RETURNING mapping_id
            """,
            (
                xcelsior_job_id,
                name,
                str(slurm_job_id),
                now,
                cluster,
                tenant_id,
                attempt_id,
                key,
                now,
            ),
        ).fetchone()
        if inserted is None:
            held = conn.execute(
                "SELECT slurm_job_id, cluster_id FROM slurm_job_mappings WHERE submission_name = %s",
                (name,),
            ).fetchone()
            if held is None or (held[0], held[1]) != (str(slurm_job_id), cluster):
                # Raised inside the transaction: a supersede above rolls back with it.
                raise SlurmAdapterError(
                    "mapping_conflict",
                    f"submission {name} is already mapped to Slurm job "
                    f"{held[0] if held else '?'} on {held[1] if held else '?'}; "
                    "a mapping is never re-pointed",
                )
    if superseded:
        log.warning(
            "Slurm job %s on %s was handed to %s while %s still held it; closed as unknown",
            slurm_job_id,
            cluster,
            name,
            superseded,
        )
    return superseded


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
    """Reconcile every open Slurm mapping now, batched per cluster.

    A thin entry point over slurm_reconciler.reconcile_open_mappings, which
    holds the rules: one squeue and at most one bounded sacct per cluster,
    a mapping closed only when its job has left the controller and
    accounting says it finished, never from silence. The old per-job loop
    ran up to two SSH calls per open mapping and closed a mapping the moment
    squeue showed a finished state, which loses requeued jobs.

    Args:
        update_callback: fn(xcelsior_job_id, job_state) for each changed
            observation, in SLURM_STATE_MAP's job vocabulary.
        cancel_callback: fn(slurm_job_id) for a mapping whose desired_state
            is ``cancelled`` and whose job isn't finished.

    Returns:
        list of (xcelsior_job_id, slurm_job_id, old_observed, new_observed).
    """
    from slurm_reconciler import reconcile_open_mappings

    return reconcile_open_mappings(update_callback=update_callback, cancel_callback=cancel_callback)


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
        xcelsior_id,
        slurm_id,
        cluster_id=meta.get("cluster_id") or DEFAULT_CLUSTER_ID,
        just_submitted=True,
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
                "WHERE xcelsior_job_id = %s "
                "ORDER BY terminal_at IS NULL DESC, submitted_at DESC NULLS LAST LIMIT 1",
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
