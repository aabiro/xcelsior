#!/usr/bin/env python3
# Xcelsior Slurm Adapter v2.2.0
# Bridges Xcelsior job scheduling with HPC Slurm clusters.
#
# Supports:
# - Translating Xcelsior jobs to sbatch scripts
# - SHARCNET / Nibi cluster specifics (partitions, GPU types, accounts)
# - Polling squeue/sacct for job status updates
# - Bidirectional sync between Xcelsior and Slurm job states

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("xcelsior-slurm")

# ── Cluster Profiles ─────────────────────────────────────────────────
# Pre-configured profiles for known clusters.
# Users can add custom profiles via XCELSIOR_SLURM_PROFILE env var or config file.

CLUSTER_PROFILES = {
    "nibi": {
        "name": "SHARCNET Nibi",
        "partition_gpu": "gpu",
        "partition_cpu": "compute",
        "account_env": "SLURM_ACCOUNT",
        "gpu_type": "a100",  # Nibi has A100-40GB nodes
        "gpus_per_node": 4,
        "max_walltime": "7-00:00:00",  # 7 days
        "modules": ["cuda/12.2", "python/3.11"],
        "container_runtime": "apptainer",
        "scratch_dir": "/scratch/{user}",
        "project_dir": "/project/{account}",
    },
    "graham": {
        "name": "Compute Canada Graham",
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
        "gpu_type": "gpu",
        "gpus_per_node": 1,
        "max_walltime": "1-00:00:00",
        "modules": [],
        "container_runtime": "docker",
        "scratch_dir": "/tmp",
        "project_dir": "/home/{user}",
    },
}

# ── GPU VRAM to Slurm GPU Count Mapping ──────────────────────────────

GPU_VRAM_TABLE = {
    "a100": 40,  # A100-40GB (80GB variant exists)
    "v100": 16,
    "t4": 16,
    "h100": 80,
    "l4": 24,
    "rtx_3090": 24,
    "rtx_4090": 24,
}


# ── Slurm Availability Check ─────────────────────────────────────────


def is_slurm_available():
    """Check if Slurm commands (sbatch, squeue, sacct) are available."""
    try:
        r = subprocess.run(
            ["sinfo", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0
    except (FileNotFoundError, PermissionError, OSError, subprocess.TimeoutExpired):
        return False


def get_cluster_info():
    """Get basic info about the Slurm cluster."""
    if not is_slurm_available():
        return None

    info = {}
    try:
        r = subprocess.run(
            ["sinfo", "-N", "--noheader", "-o", "%N %P %G %m"],
            capture_output=True,
            text=True,
            timeout=10,
        )
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


# ── Job Translation ──────────────────────────────────────────────────


def _get_profile(profile_name=None):
    """Get cluster profile by name or from environment."""
    name = profile_name or os.environ.get("XCELSIOR_SLURM_PROFILE", "generic")
    return CLUSTER_PROFILES.get(name, CLUSTER_PROFILES["generic"])


def _estimate_gpus_needed(vram_needed_gb, gpu_type):
    """Estimate how many GPUs are needed for a given VRAM requirement."""
    vram_per_gpu = GPU_VRAM_TABLE.get(gpu_type.lower(), 16)
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


def xcelsior_job_to_sbatch(job, profile_name=None, extra_env=None):
    """Translate an Xcelsior job dict into an sbatch script string.

    Args:
        job: Xcelsior job dict with keys: name, vram_needed_gb, priority, tier, job_id, etc.
        profile_name: Cluster profile name (nibi, graham, narval, generic)
        extra_env: Additional environment variables dict

    Returns:
        (script: str, metadata: dict) — sbatch script content and translation metadata
    """
    profile = _get_profile(profile_name)
    gpu_type = profile["gpu_type"]
    num_gpus = job.get("num_gpus", None)
    if num_gpus is None:
        num_gpus = _estimate_gpus_needed(job.get("vram_needed_gb", 0), gpu_type)

    walltime = _estimate_walltime(job.get("name", ""))
    qos = _priority_to_qos(job.get("priority", 0))

    account = os.environ.get(profile["account_env"], "")
    user = os.environ.get("USER", "xcelsior")

    scratch = profile["scratch_dir"].format(user=user, account=account)
    project = profile["project_dir"].format(user=user, account=account)

    job_id = job.get("job_id", "unknown")
    job_name = f"xcl-{job_id}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={profile['partition_gpu']}",
        f"#SBATCH --gres=gpu:{gpu_type}:{num_gpus}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={scratch}/xcelsior/{job_id}-%j.out",
        f"#SBATCH --error={scratch}/xcelsior/{job_id}-%j.err",
    ]

    if account:
        lines.append(f"#SBATCH --account={account}")

    # Attempt to set QoS (may not be available on all clusters)
    if qos != "default":
        lines.append(f"#SBATCH --qos={qos}")

    # Nodes and tasks
    nodes_needed = max(1, -(-num_gpus // profile["gpus_per_node"]))
    if nodes_needed > 1:
        lines.append(f"#SBATCH --nodes={nodes_needed}")
        lines.append(f"#SBATCH --ntasks-per-node=1")
    else:
        lines.append("#SBATCH --nodes=1")
        lines.append("#SBATCH --ntasks=1")

    # Memory (estimate: 8GB per GPU + 4GB base)
    mem_gb = 4 + (8 * num_gpus)
    lines.append(f"#SBATCH --mem={mem_gb}G")

    # Email notifications (optional)
    email = os.environ.get("XCELSIOR_SLURM_EMAIL", "")
    if email:
        lines.append(f"#SBATCH --mail-user={email}")
        lines.append("#SBATCH --mail-type=END,FAIL")

    lines.append("")
    lines.append("# ── Environment Setup ──")
    lines.append("set -euo pipefail")

    # Load modules
    for mod in profile["modules"]:
        lines.append(f"module load {mod}")
    if profile["modules"]:
        lines.append("")

    # Create output directory
    lines.append(f"mkdir -p {scratch}/xcelsior")
    lines.append("")

    # Environment variables
    lines.append("# ── Xcelsior Job Metadata ──")
    lines.append(f"export XCELSIOR_JOB_ID={job_id}")
    lines.append(f"export XCELSIOR_JOB_NAME=\"{job.get('name', '')}\"")
    lines.append(f"export XCELSIOR_TIER=\"{job.get('tier', 'free')}\"")
    lines.append(f"export XCELSIOR_NUM_GPUS={num_gpus}")

    if extra_env:
        lines.append("")
        for k, v in extra_env.items():
            lines.append(f'export {k}="{v}"')

    lines.append("")

    # Container execution
    container_rt = profile["container_runtime"]
    image = job.get("image", "")

    if container_rt == "apptainer" and image:
        # HPC clusters typically use Apptainer (formerly Singularity)
        sif_path = f"{scratch}/xcelsior/{job_id}.sif"
        lines.append("# ── Container Execution (Apptainer) ──")
        lines.append(f"apptainer pull --force {sif_path} docker://{image}")
        lines.append(
            f"apptainer exec --nv {sif_path} python -c 'print(\"Xcelsior job {job_id} running\")'"
        )
    elif container_rt == "docker" and image:
        lines.append("# ── Container Execution (Docker) ──")
        lines.append(f"docker pull {image}")
        gpu_flag = (
            f"--gpus '\"device=$(seq -s, 0 {num_gpus - 1})\"'" if num_gpus > 1 else "--gpus all"
        )
        lines.append(f"docker run --rm {gpu_flag} {image}")
    else:
        lines.append("# ── Inference Script ──")
        lines.append(f"echo \"Running Xcelsior job {job_id}: {job.get('name', '')}\"")
        lines.append("# Add your inference command here:")
        lines.append(f"# python inference.py --model \"{job.get('name', '')}\" --gpus {num_gpus}")

    lines.append("")
    lines.append(f'echo "Xcelsior job {job_id} completed at $(date)"')

    metadata = {
        "job_id": job_id,
        "slurm_job_name": job_name,
        "profile": profile_name or "generic",
        "num_gpus": num_gpus,
        "gpu_type": gpu_type,
        "walltime": walltime,
        "qos": qos,
        "nodes": nodes_needed,
    }

    return "\n".join(lines), metadata


# ── Slurm Submission ─────────────────────────────────────────────────


def submit_to_slurm(job, profile_name=None, extra_env=None, dry_run=False):
    """Submit an Xcelsior job to a Slurm cluster.

    Args:
        job: Xcelsior job dict
        profile_name: Cluster profile name
        extra_env: Additional environment variables
        dry_run: If True, return script without submitting

    Returns:
        dict with slurm_job_id, script, metadata, or error
    """
    script, metadata = xcelsior_job_to_sbatch(job, profile_name, extra_env)

    if dry_run:
        return {"script": script, "metadata": metadata, "dry_run": True}

    if not is_slurm_available():
        return {"error": "Slurm not available on this system"}

    # Write script to temp file and submit
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sbatch",
            prefix=f"xcl-{job.get('job_id', 'x')}-",
            delete=False,
        ) as f:
            f.write(script)
            script_path = f.name

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            return {"error": f"sbatch failed: {result.stderr.strip()}", "script": script}

        # Parse Slurm job ID from output: "Submitted batch job 12345"
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        slurm_job_id = match.group(1) if match else "unknown"

        log.info("Submitted Xcelsior job %s as Slurm job %s", job.get("job_id", "?"), slurm_job_id)

        return {
            "slurm_job_id": slurm_job_id,
            "xcelsior_job_id": job.get("job_id"),
            "script_path": script_path,
            "metadata": metadata,
        }

    except subprocess.TimeoutExpired:
        return {"error": "sbatch timed out"}
    except Exception as e:
        return {"error": str(e)}


# ── Status Polling ───────────────────────────────────────────────────

# Slurm state → Xcelsior state mapping
SLURM_STATE_MAP = {
    "PENDING": "queued",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "preempted",
    "SUSPENDED": "queued",
    "OUT_OF_MEMORY": "failed",
}


def get_slurm_job_status(slurm_job_id):
    """Get the current status of a Slurm job.

    Returns:
        dict with slurm_state, xcelsior_state, job_info
    """
    if not is_slurm_available():
        return {"error": "Slurm not available"}

    try:
        # Try squeue first (for running/pending jobs)
        r = subprocess.run(
            ["squeue", "-j", str(slurm_job_id), "--noheader", "-o", "%i %T %M %l %D %R"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if r.returncode == 0 and r.stdout.strip():
            # Format is "%i %T %M %l %D %R": the first five fields are
            # single tokens, but %R (reason / nodelist) can contain spaces
            # — e.g. "(launch failed requeued held)". Cap the split at 5 so
            # the reason is preserved intact instead of truncated to its
            # first word. squeue -j for a single id returns one line.
            parts = r.stdout.strip().split(None, 5)
            slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"
            return {
                "slurm_job_id": slurm_job_id,
                "slurm_state": slurm_state,
                "xcelsior_state": SLURM_STATE_MAP.get(slurm_state, "running"),
                "elapsed": parts[2] if len(parts) > 2 else "0:00",
                "time_limit": parts[3] if len(parts) > 3 else "?",
                "nodes": parts[4] if len(parts) > 4 else "1",
                "reason": parts[5] if len(parts) > 5 else "",
            }

        # Fall back to sacct for completed jobs
        r2 = subprocess.run(
            [
                "sacct",
                "-j",
                str(slurm_job_id),
                "--noheader",
                "--parsable2",
                "-o",
                "JobID,State,Elapsed,ExitCode,MaxRSS,MaxVMSize",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if r2.returncode == 0 and r2.stdout.strip():
            for line in r2.stdout.strip().split("\n"):
                parts = line.split("|")
                # Skip sub-step entries (e.g., "12345.batch")
                if "." not in parts[0]:
                    slurm_state = parts[1] if len(parts) > 1 else "UNKNOWN"
                    return {
                        "slurm_job_id": slurm_job_id,
                        "slurm_state": slurm_state,
                        "xcelsior_state": SLURM_STATE_MAP.get(slurm_state, "failed"),
                        "elapsed": parts[2] if len(parts) > 2 else "?",
                        "exit_code": parts[3] if len(parts) > 3 else "?",
                        "max_rss": parts[4] if len(parts) > 4 else "?",
                        "max_vmsize": parts[5] if len(parts) > 5 else "?",
                    }

        return {
            "slurm_job_id": slurm_job_id,
            "slurm_state": "UNKNOWN",
            "xcelsior_state": "failed",
            "error": "Job not found in squeue or sacct",
        }

    except subprocess.TimeoutExpired:
        return {"error": "Status poll timed out"}
    except Exception as e:
        return {"error": str(e)}


def cancel_slurm_job(slurm_job_id):
    """Cancel a Slurm job."""
    if not is_slurm_available():
        return {"error": "Slurm not available"}

    try:
        r = subprocess.run(
            ["scancel", str(slurm_job_id)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0:
            return {"cancelled": True, "slurm_job_id": slurm_job_id}
        return {"error": f"scancel failed: {r.stderr.strip()}"}
    except Exception as e:
        return {"error": str(e)}


# ── Sync Engine ──────────────────────────────────────────────────────
# Tracks mapping between Xcelsior job_ids and Slurm job_ids

_slurm_job_map: dict[str, str] = {}  # xcelsior_job_id -> slurm_job_id
SLURM_MAP_FILE = os.environ.get("XCELSIOR_SLURM_MAP", "data/slurm_jobs.json")


def _load_slurm_map():
    """Load Xcelsior↔Slurm job mapping from disk."""
    global _slurm_job_map
    if os.path.exists(SLURM_MAP_FILE):
        with open(SLURM_MAP_FILE) as f:
            _slurm_job_map = json.load(f)
    return _slurm_job_map


def _save_slurm_map():
    """Persist Xcelsior↔Slurm job mapping to disk."""
    os.makedirs(os.path.dirname(SLURM_MAP_FILE) or ".", exist_ok=True)
    with open(SLURM_MAP_FILE, "w") as f:
        json.dump(_slurm_job_map, f, indent=2)


def register_slurm_job(xcelsior_job_id, slurm_job_id):
    """Register a mapping between Xcelsior and Slurm job IDs."""
    _slurm_job_map[xcelsior_job_id] = slurm_job_id
    _save_slurm_map()


def sync_slurm_statuses(update_callback=None):
    """Poll all tracked Slurm jobs and sync status back to Xcelsior.

    Args:
        update_callback: function(xcelsior_job_id, new_status) called for each update

    Returns:
        list of (xcelsior_job_id, slurm_job_id, old_state, new_state) tuples
    """
    _load_slurm_map()
    changes = []

    for xcelsior_id, slurm_id in list(_slurm_job_map.items()):
        status = get_slurm_job_status(slurm_id)
        if "error" in status:
            continue

        new_state = status.get("xcelsior_state", "")
        if new_state and update_callback:
            update_callback(xcelsior_id, new_state)
            changes.append((xcelsior_id, slurm_id, "unknown", new_state))

        # Clean up completed/failed jobs from tracking
        if new_state in ("completed", "failed", "cancelled"):
            del _slurm_job_map[xcelsior_id]

    _save_slurm_map()
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
    register_slurm_job(xcelsior_id, slurm_id)

    meta = result.get("metadata", {})
    return (
        f"Submitted to Slurm: {slurm_id}\n"
        f"  Xcelsior job: {xcelsior_id}\n"
        f"  Cluster:      {meta.get('profile', '?')}\n"
        f"  GPUs:         {meta.get('num_gpus', '?')} x {meta.get('gpu_type', '?')}\n"
        f"  Walltime:     {meta.get('walltime', '?')}\n"
        f"  Nodes:        {meta.get('nodes', '?')}"
    )


def slurm_status_cli(xcelsior_job_id=None, slurm_job_id=None):
    """CLI-friendly wrapper for Slurm status check."""
    if slurm_job_id:
        status = get_slurm_job_status(slurm_job_id)
    elif xcelsior_job_id:
        _load_slurm_map()
        s_id = _slurm_job_map.get(xcelsior_job_id)
        if not s_id:
            return f"No Slurm job found for Xcelsior job {xcelsior_job_id}"
        status = get_slurm_job_status(s_id)
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
