# Xcelsior Worker Agent — NVMe local model cache (cold-start optimization).
#
# Extracted from worker_agent.py. Per Phase 3.4: pre-pull model weights to
# local NVMe cache on first deploy. Tiered cache hierarchy:
# VRAM → NVMe SSD → Network Storage (NFS). Fully self-contained: only
# touches os/subprocess/shutil and the shared "xcelsior-worker" logger.

import logging
import os
import subprocess
import threading

log = logging.getLogger("xcelsior-worker")

NVME_CACHE_DIR = os.environ.get("XCELSIOR_NVME_CACHE_DIR", "/mnt/nvme/model-cache")
NVME_CACHE_MAX_GB = int(os.environ.get("XCELSIOR_NVME_CACHE_MAX_GB", "200"))

_nvme_cache_lock = threading.Lock()


def nvme_cache_init():
    """Initialize local NVMe model cache directory."""
    os.makedirs(NVME_CACHE_DIR, exist_ok=True)
    log.info("NVMe model cache initialized at %s (max %d GB)", NVME_CACHE_DIR, NVME_CACHE_MAX_GB)


def nvme_cache_path(model_id: str, revision: str = "main") -> str:
    """Return the local NVMe cache path for a model."""
    safe_name = model_id.replace("/", "--")
    return os.path.join(NVME_CACHE_DIR, f"{safe_name}_{revision}")


def nvme_cache_has(model_id: str, revision: str = "main") -> bool:
    """Check if a model is already cached on local NVMe."""
    path = nvme_cache_path(model_id, revision)
    return os.path.isdir(path) and any(os.scandir(path))


def nvme_cache_size_gb() -> float:
    """Total size of the NVMe model cache in GB."""
    total = 0
    if not os.path.isdir(NVME_CACHE_DIR):
        return 0.0
    for entry in os.scandir(NVME_CACHE_DIR):
        if entry.is_dir():
            for f in os.scandir(entry.path):
                if f.is_file():
                    total += f.stat().st_size
    return total / (1024**3)


def nvme_cache_evict_lru():
    """Evict least-recently-used models from NVMe cache if over limit."""
    import shutil

    if not os.path.isdir(NVME_CACHE_DIR):
        return

    entries = []
    for entry in os.scandir(NVME_CACHE_DIR):
        if entry.is_dir():
            entries.append((entry.path, entry.stat().st_mtime))

    # Sort oldest first
    entries.sort(key=lambda e: e[1])

    while nvme_cache_size_gb() > NVME_CACHE_MAX_GB and entries:
        oldest_path, _ = entries.pop(0)
        log.info("NVMe cache evicting: %s", os.path.basename(oldest_path))
        shutil.rmtree(oldest_path, ignore_errors=True)


def nvme_prepull_model(model_id: str, revision: str = "main", source_nfs: str = "") -> bool:
    """Pre-pull model weights to local NVMe cache for fast cold starts.

    Per Phase 3.4: tiered cache — VRAM → NVMe → Network Storage.
    Downloads model from NFS shared storage or HuggingFace Hub to local NVMe.

    Args:
        model_id: Model identifier (e.g. "meta-llama/Llama-3-70B")
        revision: Model revision/branch
        source_nfs: Optional NFS path to copy from (faster than download)

    Returns:
        True if model is now available on local NVMe.
    """
    if nvme_cache_has(model_id, revision):
        log.debug("Model %s@%s already in NVMe cache", model_id, revision)
        return True

    with _nvme_cache_lock:
        # Double-check under lock
        if nvme_cache_has(model_id, revision):
            return True

        # Evict if cache is full
        nvme_cache_evict_lru()

        dest = nvme_cache_path(model_id, revision)
        os.makedirs(dest, exist_ok=True)

        # Strategy 1: Copy from NFS shared storage (fastest)
        if source_nfs and os.path.isdir(source_nfs):
            try:
                import shutil

                log.info("NVMe pre-pull: copying %s from NFS %s", model_id, source_nfs)
                for item in os.listdir(source_nfs):
                    src = os.path.join(source_nfs, item)
                    dst = os.path.join(dest, item)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                log.info("NVMe pre-pull complete: %s (from NFS)", model_id)
                return True
            except Exception as e:
                log.warning("NFS copy failed for %s: %s, trying HuggingFace", model_id, e)

        # Strategy 2: Download from HuggingFace Hub
        try:
            dl_cmd = [
                "huggingface-cli",
                "download",
                model_id,
                "--revision",
                revision,
                "--local-dir",
                dest,
                "--quiet",
            ]
            result = subprocess.run(
                dl_cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max for large models
            )
            if result.returncode == 0:
                log.info("NVMe pre-pull complete: %s (from HuggingFace)", model_id)
                return True
            else:
                log.warning("HuggingFace download failed for %s: %s", model_id, result.stderr[:200])
                return False
        except FileNotFoundError:
            log.warning("huggingface-cli not found — cannot pre-pull %s", model_id)
            return False
        except subprocess.TimeoutExpired:
            log.warning("NVMe pre-pull timed out for %s", model_id)
            return False
