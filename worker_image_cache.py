# Xcelsior Worker Agent — Docker image cache / pre-pull.
#
# Extracted from worker_agent.py (image pre-pull/cache manager section) to
# keep the agent monolith smaller. Tracks locally-cached Docker images with
# LRU eviction so popular images can be pre-pulled during idle time without
# unbounded disk growth.
#
# A couple of functions (`cache_prepull_popular`, `fetch_popular_images`) need
# agent-wide shared state owned by worker_agent.py (the shutdown event, the
# active-containers map, the API auth headers). Rather than restructure the
# whole agent around a shared context object, they import worker_agent lazily
# inside the function body — this avoids a circular import at module-load
# time (worker_agent imports *this* module at its own top level) while
# preserving the exact same shared objects/behavior as before the split.

import logging
import subprocess
import threading
import time
from os import environ

log = logging.getLogger("xcelsior-worker")

IMAGE_CACHE_MAX_GB = float(environ.get("XCELSIOR_IMAGE_CACHE_MAX_GB", "200"))
IMAGE_CACHE_EVICT_LOW_GB = float(environ.get("XCELSIOR_IMAGE_CACHE_EVICT_LOW_GB", "150"))

_image_cache_lock = threading.Lock()
# {image_tag: {"last_used": timestamp, "size_mb": float, "pull_count": int}}
_image_cache_index: dict[str, dict] = {}


def _get_local_images():
    """List locally cached Docker images with sizes."""
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}} {{.Size}}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return {}

        images = {}
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            tag, size_str = parts
            # Parse size string (e.g., "2.5GB", "850MB")
            size_mb = _parse_docker_size(size_str)
            images[tag] = size_mb
        return images
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}


def _parse_docker_size(size_str):
    """Parse Docker size string to MB."""
    size_str = size_str.strip().upper()
    try:
        if "GB" in size_str:
            return float(size_str.replace("GB", "")) * 1024
        elif "MB" in size_str:
            return float(size_str.replace("MB", ""))
        elif "KB" in size_str:
            return float(size_str.replace("KB", "")) / 1024
        elif "B" in size_str:
            return float(size_str.replace("B", "")) / (1024 * 1024)
        return 0
    except (ValueError, TypeError):
        return 0


def _total_cache_size_mb():
    """Get total size of cached images in MB."""
    with _image_cache_lock:
        return sum(entry.get("size_mb", 0) for entry in _image_cache_index.values())


def cache_track_pull(image_tag, size_mb=0):
    """Record that an image was pulled/used. Update LRU tracking."""
    with _image_cache_lock:
        if image_tag in _image_cache_index:
            _image_cache_index[image_tag]["last_used"] = time.time()
            _image_cache_index[image_tag]["pull_count"] += 1
            if size_mb > 0:
                _image_cache_index[image_tag]["size_mb"] = size_mb
        else:
            _image_cache_index[image_tag] = {
                "last_used": time.time(),
                "size_mb": size_mb,
                "pull_count": 1,
            }


def cache_evict_lru(exclude_images: set | None = None):
    """Evict least-recently-used images when cache exceeds limit.

    Evicts until cache is below IMAGE_CACHE_EVICT_LOW_GB.
    Never evicts currently-running container images or images in *exclude_images*.
    """
    import worker_agent  # lazy import — worker_agent.* is the source of truth so
    # tests/operators patching worker_agent.IMAGE_CACHE_MAX_GB still take effect.

    exclude_images = exclude_images or set()
    total_mb = _total_cache_size_mb()
    limit_mb = worker_agent.IMAGE_CACHE_MAX_GB * 1024

    if total_mb <= limit_mb:
        return 0

    target_mb = worker_agent.IMAGE_CACHE_EVICT_LOW_GB * 1024
    evicted = 0

    # Get currently running images (do not evict)
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Image}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        running_images = set(result.stdout.strip().split("\n")) if result.returncode == 0 else set()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        running_images = set()

    # Sort by last_used ascending (oldest first)
    with _image_cache_lock:
        sorted_images = sorted(
            _image_cache_index.items(),
            key=lambda x: x[1].get("last_used", 0),
        )

    for image_tag, entry in sorted_images:
        if total_mb <= target_mb:
            break
        if image_tag in running_images or image_tag in exclude_images:
            continue

        # Remove image
        try:
            rm = subprocess.run(
                ["docker", "rmi", image_tag],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if rm.returncode == 0:
                size = entry.get("size_mb", 0)
                total_mb -= size
                evicted += 1
                with _image_cache_lock:
                    _image_cache_index.pop(image_tag, None)
                log.info("Cache evicted: %s (%.0f MB freed)", image_tag, size)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return evicted


def cache_init():
    """Initialize cache index from locally available Docker images.

    Prunes dangling (<none>:<none>) images first so the cache only
    tracks real, reusable tagged images.
    """
    # Remove dangling images — they waste disk and inflate cache size
    try:
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    images = _get_local_images()
    with _image_cache_lock:
        for tag, size_mb in images.items():
            # Skip any residual untagged entries
            if "<none>" in tag:
                continue
            if tag not in _image_cache_index:
                _image_cache_index[tag] = {
                    "last_used": time.time(),
                    "size_mb": size_mb,
                    "pull_count": 0,
                }


def cache_prepull_popular(popular_images, max_concurrent=3):
    """Pre-pull popular images during idle time.

    Called from the main loop when the agent has no active work.
    Only pulls images not already cached, respecting cache limits.
    Pulls up to max_concurrent images in parallel (serialized when jobs run).

    Args:
        popular_images: List of image tags ordered by popularity.
        max_concurrent: Max simultaneous docker pulls.
    """
    import concurrent.futures

    import worker_agent  # lazy import — shares worker_agent's shutdown/active-container state

    if worker_agent._active_containers:
        log.debug("Skipping image pre-pull — %d active job(s)", len(worker_agent._active_containers))
        return

    to_pull = []
    for image_tag in popular_images:
        with _image_cache_lock:
            if image_tag in _image_cache_index:
                continue  # Already cached

        total_mb = _total_cache_size_mb()
        if total_mb >= worker_agent.IMAGE_CACHE_MAX_GB * 1024:
            log.debug("Cache full — skipping pre-pull of %s", image_tag)
            break
        to_pull.append(image_tag)

    if not to_pull:
        return

    def _pull_one(img_tag):
        if worker_agent._shutdown.is_set() or worker_agent._active_containers:
            return False
        log.info("Pre-pulling: %s", img_tag)
        try:
            with worker_agent._image_pull_lock:
                if worker_agent._active_containers:
                    return False
                pull = subprocess.Popen(
                    ["docker", "pull", img_tag],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # Poll with shutdown check so we can abort quickly on SIGTERM
                while pull.poll() is None:
                    if worker_agent._shutdown.is_set():
                        pull.terminate()
                        pull.wait(timeout=5)
                        return False
                    time.sleep(1)
                if pull.returncode == 0:
                    images = _get_local_images()
                    size_mb = images.get(img_tag, 0)
                    cache_track_pull(img_tag, size_mb)
                    log.info("Pre-pulled %s (%.0f MB)", img_tag, size_mb)
                    return True
                stderr = pull.stdout.read() if pull.stdout else ""
                log.warning(
                    "Pre-pull failed %s: %s", img_tag, stderr[:200] if stderr else "unknown"
                )
        except subprocess.TimeoutExpired:
            log.warning("Pre-pull timed out: %s (>15 min)", img_tag)
        except FileNotFoundError:
            log.error("docker not found — cannot pre-pull")
        return False

    workers = 1 if worker_agent._active_containers else max_concurrent
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_pull_one, to_pull))  # list() forces consumption, surfaces exceptions


def fetch_popular_images():
    """Ask the scheduler for popular/frequently-used images.

    GET /agent/popular-images
    Returns list of image tags.
    """
    import worker_agent  # lazy import — shares worker_agent's requests/auth helpers

    try:
        resp = worker_agent.requests.get(
            worker_agent._api_url("/agent/popular-images"),
            headers=worker_agent._api_headers(),
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("images", [])
        return []
    except worker_agent.requests.RequestException:
        return []
