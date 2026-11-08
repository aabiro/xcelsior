#!/usr/bin/env bash
# Build SchedMD's own Slurm .deb packages (slurm-smd-*) for Ubuntu 26.04, in
# a throwaway container, so no build tools land on the tower or the asus.
#
#   ./05-build-slurm-debs.sh        → hpc/slurm-local/debs/slurm-smd*_<version>_amd64.deb
#
# Why not Ubuntu's slurm-wlm: SchedMD "does not maintain or recommend" distro
# packages, and Ubuntu 26.04 still ships 25.11.2, without the eight CVE fixes
# in 25.11.8 (2026-09-02). The installers use these debs whenever debs/
# holds them (scripts/lib.sh, XCL_DEBS_DIR).
#
# The tarball is pinned by sha256. 25.11.8's value was checked on 2026-11-08:
# the tarball from download.schedmd.com unpacks to a tree identical to
# GitHub's slurm-25-11-8-1 tag. When moving to a new release, repeat that
# check before changing SLURM_SHA256.
set -euo pipefail
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM_VERSION="${SLURM_VERSION:-25.11.8}"
SLURM_SHA256="${SLURM_SHA256:-34ace13f81011add6094569d13bfc4006ad8868201c2236e2905443c7e526393}"
OUT="${XCL_DEBS_DIR:-$KIT_DIR/debs}"
log() { printf '[xcl-slurm] %s\n' "$*" >&2; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
tarball="slurm-$SLURM_VERSION.tar.bz2"
log "downloading $tarball"
curl -fsSL -m 600 -o "$work/$tarball" "https://download.schedmd.com/slurm/$tarball"
echo "$SLURM_SHA256  $work/$tarball" | sha256sum -c --quiet - || { log "sha256 mismatch for $tarball"; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT"/slurm-smd*.deb
log "building slurm-smd $SLURM_VERSION in ubuntu:26.04 (about 15 minutes)"
# libnvidia-ml-dev is not in SchedMD's Build-Depends, but configure builds
# the gres/gpu NVML plugin (AutoDetect=nvml on the tower) only when it is.
docker run --rm -v "$work:/src:ro" -v "$OUT:/out" \
    -e SLURM_VERSION="$SLURM_VERSION" -e OWNER="$(id -u):$(id -g)" ubuntu:26.04 bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends build-essential devscripts equivs fakeroot \
        bzip2 libnvidia-ml-dev >/dev/null
    mkdir /build && cd /build && tar -xjf "/src/slurm-$SLURM_VERSION.tar.bz2"
    cd "slurm-$SLURM_VERSION"
    mk-build-deps -i -r -t "apt-get -y -qq --no-install-recommends" debian/control >/dev/null
    if ! debuild -b -uc -us >/build/debuild.log 2>&1; then tail -60 /build/debuild.log; exit 1; fi
    grep -E "^(checking for nvml.h|checking for nvmlInit)" /build/debuild.log || true
    # Not grep -q: it exits at the first match, dpkg-deb dies of SIGPIPE, and
    # pipefail turns the match into a failure (this discarded a good build).
    dpkg -c /build/slurm-smd_*.deb | grep "gpu_nvml.so" >/dev/null || { echo "gpu_nvml plugin was not built" >&2; exit 1; }
    cp /build/slurm-smd*.deb /out/
    chown "$OWNER" /out/slurm-smd*.deb
'
log "built: $(cd "$OUT" && ls slurm-smd*.deb | tr '\n' ' ')"
