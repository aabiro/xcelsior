#!/usr/bin/env bash
# Rehearse the whole kit in two containers on one machine, then tear down.
#
#   ./rehearsal/run-rehearsal.sh [--keep] [--suite-args "..."]
#
# tower container: slurmctld + slurmdbd + MariaDB + slurmd, with this
#                  machine's NVIDIA GPU (--gpus all) for AutoDetect=nvml
# asus container:  slurmd only, config-less
# The suite runs on THIS host and reaches the tower container over SSH
# through xcl-slurm-gate — the same path as the real run.
#
# Deviations from the real cluster (and only these):
#   addresses 172.31.251.4/.6 instead of the tailnet; tower node CPUs=16;
#   cgroup.conf IgnoreSystemd=yes; daemons started without systemd; the GPU
#   is whatever this host has; SSH on 127.0.0.1:22022.
set -euo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$KIT/../.." && pwd)"
WORK="$KIT/rehearsal/.work"
NET=xcl-slurm-rh
SUBNET=172.31.251.0/24
T_IP=172.31.251.4
A_IP=172.31.251.6
ASUS_HOST=aaryn-ASUS-TUF-Gaming-A15-FA506IV-TUF506IV
KEEP=0
SUITE_ARGS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP=1; shift ;;
        --suite-args) SUITE_ARGS="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done

teardown() {
    docker rm -f xcl-rh-tower xcl-rh-asus >/dev/null 2>&1 || true
    docker network rm "$NET" >/dev/null 2>&1 || true
}
if [[ $KEEP == 0 ]]; then trap teardown EXIT; fi

teardown
# Fresh per-run state; earlier results/ are kept. Containers write some of
# this as root, so fall back to a throwaway container to delete it.
for d in shared asus-jobs; do
    rm -rf "${WORK:?}/$d" 2>/dev/null || docker run --rm -v "$WORK:/w" ubuntu:26.04 rm -rf "/w/$d"
done
rm -f "$WORK/adapter_key" "$WORK/adapter_key.pub" "$WORK/profiles.json"
mkdir -p "$WORK/shared" "$WORK/asus-jobs" "$WORK/results"
ssh-keygen -q -t ed25519 -N '' -C xcl-slurm-adapter -f "$WORK/adapter_key"
cp "$WORK/adapter_key.pub" "$WORK/shared/adapter_key.pub"

echo "== building images (Ubuntu 26.04, Slurm from the archive)"
docker build -q -f "$KIT/rehearsal/Dockerfile" --build-arg ROLE=tower -t xcl-slurm-rh:tower "$KIT" >/dev/null
docker build -q -f "$KIT/rehearsal/Dockerfile" --build-arg ROLE=asus -t xcl-slurm-rh:asus "$KIT" >/dev/null

docker network create --subnet "$SUBNET" "$NET" >/dev/null
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/^NVIDIA //; s/^GeForce //')"
# "RTX 2060" → "rtx_2060": a substring of NVML's nvidia_geforce_rtx_2060
GRES_TYPE="$(tr 'A-Z ' 'a-z_' <<<"${GPU_NAME:-rtx}")"
sed "s/\"gpu_type\": \"rtx_3060\"/\"gpu_type\": \"$GRES_TYPE\"/" "$KIT/conf/xcelsior-profiles.json" >"$WORK/profiles.json"
GPU_FLAGS=()
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    GPU_FLAGS=(--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility)
fi
COMMON=(--privileged --cgroupns=private -e TOWER_ADDR=$T_IP -e ASUS_ADDR=$A_IP -e GRES_TYPE=$GRES_TYPE -v "$WORK/shared:/shared" --memory 1500m)
docker run -d --name xcl-rh-tower --hostname tower --network "$NET" --ip $T_IP \
    -p 127.0.0.1:22022:22 "${GPU_FLAGS[@]}" "${COMMON[@]}" xcl-slurm-rh:tower tower >/dev/null
docker run -d --name xcl-rh-asus --hostname "$ASUS_HOST" --network "$NET" --ip $A_IP \
    -v "$WORK/asus-jobs:/var/lib/xcl-slurm-test/jobs" "${COMMON[@]}" xcl-slurm-rh:asus asus >/dev/null

echo "== waiting for both nodes"
for _ in $(seq 120); do
    [[ -e "$WORK/shared/tower.ready" && -e "$WORK/shared/asus.ready" ]] && break
    sleep 2
done
[[ -e "$WORK/shared/tower.ready" && -e "$WORK/shared/asus.ready" ]] || { docker logs xcl-rh-tower | tail -40; docker logs xcl-rh-asus | tail -20; exit 1; }
for _ in $(seq 60); do
    n=$(docker exec xcl-rh-tower sinfo -h -p xcl -N -o '%T' 2>/dev/null | grep -Ec '^(idle|mixed)$' || true)
    [[ "$n" == 2 ]] && break
    sleep 2
done
docker exec xcl-rh-tower sinfo -N -l || true

echo "== cluster verification (raw Slurm, inside the tower container)"
docker exec -u aaryn xcl-rh-tower bash -c "cd /kit/scripts && ./30-verify-cluster.sh --expect-gpu '${GPU_NAME:-RTX}' --expect-type '$GRES_TYPE'" || true

echo "== adapter suite (this host → ssh → gate → slurmctld)"
# The overrides apply only to the profile named here, and are appended to its
# ssh options; ssh uses the last -F, so the -F /dev/null below wins over the profile's.
export XCELSIOR_SLURM_SSH_PROFILE=xcl-local
export XCELSIOR_SLURM_SSH_HOST=aaryn@127.0.0.1
export XCELSIOR_SLURM_SSH_OPTIONS="-F /dev/null -p 22022 -i $WORK/adapter_key -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
set +e
# shellcheck disable=SC2086
"${PYTHON:-python3}" "$KIT/tests/validate_adapter.py" --expect-gpu "${GPU_NAME:-RTX}" --profiles-file "$WORK/profiles.json" \
    --local-output-dir "$WORK/asus-jobs" --results-dir "$WORK/results" $SUITE_ARGS
rc=$?
set -e
echo "results: $WORK/results"
exit $rc
