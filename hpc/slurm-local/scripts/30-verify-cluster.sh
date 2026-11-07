#!/usr/bin/env bash
# Confirm the cluster is healthy with raw Slurm commands, before the
# adapter suite runs. Run on the tower as aaryn (not root):
#
#   ./30-verify-cluster.sh [--expect-gpu "RTX 3060"]
#
# Prints PASS/FAIL per check and exits non-zero on any failure. These
# checks use sinfo/scontrol/srun directly on purpose: they prove the
# cluster, so a later adapter failure is the adapter's.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh" 2>/dev/null || true
set +e
EXPECT_GPU="RTX 3060"
EXPECT_TYPE="rtx_3060"
GPU_NODE=tower
CPU_NODE=asus
while [[ $# -gt 0 ]]; do
    case "$1" in
        --expect-gpu) EXPECT_GPU="$2"; shift 2 ;;
        --expect-type) EXPECT_TYPE="$2"; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done

fails=0
skips=0
check() { # name, command — exit 0 PASS, 3 SKIP, else FAIL
    local out rc
    out="$(bash -o pipefail -c "$2" 2>&1)"
    rc=$?
    if [[ $rc -eq 3 ]]; then printf 'SKIP  %-46s %s\n' "$1" "$(head -c 160 <<<"$out" | tr '\n' ' ')"; skips=$((skips + 1)); return; fi
    if [[ $rc -eq 0 ]]; then printf 'PASS  %-46s %s\n' "$1" "$(head -c 160 <<<"$out" | tr '\n' ' ')"
    else printf 'FAIL  %-46s %s\n' "$1" "$(head -c 300 <<<"$out" | tr '\n' ' ')"; fails=$((fails + 1)); fi
}

check "V-1 sinfo: partition xcl has 2 nodes" \
    "sinfo -h -p xcl -N -o '%N %T' | tee /dev/stderr | wc -l | grep -qx 2"
check "V-2 both nodes idle/mixed (not down/drain)" \
    "! sinfo -h -p xcl -N -o '%T' | grep -Eqv '^(idle|mixed|allocated)'"
check "V-3 scontrol: $GPU_NODE Gres is gpu:$EXPECT_TYPE:1" \
    "scontrol show node $GPU_NODE | grep -E 'Gres=gpu:$EXPECT_TYPE:1'"
check "V-4 scontrol: $GPU_NODE CfgTRES lists gres/gpu=1" \
    "scontrol show node $GPU_NODE | grep -E 'CfgTRES=.*gres/gpu=1'"
check "V-5 scontrol: $CPU_NODE has no GPU" \
    "scontrol show node $CPU_NODE | grep -E 'Gres=\(null\)'"
check "V-6 slurmd -G: NVML detected type $EXPECT_TYPE + /dev/nvidia0" \
    "if ! sudo -n true 2>/dev/null; then echo 'needs passwordless sudo'; exit 3; fi; sudo -n slurmd -G 2>&1 | grep -E 'Name=gpu Type=$EXPECT_TYPE Count=1 .*File=/dev/nvidia0'"
check "V-7 srun GPU step sees exactly 1 GPU ($EXPECT_GPU)" \
    "srun --chdir=/tmp -p xcl --gres=gpu:1 -t 1 nvidia-smi -L | tee /dev/stderr | grep -c '^GPU ' | grep -qx 1 && srun --chdir=/tmp -p xcl --gres=gpu:1 -t 1 nvidia-smi -L | grep -q '$EXPECT_GPU'"
check "V-8 srun without GRES on $GPU_NODE sees no GPU (ConstrainDevices)" \
    "! srun --chdir=/tmp -p xcl -w $GPU_NODE -t 1 nvidia-smi -L 2>&1 | grep -q '^GPU '"
check "V-9 srun on $CPU_NODE runs" \
    "srun --chdir=/tmp -p xcl -w $CPU_NODE -t 1 hostname"
check "V-10 sacctmgr: cluster xcl-local registered" \
    "sacctmgr -n show cluster format=cluster | grep -qw xcl-local"
check "V-11 sacct records jobs with epoch times" \
    "SLURM_TIME_FORMAT=%s sacct -X -n -S now-1hour -o JobID,State,Start,End,AllocTRES --parsable2 | tail -3 | tee /dev/stderr | grep -Eq '\|[0-9]{10}\|[0-9]{10}\|'"

if [[ -r "${XCL_STATE:-/var/lib/xcl-slurm-test}/untouchables.before" ]] && command -v docker >/dev/null; then
    # root creates the file it writes: with fs.protected_regular (Ubuntu's
    # default) root may not write a file another user owns in /tmp, and the
    # snapshot silently came out empty.
    after="$(sudo -n mktemp)"
    sudo -n bash -c "source '$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh'; xcl_snapshot_untouchables '$after'"
    check "V-12 Docker containers + runners unchanged since preflight" \
        "sudo -n test -s '$after' && sudo -n diff '${XCL_STATE:-/var/lib/xcl-slurm-test}/untouchables.before' '$after'"
    sudo -n rm -f "$after"
fi

echo
if [[ $fails -eq 0 ]]; then echo "cluster verification: ALL PASS ($skips skipped)"; else echo "cluster verification: $fails FAIL ($skips skipped)"; fi
exit $((fails > 0))
