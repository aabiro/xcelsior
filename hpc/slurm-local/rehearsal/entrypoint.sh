#!/bin/bash
# Rehearsal container entrypoint: finish the install with the kit's own
# steps, then start the daemons by hand (no systemd in a container).
set -euo pipefail
ROLE="$1"
export XCL_NO_SYSTEMD=1

# cgroup v2: move every process out of the namespace root so slurmd can
# enable controllers and create job cgroups below it (same trick as dind).
if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
    mkdir -p /sys/fs/cgroup/init
    while read -r p; do echo "$p" >/sys/fs/cgroup/init/cgroup.procs 2>/dev/null || true; done </sys/fs/cgroup/cgroup.procs
    sed -e 's/ / +/g' -e 's/^/+/' </sys/fs/cgroup/cgroup.controllers >/sys/fs/cgroup/cgroup.subtree_control || true
fi

start_munge() {
    install -d -o munge -g munge -m 0755 /run/munge
    runuser -u munge -- /usr/sbin/munged
}
wait_port() { for _ in $(seq 60); do timeout 1 bash -c "</dev/tcp/$1/$2" 2>/dev/null && return 0; sleep 1; done; echo "port $1:$2 never opened" >&2; return 1; }

case "$ROLE" in
tower)
    # configure validates the sshd drop-in with sshd -t, which needs host
    # keys and the privilege-separation dir.
    ssh-keygen -A >/dev/null
    install -d -m 0755 /run/sshd
    /kit/scripts/10-install-tower.sh configure db
    # ── rehearsal-only deviations (documented in README §Rehearsal) ──
    # 1. Both containers share this host's 16 CPUs, not the tower's 32.
    sed -i 's/CPUs=32 Boards=1 SocketsPerBoard=1 CoresPerSocket=16/CPUs=16 Boards=1 SocketsPerBoard=1 CoresPerSocket=8/' /etc/slurm/slurm.conf
    # 2. No systemd to delegate cgroups; slurmd manages its own subtree.
    grep -q IgnoreSystemd /etc/slurm/cgroup.conf || echo "IgnoreSystemd=yes" >>/etc/slurm/cgroup.conf
    # 3. The GPU is this host's, so its GRES type token replaces rtx_3060.
    if [[ -n "${GRES_TYPE:-}" ]]; then sed -i "s/Gres=gpu:rtx_3060:1/Gres=gpu:${GRES_TYPE}:1/" /etc/slurm/slurm.conf; fi
    install -m 0444 /etc/munge/munge.key /shared/munge.key
    start_munge
    install -d -o slurm -g slurm /run/slurmdbd /run/slurmctld
    /usr/sbin/slurmdbd
    wait_port 127.0.0.1 6819
    sacctmgr -n show cluster format=cluster | grep -qw xcl-local || sacctmgr -i add cluster xcl-local
    /kit/scripts/10-install-tower.sh qos
    /usr/sbin/slurmctld
    wait_port 127.0.0.1 6817
    install -d /run/slurm
    /usr/sbin/slurmd
    # Adapter key: same restricted entry the real tower gets.
    if [[ -s /shared/adapter_key.pub ]]; then
        /kit/scripts/10-install-tower.sh authorize-adapter-key /shared/adapter_key.pub "${ADAPTER_FROM:-172.31.251.1,127.0.0.1}"
    fi
    /usr/sbin/sshd
    ;;
asus)
    for _ in $(seq 120); do [[ -s /shared/munge.key ]] && break; sleep 1; done
    XCL_MUNGE_KEY_SRC=/shared/munge.key /kit/scripts/20-install-asus.sh configure
    start_munge
    wait_port "${TOWER_ADDR}" 6817
    # shellcheck disable=SC1091
    source /etc/default/slurmd
    install -d /run/slurm
    # shellcheck disable=SC2086
    /usr/sbin/slurmd $SLURMD_OPTIONS
    ;;
*) echo "role must be tower|asus" >&2; exit 2 ;;
esac
touch "/shared/$ROLE.ready"
exec tail -n +1 -F /var/log/slurm/slurmd.log /var/log/slurm/slurmctld.log /var/log/slurm/slurmdbd.log 2>/dev/null
