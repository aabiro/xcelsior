#!/usr/bin/env bash
# Install the controller + GPU node of the xcl-local test cluster on the tower.
#
#   sudo ./10-install-tower.sh all            preflight users packages configure db services
#   sudo ./10-install-tower.sh <step>...      run named steps only
#   sudo ./10-install-tower.sh authorize-adapter-key /path/to/key.pub
#
# Test/research only. Every change is recorded in /var/lib/xcl-slurm-test/
# manifest; sudo ./90-rollback.sh undoes it. Docker containers and GitHub
# runners are never touched: preflight snapshots them, 30-verify compares.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TOWER_PKGS=(munge slurmctld slurmd slurmdbd slurm-client slurm-wlm-nvml-plugin slurm-wlm-mysql-plugin mariadb-server)

step_preflight() {
    local fail=0
    chk() { if eval "$2"; then xcl_log "ok   $1"; else xcl_log "FAIL $1"; fail=1; fi; }
    chk "hostname is tower" '[[ "$(hostname -s)" == tower ]] || [[ "$XCL_NO_SYSTEMD" == 1 ]]'
    chk "Ubuntu 26.04" 'grep -q "VERSION_ID=\"26.04\"" /etc/os-release'
    chk "tailnet address $TOWER_ADDR present" 'ip -4 -o addr | grep " $TOWER_ADDR/" >/dev/null'
    chk "cgroup v2" '[[ "$(stat -fc %T /sys/fs/cgroup)" == cgroup2fs ]]'
    chk "nvidia-smi sees a GPU" 'nvidia-smi -L | grep "^GPU 0" >/dev/null'
    chk "libnvidia-ml.so.1 present" 'ldconfig -p | grep libnvidia-ml.so.1 >/dev/null'
    chk "ports 6817-6819 free" '! ss -ltnH | awk "{print \$4}" | grep -E ":(6817|6818|6819)$" >/dev/null'
    chk "clock NTP-synchronised (munge TTL)" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || timedatectl show -p NTPSynchronized --value | grep -x yes >/dev/null'
    chk ">= 3 GB free on /" '[[ $(df --output=avail -k / | tail -1) -gt 3000000 ]]'
    chk "uid/gid $SLURM_UID free or already slurm" '! getent passwd $SLURM_UID >/dev/null || [[ "$(getent passwd $SLURM_UID | cut -d: -f1)" == slurm ]]'
    chk "uid/gid $MUNGE_UID free or already munge" '! getent passwd $MUNGE_UID >/dev/null || [[ "$(getent passwd $MUNGE_UID | cut -d: -f1)" == munge ]]'
    chk "user aaryn is uid 1000 (must match asus)" '[[ "$(id -u aaryn 2>/dev/null)" == 1000 ]]'
    xcl_init_state
    [[ -e "$XCL_STATE/untouchables.before" ]] || xcl_snapshot_untouchables "$XCL_STATE/untouchables.before"
    xcl_log "snapshot of Docker containers + runners: $XCL_STATE/untouchables.before"
    [[ $fail == 0 ]] || xcl_die "preflight failed; nothing was changed"
}

step_users() {
    xcl_init_state
    xcl_create_system_user slurm "$SLURM_UID" /var/lib/slurm
    xcl_create_system_user munge "$MUNGE_UID" /nonexistent
}

step_packages() {
    xcl_init_state
    xcl_note_leftovers
    # Present before mariadb-server's first start, so it never opens TCP.
    xcl_put_file "$CONF_DIR/mariadb-xcl-slurm.cnf" /etc/mysql/mariadb.conf.d/90-xcl-slurm.cnf 0644 root:root
    apt-get update -qq
    local had_key=0
    [[ -e /etc/munge/munge.key ]] && had_key=1
    xcl_apt_install "${TOWER_PKGS[@]}"
    # munge's postinst generates a key; it is ours to remove on rollback.
    if [[ $had_key == 0 && -e /etc/munge/munge.key ]]; then
        xcl_record file /etc/munge/munge.key
    fi
}

step_configure() {
    xcl_init_state
    xcl_put_dir /etc/slurm 0755 root:root
    xcl_put_file "$CONF_DIR/slurm.conf" /etc/slurm/slurm.conf 0644 root:root
    xcl_put_file "$CONF_DIR/gres.conf" /etc/slurm/gres.conf 0644 root:root
    xcl_put_file "$CONF_DIR/cgroup.conf" /etc/slurm/cgroup.conf 0644 root:root
    xcl_put_dir /var/spool/slurmctld 0700 slurm:slurm
    xcl_put_dir /var/spool/slurmd 0755 root:root
    xcl_put_dir /var/log/slurm 0755 slurm:slurm
    xcl_put_dir "$XCL_JOBS_DIR" 1777 root:root

    if [[ ! -s /etc/munge/munge.key ]]; then
        xcl_put_dir /etc/munge 0700 munge:munge
        mungekey --create --keyfile /etc/munge/munge.key
        xcl_record file /etc/munge/munge.key
    fi
    chown munge:munge /etc/munge/munge.key
    chmod 0400 /etc/munge/munge.key

    if [[ ! -s /etc/slurm/slurmdbd.conf ]]; then
        local pass tmp
        pass="$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')"
        tmp="$(mktemp)"
        sed "s/@STORAGE_PASS@/$pass/" "$CONF_DIR/slurmdbd.conf.template" >"$tmp"
        xcl_put_file "$tmp" /etc/slurm/slurmdbd.conf 0600 slurm:slurm
        rm -f "$tmp"
    fi

    # Forced-command gate for the adapter's restricted SSH key.
    xcl_put_file "$KIT_DIR/scripts/xcl-slurm-gate" /usr/local/libexec/xcl-slurm-gate 0755 root:root
}

storage_pass() { sed -n 's/^StoragePass=//p' /etc/slurm/slurmdbd.conf; }

mariadb_running() { mariadb-admin --protocol=socket ping >/dev/null 2>&1; }

step_db() {
    xcl_init_state
    if xcl_have_systemd; then
        xcl_service_start mariadb
    elif ! mariadb_running; then
        install -d -o mysql -g mysql /run/mysqld
        (mariadbd-safe >/dev/null 2>&1 &)
    fi
    for _ in $(seq 30); do mariadb_running && break; sleep 1; done
    mariadb_running || xcl_die "mariadb did not come up"
    local pass
    pass="$(storage_pass)"
    mariadb --protocol=socket <<SQL
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY '$pass';
ALTER USER 'slurm'@'localhost' IDENTIFIED BY '$pass';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
SQL
    xcl_log "accounting database ready (socket only)"
}

# The cluster must exist in slurmdbd before slurmctld first registers.
register_cluster() {
    sacctmgr -n show cluster format=cluster | grep -w xcl-local >/dev/null || sacctmgr -i add cluster xcl-local
}

step_services() {
    xcl_have_systemd || { xcl_log "no systemd: daemons are started by the caller"; return 0; }
    xcl_service_start munge
    xcl_service_start mariadb
    xcl_service_start slurmdbd
    xcl_wait_tcp 127.0.0.1 6819 30 || xcl_die "slurmdbd not listening on 6819 (journalctl -u slurmdbd)"
    register_cluster
    xcl_service_start slurmctld
    xcl_wait_tcp 127.0.0.1 6817 30 || xcl_die "slurmctld not listening on 6817 (see /var/log/slurm/slurmctld.log)"
    xcl_service_start slurmd
    xcl_log "tower services up; boot autostart: $([[ $XCL_ENABLE_AT_BOOT == 1 ]] && echo on || echo off)"
}

# Append the adapter's public key to aaryn's authorized_keys, restricted to
# the gate and to the asus tailnet address. Same shape as an Alliance
# robot-node key: restrict,from=...,command=...
step_authorize_adapter_key() {
    local pub="${1:?usage: authorize-adapter-key KEY.pub [FROM]}" from="${2:-$ASUS_ADDR}"
    xcl_init_state
    local ak=/home/aaryn/.ssh/authorized_keys
    install -d -m 0700 -o aaryn -g aaryn /home/aaryn/.ssh
    touch "$ak"; chown aaryn:aaryn "$ak"; chmod 0600 "$ak"
    local key
    key="$(awk '{print $1, $2}' "$pub")"
    if ! grep -qF "$key" "$ak"; then
        printf 'restrict,from="%s",command="/usr/local/libexec/xcl-slurm-gate" %s xcl-slurm-adapter\n' \
            "$from" "$key" >>"$ak"
    fi
    xcl_record authkey "$ak"
    xcl_log "adapter key authorised (forced command: xcl-slurm-gate, from $from)"
}

main() {
    xcl_require_root
    [[ $# -gt 0 ]] || set -- all
    while [[ $# -gt 0 ]]; do
        case "$1" in
            all) step_preflight; step_users; step_packages; step_configure; step_db; step_services ;;
            preflight | users | packages | configure | db | services) "step_$1" ;;
            authorize-adapter-key) shift; step_authorize_adapter_key "$@"; return ;;
            *) xcl_die "unknown step: $1" ;;
        esac
        shift
    done
}
main "$@"
