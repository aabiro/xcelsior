#!/usr/bin/env bash
# Install the controller + GPU node of the xcl-local test cluster on the tower.
#
#   sudo ./10-install-tower.sh all            preflight users packages configure db services
#   sudo ./10-install-tower.sh <step>...      run named steps only (qos: the spot QoS)
#   sudo ./10-install-tower.sh authorize-adapter-key /path/to/key.pub [FROM]
#
# Test/research only. Every change is recorded in /var/lib/xcl-slurm-test/
# manifest; sudo ./90-rollback.sh undoes it. Docker containers and GitHub
# runners are never touched: preflight snapshots them, 30-verify compares.
# Site values (addresses, your account) come from ../site.env.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

TOWER_PKGS=(munge slurmctld slurmd slurmdbd slurm-client slurm-wlm-nvml-plugin slurm-wlm-mysql-plugin mariadb-server)

step_preflight() {
    local fail=0
    chk() { if eval "$2"; then xcl_log "ok   $1"; else xcl_log "FAIL $1"; fail=1; fi; }
    chk "hostname is tower" '[[ "$(hostname -s)" == tower ]] || [[ "$XCL_NO_SYSTEMD" == 1 ]]'
    # Slurm daemons bind every interface; the firewall is what keeps
    # 6817-6819 to the tailnet. Refuse to install where it isn't up.
    chk "ufw active, default deny incoming" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || ufw status verbose | grep -q "Default: deny (incoming)"'
    chk "sshd present (adapter login)" '[[ -x /usr/sbin/sshd ]]'
    chk "TOWER_MEM_MB ($TOWER_MEM_MB) fits the installed RAM" 'xcl_mem_fits "$TOWER_MEM_MB"'
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
    chk "uid/gid $SVC_UID free or already $SVC_USER" '! getent passwd $SVC_UID >/dev/null || [[ "$(getent passwd $SVC_UID | cut -d: -f1)" == "$SVC_USER" ]]'
    # Jobs run as $SVC_USER; your home holds unrestricted keys.
    chk "home of $OPERATOR_USER closed to other users" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || (( 8#$(stat -c %a "$(getent passwd "$OPERATOR_USER" | cut -d: -f6)") % 8 == 0 ))'
    xcl_init_state
    [[ -e "$XCL_STATE/untouchables.before" ]] || xcl_snapshot_untouchables "$XCL_STATE/untouchables.before"
    xcl_log "snapshot of Docker containers + runners: $XCL_STATE/untouchables.before"
    [[ $fail == 0 ]] || xcl_die "preflight failed; nothing was changed"
}

step_users() {
    xcl_init_state
    xcl_create_system_user slurm "$SLURM_UID" /var/lib/slurm
    xcl_create_system_user munge "$MUNGE_UID" /nonexistent
    xcl_create_service_user
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
    xcl_service_dirs
    xcl_put_file "$KIT_DIR/scripts/xcl-node-health" "$XCL_HEALTH" 0755 root:root

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

    # Forced-command gate for the adapter's restricted SSH key, and the sshd
    # Match block that makes it the only thing $SVC_USER can run.
    xcl_put_file "$KIT_DIR/scripts/xcl-slurm-gate" "$XCL_GATE" 0755 root:root
    xcl_put_file "$CONF_DIR/sshd-xcl-slurm-test.conf" "$XCL_SSHD_DROPIN" 0644 root:root
    xcl_record sshd reload
    if ! /usr/sbin/sshd -t; then
        rm -f "$XCL_SSHD_DROPIN"
        xcl_die "sshd rejects $XCL_SSHD_DROPIN; removed it, sshd untouched"
    fi
    if xcl_have_systemd && systemctl is-active --quiet ssh; then
        systemctl reload ssh
    fi
}

storage_pass() { sed -n 's/^StoragePass=//p' /etc/slurm/slurmdbd.conf; }

mariadb_running() { mariadb-admin --protocol=socket ping >/dev/null 2>&1; }

# The package's postinst creates the system tables with mariadb-install-db,
# run as root and dropping to mysql. MariaDB's own AppArmor profile denies
# that uid change once the profile is loaded, which it still is after a
# purge, so a reinstall left the data dir without them (2026-11-08). Run
# as mysql there is no uid change to deny.
mariadb_system_tables() {
    local datadir
    datadir="$(my_print_defaults --mysqld 2>/dev/null | sed -n 's/^--datadir=//p' | tail -1)"
    datadir="${datadir:-/var/lib/mariadb}"
    [[ -d "$datadir/mysql" ]] && return 0
    xcl_log "MariaDB system tables missing in $datadir; creating them as mysql"
    runuser -u mysql -- mariadb-install-db --datadir="$datadir" --skip-test-db \
        --auth-root-authentication-method=socket >"$XCL_STATE/mariadb-install-db.log" 2>&1 ||
        xcl_die "mariadb-install-db failed: see $XCL_STATE/mariadb-install-db.log"
}

step_db() {
    xcl_init_state
    mariadb_system_tables
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

# The spot QoS slurm.conf's preempt/qos acts on: normal jobs requeue spot
# ones. Lives in the accounting DB, which rollback removes with MariaDB.
step_qos() {
    sacctmgr -n -P show qos format=name | grep -qx spot || sacctmgr -i add qos spot
    sacctmgr -i modify qos normal set preempt=spot >/dev/null
    xcl_log "QoS spot ready (preempted by normal, requeued)"
}

step_services() {
    xcl_have_systemd || { xcl_log "no systemd: daemons are started by the caller"; return 0; }
    xcl_service_start munge
    xcl_service_start mariadb
    xcl_service_start slurmdbd
    xcl_wait_tcp 127.0.0.1 6819 30 || xcl_die "slurmdbd not listening on 6819 (journalctl -u slurmdbd)"
    register_cluster
    step_qos
    xcl_service_start slurmctld
    xcl_wait_tcp 127.0.0.1 6817 30 || xcl_die "slurmctld not listening on 6817 (see /var/log/slurm/slurmctld.log)"
    xcl_service_start slurmd
    xcl_log "tower services up; boot autostart: $([[ $XCL_ENABLE_AT_BOOT == 1 ]] && echo on || echo off)"
}

# Authorise the adapter's public key for $SVC_USER, restricted to the gate
# and the asus tailnet address, expiring after XCL_KEY_DAYS (default 90).
# Same shape as an Alliance robot-node key (restrict,from=,command=). The
# file is root-owned and outside any home, so no job can add a key to it;
# writing it replaces any earlier adapter key.
step_authorize_adapter_key() {
    local pub="${1:?usage: authorize-adapter-key KEY.pub [FROM]}" from="${2:-$ASUS_ADDR}"
    xcl_init_state
    local key expiry tmp
    key="$(awk '{print $1, $2}' "$pub")"
    [[ "$key" =~ ^ssh-ed25519\ [A-Za-z0-9+/=]+$ ]] || xcl_die "not an ed25519 public key: $pub"
    [[ "$from" =~ ^[0-9.,/]+$ ]] || xcl_die "FROM must be IPv4 addresses or CIDRs: $from"
    expiry="$(date -u -d "+${XCL_KEY_DAYS:-90} days" +%Y%m%d)"
    xcl_put_dir "$XCL_AUTHKEYS_DIR" 0755 root:root
    tmp="$(mktemp)"
    printf 'restrict,from="%s",expiry-time="%s",command="%s" %s xcl-slurm-adapter\n' \
        "$from" "$expiry" "$XCL_GATE" "$key" >"$tmp"
    install -m 0644 -o root -g root "$tmp" "$XCL_AUTHKEYS"
    rm -f "$tmp"
    xcl_record file "$XCL_AUTHKEYS"
    xcl_log "adapter key authorised for $SVC_USER (gate only, from $from, expires $expiry)"
}

main() {
    xcl_require_root
    xcl_require_site
    [[ $# -gt 0 ]] || set -- all
    while [[ $# -gt 0 ]]; do
        case "$1" in
            all) step_preflight; step_users; step_packages; step_configure; step_db; step_services ;;
            preflight | users | packages | configure | db | services | qos) "step_$1" ;;
            authorize-adapter-key) shift; step_authorize_adapter_key "$@"; return ;;
            *) xcl_die "unknown step: $1" ;;
        esac
        shift
    done
}
main "$@"
