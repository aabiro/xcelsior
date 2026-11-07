#!/usr/bin/env bash
# Install the CPU compute node of the xcl-local test cluster on the ASUS.
#
#   sudo ./20-install-asus.sh all     preflight users packages configure services
#   sudo ./20-install-asus.sh <step>...
#
# Run after the tower is up. Config-less: slurmd fetches slurm.conf,
# gres.conf and cgroup.conf from slurmctld, so no Slurm config files are
# written here. The munge key is copied from the tower over SSH (or from
# XCL_MUNGE_KEY_SRC in the rehearsal).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ASUS_PKGS=(munge slurmd slurm-client)
TOWER_SSH="${TOWER_SSH:-aaryn@$TOWER_ADDR}"

step_preflight() {
    local fail=0
    chk() { if eval "$2"; then xcl_log "ok   $1"; else xcl_log "FAIL $1"; fail=1; fi; }
    chk "hostname matches NodeHostname in slurm.conf" 'grep -q "NodeHostname=$(hostname -s) " "$CONF_DIR/slurm.conf"'
    chk "Ubuntu 26.04" 'grep -q "VERSION_ID=\"26.04\"" /etc/os-release'
    chk "tailnet address $ASUS_ADDR present" 'ip -4 -o addr | grep -q " $ASUS_ADDR/"'
    chk "cgroup v2" '[[ "$(stat -fc %T /sys/fs/cgroup)" == cgroup2fs ]]'
    chk "port 6818 free" '! ss -ltnH | awk "{print \$4}" | grep -Eq ":6818$"'
    chk "clock NTP-synchronised" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || timedatectl show -p NTPSynchronized --value | grep -qx yes'
    chk "uid $SLURM_UID free or already slurm" '! getent passwd $SLURM_UID >/dev/null || [[ "$(getent passwd $SLURM_UID | cut -d: -f1)" == slurm ]]'
    chk "uid $MUNGE_UID free or already munge" '! getent passwd $MUNGE_UID >/dev/null || [[ "$(getent passwd $MUNGE_UID | cut -d: -f1)" == munge ]]'
    chk "user aaryn is uid 1000 (must match tower)" '[[ "$(id -u aaryn 2>/dev/null)" == 1000 ]]'
    chk "slurmctld on $TOWER_ADDR:6817 reachable" 'xcl_wait_tcp "$TOWER_ADDR" 6817 3'
    xcl_init_state
    [[ -e "$XCL_STATE/untouchables.before" ]] || xcl_snapshot_untouchables "$XCL_STATE/untouchables.before"
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
    apt-get update -qq
    local had_key=0
    [[ -e /etc/munge/munge.key ]] && had_key=1
    xcl_apt_install "${ASUS_PKGS[@]}"
    # munge's postinst generates a key; it is ours to remove on rollback.
    if [[ $had_key == 0 && -e /etc/munge/munge.key ]]; then
        xcl_record file /etc/munge/munge.key
    fi
}

step_configure() {
    xcl_init_state
    local tmp
    tmp="$(mktemp)"
    if [[ -n "${XCL_MUNGE_KEY_SRC:-}" ]]; then
        cp "$XCL_MUNGE_KEY_SRC" "$tmp"
    else
        # aaryn's own (unrestricted) key on the tower, not the adapter key.
        sudo -u aaryn ssh -o BatchMode=yes "$TOWER_SSH" 'sudo -n cat /etc/munge/munge.key' >"$tmp"
    fi
    [[ -s "$tmp" ]] || xcl_die "could not fetch the munge key from the tower"
    xcl_put_dir /etc/munge 0700 munge:munge
    if [[ -e /etc/munge/munge.key ]] && ! grep -qxF "file /etc/munge/munge.key" "$XCL_MANIFEST"; then
        install -d "$XCL_BACKUP/etc/munge"; cp -a /etc/munge/munge.key "$XCL_BACKUP/etc/munge/munge.key"
        xcl_record backup /etc/munge/munge.key
    fi
    install -m 0400 -o munge -g munge "$tmp" /etc/munge/munge.key
    rm -f "$tmp"
    xcl_record file /etc/munge/munge.key

    printf 'SLURMD_OPTIONS="--conf-server %s:6817"\n' "$TOWER_ADDR" >"${tmp}.d"
    xcl_put_file "${tmp}.d" /etc/default/slurmd 0644 root:root
    rm -f "${tmp}.d"
    xcl_put_dir /var/spool/slurmd 0755 root:root
    xcl_put_dir /var/log/slurm 0755 slurm:slurm
    xcl_put_dir "$XCL_JOBS_DIR" 1777 root:root

    # Firewall: default-deny inbound here. slurmctld on the tower must reach
    # slurmd on 6818 — allow exactly that, on the tailnet interface only.
    if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
        local rule="allow in on tailscale0 from $TOWER_ADDR to any port 6818 proto tcp"
        # shellcheck disable=SC2086
        ufw $rule comment 'xcl-slurm-test' >/dev/null
        xcl_record ufw "$rule"
        xcl_log "ufw: $rule"
    fi
}

step_services() {
    xcl_have_systemd || { xcl_log "no systemd: daemons are started by the caller"; return 0; }
    xcl_service_start munge
    xcl_service_start slurmd
    xcl_log "asus services up; boot autostart: $([[ $XCL_ENABLE_AT_BOOT == 1 ]] && echo on || echo off)"
}

main() {
    xcl_require_root
    [[ $# -gt 0 ]] || set -- all
    while [[ $# -gt 0 ]]; do
        case "$1" in
            all) step_preflight; step_users; step_packages; step_configure; step_services ;;
            preflight | users | packages | configure | services) "step_$1" ;;
            *) xcl_die "unknown step: $1" ;;
        esac
        shift
    done
}
main "$@"
