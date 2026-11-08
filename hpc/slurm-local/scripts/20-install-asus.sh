#!/usr/bin/env bash
# Install the CPU compute node of the xcl-local test cluster on the ASUS.
#
#   sudo ./20-install-asus.sh all     preflight users packages configure services
#   sudo ./20-install-asus.sh <step>...
#   sudo ./20-install-asus.sh adapter-client   adapter key + SSH alias + tower authorisation
#
# Run after the tower is up. Config-less: slurmd fetches slurm.conf,
# gres.conf and cgroup.conf from slurmctld, so no Slurm config files are
# written here. The munge key is copied from the tower over SSH (or from
# XCL_MUNGE_KEY_SRC in the rehearsal). Site values come from ../site.env.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ASUS_PKGS=(munge slurmd slurm-client)
# Your own (unrestricted) login on the tower: munge key, adapter key setup.
TOWER_SSH="${TOWER_SSH:-${OPERATOR_USER:-}@${TOWER_ADDR:-}}"

step_preflight() {
    local fail=0
    chk() { if eval "$2"; then xcl_log "ok   $1"; else xcl_log "FAIL $1"; fail=1; fi; }
    chk "hostname is ASUS_HOSTNAME from site.env" '[[ "$(hostname -s)" == "$ASUS_HOSTNAME" ]]'
    chk "ASUS_MEM_MB ($ASUS_MEM_MB) fits the installed RAM" 'xcl_mem_fits "$ASUS_MEM_MB"'
    chk "Ubuntu 26.04" 'grep -q "VERSION_ID=\"26.04\"" /etc/os-release'
    chk "tailnet address $ASUS_ADDR present" 'ip -4 -o addr | grep " $ASUS_ADDR/" >/dev/null'
    chk "cgroup v2" '[[ "$(stat -fc %T /sys/fs/cgroup)" == cgroup2fs ]]'
    chk "port 6818 free" '! ss -ltnH | awk "{print \$4}" | grep -E ":6818$" >/dev/null'
    chk "clock NTP-synchronised" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || timedatectl show -p NTPSynchronized --value | grep -x yes >/dev/null'
    chk "uid $SLURM_UID free or already slurm" '! getent passwd $SLURM_UID >/dev/null || [[ "$(getent passwd $SLURM_UID | cut -d: -f1)" == slurm ]]'
    chk "uid $MUNGE_UID free or already munge" '! getent passwd $MUNGE_UID >/dev/null || [[ "$(getent passwd $MUNGE_UID | cut -d: -f1)" == munge ]]'
    chk "uid/gid $SVC_UID free or already $SVC_USER (must match tower)" '! getent passwd $SVC_UID >/dev/null || [[ "$(getent passwd $SVC_UID | cut -d: -f1)" == "$SVC_USER" ]]'
    # Jobs land here as $SVC_USER; your home holds the adapter key and your
    # unrestricted keys.
    chk "home of $OPERATOR_USER closed to other users" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || (( 8#$(stat -c %a "$(getent passwd "$OPERATOR_USER" | cut -d: -f6)") % 8 == 0 ))'
    # A laptop that sleeps takes its jobs with it.
    chk "sleep/suspend/hibernate masked" '[[ "$XCL_NO_SYSTEMD" == 1 ]] || ! systemctl show -p LoadState sleep.target suspend.target hibernate.target hybrid-sleep.target | grep -qv "=masked"'
    chk "slurmctld on $TOWER_ADDR:6817 reachable" 'xcl_wait_tcp "$TOWER_ADDR" 6817 3'
    xcl_init_state
    [[ -e "$XCL_STATE/untouchables.before" ]] || xcl_snapshot_untouchables "$XCL_STATE/untouchables.before"
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
        # Your own (unrestricted) key on the tower, not the adapter key.
        sudo -u "$OPERATOR_USER" ssh -o BatchMode=yes "$TOWER_SSH" 'sudo -n cat /etc/munge/munge.key' >"$tmp"
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
    xcl_service_dirs
    xcl_put_file "$KIT_DIR/scripts/xcl-node-health" "$XCL_HEALTH" 0755 root:root

    # Firewall: default-deny inbound here. slurmctld on the tower must reach
    # slurmd on 6818, and a step's slurmstepd must reach an srun started
    # here (SrunPortRange) — allow exactly those, from the tower, on the
    # tailnet interface only.
    if command -v ufw >/dev/null && ufw status | grep "Status: active" >/dev/null; then
        local rule
        for rule in "allow in on tailscale0 from $TOWER_ADDR to any port 6818 proto tcp" \
            "allow in on tailscale0 from $TOWER_ADDR to any port $XCL_SRUN_PORTS proto tcp"; do
            # shellcheck disable=SC2086
            ufw $rule comment 'xcl-slurm-test' >/dev/null
            xcl_record ufw "$rule"
            xcl_log "ufw: $rule"
        done
    fi
}

step_services() {
    xcl_have_systemd || { xcl_log "no systemd: daemons are started by the caller"; return 0; }
    xcl_service_start munge
    xcl_service_start slurmd
    xcl_log "asus services up; boot autostart: $([[ $XCL_ENABLE_AT_BOOT == 1 ]] && echo on || echo off)"
}

# The adapter's client side, on this machine: its own key (never reused for
# interactive login, as Alliance requires for automation keys) and its own
# ssh config file, passed with -F. Never an alias in ~/.ssh/config: there,
# IdentityFile entries accumulate across Host blocks, so a "Host *" key is
# offered first and the tower accepts it unrestricted (found on the real
# run). It logs in as $SVC_USER, never as you. Both files are recorded, so
# rollback removes them. Your home must stay closed to other users: jobs
# run on this machine too, as $SVC_USER.
OPERATOR_HOME="$(getent passwd "${OPERATOR_USER:-}" | cut -d: -f6 || true)"
ADAPTER_KEY="$OPERATOR_HOME/.ssh/xcl_slurm_adapter"
ADAPTER_SSH_CONFIG="$OPERATOR_HOME/.ssh/xcl_slurm_adapter.conf"

step_adapter_client() {
    xcl_init_state
    [[ -n "$OPERATOR_HOME" ]] || xcl_die "no home directory for $OPERATOR_USER"
    install -d -m 0700 -o "$OPERATOR_USER" -g "$OPERATOR_USER" "$OPERATOR_HOME/.ssh"
    if [[ ! -e "$ADAPTER_KEY" ]]; then
        sudo -u "$OPERATOR_USER" ssh-keygen -q -t ed25519 -N '' -C xcl-slurm-adapter -f "$ADAPTER_KEY"
        xcl_record file "$ADAPTER_KEY"
        xcl_record file "$ADAPTER_KEY.pub"
    fi
    # Rewritten every run: an older kit wrote "User aaryn" here.
    [[ -e "$ADAPTER_SSH_CONFIG" ]] || xcl_record file "$ADAPTER_SSH_CONFIG"
    cat >"$ADAPTER_SSH_CONFIG" <<CFG
# xcl-slurm-test: ssh config used only by the Slurm adapter (ssh -F this-file).
Host xcl-tower
    HostName $TOWER_ADDR
    User $SVC_USER
    IdentityFile $ADAPTER_KEY
    IdentitiesOnly yes
    IdentityAgent none
    RequestTTY no
    BatchMode yes
CFG
    chown "$OPERATOR_USER:$OPERATOR_USER" "$ADAPTER_SSH_CONFIG"
    chmod 0600 "$ADAPTER_SSH_CONFIG"
    # Authorise it on the tower, restricted to the gate and this address.
    sudo -u "$OPERATOR_USER" scp -q -o BatchMode=yes "$ADAPTER_KEY.pub" "$TOWER_SSH:/tmp/xcl_slurm_adapter.pub"
    sudo -u "$OPERATOR_USER" ssh -o BatchMode=yes "$TOWER_SSH" \
        "cd ~/xcl-slurm-kit/scripts && sudo -n ./10-install-tower.sh authorize-adapter-key /tmp/xcl_slurm_adapter.pub $ASUS_ADDR && rm -f /tmp/xcl_slurm_adapter.pub"
    # Prove the restriction before declaring success.
    local ok refused
    ok="$(sudo -u "$OPERATOR_USER" ssh -F "$ADAPTER_SSH_CONFIG" xcl-tower sinfo --version 2>&1)" ||
        xcl_die "adapter key cannot run sinfo through the gate: $ok"
    refused="$(sudo -u "$OPERATOR_USER" ssh -F "$ADAPTER_SSH_CONFIG" xcl-tower id 2>&1 || true)"
    [[ "$refused" == *"xcl-slurm-gate: refused"* ]] ||
        xcl_die "adapter key is NOT restricted (got: $refused)"
    xcl_log "adapter key ready and restricted: '$ok'; 'id' refused by the gate"
}

main() {
    xcl_require_root
    xcl_require_site
    [[ $# -gt 0 ]] || set -- all
    while [[ $# -gt 0 ]]; do
        case "$1" in
            all) step_preflight; step_users; step_packages; step_configure; step_services ;;
            preflight | users | packages | configure | services) "step_$1" ;;
            adapter-client) step_adapter_client ;;
            *) xcl_die "unknown step: $1" ;;
        esac
        shift
    done
}
main "$@"
