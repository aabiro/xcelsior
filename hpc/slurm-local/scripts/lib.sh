#!/usr/bin/env bash
# Shared helpers for the xcl-local Slurm validation cluster scripts.
#
# Every change to the machine is appended to $XCL_MANIFEST as one line,
# "<kind> <value>", so 90-rollback.sh can undo exactly what was done and
# nothing else. Kinds: user, pkg, file, dir, backup, ufw, service, leftover
# (package-created paths removed after the purge), sshd (reload sshd after
# removing files), and authkey / sshconfig (written by older kit versions).

set -euo pipefail
# NB: never pipe a long output into "grep -q" here: grep exits at the first
# match, the writer gets SIGPIPE, and pipefail turns a match into a failure
# (found on the real tower: "ldconfig -p | grep -q" failed with the lib present).

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF_DIR="$KIT_DIR/conf"
XCL_STATE="${XCL_STATE:-/var/lib/xcl-slurm-test}"
XCL_MANIFEST="$XCL_STATE/manifest"
XCL_BACKUP="$XCL_STATE/backup"
XCL_JOBS_DIR="$XCL_STATE/jobs"

# Site-specific values (tailnet addresses, the asus hostname, your own
# account) live in site.env beside this kit, which git ignores: the
# repository is public. See site.env.example. Values already in the
# environment win, so the rehearsal can set its own.
XCL_SITE_ENV="${XCL_SITE_ENV:-$KIT_DIR/site.env}"
XCL_SITE_KEYS=(TOWER_ADDR ASUS_ADDR ASUS_HOSTNAME OPERATOR_USER TOWER_MEM_MB ASUS_MEM_MB)
if [[ -r "$XCL_SITE_ENV" ]]; then
    while IFS='=' read -r _k _v; do
        [[ " ${XCL_SITE_KEYS[*]} " == *" $_k "* ]] || continue
        _v="${_v%%#*}"
        _v="${_v//[[:space:]]/}"
        [[ -n "${!_k:-}" ]] || printf -v "$_k" '%s' "$_v"
    done <"$XCL_SITE_ENV"
    unset _k _v
fi
# Memory each node offers Slurm (slurm.conf RealMemory, MB): deliberately far
# below the hardware, so jobs can't crowd out what already runs there. Never
# above the installed RAM: slurmd would register short and the node drains.
TOWER_MEM_MB="${TOWER_MEM_MB:-8192}"
ASUS_MEM_MB="${ASUS_MEM_MB:-4096}"

# xcl_require_site — die unless every site value is present and well formed.
xcl_require_site() {
    local ipv4='^([0-9]{1,3}\.){3}[0-9]{1,3}$'
    [[ "${TOWER_ADDR:-}" =~ $ipv4 ]] || xcl_die "TOWER_ADDR missing or not an IPv4 address (set it in $XCL_SITE_ENV)"
    [[ "${ASUS_ADDR:-}" =~ $ipv4 ]] || xcl_die "ASUS_ADDR missing or not an IPv4 address (set it in $XCL_SITE_ENV)"
    [[ "${ASUS_HOSTNAME:-}" =~ ^[A-Za-z0-9][A-Za-z0-9-]{0,62}$ ]] || xcl_die "ASUS_HOSTNAME missing or invalid (set it in $XCL_SITE_ENV)"
    [[ "${OPERATOR_USER:-}" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] || xcl_die "OPERATOR_USER missing or invalid (set it in $XCL_SITE_ENV)"
    [[ "$TOWER_MEM_MB" =~ ^[1-9][0-9]{2,6}$ ]] || xcl_die "TOWER_MEM_MB must be a whole number of MB"
    [[ "$ASUS_MEM_MB" =~ ^[1-9][0-9]{2,6}$ ]] || xcl_die "ASUS_MEM_MB must be a whole number of MB"
}

# xcl_mem_fits MB — true when this machine has at least MB of RAM.
xcl_mem_fits() {
    (($1 <= $(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)))
}

SLURM_UID="${SLURM_UID:-970}"
MUNGE_UID="${MUNGE_UID:-971}"
# The adapter's service account: jobs submitted through the gate run as it,
# on both nodes. Never the operator's own account, which has sudo.
SVC_USER="${SVC_USER:-xclslurm}"
SVC_UID="${SVC_UID:-972}"
# Root-owned and not writable by the service account, so a job can't plant
# ~/.bashrc, ~/.ssh or ~/.ssh/rc there.
SVC_HOME="$XCL_STATE/svc-home"
XCL_GATE=/usr/local/libexec/xcl-slurm-gate
XCL_HEALTH=/usr/local/libexec/xcl-node-health
# The adapter key's authorized_keys file: root-owned, outside any home.
XCL_AUTHKEYS_DIR=/etc/ssh/xcl-slurm-test
XCL_AUTHKEYS="$XCL_AUTHKEYS_DIR/authorized_keys"
XCL_SSHD_DROPIN=/etc/ssh/sshd_config.d/50-xcl-slurm-test.conf
# What xcl-node-health does about GPU processes Slurm can't see: "log"
# (default; the tower's GPU is shared by design) or "drain".
XCL_GPU_CONTENTION="${XCL_GPU_CONTENTION:-log}"
[[ "$XCL_GPU_CONTENTION" =~ ^(log|drain)$ ]] || { echo "XCL_GPU_CONTENTION must be log or drain" >&2; exit 1; }
# srun listens on these; the asus firewall admits them from the tower.
XCL_SRUN_PORTS=60001:60100
# SchedMD's Slurm packages, built by 05-build-slurm-debs.sh. The installers
# take Slurm from here only, never from Ubuntu's archive (25.11.2 there).
XCL_DEBS_DIR="${XCL_DEBS_DIR:-$KIT_DIR/debs}"
XCL_MIN_SLURM=25.11.8

# xcl_slurm_debs NAME... — the .deb file of each named slurm-smd package.
xcl_slurm_debs() {
    local name f
    for name in "$@"; do
        f="$(compgen -G "$XCL_DEBS_DIR/${name}_*_amd64.deb" | head -1)"
        [[ -n "$f" ]] || xcl_die "no ${name} package in $XCL_DEBS_DIR: run scripts/05-build-slurm-debs.sh"
        printf '%s\n' "$f"
    done
}
# Set by the container rehearsal only: no systemd, daemons started by hand.
XCL_NO_SYSTEMD="${XCL_NO_SYSTEMD:-0}"
# Default: test services do NOT start at boot. A reboot leaves the box as
# it was before the cluster existed; 40-start.sh brings it back.
XCL_ENABLE_AT_BOOT="${XCL_ENABLE_AT_BOOT:-0}"

xcl_log() { printf '[xcl-slurm] %s\n' "$*" >&2; }
xcl_die() { printf '[xcl-slurm] ERROR: %s\n' "$*" >&2; exit 1; }

xcl_require_root() {
    [[ $EUID -eq 0 ]] || xcl_die "run with sudo"
}

xcl_init_state() {
    install -d -m 0755 "$XCL_STATE" "$XCL_BACKUP"
    touch "$XCL_MANIFEST"
}

# xcl_record KIND VALUE — append once.
xcl_record() {
    local line="$1 $2"
    grep -qxF -- "$line" "$XCL_MANIFEST" 2>/dev/null || printf '%s\n' "$line" >>"$XCL_MANIFEST"
}

# Fill the @PLACEHOLDERS@ in a committed config from site.env. Refuses to
# leave any behind: a literal "@TOWER_ADDR@" in slurm.conf fails late and
# confusingly.
xcl_render() {
    xcl_require_site
    local out
    out="$(sed -e "s/@TOWER_ADDR@/${TOWER_ADDR}/g" -e "s/@ASUS_ADDR@/${ASUS_ADDR}/g" \
        -e "s/@ASUS_HOSTNAME@/${ASUS_HOSTNAME}/g" -e "s/@SVC_USER@/${SVC_USER}/g" \
        -e "s/@GPU_CONTENTION@/${XCL_GPU_CONTENTION}/g" \
        -e "s/@TOWER_MEM_MB@/${TOWER_MEM_MB}/g" -e "s/@ASUS_MEM_MB@/${ASUS_MEM_MB}/g" \
        -e "s/@GATE@/${XCL_GATE//\//\\/}/g" -e "s/@AUTHKEYS@/${XCL_AUTHKEYS//\//\\/}/g" "$1")"
    if grep -oE '@[A-Z_]+@' <<<"$out" >/dev/null; then
        xcl_die "unfilled placeholder in $1: $(grep -oE '@[A-Z_]+@' <<<"$out" | sort -u | tr '\n' ' ')"
    fi
    printf '%s\n' "$out"
}

# xcl_kit_made PATH — true when PATH, or a directory above it, didn't exist
# before the kit installed its packages (recorded as a leftover).
xcl_kit_made() {
    local p="$1"
    while [[ "$p" != / && -n "$p" ]]; do
        grep -qxF -- "leftover $p" "$XCL_MANIFEST" 2>/dev/null && return 0
        p="$(dirname "$p")"
    done
    return 1
}

# xcl_put_file SRC DEST MODE OWNER[:GROUP] — install a rendered copy.
# An existing DEST is backed up first and restored by rollback, unless the
# kit's own packages created it: restoring that after the purge would bring
# it back (the asus's /etc/default/slurmd, found by the 2026-11-08 rollback).
xcl_put_file() {
    local src="$1" dest="$2" mode="$3" owner="$4"
    if [[ -e "$dest" ]] && ! grep -qxF -- "file $dest" "$XCL_MANIFEST" && ! xcl_kit_made "$dest"; then
        local bk="$XCL_BACKUP$dest"
        install -d "$(dirname "$bk")"
        cp -a "$dest" "$bk"
        xcl_record backup "$dest"
    fi
    install -d "$(dirname "$dest")"
    local tmp
    tmp="$(mktemp)"
    xcl_render "$src" >"$tmp"
    install -m "$mode" -o "${owner%%:*}" -g "${owner##*:}" "$tmp" "$dest"
    rm -f "$tmp"
    xcl_record file "$dest"
}

# xcl_put_dir PATH MODE OWNER[:GROUP] — create if missing; record if we did.
xcl_put_dir() {
    local path="$1" mode="$2" owner="$3"
    if [[ ! -d "$path" ]]; then
        install -d -m "$mode" -o "${owner%%:*}" -g "${owner##*:}" "$path"
        xcl_record dir "$path"
    else
        chown "${owner%%:*}:${owner##*:}" "$path"
        chmod "$mode" "$path"
    fi
}

# Fixed UIDs/GIDs: slurmd rejects RPCs whose SlurmUser UID differs from its
# own, so "slurm" must have the same UID on both nodes. The packages would
# otherwise pick whatever system UID is free on each machine.
xcl_create_system_user() {
    local name="$1" uid="$2" home="$3"
    if getent passwd "$name" >/dev/null; then
        local have
        have="$(id -u "$name")"
        [[ "$have" == "$uid" ]] || xcl_die "user $name exists with uid $have, expected $uid (other node must match)"
        return 0
    fi
    getent passwd "$uid" >/dev/null && xcl_die "uid $uid already taken by $(getent passwd "$uid" | cut -d: -f1)"
    getent group "$uid" >/dev/null && xcl_die "gid $uid already taken by $(getent group "$uid" | cut -d: -f1)"
    groupadd --system --gid "$uid" "$name"
    useradd --system --uid "$uid" --gid "$uid" --home-dir "$home" --no-create-home \
        --shell /usr/sbin/nologin --comment "xcl-slurm-test $name" "$name"
    xcl_record user "$name"
    xcl_log "created system user $name ($uid)"
}

# The service account. /bin/sh (dash) reads no startup files when sshd runs
# the forced command; bash would read ~/.bashrc first. Password "*" means no
# password, not "locked" ("!"), which sshd refuses for key logins too.
xcl_create_service_user() {
    if getent passwd "$SVC_USER" >/dev/null; then
        [[ "$(id -u "$SVC_USER")" == "$SVC_UID" ]] || xcl_die "user $SVC_USER exists with uid $(id -u "$SVC_USER"), expected $SVC_UID"
        return 0
    fi
    getent passwd "$SVC_UID" >/dev/null && xcl_die "uid $SVC_UID already taken by $(getent passwd "$SVC_UID" | cut -d: -f1)"
    getent group "$SVC_UID" >/dev/null && xcl_die "gid $SVC_UID already taken by $(getent group "$SVC_UID" | cut -d: -f1)"
    groupadd --system --gid "$SVC_UID" "$SVC_USER"
    useradd --system --uid "$SVC_UID" --gid "$SVC_UID" --home-dir "$SVC_HOME" --no-create-home \
        --shell /bin/sh --password '*' --comment "xcl-slurm-test adapter service account" "$SVC_USER"
    xcl_record user "$SVC_USER"
    xcl_log "created service account $SVC_USER ($SVC_UID)"
}

# Directories every node needs for jobs run as the service account. The
# jobs dir belongs to it, readable by your group (setgid, so job output
# inherits the group) and by no one else: the validation suite reads the
# asus's job output directly.
xcl_service_dirs() {
    xcl_put_dir "$SVC_HOME" 0755 root:root
    xcl_put_dir "$XCL_JOBS_DIR" 2750 "$SVC_USER:$(id -gn "$OPERATOR_USER")"
}

xcl_installed_packages() {
    dpkg-query -W -f='${db:Status-Status} ${Package}\n' 2>/dev/null | awk '$1=="installed"{print $2}' | sort -u
}

# Directories the packages may create and leave behind after a purge
# (MariaDB 11.8 on Ubuntu 26.04 keeps data in /var/lib/mariadb, not mysql).
XCL_PKG_LEFTOVERS=(/etc/mysql /var/lib/mariadb /var/lib/mysql /var/log/mysql /run/mysqld
    /etc/munge /var/lib/munge /var/log/munge /run/munge /var/lib/slurm /var/log/slurm /etc/slurm
    /etc/default/slurmd)

# Record which of XCL_PKG_LEFTOVERS do not exist yet. Call before writing
# anything into them (the tower writes /etc/mysql/... before apt runs).
xcl_note_leftovers() {
    local path
    for path in "${XCL_PKG_LEFTOVERS[@]}"; do
        [[ -e "$path" ]] || xcl_record leftover "$path"
    done
}

# xcl_apt_install PKG... — install with service auto-start suppressed, and
# record exactly the packages that were not installed before (dependencies
# included). Rollback purges that list and nothing else: no autoremove.
# System users the packages create (mysql) and package directories that did
# not exist before are recorded too, for removal after the purge.
xcl_apt_install() {
    local pre post users_pre path
    pre="$(mktemp)"; post="$(mktemp)"; users_pre="$(mktemp)"
    xcl_installed_packages >"$pre"
    getent passwd | cut -d: -f1 | sort >"$users_pre"
    xcl_note_leftovers
    local policy=/usr/sbin/policy-rc.d created_policy=0
    if [[ ! -e $policy ]]; then
        printf '#!/bin/sh\nexit 101\n' >"$policy"
        chmod 0755 "$policy"
        created_policy=1
    fi
    # shellcheck disable=SC2064
    trap "[[ $created_policy == 1 ]] && rm -f $policy" EXIT
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "$@"
    [[ $created_policy == 1 ]] && rm -f "$policy"
    trap - EXIT
    xcl_installed_packages >"$post"
    local p
    while read -r p; do
        [[ -n "$p" ]] && xcl_record pkg "$p"
    done < <(comm -13 "$pre" "$post")
    while read -r u; do
        [[ -n "$u" ]] && xcl_record user "$u"
    done < <(comm -13 "$users_pre" <(getent passwd | cut -d: -f1 | sort))
    rm -f "$pre" "$post" "$users_pre"
}

xcl_have_systemd() {
    [[ "$XCL_NO_SYSTEMD" != 1 ]] && [[ -d /run/systemd/system ]]
}

# xcl_service_start NAME — start for this session; enable at boot only when
# asked. Packages enable their units at install, so default is to disable.
xcl_service_start() {
    local svc="$1"
    xcl_have_systemd || return 0
    systemctl start "$svc"
    if [[ "$XCL_ENABLE_AT_BOOT" == 1 ]]; then
        systemctl enable "$svc" >/dev/null 2>&1
    else
        systemctl disable "$svc" >/dev/null 2>&1 || true
    fi
    xcl_record service "$svc"
}

# Snapshot of things this kit must never change, compared by 30-verify.
# Containers that churn by design are left out: the tower's per-job CI
# runner container is created and destroyed every few seconds (63 cycles an
# hour on 2026-11-08), so it never matches between two snapshots.
XCL_UNTOUCHABLE_IGNORE="${XCL_UNTOUCHABLE_IGNORE:-^xcelsior-ci-runner-}"
xcl_snapshot_untouchables() {
    local out="$1"
    {
        echo "# docker containers: id name status restart-count (ignoring /$XCL_UNTOUCHABLE_IGNORE/)"
        if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
            docker ps -a --format '{{.ID}} {{.Names}} {{.State}}' | sort | while read -r id name state; do
                [[ "$name" =~ $XCL_UNTOUCHABLE_IGNORE ]] && continue
                echo "$id $name $state $(docker inspect -f '{{.RestartCount}} {{.State.StartedAt}}' "$id" 2>/dev/null)"
            done
        fi
        echo "# runner services"
        systemctl list-units --type=service --all --no-legend 'actions.runner.*' 2>/dev/null | awk '{print $1, $3, $4}' || true
    } >"$out"
}

xcl_wait_tcp() {
    local host="$1" port="$2" tries="${3:-30}"
    for _ in $(seq "$tries"); do
        timeout 2 bash -c "</dev/tcp/$host/$port" 2>/dev/null && return 0
        sleep 1
    done
    return 1
}
