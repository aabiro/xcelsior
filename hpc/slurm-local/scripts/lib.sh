#!/usr/bin/env bash
# Shared helpers for the xcl-local Slurm validation cluster scripts.
#
# Every change to the machine is appended to $XCL_MANIFEST as one line,
# "<kind> <value>", so 90-rollback.sh can undo exactly what was done and
# nothing else. Kinds: user, pkg, file, dir, backup, ufw, authkey, service,
# leftover (package-created paths removed after the purge), sshconfig.

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

TOWER_ADDR="${TOWER_ADDR:-100.64.0.4}"
ASUS_ADDR="${ASUS_ADDR:-100.64.0.6}"
SLURM_UID="${SLURM_UID:-970}"
MUNGE_UID="${MUNGE_UID:-971}"
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

# Rewrite the committed addresses when the rehearsal (or a changed tailnet)
# needs different ones. A no-op with the defaults.
xcl_render() {
    sed -e "s/100\.64\.0\.4/${TOWER_ADDR}/g" -e "s/100\.64\.0\.6/${ASUS_ADDR}/g" "$1"
}

# xcl_put_file SRC DEST MODE OWNER[:GROUP] — install a rendered copy.
# An existing DEST is backed up first and restored by rollback.
xcl_put_file() {
    local src="$1" dest="$2" mode="$3" owner="$4"
    if [[ -e "$dest" ]] && ! grep -qxF -- "file $dest" "$XCL_MANIFEST"; then
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

xcl_installed_packages() {
    dpkg-query -W -f='${db:Status-Status} ${Package}\n' 2>/dev/null | awk '$1=="installed"{print $2}' | sort -u
}

# Directories the packages may create and leave behind after a purge
# (MariaDB 11.8 on Ubuntu 26.04 keeps data in /var/lib/mariadb, not mysql).
XCL_PKG_LEFTOVERS=(/etc/mysql /var/lib/mariadb /var/lib/mysql /var/log/mysql /run/mysqld
    /etc/munge /var/lib/munge /var/log/munge /run/munge /var/lib/slurm /var/log/slurm /etc/slurm)

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
xcl_snapshot_untouchables() {
    local out="$1"
    {
        echo "# docker containers: id name status restart-count"
        if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
            docker ps -a --format '{{.ID}} {{.Names}} {{.State}}' | sort | while read -r id name state; do
                echo "$id $name $state $(docker inspect -f '{{.RestartCount}} {{.State.StartedAt}}' "$id")"
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
