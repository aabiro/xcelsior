#!/usr/bin/env bash
# Undo everything the install scripts did on THIS node, from the manifest.
#
#   sudo ./90-rollback.sh --dry-run     print the plan, change nothing
#   sudo ./90-rollback.sh               do it
#
# Order: stop services → remove adapter key lines older kits wrote into a
# home → delete the ufw rules → remove created files and dirs (the gate,
# the sshd drop-in, the adapter's authorized_keys) and reload sshd → purge
# exactly the packages the install added → restore backed-up files →
# delete the users (slurm, munge, the xclslurm service account) → remove
# /var/lib/xcl-slurm-test (backups under it included: copy them off first). Package directories (MariaDB's data dir,
# /etc/mysql, munge's state) are removed only if they did not exist before.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
xcl_require_root
DRY=0
[[ "${1:-}" == --dry-run ]] && DRY=1
[[ -s "$XCL_MANIFEST" ]] || xcl_die "no manifest at $XCL_MANIFEST — nothing recorded, nothing to undo"

run() { if [[ $DRY == 1 ]]; then echo "would: $*"; else xcl_log "$*"; "$@"; fi; }
kind() { awk -v k="$1" '$1==k { $1=""; sub(/^ /,""); print }' "$XCL_MANIFEST" | tac; }

for svc in slurmd slurmctld slurmdbd mariadb munge; do
    if kind service | grep -qx "$svc" || kind pkg | grep -qx "${svc/mariadb/mariadb-server}"; then
        systemctl is-active --quiet "$svc" 2>/dev/null && run systemctl stop "$svc"
    fi
done
# own_pids NAME — this machine's own NAME processes, never a container's.
# pkill -x matched by name across every PID namespace: on the tower that
# included mempool's mariadbd inside Docker (found by the 2026-11-08 dry run,
# before any rollback ran there).
own_pids() {
    local host p
    host="$(readlink /proc/1/ns/pid)"
    for p in $(pgrep -x "$1"); do
        [[ "$(readlink "/proc/$p/ns/pid" 2>/dev/null)" == "$host" ]] || continue
        grep -qE 'docker-|containerd|libpod|kubepods' "/proc/$p/cgroup" 2>/dev/null && continue
        echo "$p"
    done
}

# Stray daemons (rehearsal / manual starts), only those our packages own.
for d in slurmd slurmctld slurmdbd mariadbd munged; do
    case $d in
        mariadbd) kind pkg | grep -qx mariadb-server || continue ;;
        munged) kind pkg | grep -qx munge || continue ;;
    esac
    mapfile -t pids < <(own_pids "$d")
    if [[ ${#pids[@]} -gt 0 ]]; then
        run kill "${pids[@]}"
        [[ $DRY == 1 ]] || for _ in $(seq 20); do kill -0 "${pids[@]}" 2>/dev/null || break; sleep 1; done
    fi
done

while read -r ak; do
    [[ -f "$ak" ]] || continue
    if [[ $DRY == 1 ]]; then echo "would: remove 'xcl-slurm-adapter' line from $ak"
    else sed -i '/ xcl-slurm-adapter$/d' "$ak"; xcl_log "removed adapter key from $ak"; fi
done < <(kind authkey)

while read -r cfg; do
    [[ -f "$cfg" ]] || continue
    if [[ $DRY == 1 ]]; then echo "would: remove the xcl-slurm-test alias block from $cfg"
    else
        sed -i '/^# >>> xcl-slurm-test adapter alias >>>$/,/^# <<< xcl-slurm-test adapter alias <<<$/d' "$cfg"
        xcl_log "removed adapter alias from $cfg"
    fi
done < <(kind sshconfig)

while read -r rule; do
    [[ -n "$rule" ]] || continue
    # shellcheck disable=SC2086
    run ufw delete $rule || true
done < <(kind ufw)

# Files and dirs first, so package purges can remove their now-empty dirs.
while read -r f; do
    grep -qxF "backup $f" "$XCL_MANIFEST" && continue
    [[ -e "$f" ]] && run rm -f "$f"
done < <(kind file)

while read -r d; do
    [[ -d "$d" ]] && run rm -rf "$d"
done < <(kind dir)

# The sshd drop-in is gone now; have sshd drop the service account's block.
if [[ -n "$(kind sshd)" ]] && xcl_have_systemd && systemctl is-active --quiet ssh; then
    if [[ $DRY == 1 ]]; then echo "would: sshd -t && systemctl reload ssh"
    else /usr/sbin/sshd -t && systemctl reload ssh && xcl_log "sshd reloaded without the xcl-slurm-test block"; fi
fi

mapfile -t PKGS < <(kind pkg)
if [[ ${#PKGS[@]} -gt 0 ]]; then
    run env DEBIAN_FRONTEND=noninteractive apt-get purge -y "${PKGS[@]}"
fi
# Package directories that did not exist before the install (MariaDB's data
# dir, /etc/mysql, munge's state) — dpkg leaves them when they are not empty.
while read -r d; do
    [[ -e "$d" ]] && run rm -rf "$d"
done < <(kind leftover)

# Files that existed before the kit touched them go back as they were.
while read -r f; do
    [[ -e "$XCL_BACKUP$f" ]] && run cp -a "$XCL_BACKUP$f" "$f"
done < <(kind backup)

while read -r u; do
    getent passwd "$u" >/dev/null && run userdel "$u"
    getent group "$u" >/dev/null && run groupdel "$u"
done < <(kind user)

if [[ $DRY == 1 ]]; then
    echo "would: rm -rf $XCL_STATE"
else
    cp "$XCL_MANIFEST" "/root/xcl-slurm-test-manifest.$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
    rm -rf "$XCL_STATE"
    xcl_log "rollback complete (manifest copy kept in /root)"
fi
