#!/usr/bin/env bash
# Undo everything the install scripts did on THIS node, from the manifest.
#
#   sudo ./90-rollback.sh --dry-run     print the plan, change nothing
#   sudo ./90-rollback.sh               do it
#
# Order: stop services → remove the adapter's authorized_keys line → delete
# the ufw rule → remove created files and dirs → purge exactly the packages
# the install added → restore backed-up files → delete the users → remove
# /var/lib/xcl-slurm-test. Package directories (MariaDB's data dir,
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
# Stray daemons (rehearsal / manual starts), only those our packages own.
for d in slurmd slurmctld slurmdbd mariadbd munged; do
    case $d in
        mariadbd) kind pkg | grep -qx mariadb-server || continue ;;
        munged) kind pkg | grep -qx munge || continue ;;
    esac
    if pgrep -x "$d" >/dev/null; then
        run pkill -x "$d"
        [[ $DRY == 1 ]] || for _ in $(seq 20); do pgrep -x "$d" >/dev/null || break; sleep 1; done
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
