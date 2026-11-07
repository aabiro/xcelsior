#!/usr/bin/env bash
# Stop the test cluster's services on this node without uninstalling.
#   sudo ./45-stop.sh tower|asus
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
xcl_require_root
case "${1:-}" in
    tower) for s in slurmd slurmctld slurmdbd mariadb munge; do systemctl stop "$s" 2>/dev/null || true; done ;;
    asus) for s in slurmd munge; do systemctl stop "$s" 2>/dev/null || true; done ;;
    *) xcl_die "usage: $0 tower|asus" ;;
esac
xcl_log "stopped"
