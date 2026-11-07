#!/usr/bin/env bash
# Start the test cluster's services on this node (after a reboot: they do
# not start at boot unless XCL_ENABLE_AT_BOOT=1 was set at install).
#   sudo ./40-start.sh tower|asus
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
xcl_require_root
case "${1:-}" in
    tower) for s in munge mariadb slurmdbd slurmctld slurmd; do systemctl start "$s"; done ;;
    asus) for s in munge slurmd; do systemctl start "$s"; done ;;
    *) xcl_die "usage: $0 tower|asus" ;;
esac
xcl_log "started"
