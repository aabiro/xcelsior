#!/bin/sh
# Allocate metrics storage before Python imports prometheus_client. Each API
# master gets private files; worker exits retain counters within that generation.
set -eu
umask 077

PROMETHEUS_MULTIPROC_DIR="$(mktemp -d "${TMPDIR:-/tmp}/xcelsior-prometheus.XXXXXXXX")"
XCELSIOR_METRICS_GENERATION_STARTED="$(date +%s.%N)"
export PROMETHEUS_MULTIPROC_DIR XCELSIOR_METRICS_GENERATION_STARTED

if [ "$#" -eq 0 ]; then
    set -- gunicorn api:app -c gunicorn.conf.py
fi
exec "$@"
