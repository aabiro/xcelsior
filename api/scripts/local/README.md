# scripts/local/

Local-only staging area for scripts that are deployed elsewhere (tower, VPS) and
not intended to be committed to the repo. Everything here except this README and
.gitignore is ignored by git.

Source of truth for the deployed copies. If you delete one of these files,
recover from the live host or recreate from `docs/3060-utilization-plan.md`.

## What's here
- `xcm/xcm.py`, `xcm/schema.sql` — deployed on tower at `/usr/local/bin/xcm` and `/srv/models/_meta/schema.sql`
- `tower-serverless/server.py` — deployed on tower at `/opt/tower-serverless/server.py`
- `tier3-watchdog.sh` — deployed on tower at `/usr/local/bin/tier3-watchdog.sh`
