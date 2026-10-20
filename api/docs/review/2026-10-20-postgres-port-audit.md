# PostgreSQL port audit — 2026-10-20

The running Xcelsior API, maintenance/billing worker, scheduler and SSH gateway
all have their shared application DSN set to **localhost:5434/xcelsior**.
`pg_stat_activity` independently confirms Xcelsior API, billing, scheduler and
SSH-gateway connections on 5434. The MCP container has no PostgreSQL DSN.

Port **5432** is also live on this host: it serves the separate SPIRE database
and PixelSpark databases. Their connections were observed there; no Xcelsior
application connections were observed on 5432 during this check. The cluster
also contains a database named `xcelsior` with different row counts, so connecting successfully
with the right database name alone does not prove the correct cluster.

The workspace `.env` explicitly selects 5434 in its port setting and all three
shared DSN aliases. `.env.test` selects 5432 and separate test databases. No
credentials or complete DSNs are included in this report.

## Application and operation references

| Location | Meaning |
| --- | --- |
| `core/db.py:47` | Shared resolver's developer fallback when every configured DSN is absent. Production's configured DSN overrides it. |
| `alembic.ini:7` | Migration fallback URL, overridden by the environment in `api/migrations/env.py`. |
| `docker-compose.yml:32,523,741,770` | Interpolated application DSNs default the configurable PostgreSQL port to 5432. Production renders these to 5434. |
| `docker-compose.yml:1039` | PostgreSQL exporter URI uses the same configurable default. The optional Compose exporter is separate from the running native monitoring services. |
| `api/scripts/backup-db.sh` | Port default. The relocated script was looking for `api/.env`, missing the root configuration. Corrected to read the project root as data. |
| `api/scripts/restore-db.sh:19` | Manual restore port default. It had the same relocated dotenv path problem and executed dotenv content. Corrected to read only the selected database settings as data. No destructive manual restore was executed. |
| `api/scripts/restore-db-drill.sh` | PostgreSQL administration commands previously inherited libpq's implicit default 5432. The script now exports the configured application port, validates it, and preserves an explicit `PGPORT` override. |
| `api/scripts/deploy.sh:1912` | Explicit port for the local test-stack migration, within `deploy_test_local`. |
| `api/scripts/setup_pytest_db.sh:53` | Test bootstrap URL used to read the Alembic head. Its administration commands also use libpq's port configuration. |
| `api/infra/spire/rollout-enrollment.sh:54` | SPIRE's separate PostgreSQL datastore on 5432. |
| `api/scripts/provision_spire_datastore.sh:54` | SPIRE datastore DSN printed by the provisioner. This command was not run. |
| `hosts/ssh_gateway.py:114` | Comment explaining that a missing production DSN must fail rather than use the developer default. This line opens no connection. |
| `api/scripts/bootstrap_pg_from_empty.sh:9` | Usage example only. |
| `api/scripts/incident/audit_operator_clients.py:10` | Usage example only. |

`.env.example`, CI service definitions and architecture documentation contain
additional examples and test defaults. The
[location inventory](2026-10-20-postgres-port-audit-locations.json) records all
whole-token occurrences in tracked text before adding this report, including
their line numbers. Font bytes, image coordinates and dependency hashes are
not PostgreSQL references.

The scan also covered PostgreSQL commands without a literal port. CI schema
export and pytest database setup use libpq port selection; the test-container
initializer uses its local PostgreSQL socket. SPIRE host registration uses the
explicit application DSN. Role audits, role provisioning and partition
maintenance require a supplied DSN; they do not independently hardcode 5432.

## Reproduced operations defects

The first restore drill created and restored its disposable database on 5432,
then the migration container correctly attempted to connect to the disposable
name on 5434. The migration guard stopped before Alembic. Cleanup removed the
disposable database; neither production database was restored over.

Both database service units still referred to the old `/opt/xcelsior/scripts/`
paths after scripts moved under `api/scripts/`. The backup unit path and both
units' documentation paths are corrected. Backup's project configuration and
manual restore's project configuration now work from either script layout.

Three workflow regressions failed before the first fixes; both manual-restore
cases separately failed before their fix. All six workflow cases then passed,
using fake administration commands and an explicit refusal at the manual
restore confirmation prompt. The larger frozen application gate had already
passed 618 tests against an isolated PostgreSQL cluster. Production backup and
restore execution evidence is recorded in the cutover audit follow-up report.
