# WAL archive and point-in-time recovery qualification

The candidate adds a private, checksum-verified **local WAL spool** and a real
PostgreSQL 16 recovery test. Production archiving, independent encrypted offsite
storage and failover are separate deployment requirements.

`api/scripts/wal_archive.py` publishes data and a checksum receipt together in
an atomic directory. PostgreSQL retries accept only an identical, verified
object. A collision or corrupt receipt/data fails without replacing the
original archive or restore destination. Cluster system identifiers separate
WAL namespaces. Filenames, private directory modes, regular files and symlink
boundaries are checked. Files and directory entries are fsynced. There is no
automatic WAL deletion or guessed retention window.

`tests/test_wal_recovery.py` passed **seven checks** through the isolated
security runner. Its integration case creates a checksum-enabled, Unix-socket
PostgreSQL cluster, takes a streamed physical base backup and runs
`pg_verifybackup`. It commits another marker, creates a named restore point,
then commits a marker that must be excluded. A separate restored cluster
fetches WAL from the archive, promotes at that point and contains exactly the
first two markers. Both clusters stop successfully. Corruption, duplicate
publication, conflicting content, unsafe names and symlinks have negative
checks. The outer disposable PostgreSQL cluster also stops successfully.

Raw evidence: `artifacts/control-plane-open-work-20261024/wal-pitr-v2/`.
The first run exposed a test socket pathname exceeding the Unix kernel limit;
the final test uses uniquely owned private short socket directories and removes
them after stopping the two clusters. The failed run is diagnostic evidence.

This follows PostgreSQL's documented requirement for a base backup plus a
continuous WAL sequence and a separately configured recovery target. See the
[PostgreSQL 16 recovery documentation](https://www.postgresql.org/docs/16/continuous-archiving.html)
and [base-backup reference](https://www.postgresql.org/docs/16/app-pgbasebackup.html).
Configuration files, object-store bytes and projection rebuilds require their
own recovery inventory and acceptance.

For production, resolve an independent backup destination and its credential
boundary, retain a verified physical base backup, configure the actual
Xcelsior PostgreSQL cluster (port 5434), monitor archiver failures/WAL disk
growth, and exercise recovery on a separate host. Enabling `archive_mode`
requires a planned restart of that cluster. The local test does not establish
production RPO/RTO, region-loss recovery or encryption/offsite retention.
