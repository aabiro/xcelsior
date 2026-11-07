# Local Slurm validation: tower + ASUS

Test and research only. Nothing here is a production deployment.

This kit builds a two-node Slurm cluster on hardware you already own. It then drives that cluster through the Xcelsior Slurm adapter (`workers/slurm_adapter.py`), never through raw `sbatch`. The result is the evidence behind "tested locally, seeking scale validation" in a later SHARCNET proposal. Public wording stays **"Slurm-native, integration-ready"** until a partner route lands.

| File | What it is |
|---|---|
| `README.md` | This plan: steps, what changes on each machine, rollback, risks |
| `CHECKLIST.md` | Pass/fail checklist for every check and test case |
| `RESULTS_TEMPLATE.md` | Short results report to attach to the SHARCNET proposal |
| `RESULTS_2026-11-07.md`, `CHECKLIST_RESULTS_2026-11-07.md` | The filled report and checklist from the real run |
| `evidence/r1107050537/` | That run's results.json, pod listing and job outputs |
| `conf/slurm.conf`, `gres.conf`, `cgroup.conf` | The cluster configuration |
| `conf/slurmdbd.conf.template`, `mariadb-xcl-slurm.cnf` | Accounting (needed for metering) |
| `conf/xcelsior-profiles.json` | The adapter profile `xcl-local` |
| `conf/pod-declared.json` | What a facility would declare for the pod listing |
| `scripts/10-install-tower.sh`, `20-install-asus.sh` | Installers; every change is recorded in a manifest |
| `scripts/30-verify-cluster.sh` | Raw-Slurm health checks (V-1 … V-12) |
| `scripts/90-rollback.sh` | Undoes exactly what the manifest recorded |
| `scripts/40-start.sh`, `45-stop.sh` | Start or stop the test services (nothing starts at boot) |
| `scripts/xcl-slurm-gate` | Forced-command gate for the adapter's SSH key |
| `tests/validate_adapter.py` | The adapter validation suite (ENV, GPU, CPU, LIFE, FAIL, METER, POD, SYNC) |
| `rehearsal/` | Runs the whole kit in two containers first, on one machine |

The production roadmap that follows this kit is in [../IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).

## What this proves, and what it does not

It proves that the adapter can do the following against a real Slurm 25.11 controller:

- submit GPU and CPU work and have GRES honoured and enforced;
- map every lifecycle state correctly, from both `squeue` and `sacct`;
- turn every bad request into a machine-readable error in under a second, with no job created and no hang;
- produce per-second metering records in the exact shape `billing.UsageMeter` takes;
- describe the cluster as a pod listing whose specs are verified by jobs the scheduler itself ran.

It does this over the transport a real academic cluster demands: one restricted SSH key whose forced command allows only the adapter's command shapes. That is how Alliance automation ("robot") nodes admit MFA-exempt automated access.

It does not prove scale, multi-node MPI, a real fabric (InfiniBand or RoCE), a shared filesystem, or Alliance-specific policy. Alliance-specific policy means automatic partition assignment, allocation accounts and robot-node approval. Those are the open questions for scale validation. One policy constraint is already known: Alliance automation nodes "are not suitable for usage of AI agents", so a scale test would carry scripted adapter traffic only.

## Topology

| | tower | asus |
|---|---|---|
| Role | slurmctld, slurmdbd, MariaDB, slurmd | slurmd (config-less) |
| Hardware | Ryzen 9 5950X (16C/32T), 15 GB, RTX 3060 12 GB | Ryzen 9 4900H (8C/16T), 15 GB; CPU-only here |
| Tailnet | 100.64.0.4 | 100.64.0.6 |
| Offered to Slurm | 8 GB RAM, 8 CPUs per job, 1 GPU | 4 GB RAM, 8 CPUs per job |
| Firewall change | none (tailscale0 already allowed) | allow 6818/tcp from 100.64.0.4 on tailscale0 |

Partition `xcl` spans both nodes. MaxTime is 1 hour, and the GRES type is `rtx_3060`, typed the way Alliance clusters type theirs (`h100`). All Slurm and Munge traffic uses the tailnet addresses. MagicDNS is not relied on.

## What changes on each machine

Every item below is written to `/var/lib/xcl-slurm-test/manifest` as it happens. `90-rollback.sh` reads that file in reverse.

**tower**
- Packages: munge, slurmctld, slurmd, slurmdbd, slurm-client, slurm-wlm-nvml-plugin, slurm-wlm-mysql-plugin and mariadb-server, plus their dependencies. That is about 60 packages. A simulated install on 2026-11-07 touched no NVIDIA package and removed nothing.
- Users: `slurm` (uid/gid 970) and `munge` (971) are created first, with fixed IDs so both nodes agree. The package also creates `mysql`.
- Files: `/etc/slurm/{slurm,gres,cgroup,slurmdbd}.conf`, `/etc/munge/munge.key`, `/etc/mysql/mariadb.conf.d/90-xcl-slurm.cnf` and `/usr/local/libexec/xcl-slurm-gate`.
- Directories: `/var/spool/slurmctld`, `/var/spool/slurmd`, `/var/log/slurm`, `/var/lib/xcl-slurm-test/jobs` (job output) and MariaDB's `/var/lib/mariadb`.
- One line in `~aaryn/.ssh/authorized_keys`, marked `xcl-slurm-adapter`. That key can only run the gate, and only from 100.64.0.6.
- A copy of this kit at `~/xcl-slurm-kit`, because the tower has no repo checkout. It is not in the manifest; delete it after rollback.
- Services: munge, mariadb, slurmdbd, slurmctld and slurmd. They are **disabled at boot** by default, so a reboot returns the tower to its pre-cluster state. `40-start.sh tower` brings them back.
- Not changed: Docker and its containers, the GitHub Actions runners, the NVIDIA driver and ufw. MariaDB opens no TCP port; `skip-networking` is set because mempool-db already uses 3306 inside Docker. Preflight snapshots every container's ID, state, restart count and start time, plus the runner units. V-12 proves they are unchanged.

**asus**
- Packages: munge, slurmd and slurm-client, about 30 with dependencies. The NVML plugin is not installed, so the RTX 2060 is never offered.
- Users: `slurm` (970) and `munge` (971).
- Files: `/etc/munge/munge.key` (copied from the tower) and `/etc/default/slurmd` (`--conf-server 100.64.0.4:6817`).
- Directories: `/var/spool/slurmd`, `/var/log/slurm` and `/var/lib/xcl-slurm-test/jobs`.
- One ufw rule: `allow in on tailscale0 from 100.64.0.4 to any port 6818 proto tcp`.
- Services: munge and slurmd, disabled at boot.
- The adapter's client side (`adapter-client` step): `~/.ssh/xcl_slurm_adapter` and its `.pub`, plus `~/.ssh/xcl_slurm_adapter.conf`, an ssh config used only by the adapter through `-F`. Your `~/.ssh/config` is never edited.

## Step 0: rehearse in containers (on the ASUS, ~6 minutes)

```bash
cd hpc/slurm-local
./rehearsal/run-rehearsal.sh            # builds, runs everything, tears down
./rehearsal/run-rehearsal.sh --keep     # leave the containers up to poke at
```

The rehearsal runs two privileged containers on a private network, `172.31.251.0/24`. They run the kit's own install steps and configs, raw verification and the full suite. The suite reaches the "tower" container over SSH through the gate, which is the same path as the real run. It changes nothing on the host except two Docker images (`xcl-slurm-rh:tower`, `xcl-slurm-rh:asus`) and a scratch directory. Remove them with `docker rmi xcl-slurm-rh:tower xcl-slurm-rh:asus`.

Deviations from the real cluster, and only these:

- container addresses instead of the tailnet;
- the tower node is declared with the host's 16 CPUs;
- `IgnoreSystemd=yes` in cgroup.conf, and daemons start without systemd;
- the GPU is the host's (RTX 2060 on the ASUS), so its type token replaces `rtx_3060`;
- SSH arrives on 127.0.0.1:22022.

**Real-hardware record, 2026-11-07 (run `r1107050537`):** V-1 to V-12 all passed, V-6 and V-12 included. All 24 suite cases passed, SYNC-1 included. Docker containers and runners were unchanged after install and again after the suite. The real run found three kit defects, all fixed:
- `ldconfig -p | grep -q` failed under pipefail through SIGPIPE;
- V-12's snapshot came out empty because of `fs.protected_regular`;
- a `~/.ssh/config` alias let a `Host *` key bypass the restricted key.

The cluster was left running, with services disabled at boot. Results are in `RESULTS_2026-11-07.md`.

**Rehearsal record, 2026-11-07:** V-1 to V-11 passed. V-6 was skipped because it needs sudo inside the container. All 24 suite cases passed, including the optional SYNC-1 against the pytest database. Install followed by rollback left the package set identical to pre-install, with no users, files, directories or daemons left. The rehearsal found three real defects, all now fixed:
- plain `scancel` exits 0 silently for unknown ids;
- the munge key and MariaDB's `/var/lib/mariadb` and `mysql` user would have survived rollback;
- an untyped `Gres=gpu:1` makes Slurm discard the NVML type.

## Step 1: install the tower

```bash
# from the ASUS: the tower has no repo checkout, so copy the kit over
rsync -a --exclude rehearsal/.work --exclude results hpc/slurm-local/ aaryn@100.64.0.4:~/xcl-slurm-kit/

# on the tower
cd ~/xcl-slurm-kit/scripts
sudo ./10-install-tower.sh preflight      # read-only checks + Docker/runner snapshot; changes nothing
sudo ./10-install-tower.sh all > ~/xcl-install-tower.log 2>&1   # log to a file; never pipe apt into head
```

Expect `tower services up; boot autostart: off`. If any step fails, fix the cause and re-run `all`. Every step is idempotent.

## Step 2: install the ASUS

```bash
# on the ASUS
cd hpc/slurm-local/scripts
sudo ./20-install-asus.sh preflight       # includes: can it reach slurmctld on 100.64.0.4:6817?
sudo ./20-install-asus.sh all
```

The munge key is fetched over SSH with your normal key, as `aaryn@100.64.0.4`, using `sudo -n cat`.

## Step 3: verify the cluster with raw Slurm

```bash
# on the tower, as aaryn (not root)
cd ~/xcl-slurm-kit/scripts && ./30-verify-cluster.sh
```

All of V-1 to V-12 must pass. Together they show:
- both nodes are idle in `xcl`;
- the tower's GRES is `gpu:rtx_3060:1`, and NVML detected it on `/dev/nvidia0`;
- an `srun` with GRES sees exactly one RTX 3060, and an `srun` without GRES sees none;
- the ASUS runs jobs;
- accounting records epoch timestamps;
- Docker containers and runners are unchanged.

The adapter is not involved yet, so a later adapter failure is the adapter's.

## Step 4: give the adapter its restricted key

```bash
# on the ASUS
sudo ./20-install-asus.sh adapter-client
ssh -F ~/.ssh/xcl_slurm_adapter.conf xcl-tower sinfo --version   # allowed: "slurm-wlm 25.11.2"
ssh -F ~/.ssh/xcl_slurm_adapter.conf xcl-tower id                # refused: "xcl-slurm-gate: refused: command 'id' not allowed"
```

The step does four things, all recorded:
1. It creates a key used for nothing else.
2. It writes a dedicated ssh config, used through `-F`.
3. It authorises the key on the tower as `restrict,from="100.64.0.6",command="/usr/local/libexec/xcl-slurm-gate"`.
4. It refuses to finish unless `sinfo` works and `id` is refused.

The adapter profile passes the same `-F` file. **Do not add an alias to `~/.ssh/config` instead.** IdentityFile entries accumulate across every matching Host block, so a `Host *` key is offered first. On the real run, the tower accepted that unrestricted key and `id` ran.

The gate allows only these command shapes:
- `sinfo`, `squeue` and `sacct`;
- `scontrol show …`;
- `sbatch --parsable`, with the script on stdin;
- `scancel [-v] <id>`;
- `tail -c N -- <path under /var/lib/xcl-slurm-test/jobs/>`;
- `SLURM_TIME_FORMAT` as the only allowed environment variable.

It logs every allow and deny to the journal under the tag `xcl-slurm-gate`.

## Step 5: run the adapter validation suite

```bash
# on the ASUS, from the repo root
python3 hpc/slurm-local/tests/validate_adapter.py
# one case:      --only GPU-1
# with the reconcile loop against the pytest DB (never prod):
( set -a; source .env.test; set +a; export XCELSIOR_SKIP_DOTENV=1
  .venv/bin/python hpc/slurm-local/tests/validate_adapter.py --with-db )
```

It takes about 4 minutes. Results go to `hpc/slurm-local/results/<run-id>/`:
- `checklist.md`, a filled pass/fail table;
- `results.json`, with evidence for every case: job ids, state traces, accounting rows and outputs;
- `pod-listing.json`, the reference pod spec;
- `outputs/*.txt`.

The exit status is 0 only if every executed case passed. SYNC-1 refuses to run unless `.env` loading is off and the database name contains `test` or `dev`.

The suite runs on the ASUS for a reason. The cluster has no shared filesystem, so a job on the ASUS writes its output on the ASUS, and the suite reads it there (`--local-output-dir`). Tower jobs are read through the gate. Real clusters share `/scratch`, so there the adapter reads everything through the login node.

## Step 6: record the results

Copy `results/<run-id>/checklist.md` into `CHECKLIST.md`'s results column, or attach it. Then fill in `RESULTS_TEMPLATE.md`. Keep the run directory: it is the evidence.

## Rollback

```bash
# on each machine (ASUS first, then tower)
sudo ./90-rollback.sh --dry-run    # prints every action
sudo ./90-rollback.sh
```

Rollback runs in this order:
1. stop the services;
2. remove the adapter's authorized_keys line;
3. delete the ufw rule;
4. remove recorded files and directories;
5. purge exactly the recorded packages (never `autoremove`);
6. restore any backed-up file;
7. remove package directories that did not exist before (`/var/lib/mariadb`, `/etc/mysql`, munge's state);
8. delete the `slurm`, `munge` and `mysql` users;
9. remove `/var/lib/xcl-slurm-test`.

A copy of the manifest is kept in `/root`. The rehearsal checked this end to end: the package set after rollback was identical to the one before install.

To pause without uninstalling: `sudo ./45-stop.sh tower` and `sudo ./45-stop.sh asus`.

## Risks and how the kit handles them

- **The tower runs production-adjacent services.** Slurm may allocate at most 8 GB, 8 CPUs per job and 1 hour per job. Job steps are cgroup-confined, so cores, RAM, swap and devices are all limited. MariaDB has no TCP port. Services don't start at boot. Docker and the runners are snapshotted and compared.
- **The tower's disk is 93% full** (130 GB free). The install adds about 300 MB, and logs are small. Preflight requires 3 GB free.
- **The ASUS crashes several times a day** (pstore panics; failing RAM or CPU), and it runs with heavy swap. With `ReturnToService=2` it rejoins once slurmd re-registers. Don't read a case that failed because the node went DOWN as an adapter failure. Check `sinfo -R`, then re-run that case with `--only`. RealMemory is set to 4 GB.
- **Munge is clock-sensitive** (5-minute TTL). Preflight requires NTP sync on both nodes.
- **On the ASUS, Slurm jobs can still open the RTX 2060.** Slurm only fences devices it manages as GRES. That's harmless for the CPU tests, but don't treat the ASUS as GPU-isolated.

## Troubleshooting

| Symptom | Check |
|---|---|
| `asus` is `down*` / `unknown` | `sudo systemctl status slurmd` on the ASUS; ufw rule present; `munge -n \| ssh aaryn@100.64.0.4 unmunge` |
| tower `drain` with `gres/gpu count` | `sudo slurmd -G` on the tower; `nvidia-smi`; the NVML plugin installed |
| `Invalid account or account/partition combination` | not expected (no `AccountingStorageEnforce`); check slurm.conf was not edited |
| `sacct` says accounting is disabled | `systemctl status slurmdbd`; `sacctmgr show cluster` lists `xcl-local` |
| `Security violation` in slurmd log | the `slurm` uid differs between nodes (`id slurm` on both must be 970) |
| suite: `transport_error` | `ssh xcl-tower sinfo --version`; the key line has `from="100.64.0.6"` |
