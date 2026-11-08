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
| `site.env.example` | Your addresses, the asus hostname and your account; copy to `site.env` (git-ignored) |
| `conf/slurmdbd.conf.template`, `mariadb-xcl-slurm.cnf` | Accounting (needed for metering) |
| `conf/sshd-xcl-slurm-test.conf` | sshd Match block for the adapter's service account (tower) |
| `conf/xcelsior-profiles.json` | The adapter profile `xcl-local` |
| `conf/pod-declared.json` | What a facility would declare for the pod listing |
| `scripts/10-install-tower.sh`, `20-install-asus.sh` | Installers; every change is recorded in a manifest |
| `scripts/30-verify-cluster.sh` | Raw-Slurm health checks (V-1 … V-12) |
| `scripts/90-rollback.sh` | Undoes exactly what the manifest recorded |
| `scripts/40-start.sh`, `45-stop.sh` | Start or stop the test services (nothing starts at boot) |
| `scripts/xcl-slurm-gate` | Forced-command gate for the adapter's SSH key (tests: `tests/test_slurm_gate.py`) |
| `scripts/xcl-node-health` | Slurm HealthCheckProgram on both nodes |
| `scripts/50-backup.sh` | Backs up the accounting DB, slurmctld state, `/etc/slurm` and the munge key |
| `tests/validate_adapter.py` | The adapter validation suite (ENV, GPU, CPU, LIFE, FAIL, SEC, METER, POD, SYNC) |
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

## Changes since the 2026-11-07 run (Phase 0 of the blueprint)

The 2026-11-07 run proved the adapter, but jobs ran as your own account and the gate passed batch scripts through unread. These kit changes have **not yet been run on the real machines**; the cluster there is still the 2026-11-07 install until it is rolled back and reinstalled (Step 0 onwards).

- **A service account, `xclslurm` (uid 972).** Jobs submitted through the gate run as it on both nodes, never as you. Its shell is `/bin/sh`, which reads no startup files when sshd runs the gate, and its home is root-owned. The adapter's key sits in a root-owned file outside any home, with an `expiry-time`, and an sshd Match block forces the gate for that account whatever the client asks. Before, one job could append to `~/.ssh/authorized_keys` or `~/.bashrc` and turn the restricted key into a login, as an account with sudo.
- **The gate checks exact command shapes and reads batch scripts.** It refuses `squeue -a`, `-i`, `scontrol` beyond `ping`, any `#SBATCH` directive outside the adapter's set (so `--output` can't target `~/.ssh`), and reads output files with `O_NOFOLLOW`. It logs every admitted script's sha256, and slurmdbd now keeps the scripts (`AccountingStoreFlags=job_script`).
- **Config fixes:** `innodb_snapshot_isolation=OFF` (SchedMD requires it; MariaDB 11.8's default would let slurmdbd stop), a health check, `SrunPortRange` with the asus firewall rule for it, retention limits in slurmdbd, `job_comment` so sacct can find the adapter's tags, and a `spot` QoS that normal jobs preempt (`PreemptMode=REQUEUE`), so the requeue paths can be tested.
- **Site values moved out of the repository** into `site.env` (git-ignored): the repository is public.
- **Two new suite cases**, SEC-1 and SEC-2, prove the service account and the script check on the real cluster.

Still open from Phase 0: Slurm itself. Ubuntu 26.04 ships 25.11.2, which lacks the eight CVE fixes in SchedMD's 25.11.8 (2026-09-02), and Ubuntu has published no update. That is acceptable for this single-operator lab on the tailnet and not for anything with tenants.

## Topology

| | tower | asus |
|---|---|---|
| Role | slurmctld, slurmdbd, MariaDB, slurmd | slurmd (config-less) |
| Hardware | Ryzen 9 5950X (16C/32T), 15 GB, RTX 3060 12 GB | Ryzen 9 4900H (8C/16T), 15 GB; CPU-only here |
| Tailnet | `TOWER_ADDR` in `site.env` | `ASUS_ADDR` in `site.env` |
| Offered to Slurm | 8 GB RAM (`TOWER_MEM_MB`; 16 GB after the RAM upgrade), 8 CPUs, 1 GPU (shared, see below) | 4 GB RAM (`ASUS_MEM_MB`), 8 CPUs |
| Firewall change | none: ufw already denies incoming except on tailscale0 (preflight checks it) | allow 6818/tcp and 60001-60100/tcp (srun) from the tower on tailscale0 |

The 8 CPUs are a cap on all of the partition's jobs together on that node (`MaxCPUsPerNode`), not a per-job limit. A second partition would get its own 8, so switch to `CoreSpecCount` on the node lines before adding one.

**The tower's RTX 3060 is not Slurm's alone.** On 2026-11-08 it was in use by `nvidia-cuda-mps-server`, `pxl-publisher` (ffmpeg), `pe-test-gpu-worker` and a Docker container, all outside Slurm. Slurm can't see or fence them. GPU jobs here share the device, so treat GPU timings from this node as shared-GPU numbers. The health check logs these processes; it doesn't drain on them, because the node would never come back (`XCL_GPU_CONTENTION=drain` at install changes that).

Partition `xcl` spans both nodes. MaxTime is 1 hour, and the GRES type is `rtx_3060`, typed the way Alliance clusters type theirs (`h100`). All Slurm and Munge traffic uses the tailnet addresses. MagicDNS is not relied on.

## What changes on each machine

Every item below is written to `/var/lib/xcl-slurm-test/manifest` as it happens. `90-rollback.sh` reads that file in reverse.

**tower**
- Packages: munge, slurmctld, slurmd, slurmdbd, slurm-client, slurm-wlm-nvml-plugin, slurm-wlm-mysql-plugin and mariadb-server, plus their dependencies. That is about 60 packages. A simulated install on 2026-11-07 touched no NVIDIA package and removed nothing.
- Users: `slurm` (uid/gid 970), `munge` (971) and the service account `xclslurm` (972) are created first, with fixed IDs so both nodes agree. The package also creates `mysql`.
- Files: `/etc/slurm/{slurm,gres,cgroup,slurmdbd}.conf`, `/etc/munge/munge.key`, `/etc/mysql/mariadb.conf.d/90-xcl-slurm.cnf`, `/usr/local/libexec/xcl-slurm-gate`, `/usr/local/libexec/xcl-node-health` and `/etc/ssh/sshd_config.d/50-xcl-slurm-test.conf` (sshd is reloaded after `sshd -t` accepts it; the Match block ends with that file, so no other account is affected).
- Directories: `/var/spool/slurmctld`, `/var/spool/slurmd`, `/var/log/slurm`, `/var/lib/xcl-slurm-test/jobs` (job output: `xclslurm`, your group, mode 2750), `/var/lib/xcl-slurm-test/svc-home` (root-owned) and MariaDB's `/var/lib/mariadb`.
- `/etc/ssh/xcl-slurm-test/authorized_keys`, root-owned, holding the adapter's key for `xclslurm`. That key can only run the gate, only from the asus, and expires after 90 days.
- In the accounting DB: QoS `spot`, which `normal` may preempt.
- A copy of this kit at `~/xcl-slurm-kit`, because the tower has no repo checkout. It is not in the manifest; delete it after rollback.
- Services: munge, mariadb, slurmdbd, slurmctld and slurmd. They are **disabled at boot** by default, so a reboot returns the tower to its pre-cluster state. `40-start.sh tower` brings them back.
- Not changed: Docker and its containers, the GitHub Actions runners, the NVIDIA driver and ufw. MariaDB opens no TCP port; `skip-networking` is set because mempool-db already uses 3306 inside Docker. Preflight snapshots every container's ID, state, restart count and start time, plus the runner units. V-12 proves they are unchanged.

**asus**
- Packages: munge, slurmd and slurm-client, about 30 with dependencies. The NVML plugin is not installed, so the RTX 2060 is never offered.
- Users: `slurm` (970), `munge` (971) and `xclslurm` (972).
- Files: `/etc/munge/munge.key` (copied from the tower), `/etc/default/slurmd` (`--conf-server <tower>:6817`) and `/usr/local/libexec/xcl-node-health`.
- Directories: `/var/spool/slurmd`, `/var/log/slurm`, `/var/lib/xcl-slurm-test/jobs` and `/var/lib/xcl-slurm-test/svc-home`.
- Two ufw rules, from the tower on tailscale0: port 6818 (slurmd) and 60001-60100 (srun).
- Services: munge and slurmd, disabled at boot.
- The adapter's client side (`adapter-client` step): `~/.ssh/xcl_slurm_adapter` and its `.pub`, plus `~/.ssh/xcl_slurm_adapter.conf`, an ssh config used only by the adapter through `-F`, logging in as `xclslurm`. Your `~/.ssh/config` is never edited. Your home must stay closed to other users (preflight checks it): jobs run on this machine too.

## Step 0: rehearse in containers (on the ASUS, ~6 minutes)

```bash
cd hpc/slurm-local
./rehearsal/run-rehearsal.sh            # builds, runs everything, tears down
./rehearsal/run-rehearsal.sh --keep     # leave the containers up to poke at
```

The rehearsal runs two privileged containers on a private network, `172.31.251.0/24`. They run the kit's own install steps and configs, raw verification and the full suite. The suite reaches the "tower" container over SSH through the gate, which is the same path as the real run. It changes nothing on the host except two Docker images (`xcl-slurm-rh:tower`, `xcl-slurm-rh:asus`) and a scratch directory. Remove them with `docker rmi xcl-slurm-rh:tower xcl-slurm-rh:asus`.

Deviations from the real cluster, and only these:

- container addresses instead of the tailnet, and the asus container is named `xcl-rh-asus` (site values come from the environment, not `site.env`);
- the tower node is declared with the host's 16 CPUs;
- `IgnoreSystemd=yes` in cgroup.conf, and daemons start without systemd;
- the GPU is the host's (RTX 2060 on the ASUS), so its type token replaces `rtx_3060`;
- SSH arrives on 127.0.0.1:22022.

**Rehearsal record, 2026-11-08 (the Phase 0 kit, run `r1108022520`):** V-1 to V-11 passed (V-6 skipped: sudo inside the container). 26 of 26 suite cases passed, SEC-1 and SEC-2 included: the job ran as `xclslurm` and got `Permission denied` writing its own `~/.ssh`, `id` was still refused afterwards, and scripts with `--chdir` or `--output` aimed at `~/.ssh` were refused before sbatch. SYNC-1 was skipped (no `--with-db`). The first attempt stopped on a placeholder written in a slurm.conf comment, now caught by `tests/test_slurm_gate.py`.

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
# on the ASUS, once: your site values (git-ignored)
cp hpc/slurm-local/site.env.example hpc/slurm-local/site.env && $EDITOR hpc/slurm-local/site.env

# from the ASUS: the tower has no repo checkout, so copy the kit over (site.env included)
rsync -a --exclude rehearsal/.work --exclude results hpc/slurm-local/ <you>@<tower-addr>:~/xcl-slurm-kit/

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
sudo ./20-install-asus.sh preflight       # includes: can it reach slurmctld on <tower>:6817?
sudo ./20-install-asus.sh all
```

The munge key is fetched over SSH with your normal key, as `OPERATOR_USER@TOWER_ADDR`, using `sudo -n cat`.

## Step 3: verify the cluster with raw Slurm

```bash
# on the tower, as yourself (not root)
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
2. It writes a dedicated ssh config, used through `-F`, that logs in as `xclslurm`.
3. It authorises the key on the tower in the root-owned `/etc/ssh/xcl-slurm-test/authorized_keys` as `restrict,from="<asus>",expiry-time="<+90 days>",command="/usr/local/libexec/xcl-slurm-gate"`. Re-run the step to renew it.
4. It refuses to finish unless `sinfo` works and `id` is refused.

The adapter profile passes the same `-F` file, and the adapter refuses any remote run without one. **Do not add an alias to `~/.ssh/config` instead.** IdentityFile entries accumulate across every matching Host block, so a `Host *` key is offered first. On the 2026-11-07 run, the tower accepted that unrestricted key and `id` ran.

The gate admits only the exact argv shapes the adapter sends, options and order included:
- `sinfo --version`, and `sinfo -N --noheader [-p <partition>] -o <format>`;
- `squeue -j <id> --noheader -o <format>`;
- `sacct -j <id> [-X] --noheader --parsable2 [--noconvert] -o <fields>`, optionally after `env SLURM_TIME_FORMAT=%s`;
- `scontrol ping` and `scontrol --version`;
- `sbatch --parsable`, with the script on stdin. Every `#SBATCH` line must be one the adapter writes, `--output`/`--error` must be the job's own files in the jobs directory, and sbatch runs from that directory;
- `scancel [-v] <id>`;
- `tail -c N -- /var/lib/xcl-slurm-test/jobs/xcl-<job>-<id>.out|err`, read by the gate itself with `O_NOFOLLOW`.

Quotes and shell metacharacters are refused, not interpreted. It logs every allow and deny to the journal under the tag `xcl-slurm-gate`, with each admitted script's sha256. Jobs run as `xclslurm`, so a script's body still runs arbitrary code as that account; isolating it is Phase 3 of the plan.

## Step 5: run the adapter validation suite

```bash
# on the ASUS, from the repo root
python3 hpc/slurm-local/tests/validate_adapter.py
# one case:      --only GPU-1
# with the reconcile loop against the pytest DB (never prod):
( set -a; source .env.test; set +a; export XCELSIOR_SKIP_DOTENV=1
  .venv/bin/python hpc/slurm-local/tests/validate_adapter.py --with-db )
```

It takes about 5 minutes. Results go to `hpc/slurm-local/results/<run-id>/`:
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
2. remove adapter key lines an older kit wrote into your `authorized_keys`;
3. delete the ufw rules;
4. remove recorded files and directories (the gate, the health check, the sshd drop-in, the adapter's authorized_keys), then reload sshd;
5. purge exactly the recorded packages (never `autoremove`);
6. restore any backed-up file;
7. remove package directories that did not exist before (`/var/lib/mariadb`, `/etc/mysql`, munge's state);
8. delete the `slurm`, `munge`, `xclslurm` and `mysql` users;
9. remove `/var/lib/xcl-slurm-test`, including any backups under it.

A copy of the manifest is kept in `/root`. The rehearsal checked this end to end: the package set after rollback was identical to the one before install.

To pause without uninstalling: `sudo ./45-stop.sh tower` and `sudo ./45-stop.sh asus`.

## After the tower's RAM upgrade

The tower goes to at least 32 GB in late November 2026. Slurm can't be told before the RAM is in: a node with less memory than its `RealMemory` registers short and is drained. Afterwards:

1. In `site.env`, set `TOWER_MEM_MB=16384`, half the RAM as today (the rest stays with Docker, the runners and the GPU services). Then rsync the kit to the tower.
2. On the tower: `sudo ./10-install-tower.sh preflight` (it refuses a value above the installed RAM), then `sudo ./10-install-tower.sh configure`.
3. Restart rather than `scontrol reconfigure`, which can fail with cgroup v2 on 25.11.2: `sudo systemctl restart slurmctld slurmd` on the tower, then `sudo systemctl restart slurmd` on the asus.
4. `scontrol show node tower` shows the new `RealMemory`; re-run `30-verify-cluster.sh`.

## Backups

```bash
# on the tower
sudo ./50-backup.sh    # → /var/lib/xcl-slurm-test/backups/<UTC time>.tar.gz, mode 0600
```

The archive holds a consistent dump of the accounting database, slurmctld's state, `/etc/slurm` and the munge key, so treat it as a secret. Rollback deletes `/var/lib/xcl-slurm-test`: copy archives off the machine first. Retention inside the database is set in `slurmdbd.conf` (job records 24 months, steps 6 months).

## Risks and how the kit handles them

- **The tower runs production-adjacent services.** Slurm may allocate at most 8 GB and 8 CPUs on it in total, and 1 hour per job. Job steps are cgroup-confined, so cores, RAM, swap and devices are all limited. Jobs run as `xclslurm`, which has no sudo and can't read your home. MariaDB has no TCP port. Services don't start at boot. Docker and the runners are snapshotted and compared.
- **The GPU is shared** with services outside Slurm (see Topology). Slurm's device fencing keeps jobs without GRES off it; it can't keep those services off it.
- **Slurm 25.11.2 is behind on security fixes** (see the Phase 0 notes above). Don't give anyone else a way to submit jobs here.
- **The tower's disk is 93% full** (130 GB free). The install adds about 300 MB, and logs are small. Preflight requires 3 GB free.
- **The ASUS crashes several times a day** (pstore panics; failing RAM or CPU), and it runs with heavy swap. With `ReturnToService=2` it rejoins once slurmd re-registers. Don't read a case that failed because the node went DOWN as an adapter failure. Check `sinfo -R`, then re-run that case with `--only`. RealMemory is set to 4 GB.
- **Munge is clock-sensitive** (5-minute TTL). Preflight requires NTP sync on both nodes.
- **On the ASUS, Slurm jobs can still open the RTX 2060.** Slurm only fences devices it manages as GRES. That's harmless for the CPU tests, but don't treat the ASUS as GPU-isolated.
- **The ASUS is a laptop.** Preflight refuses to install unless the sleep, suspend and hibernate targets are masked.

## Troubleshooting

| Symptom | Check |
|---|---|
| `asus` is `down*` / `unknown` | `sudo systemctl status slurmd` on the ASUS; ufw rule present; `munge -n \| ssh <you>@<tower-addr> unmunge` |
| a node `drain`ed with `xcl-health:` | the reason names the check; `journalctl -t xcl-node-health`; it resumes on its own once the check passes |
| tower `drain` with `gres/gpu count` | `sudo slurmd -G` on the tower; `nvidia-smi`; the NVML plugin installed |
| `Invalid account or account/partition combination` | not expected (no `AccountingStorageEnforce`); check slurm.conf was not edited |
| `sacct` says accounting is disabled | `systemctl status slurmdbd`; `sacctmgr show cluster` lists `xcl-local` |
| `Security violation` in slurmd log | the `slurm` uid differs between nodes (`id slurm` on both must be 970) |
| suite: `transport_error` | `ssh -F ~/.ssh/xcl_slurm_adapter.conf xcl-tower sinfo --version`; `/etc/ssh/xcl-slurm-test/authorized_keys` on the tower has the asus's `from=` and an `expiry-time` not yet passed |
