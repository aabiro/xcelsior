"""The xcl-local SSH gate: exactly the adapter's command shapes, nothing else.

The gate (hpc/slurm-local/scripts/xcl-slurm-gate) is the only thing standing
between the adapter's restricted key and a shell on the cluster. These tests
pin both directions: every command the adapter really sends for the lab
profile is accepted, and the bypass shapes found in the Phase 0 security
review are refused (`scontrol -M show update`, `squeue -a`, `-i`, env-prefix
tricks, symlink swaps, #SBATCH --output into ~/.ssh).
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from importlib.machinery import SourceFileLoader

import pytest

import slurm_adapter as _sa

ROOT = pathlib.Path(__file__).resolve().parent.parent
KIT = ROOT / "hpc" / "slurm-local"
GATE_PATH = KIT / "scripts" / "xcl-slurm-gate"


def _load_gate():
    loader = SourceFileLoader("xcl_slurm_gate", str(GATE_PATH))
    spec = importlib.util.spec_from_loader("xcl_slurm_gate", loader)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


gate = _load_gate()
JOBS = gate.JOBS_DIR


@pytest.fixture
def lab_profile(monkeypatch):
    """The kit's own profile file, so a drift between it and the gate fails here."""
    monkeypatch.setenv(_sa.PROFILES_FILE_ENV, str(KIT / "conf" / "xcelsior-profiles.json"))
    for var in ("XCELSIOR_SLURM_SSH_HOST", "XCELSIOR_SLURM_SSH_OPTIONS", "SLURM_ACCOUNT"):
        monkeypatch.delenv(var, raising=False)
    return "xcl-local"


class _Remote:
    """subprocess.run stand-in for ssh: records the remote command and stdin."""

    def __init__(self):
        self.sent = []

    def __call__(self, argv, *a, input=None, **kw):
        cmd = argv[-1]
        self.sent.append((cmd, input))
        out = ""
        if cmd.startswith("sbatch"):
            out = "41\n"
        elif cmd.startswith("scontrol ping"):
            out = "Slurmctld(primary) at tower is UP\n"
        return subprocess.CompletedProcess(argv, 0, out, "")


def _job(**kw):
    job = {"job_id": "g-1", "name": "probe", "priority": 0, "tier": "free", "num_gpus": 0}
    job.update(kw)
    return job


class TestAdapterAndGateAgree:
    def test_every_command_the_adapter_sends_is_accepted(self, lab_profile, monkeypatch):
        remote = _Remote()
        monkeypatch.setattr(subprocess, "run", remote)
        runner = _sa.get_runner(_sa._get_profile(lab_profile))
        assert runner.remote and "-F" in runner.ssh_options

        _sa.is_slurm_available(runner)
        _sa.controller_health(profile_name=lab_profile)
        _sa.get_cluster_info(runner)
        _sa.get_cluster_inventory(lab_profile)
        _sa.get_cluster_inventory(lab_profile, partition="xcl")
        _sa.submit_to_slurm(_job(command="hostname"), profile_name=lab_profile)
        _sa.get_slurm_job_status("41", profile_name=lab_profile)  # squeue, then sacct
        _sa.get_slurm_job_accounting("41", profile_name=lab_profile)
        _sa.get_slurm_job_output("41", "g-1", profile_name=lab_profile)
        _sa.get_slurm_job_output("41", "g-1", profile_name=lab_profile, stream="err")
        _sa.cancel_slurm_job("41", profile_name=lab_profile)

        progs = {
            cmd.split()[0] if not cmd.startswith("env ") else cmd.split()[2]
            for cmd, _ in remote.sent
        }
        assert progs == {"sinfo", "scontrol", "sbatch", "squeue", "sacct", "tail", "scancel"}
        for cmd, stdin in remote.sent:
            gate.parse(cmd)  # raises Deny on any mismatch
            if stdin is not None:
                gate.check_script(stdin)

    @pytest.mark.parametrize(
        "job",
        [
            _job(),
            _job(num_gpus=1, command="nvidia-smi -L"),
            _job(num_gpus=2, command="hostname"),  # multi-node on the lab profile
            _job(nodelist="asus", cpus=2, mem_gb=1, walltime="00:05:00"),
            _job(name='x"; touch /tmp/p; echo "', tier="$(id)"),
        ],
    )
    def test_every_script_shape_passes_the_directive_check(self, lab_profile, monkeypatch, job):
        monkeypatch.setenv("SLURM_ACCOUNT", "lab")
        script, _ = _sa.xcelsior_job_to_sbatch(job, lab_profile)
        gate.check_script(script)


class TestRefusedCommands:
    @pytest.mark.parametrize(
        "cmd",
        [
            "",
            "id",
            "bash -c id",
            "sinfo --version; id",
            "sinfo $(id)",
            "sinfo  --version",
            "squeue -j 1 --noheader -o '%i'",
            # scontrol: only ping and --version; option values can't smuggle verbs
            "scontrol -M show update NodeName=tower State=DOWN",
            "scontrol -u show update",
            "scontrol show job 1",
            "scontrol show config",
            "scontrol update NodeName=tower State=DOWN",
            # read commands: exact shapes only
            "squeue",
            "squeue -a",
            "squeue -j 1 --noheader -o %i -a",
            "squeue -i 5 -j 1 --noheader -o %i",
            "squeue -j 1 --noheader -o %i --iterate=1",
            "squeue -u root --noheader -o %i",
            "sinfo -i 1",
            "sinfo -N --noheader -o %N -i 1",
            "sacct -a",
            "sacct -j 1 --noheader --parsable2 -o JobID -a",
            "sacct -j 1 --noheader --parsable2 -o JobID,User,Comment%999",
            # env prefix: one variable, one program
            "env SLURM_TIME_FORMAT=%s sbatch --parsable",
            "env FOO=1 sacct -j 1 --noheader --parsable2 -o JobID",
            "env SLURM_TIME_FORMAT=%s env X=1 sacct -j 1 --noheader --parsable2 -o JobID",
            "env SLURM_TIME_FORMAT=evil sacct -j 1 --noheader --parsable2 -o JobID",
            "env -i sinfo --version",
            "env LD_PRELOAD=/tmp/x.so sinfo --version",
            # writes
            "sbatch",
            "sbatch --parsable --wrap=id",
            "sbatch --parsable /tmp/script.sh",
            "scancel 1 2",
            "scancel -u aaryn",
            "scancel --signal=KILL 1",
            # output reads: the jobs dir and the adapter's file names only
            "tail -c 10 -- /etc/passwd",
            f"tail -c 10 -- {JOBS}/../../../etc/passwd",
            f"tail -c 10 -- {JOBS}/notes.txt",
            f"tail -c 10 -- {JOBS}/xcl-a-1.out extra",
            f"tail -n 10 -- {JOBS}/xcl-a-1.out",
            f"cat {JOBS}/xcl-a-1.out",
        ],
    )
    def test_refused(self, cmd):
        with pytest.raises(gate.Deny):
            gate.parse(cmd)


def _script(*directives, body="echo hi\n"):
    head = [
        "#!/bin/bash",
        "#SBATCH --job-name=xcl-a",
        f"#SBATCH --output={JOBS}/xcl-a-%j.out",
        f"#SBATCH --error={JOBS}/xcl-a-%j.err",
        *directives,
    ]
    return "\n".join(head) + "\n" + body


class TestScriptDirectives:
    def test_minimal_script_passes(self):
        gate.check_script(_script())

    @pytest.mark.parametrize(
        "text",
        [
            _script("#SBATCH --chdir=/tmp"),
            _script("#SBATCH --uid=0"),
            _script("#SBATCH --reservation=other"),
            _script("#SBATCH --container=/x"),
            _script("#SBATCH --nodes 1"),  # space form
            _script("#SBATCH --nodes=1 --ntasks=2"),  # two options on one line
            _script("#sbatch --chdir=/"),
            _script(" #SBATCH --chdir=/"),
            _script("#SBATCH --job-name=xcl-b"),  # repeated
            _script(body="echo hi\n#SBATCH --chdir=/\n"),  # after the first command
            "\n".join(  # output aimed at the service account's keys
                [
                    "#!/bin/bash",
                    "#SBATCH --job-name=xcl-a",
                    "#SBATCH --output=/home/xclslurm/.ssh/authorized_keys",
                    f"#SBATCH --error={JOBS}/xcl-a-%j.err",
                ]
            ),
            "\n".join(  # output for a different job name
                [
                    "#!/bin/bash",
                    "#SBATCH --job-name=xcl-a",
                    f"#SBATCH --output={JOBS}/xcl-b-%j.out",
                    f"#SBATCH --error={JOBS}/xcl-a-%j.err",
                ]
            ),
            "#!/bin/bash\n#SBATCH --job-name=xcl-a\n",  # no --output/--error
            "echo no shebang\n",
        ],
    )
    def test_refused(self, text):
        with pytest.raises(gate.Deny):
            gate.check_script(text)


class TestTailReads:
    @pytest.fixture
    def jobs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gate, "JOBS_DIR", str(tmp_path))
        return tmp_path

    def test_reads_the_last_bytes(self, jobs):
        (jobs / "xcl-a-1.out").write_bytes(b"0123456789")
        assert gate.read_tail(f"{jobs}/xcl-a-1.out", 4) == b"6789"
        assert gate.read_tail(f"{jobs}/xcl-a-1.out", 100) == b"0123456789"

    def test_symlink_swapped_in_is_not_followed(self, jobs, tmp_path_factory):
        secret = tmp_path_factory.mktemp("elsewhere") / "secret"
        secret.write_text("private")
        (jobs / "xcl-a-1.out").symlink_to(secret)
        with pytest.raises(OSError):
            gate.read_tail(f"{jobs}/xcl-a-1.out", 100)

    def test_fifo_is_refused_without_blocking(self, jobs):
        os.mkfifo(jobs / "xcl-a-1.out")
        with pytest.raises(gate.Deny):
            gate.read_tail(f"{jobs}/xcl-a-1.out", 100)


class TestGateProcess:
    """Run the installed shape: sshd sets SSH_ORIGINAL_COMMAND and execs the gate."""

    def _run(self, cmd, stdin=""):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "SSH_ORIGINAL_COMMAND": cmd}
        return subprocess.run(
            [sys.executable, "-I", str(GATE_PATH)],
            input=stdin,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_refusal_exits_126_with_the_marker_the_installer_checks(self):
        r = self._run("id")
        assert r.returncode == gate.DENY_EXIT
        assert "xcl-slurm-gate: refused" in r.stderr

    def test_bad_script_never_reaches_sbatch(self):
        r = self._run("sbatch --parsable", _script("#SBATCH --chdir=/"))
        assert r.returncode == gate.DENY_EXIT
        assert "--chdir" in r.stderr

    def test_profile_file_matches_the_gate_paths(self):
        prof = json.loads((KIT / "conf" / "xcelsior-profiles.json").read_text())["profiles"]
        assert prof["xcl-local"]["output_dir"] == gate.JOBS_DIR
