"""Real PostgreSQL base backup, WAL replay and named recovery-point proof."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

import psycopg
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "api/scripts/wal_archive.py"
spec = importlib.util.spec_from_file_location("wal_archive", SCRIPT)
wal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wal)
SEGMENT = "000000010000000000000001"


def test_archive_retry_collision_corruption_and_private_files(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"recorded WAL fixture")
    repository = tmp_path / "archive"
    receipt = wal.archive(repository, "123", source, SEGMENT)
    assert wal.archive(repository, "123", source, SEGMENT) == receipt
    obj = repository / "123" / SEGMENT
    assert obj.stat().st_mode & 0o777 == 0o700
    assert (obj / "data").stat().st_mode & 0o777 == 0o600
    destination = tmp_path / "restore"
    wal.restore(repository, "123", SEGMENT, destination)
    assert destination.read_bytes() == source.read_bytes()
    source.write_bytes(b"different WAL")
    with pytest.raises(ValueError, match="collision"):
        wal.archive(repository, "123", source, SEGMENT)
    (obj / "data").write_bytes(b"corrupt")
    before = destination.read_bytes()
    with pytest.raises(ValueError, match="checksum"):
        wal.restore(repository, "123", SEGMENT, destination)
    assert destination.read_bytes() == before


@pytest.mark.parametrize(
    "name",
    [
        SEGMENT,
        SEGMENT + ".partial",
        SEGMENT + ".00000028.backup",
        "00000002.history",
    ],
)
def test_archive_accepts_every_name_postgres_archives(tmp_path, name):
    # PostgreSQL archives the old timeline's last segment as `.partial` after a
    # recovery or promotion; refusing it stalls every later segment.
    source = tmp_path / "source"
    source.write_bytes(b"wal")
    wal.archive(tmp_path / "repo", "123", source, name)
    destination = tmp_path / "restored"
    wal.restore(tmp_path / "repo", "123", name, destination)
    assert destination.read_bytes() == b"wal"


@pytest.mark.parametrize(
    "name", ["../escape", "/tmp/WAL", "", "00000001.history/escape", SEGMENT + ".partial/x"]
)
def test_archive_refuses_untrusted_names(tmp_path, name):
    with pytest.raises(ValueError):
        wal.archive(tmp_path / "repo", "123", tmp_path / "source", name)


def test_archive_refuses_symlink_source_and_namespace(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"wal")
    link = tmp_path / "link"
    link.symlink_to(source)
    with pytest.raises(OSError):
        wal.archive(tmp_path / "repo", "123", link, SEGMENT)
    directory = tmp_path / "outside"
    directory.mkdir()
    (tmp_path / "repo" / "123").rmdir()
    (tmp_path / "repo" / "123").symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError, match="private"):
        wal.archive(tmp_path / "repo", "123", source, SEGMENT)


@pytest.mark.timeout(120)
def test_real_base_backup_replays_only_through_the_named_restore_point(tmp_path):
    # This creates its own checksum-enabled cluster under the security runner's
    # scratch space. It never connects to or reconfigures an existing cluster.
    assert os.environ["XCELSIOR_SKIP_DOTENV"] == "1"
    if "xcl-sec-pg-" not in str(tmp_path):
        # Only `api/scripts/run_security_gates.py` provides that scratch space,
        # and it fails any run with a skip, so this cannot be skipped there.
        pytest.skip("runs only under api/scripts/run_security_gates.py")
    binary = Path(shutil.which("pg_config")).resolve().parent
    if not (binary / "initdb").exists():
        binary = Path("/usr/lib/postgresql/16/bin")
    assert (binary / "initdb").is_file()
    source, replica, backup = [tmp_path / name for name in ("source", "restored", "base")]
    # Unix socket names have a short kernel limit; pytest's nested scratch path
    # is too long. Only these uniquely owned mode-0700 socket directories use /tmp.
    short_sockets = tempfile.TemporaryDirectory(prefix="xcl-pitr-", dir="/tmp")
    source_socket = Path(short_sockets.name) / "source"
    replica_socket = Path(short_sockets.name) / "restored"
    repository = tmp_path / "archive"
    for directory in (source_socket, replica_socket, repository):
        directory.mkdir(mode=0o700)
    child_env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}

    def command(name, *args, check=True):
        result = subprocess.run(
            [str(binary / name), *map(str, args)],
            env=child_env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if check:
            logs = "\n".join(path.read_text()[-4000:] for path in tmp_path.glob("*.log"))
            assert result.returncode == 0, (result.stdout + result.stderr + logs)[-6000:]
        return result

    source_started = replica_started = False
    try:
        command("initdb", "-D", source, "-U", "recovery_test", "--auth=trust", "--data-checksums")
        # Read the actual new cluster's identity, never reuse another cluster's
        # WAL namespace even if its segment names happen to be the same.
        control = command("pg_controldata", source).stdout
        system_id = next(
            line.split(":", 1)[1].strip()
            for line in control.splitlines()
            if line.startswith("Database system identifier:")
        )

        def archive_command(action):
            return shlex.join(
                [
                    sys.executable,
                    str(SCRIPT),
                    action,
                    "--directory",
                    str(repository),
                    "--system-id",
                    system_id,
                    "--name",
                    "%f",
                    "--path",
                    "%p",
                ]
            )

        def setting(value):
            return "'" + str(value).replace("'", "''") + "'"

        with (source / "postgresql.conf").open("a") as config:
            config.write(
                "\nlisten_addresses=''\nmax_connections=20\nshared_buffers='16MB'\nwal_level=replica\narchive_mode=on\n"
            )
            config.write("unix_socket_directories=" + setting(source_socket) + "\n")
            config.write("archive_command=" + setting(archive_command("archive")) + "\n")
        command("pg_ctl", "-D", source, "-l", tmp_path / "source.log", "-w", "start")
        source_started = True
        dsn = f"host={source_socket} user=recovery_test dbname=postgres"
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE TABLE recovery_markers(marker text PRIMARY KEY)")
            conn.execute("INSERT INTO recovery_markers VALUES('base')")
        command(
            "pg_basebackup",
            "-h",
            source_socket,
            "-U",
            "recovery_test",
            "-D",
            backup,
            "--wal-method=stream",
            "--checkpoint=fast",
            "--manifest-checksums=SHA256",
        )
        command("pg_verifybackup", backup)
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("INSERT INTO recovery_markers VALUES('keep-after-backup')")
            conn.execute("SELECT pg_create_restore_point('qualified_recovery_target')")
            conn.execute("INSERT INTO recovery_markers VALUES('exclude-after-target')")
            segment = conn.execute("SELECT pg_walfile_name(pg_current_wal_lsn())").fetchone()[0]
            conn.execute("SELECT pg_switch_wal()")
        deadline = time.monotonic() + 20
        while not (repository / system_id / segment).is_dir():
            assert time.monotonic() < deadline, (tmp_path / "source.log").read_text()[-3000:]
            time.sleep(0.1)
        wal.verify(repository / system_id, segment)
        shutil.copytree(backup, replica)
        # Force recovery to obtain WAL from the archived objects. A base backup
        # that merely boots with its bundled WAL is not a PITR proof.
        for item in (replica / "pg_wal").iterdir():
            if item.is_file():
                item.unlink()
        with (replica / "postgresql.conf").open("a") as config:
            config.write("\narchive_mode=off\narchive_command=''\n")
            config.write("unix_socket_directories=" + setting(replica_socket) + "\n")
            config.write("restore_command=" + setting(archive_command("restore")) + "\n")
            config.write(
                "recovery_target_name='qualified_recovery_target'\nrecovery_target_action='promote'\n"
            )
        (replica / "recovery.signal").touch()
        command("pg_ctl", "-D", replica, "-l", tmp_path / "restore.log", "-w", "start")
        replica_started = True
        with psycopg.connect(
            f"host={replica_socket} user=recovery_test dbname=postgres", autocommit=True
        ) as conn:
            deadline = time.monotonic() + 15
            while conn.execute("SELECT pg_is_in_recovery()").fetchone()[0]:
                assert time.monotonic() < deadline
                time.sleep(0.1)
            assert conn.execute(
                "SELECT marker FROM recovery_markers ORDER BY marker"
            ).fetchall() == [("base",), ("keep-after-backup",)]
            assert conn.execute("SHOW data_checksums").fetchone() == ("on",)
    finally:
        cleanup = []
        for data, started in ((replica, replica_started), (source, source_started)):
            if started or (data / "postmaster.pid").exists():
                result = command("pg_ctl", "-D", data, "-m", "immediate", "-w", "stop", check=False)
                cleanup.append(result.returncode)
        (tmp_path / "cleanup.json").write_text(json.dumps({"stop_exit_codes": cleanup}))
        short_sockets.cleanup()
        assert all(code == 0 for code in cleanup)
