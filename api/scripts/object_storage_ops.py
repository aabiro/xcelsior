"""Upload a database backup or fetch the latest verified Postgres dump."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import sys
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError

from workers.object_storage import ObjectStorageError, ObjectStoreBackend

BACKUP_SUFFIX = {"postgres": ".dump", "redis": ".rdb"}


def upload_backup(store: ObjectStoreBackend, kind: str, source: Path) -> None:
    stem = source.stem
    if source.suffix != BACKUP_SUFFIX[kind] or not re.fullmatch(r"[A-Za-z0-9_.-]+", stem):
        raise ObjectStorageError("Invalid backup filename")
    key = f"backups/{kind}/{stem}-{uuid4().hex}{source.suffix}"
    store.upload(key, source)
    print(f"Uploaded {kind} backup {key}")


def fetch_latest_postgres(store: ObjectStoreBackend, destination_dir: Path) -> Path:
    db_name = os.environ.get("XCELSIOR_OBJECT_STORE_BACKUP_DB", "xcelsior")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", db_name):
        raise ObjectStorageError("Invalid backup database name")
    prefix = f"backups/postgres/{db_name}_"
    pattern = re.compile(re.escape(prefix) + r"[0-9]{8}_[0-9]{6}-[a-f0-9]{32}\.dump")
    keys = [key for key in store.keys(prefix) if pattern.fullmatch(key)]
    if not keys:
        raise ObjectStorageError("No Postgres backup found in object storage")
    key = max(keys)
    target = destination_dir / Path(key).name
    stored = store.download(key, target)
    target.with_suffix(".dump.sha256").write_text(
        f"{stored.sha256}  {target.name}\n", encoding="ascii"
    )
    return target


def backup_redis(store: ObjectStoreBackend, url: str) -> None:
    config = urlsplit(url)
    if config.scheme not in {"redis", "rediss"} or not config.hostname:
        raise ObjectStorageError("Invalid Redis backup URL")
    command = ["redis-cli", "-h", config.hostname, "-p", str(config.port or 6379)]
    if config.scheme == "rediss":
        command.append("--tls")
    if config.username:
        command.extend(("--user", unquote(config.username)))
    child_env = os.environ.copy()
    child_env.pop("REDISCLI_AUTH", None)
    if config.password:
        child_env["REDISCLI_AUTH"] = unquote(config.password)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with TemporaryDirectory(prefix="xcelsior-redis-backup-") as directory:
        snapshot = Path(directory) / f"redis_{timestamp}.rdb"
        for args in (
            [*command, "--rdb", str(snapshot)],
            ["redis-check-rdb", str(snapshot)],
        ):
            subprocess.run(args, env=child_env, timeout=180, check=True, capture_output=True)
        upload_backup(store, "redis", snapshot)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    upload = commands.add_parser("put")
    upload.add_argument("kind", choices=BACKUP_SUFFIX)
    upload.add_argument("file", type=Path)
    fetch = commands.add_parser("latest-postgres")
    fetch.add_argument("directory", type=Path)
    commands.add_parser("backup-redis")
    args = parser.parse_args()
    os.umask(0o077)
    try:
        store = ObjectStoreBackend.from_env()
        if args.action == "put":
            upload_backup(store, args.kind, args.file)
        elif args.action == "backup-redis":
            backup_redis(store, os.environ["XCELSIOR_REDIS_BACKUP_URL"])
        else:
            print(fetch_latest_postgres(store, args.directory))
    except (
        BotoCoreError,
        ClientError,
        KeyError,
        ObjectStorageError,
        OSError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Object storage {args.action} failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
