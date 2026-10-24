#!/usr/bin/env python3
"""Private, checksum-verified local WAL spool for PostgreSQL archive/restore.

Each object is published as an atomic directory containing data and its receipt.
This is a local spool, not an encrypted independent offsite backup. It never
deletes archived WAL; retention needs a verified base-backup recovery window.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

NAME = re.compile(r"(?:[0-9A-F]{24}(?:\.[0-9A-F]{8}\.backup)?|[0-9A-F]{8}\.history)\Z")


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def checked_file(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("archive input must be a regular file")
    return os.fdopen(fd, "rb")


def digest(path):
    with checked_file(path) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def namespace(directory, system_id):
    if not re.fullmatch(r"[0-9]{1,20}", system_id):
        raise ValueError("a PostgreSQL system identifier is required")
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_mode & 0o022:
        raise ValueError("WAL repository must be a private non-symlink directory")
    root = directory / system_id
    root.mkdir(mode=0o700, exist_ok=True)
    if root.is_symlink() or root.stat().st_mode & 0o022:
        raise ValueError("WAL namespace must be private")
    return root


def verify(root, name):
    if not NAME.fullmatch(name):
        raise ValueError("invalid PostgreSQL archive filename")
    obj = root / name
    if obj.is_symlink() or not obj.is_dir():
        raise ValueError("archive object unavailable")
    with checked_file(obj / "receipt.json") as source:
        receipt = json.load(source)
    data = obj / "data"
    if receipt != {"name": name, "bytes": data.stat().st_size, "sha256": digest(data)}:
        raise ValueError("WAL archive checksum or receipt mismatch")
    return data, receipt


def archive(directory, system_id, source, name):
    if not NAME.fullmatch(name):
        raise ValueError("invalid PostgreSQL archive filename")
    root = namespace(directory, system_id)
    target = root / name
    with tempfile.TemporaryDirectory(prefix=".pending-", dir=root) as temporary:
        staging = Path(temporary) / "object"
        staging.mkdir(mode=0o700)
        with checked_file(source) as src, (staging / "data").open("xb") as dst:
            os.chmod(staging / "data", 0o600)
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        receipt = {
            "name": name,
            "bytes": (staging / "data").stat().st_size,
            "sha256": digest(staging / "data"),
        }
        with (staging / "receipt.json").open("x") as out:
            os.chmod(staging / "receipt.json", 0o600)
            json.dump(receipt, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        sync_directory(staging)
        try:
            os.rename(staging, target)
        except OSError:
            # PG retries after uncertain completion. An identical fully verified
            # object is safe; an existing different object must never be replaced.
            _, prior = verify(root, name)
            if prior != receipt:
                raise ValueError("archive filename collision; original retained")
        sync_directory(root)
    return receipt


def restore(directory, system_id, name, destination):
    root = namespace(directory, system_id)
    source, receipt = verify(root, name)
    destination = Path(destination)
    with tempfile.NamedTemporaryFile(
        prefix=".wal-restore-", dir=destination.parent, delete=False
    ) as out:
        temporary = Path(out.name)
        try:
            with checked_file(source) as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
            if digest(temporary) != receipt["sha256"]:
                raise ValueError("WAL changed during restore; destination retained")
            os.replace(temporary, destination)
            sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("archive", "restore"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--system-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--path", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    try:
        if args.action == "archive":
            archive(args.directory, args.system_id, args.path, args.name)
        else:
            restore(args.directory, args.system_id, args.name, args.path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"WAL {args.action} refused ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
