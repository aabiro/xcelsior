"""Stage verified object copies from a quiesced NFS export."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from botocore.exceptions import BotoCoreError, ClientError
import psycopg

from workers.object_storage import ObjectStorageError, ObjectStoreBackend, StoredObject

_VOLUME_ID = re.compile(r"vol-[a-f0-9]{12}")


def _source_members(export: Path, volume_id: str, encrypted: bool) -> list[str]:
    member = f"{volume_id}.img" if encrypted else volume_id
    source = export / member
    if source.is_symlink() or not (source.is_file() if encrypted else source.is_dir()):
        raise ObjectStorageError("The expected volume source is missing or unsafe")
    if encrypted:
        attached = subprocess.run(
            ["losetup", "--associated", str(source)], capture_output=True, check=True, text=True
        )
        if attached.stdout.strip():
            raise ObjectStorageError("Close the LUKS mapping and detach the loop device first")
    members = [member]
    snapshots = export / "_snapshots" / volume_id
    if snapshots.is_symlink() or snapshots.parent.is_symlink():
        raise ObjectStorageError("Unexpected symlink in the snapshot location")
    if snapshots.is_dir():
        members.append(f"_snapshots/{volume_id}")
    return members


def _archive(export: Path, members: list[str], output: Path) -> None:
    subprocess.run(
        [
            "tar",
            "--create",
            "--gzip",
            "--sparse",
            "--acls",
            "--xattrs",
            "--numeric-owner",
            "--one-file-system",
            "--file",
            str(output),
            "--directory",
            str(export),
            "--",
            *members,
        ],
        capture_output=True,
        check=True,
    )


def _volume_row(conn, volume_id: str, *, lock: bool = False) -> tuple[bool, str]:
    row = conn.execute(
        "SELECT encrypted, status FROM volumes WHERE volume_id = %s AND storage_type = 'nfs'"
        + (" FOR UPDATE" if lock else ""),
        (volume_id,),
    ).fetchone()
    if row is None:
        raise ObjectStorageError("No matching NFS volume")
    active = conn.execute(
        "SELECT 1 FROM volume_attachments WHERE volume_id = %s AND detached_at = 0 LIMIT 1",
        (volume_id,),
    ).fetchone()
    if row[1] != "available" or active:
        raise ObjectStorageError("The volume must be available and detached")
    return row


def stage_volume(conn, store: ObjectStoreBackend, export: Path, volume_id: str, scratch: Path):
    if not _VOLUME_ID.fullmatch(volume_id):
        raise ObjectStorageError("Invalid volume ID")
    with conn.transaction():
        encrypted, _ = _volume_row(conn, volume_id)
    members = _source_members(export, volume_id, encrypted)
    archive = scratch / f"{volume_id}.tar.gz"
    _archive(export, members, archive)
    key = f"volumes/{volume_id}/nfs-checkpoints/v1/{uuid4().hex}.tar.gz"
    stored = store.upload(key, archive)
    store.verify(stored)
    with conn.transaction():
        if _volume_row(conn, volume_id, lock=True)[0] != encrypted:
            raise ObjectStorageError("The source changed during staging")
        conn.execute(
            "INSERT INTO volume_object_checkpoints "
            "(volume_id, object_key, bucket, size_bytes, sha256, encrypted) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (volume_id, key, store.bucket, stored.size_bytes, stored.sha256, encrypted),
        )
    return stored


def fetch_volume(
    conn, store: ObjectStoreBackend, volume_id: str, destination_dir: Path
) -> StoredObject:
    if not _VOLUME_ID.fullmatch(volume_id):
        raise ObjectStorageError("Invalid volume ID")
    row = conn.execute(
        "SELECT object_key, size_bytes, sha256 FROM volume_object_checkpoints"
        " WHERE volume_id = %s AND bucket = %s ORDER BY created_at DESC, object_key DESC LIMIT 1",
        (volume_id, store.bucket),
    ).fetchone()
    if row is None:
        raise ObjectStorageError("No verified object copy for that volume and bucket")
    expected = StoredObject(*row)
    if not re.fullmatch(
        rf"volumes/{volume_id}/nfs-checkpoints/v1/[a-f0-9]{{32}}\.tar\.gz", expected.key
    ):
        raise ObjectStorageError("Invalid stored checkpoint key")
    with TemporaryDirectory(dir=destination_dir) as staging:
        target = Path(destination_dir) / f"{volume_id}.tar.gz"
        temporary = Path(staging) / target.name
        if store.download(expected.key, temporary) != expected:
            raise ObjectStorageError("Checkpoint metadata does not match the database")
        os.link(temporary, target)
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--source-quiesced", action="store_true")
    parser.add_argument("--fetch-volume", help="Download one verified archive into --scratch")
    args = parser.parse_args()
    if args.apply and not args.source_quiesced:
        parser.error("--apply requires --source-quiesced after writes are frozen")
    if not args.fetch_volume and not args.export:
        parser.error("--export is required when staging checkpoints")
    if args.fetch_volume and (args.apply or args.export or args.volume or args.source_quiesced):
        parser.error("--fetch-volume takes only --scratch")
    os.umask(0o077)
    try:
        scratch = args.scratch.resolve(strict=True)
        if not scratch.is_dir():
            raise ObjectStorageError("Use a local scratch directory")
        with psycopg.connect(os.environ["XCELSIOR_VOLUME_MIGRATION_DSN"], autocommit=True) as conn:
            if args.fetch_volume:
                store = ObjectStoreBackend.from_env()
                fetch_volume(conn, store, args.fetch_volume, scratch)
                print(f"Verified archive at {scratch / (args.fetch_volume + '.tar.gz')}")
                return 0
            export = args.export.resolve(strict=True)
            if not export.is_dir() or scratch.is_relative_to(export):
                raise ObjectStorageError("Use a separate local scratch directory")
            volume_ids = sorted(set(args.volume)) or [
                row[0]
                for row in conn.execute(
                    "SELECT volume_id FROM volumes WHERE storage_type = 'nfs' AND status != 'deleted'"
                    " ORDER BY volume_id"
                )
            ]
            if not args.apply:
                print(
                    f"{len(volume_ids)} volume(s) selected. Add --apply --source-quiesced to stage."
                )
                return 0
            store = ObjectStoreBackend.from_env()
            for volume_id in volume_ids:
                with TemporaryDirectory(dir=scratch) as directory:
                    stored = stage_volume(conn, store, export, volume_id, Path(directory))
                    print(f"{volume_id}: verified and recorded {stored.key}")
    except (
        BotoCoreError,
        ClientError,
        KeyError,
        ObjectStorageError,
        OSError,
        psycopg.Error,
        subprocess.SubprocessError,
    ) as exc:
        print(f"Volume staging failed ({type(exc).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
