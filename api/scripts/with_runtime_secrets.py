"""Fetch the control-plane runtime secret and exec the requested command."""

import json
import os
from pathlib import Path
import stat
import sys
from urllib.parse import quote

SECRET_ID_FILE = Path("/etc/xcelsior/runtime-secret-id")
BREAK_GLASS_FILE = Path("/run/xcelsior/runtime-break-glass.env")
OBJECT_KEYS = frozenset(
    {
        "XCELSIOR_OBJECT_STORE_ACCESS_KEY",
        "XCELSIOR_OBJECT_STORE_SECRET_KEY",
        "XCELSIOR_OBJECT_STORE_SESSION_TOKEN",
    }
)
REQUIRED_KEYS = frozenset({"XCELSIOR_POSTGRES_PASSWORD", "XCELSIOR_SECRETS_KEY"})
ALLOWED_KEYS = REQUIRED_KEYS | OBJECT_KEYS


def _validate(raw: dict) -> dict[str, str]:
    if (
        not isinstance(raw, dict)
        or not REQUIRED_KEYS <= raw.keys() <= ALLOWED_KEYS
        or any(not isinstance(value, str) or not value or "\0" in value for value in raw.values())
        or bool(raw.get("XCELSIOR_OBJECT_STORE_ACCESS_KEY"))
        != bool(raw.get("XCELSIOR_OBJECT_STORE_SECRET_KEY"))
        or (
            raw.get("XCELSIOR_OBJECT_STORE_SESSION_TOKEN")
            and not raw.get("XCELSIOR_OBJECT_STORE_ACCESS_KEY")
        )
    ):
        raise ValueError("Runtime secret has invalid keys or values")
    from cryptography.fernet import Fernet

    Fernet(raw["XCELSIOR_SECRETS_KEY"].encode())
    return {
        **raw,
        "XCELSIOR_POSTGRES_DSN_PASSWORD": quote(raw["XCELSIOR_POSTGRES_PASSWORD"], safe=""),
    }


def _read_break_glass() -> dict[str, str]:
    # This file is explicitly selected and lives on tmpfs. Treat it as data.
    fd = os.open(BREAK_GLASS_FILE, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, encoding="utf-8") as file:
        metadata = os.fstat(file.fileno())
        if metadata.st_uid != 0 or stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("Break-glass file must be owned by root with mode 0600")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Break-glass source must be a regular file")
        import time

        if not 0 <= time.time() - metadata.st_mtime < 3600:
            raise ValueError("Break-glass file has expired")
        raw = dict(
            line.strip().split("=", 1)
            for line in file
            if line.strip() and not line.lstrip().startswith("#")
        )
    return _validate(raw)


def runtime_secrets(secret_id: str) -> dict[str, str]:
    import boto3
    from botocore.config import Config

    client = boto3.client(
        "secretsmanager",
        config=Config(connect_timeout=4, read_timeout=8, retries={"total_max_attempts": 3}),
    )
    return _validate(json.loads(client.get_secret_value(SecretId=secret_id)["SecretString"]))


def main() -> int:
    if len(sys.argv) < 2:
        print("A runtime command is required", file=sys.stderr)
        return 2
    try:
        if os.environ.get("XCELSIOR_USE_RUNTIME_BREAK_GLASS") == "1":
            injected = _read_break_glass()
        else:
            secret_id = os.environ.get("XCELSIOR_RUNTIME_SECRET_ID")
            if not secret_id and SECRET_ID_FILE.exists():
                secret_id = SECRET_ID_FILE.read_text(encoding="utf-8").strip()
                if not secret_id:
                    raise ValueError("Runtime secret ID file is empty")
            injected = runtime_secrets(secret_id) if secret_id else {}
        env = os.environ.copy()
        if injected:
            # Do not mix a fresh role with stale static credentials.
            for key in OBJECT_KEYS:
                env[key] = ""
            env.update(injected)
        os.execvpe(sys.argv[1], sys.argv[1:], env)
    except Exception as exc:
        print(f"Runtime command could not start ({type(exc).__name__})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
