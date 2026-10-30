#!/usr/bin/env python3
"""Host deploy entrypoint: inject migration credentials into one-shot jobs only."""

import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MAINTENANCE_ENV = Path("/etc/xcelsior/db-maintenance.env")
MIGRATOR_KEY = "XCELSIOR_POSTGRES_DSN_MIGRATOR"


def run(project=ROOT, credential_file=MAINTENANCE_ENV, *, target="head", import_compute_catalog=False) -> int:
    if target != "head" and not re.fullmatch(r"[0-9]{3}", target):
        raise RuntimeError("migration target must be head or an absolute three-digit revision")
    if import_compute_catalog and target != "head":
        raise RuntimeError("compute catalog import requires migration head")
    env = os.environ.copy()
    if credential_file.exists():
        if credential_file.stat().st_mode & 0o077:
            raise RuntimeError("database maintenance credential file must be private")
        # Read the one allowed secret as data. Never execute a dotenv file or
        # accept unrelated environment changes from the maintenance job.
        for raw in credential_file.read_text().splitlines():
            key, separator, value = raw.partition("=")
            if separator and key.strip() == MIGRATOR_KEY:
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                env[MIGRATOR_KEY] = value
    config = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if config.returncode:
        raise RuntimeError("could not validate the migration Compose configuration")
    api_env = json.loads(config.stdout)["services"]["api"]["environment"]
    strict = str(api_env.get("XCELSIOR_DB_ROLES_REQUIRED", "0")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    credential = env.get(MIGRATOR_KEY)
    if strict and not credential:
        raise RuntimeError("strict-role deploy requires the private migration credential")
    command = ["docker", "compose", "run", "--rm", "--no-deps", "-T"]
    if credential:
        # Key-only argv: the value stays in the child's one-shot environment.
        command.extend(["-e", MIGRATOR_KEY, "-e", "XCELSIOR_DB_ROLES_REQUIRED=1"])
    migration = [*command, "api", "python", "-m", "alembic", "upgrade", target]
    result = subprocess.run(migration, cwd=project, env=env, check=False).returncode
    if result or not import_compute_catalog:
        return result
    print("Importing verified Stripe compute Price identities...", flush=True)
    catalog = [*command, "api", "python", "api/scripts/sync_stripe_service_tiers.py",
               "--output", "billing/stripe_catalog.json", "--import-dsn-env",
               MIGRATOR_KEY if credential else "XCELSIOR_POSTGRES_DSN"]
    return subprocess.run(catalog, cwd=project, env=env, check=False).returncode


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", default="head", help="absolute revision for staged writer handoffs"
    )
    parser.add_argument("--import-compute-catalog", action="store_true")
    args = parser.parse_args()
    try:
        raise SystemExit(run(target=args.target, import_compute_catalog=args.import_compute_catalog))
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
