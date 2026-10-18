#!/usr/bin/env python3
"""Reorganize xcelsior repo root into packages. Review before running."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES: list[tuple[str, list[str], str]] = [
    (
        "archive",
        [
            "link-checker-results-2026-07-02.csv",
            "link-checker-results-2026-07-03.csv",
            "UNTESTED_ENDPOINTS.md",
            "HANDOFF.md",
            "XCELSIOR_UI_UX_AUDIT_2026-06-22.md",
            "SPOT_ROLLOUT.md",
            "SPOT_RUNBOOK.md",
            "google_support.md",
            "compute_scores.json",
            "eval-baseline.json",
            ".posthog-events.json",
        ],
        "chore: archive root-level junk and audit docs",
    ),
    (
        "billing",
        [
            "billing.py",
            "money.py",
            "bitcoin.py",
            "stripe_connect.py",
            "stripe_catalog.py",
            "stripe_meters.py",
            "stripe_tax.py",
            "paypal_connect.py",
            "provider_settlement.py",
            "marketplace.py",
            "pricing_reference.py",
            "reserved_pricing.py",
            "spot_pricing.py",
            "config/stripe_catalog.json",
        ],
        "refactor: extract billing package",
    ),
    (
        "workers",
        [
            "worker_agent.py",
            "worker_privileged.py",
            "worker_image_cache.py",
            "worker_luks_volumes.py",
            "worker_nfs.py",
            "worker_nvme_cache.py",
            "bg_worker.py",
            "scheduler.py",
            "spot",
            "spot_metrics.py",
            "inference.py",
            "cloudburst.py",
            "lightning.py",
            "slurm_adapter.py",
            "criu_hosts.py",
            "nvml_telemetry.py",
            "volumes.py",
            "reaper.py",
        ],
        "refactor: extract workers package",
    ),
    (
        "auth",
        [
            "oauth_service.py",
            "oauth_registration.py",
            "oauth_delegation.py",
            "security.py",
        ],
        "refactor: extract auth package",
    ),
    (
        "hosts",
        [
            "host_acceptance.py",
            "host_admission.py",
            "host_attestation.py",
            "host_key_fingerprint.py",
            "host_metadata.py",
            "host_profiles.py",
            "egress_isolation.py",
            "environment_fingerprint.py",
            "instance_credentials.py",
            "registry_health.py",
            "verification.py",
            "verification_sweep.py",
            "ssh_gateway.py",
        ],
        "refactor: extract hosts package",
    ),
    (
        "privacy",
        [
            "privacy.py",
            "privacy_deletion.py",
            "privacy_sinks.py",
            "log_pii_filter.py",
            "abuse_policy.py",
        ],
        "refactor: extract privacy package",
    ),
    (
        "api",
        [
            "api.py",
            "split_api.py",
            "public_http.py",
            "routes",
            "frontend",
            "templates",
            "public",
            "nginx",
            "desktop",
            "wizard",
            "sprites",
            "fern",
            "mcp",
            "mcp-cli",
            "serverless",
            "infra",
            "docs",
            "support_docs",
            "runbooks",
            "scripts",
            "migrations",
            "data",
            "analytics",
            "config",
        ],
        "refactor: extract api package",
    ),
    (
        "core",
        [
            "cli.py",
            "db.py",
            "events.py",
            "chat.py",
            "artifacts.py",
            "inference_store.py",
            "cache_keys.py",
            "env_config.py",
            "structured_logging.py",
            "metrics_catalog.py",
            "slo_definitions.py",
            "sla.py",
            "reputation.py",
            "demo_account.py",
            "user_avatars.py",
            "meta_capi.py",
            "web_push.py",
            "agent_preempt.py",
            "ai_assistant.py",
            "ai_assistant_config.py",
            "fix_sprites.py",
            "provider_expectations.py",
        ],
        "refactor: extract core package",
    ),
]

ROOT_KEEP = {
    ".dockerignore",
    ".env.audit.example",
    ".env.example",
    ".env.paypal-live.example",
    ".env.paypal-sandbox.example",
    ".env.stripe-live.example",
    ".env.worker.example",
    ".github",
    ".gitignore",
    ".pre-commit-config.yaml",
    ".vscode",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "Dockerfile.example",
    "LICENSE",
    "PROJECT.md",
    "README.md",
    "alembic.ini",
    "control_plane",
    "docker-compose.staging.yml",
    "docker-compose.yml",
    "gunicorn.conf.py",
    "llms.txt",
    "pyproject.toml",
    "pyrightconfig.json",
    "reorganize.py",
    "requirements.txt",
    "run-dev.sh",
    "run-tests.sh",
    "setup.sh",
    "tests",
    "uv.lock",
    "xcelsior-api.service",
    "xcelsior-health.service",
    "xcelsior-worker.service",
    "xcelsior_icon_app_gradient.png",
    "xcelsior_icon_gradient.png",
    "xcelsior_icon_regular.png",
}


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize xcelsior repository root into focused package directories."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview planned moves without making filesystem or Git changes (default).",
    )
    mode.add_argument(
        "--apply",
        "--execute",
        dest="apply",
        action="store_true",
        help="Execute git mv operations and stage/commit per stage.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        help="Run only specific stage number(s) (1-8) or package name(s) (e.g. billing).",
    )
    parser.add_argument(
        "--no-commit",
        action="store_true",
        help="When used with --apply/--execute, perform git mv without running git commit.",
    )
    return parser.parse_args(argv)


def select_stages(
    stage_filters: list[str] | None,
) -> list[tuple[int, str, list[str], str]]:
    indexed = [
        (idx, dest, items, msg)
        for idx, (dest, items, msg) in enumerate(STAGES, start=1)
    ]
    if not stage_filters:
        return indexed

    wanted = {token.strip().rstrip("/") for token in stage_filters}
    selected = [
        entry
        for entry in indexed
        if str(entry[0]) in wanted or entry[1] in wanted
    ]
    if not selected:
        valid = ", ".join(f"{i}:{dest}" for i, dest, _, _ in indexed)
        raise SystemExit(f"ERROR: No stages matched {stage_filters!r}. Valid stages: {valid}")
    return selected


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    dry = not args.apply

    if dry:
        print("DRY RUN — no files will be moved (pass --apply or --execute to run)\n")

    selected = select_stages(args.stage)

    # Safety check: ensure no root-keep item appears in the move map.
    mapped_all = {item for _, items, _ in STAGES for item in items}
    keep_violations = sorted(mapped_all & ROOT_KEEP)
    if keep_violations:
        print(f"ERROR: Root-keep violation(s) in move map: {keep_violations}", file=sys.stderr)
        return 1

    total_planned = 0
    total_already_moved = 0
    total_missing = 0
    total_collisions = 0

    for stage_num, dest_name, items, commit_msg in selected:
        dest_dir = ROOT / dest_name
        movable: list[str] = []
        already_moved: list[str] = []
        missing: list[str] = []
        collisions: list[str] = []

        for item in items:
            src_exists = (ROOT / item).exists()
            dst_exists = (dest_dir / Path(item).name).exists()
            if src_exists and dst_exists:
                collisions.append(item)
            elif src_exists:
                movable.append(item)
            elif dst_exists:
                already_moved.append(item)
            else:
                missing.append(item)

        print(f"=== Stage {stage_num}: {dest_name}/ ({len(movable)}/{len(items)} items to move) ===")
        for item in already_moved:
            print(f"  INFO already at destination: {dest_name}/{Path(item).name}")
        for item in missing:
            print(f"  WARN missing source: {item}")
        for item in collisions:
            print(f"  WARN destination collision, skipping: {dest_name}/{Path(item).name}")

        total_already_moved += len(already_moved)
        total_missing += len(missing)
        total_collisions += len(collisions)

        if not movable:
            print("  (nothing to move in this stage)\n")
            continue

        total_planned += len(movable)

        if dry:
            for item in movable:
                target_rel = f"{dest_name}/{Path(item).name}"
                print(f"  git mv {item} -> {target_rel}")
            print(f"  commit: {commit_msg!r}\n")
        else:
            if missing or collisions:
                print(
                    f"ERROR: Refusing to execute Stage {stage_num} ({dest_name}) due to "
                    f"{len(missing)} missing source(s) and {len(collisions)} collision(s).",
                    file=sys.stderr,
                )
                return 1
            dest_dir.mkdir(parents=True, exist_ok=True)
            if dest_name in {"workers", "auth", "hosts", "core"}:
                init_file = dest_dir / "__init__.py"
                if not init_file.exists():
                    init_file.write_text(f'"""Xcelsior {dest_name} package."""\n', encoding="utf-8")
                    run(["git", "add", str(init_file.relative_to(ROOT))])
            for item in movable:
                src = ROOT / item
                dst = dest_dir / Path(item).name
                run(["git", "mv", str(src.relative_to(ROOT)), str(dst.relative_to(ROOT))])
            if not args.no_commit:
                run(["git", "commit", "-m", commit_msg])
            print()

    print(
        f"Summary: {total_planned} move(s) across {len(selected)} stage(s), "
        f"{total_already_moved} already moved, "
        f"{total_missing} missing source(s), {total_collisions} destination collision(s)."
    )
    if total_missing or total_collisions:
        return 1
    if dry:
        print("Re-run with --apply (or --execute) to execute.")
    else:
        print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
