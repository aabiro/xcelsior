#!/usr/bin/env python3
"""Cross-check Mac AI/ML docs §10 xcelsior rows against row-evidence.json."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path(
    sys.argv[1]
    if len(sys.argv) > 1
    else __import__("os").environ.get(
        "XCELSIOR_GOAL_SCRATCH", "/tmp/grok-goal-6f86c7cfe9c2/implementer"
    )
)
MAC_HOST = __import__("os").environ.get("XCELSIOR_MAC_HOST", "aaryn@100.64.0.3")
REGISTRY_MD = ROOT.parent / "pxl-registry" / "docs" / "AI_ML_MASTER_INDEX_BY_REPO.md"
EVIDENCE_PATH = SCRATCH / "row-evidence.json"


def _read_mac_section() -> str:
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        MAC_HOST,
        "sed -n '/## 10. `xcelsior`/,/^## 11/p' ~/Projects/pxl-registry/docs/AI_ML_MASTER_INDEX_BY_REPO.md",
    ]
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return f"MAC_UNREACHABLE: {exc}"


def _parse_checkboxes(section: str) -> dict[int, bool]:
    rows: dict[int, bool] = {}
    for line in section.splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|\s*-\s*\[([ x])\]", line)
        if m:
            rows[int(m.group(1))] = m.group(2) == "x"
    return rows


def _load_evidence() -> dict:
    if not EVIDENCE_PATH.is_file():
        return {}
    with EVIDENCE_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _evidence_checked(evidence: dict) -> dict[int, bool]:
    rows = evidence.get("rows") or {}
    out: dict[int, bool] = {}
    for rank in range(1, 32):
        row = rows.get(str(rank)) or {}
        out[rank] = row.get("status") == "verified"
    return out


def main() -> int:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    evidence = _load_evidence()
    evidence_rows = _evidence_checked(evidence)
    mac_section = _read_mac_section()
    mac_rows = _parse_checkboxes(mac_section)
    local_section = ""
    if REGISTRY_MD.is_file():
        full = REGISTRY_MD.read_text(encoding="utf-8")
        m = re.search(r"## 10\. `xcelsior`.*?(?=## 11\.)", full, re.DOTALL)
        local_section = m.group(0) if m else ""
    local_rows = _parse_checkboxes(local_section)

    lines = [
        f"=== AI/ML §10 xcelsior cross-check ({evidence.get('closure_policy', 'engineering_partial')}) ===",
        f"mac_host={MAC_HOST}",
        f"local_registry_md={REGISTRY_MD}",
        f"evidence={EVIDENCE_PATH}",
        f"verified_rows={evidence.get('verified_rows', [])}",
        f"blocked_rows={evidence.get('blocked_rows', [])}",
        "",
        "evidence_checkboxes:",
    ]
    for rank in sorted(evidence_rows):
        row = (evidence.get("rows") or {}).get(str(rank)) or {}
        reason = row.get("reason", "")
        lines.append(
            f"  {rank}: {'[x]' if evidence_rows[rank] else '[ ]'}  # {reason[:80]}"
        )
    lines.append("")
    lines.append("mac_checkboxes:")
    for rank in sorted(mac_rows):
        lines.append(f"  {rank}: {'[x]' if mac_rows[rank] else '[ ]'}")
    lines.append("")
    lines.append("local_registry_checkboxes:")
    for rank in sorted(local_rows):
        lines.append(f"  {rank}: {'[x]' if local_rows[rank] else '[ ]'}")
    lines.append("")
    mismatches = []
    for rank in range(1, 32):
        ev = evidence_rows.get(rank)
        mac = mac_rows.get(rank)
        local = local_rows.get(rank)
        if ev is not None and mac is not None and ev != mac:
            mismatches.append(f"row {rank}: evidence={ev} mac_md={mac}")
        if ev is not None and local is not None and ev != local:
            mismatches.append(f"row {rank}: evidence={ev} local_md={local}")
    lines.append("mismatches:")
    lines.extend(f"  - {m}" for m in mismatches) if mismatches else lines.append("  (none)")
    lines.append("")
    lines.append("docx_on_mac:")
    try:
        docx_list = subprocess.check_output(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                MAC_HOST,
                "find ~/Projects ~/Documents ~/Downloads /mnt/storage/projects -maxdepth 8 \\( -iname 'AI_ML*.docx' -o -iname '*MASTER_INDEX*.docx' -o -iname 'AI_ML*.doc' -o -iname '*AI_ML*.docx' \\) 2>/dev/null | head -40",
            ],
            text=True,
            timeout=25,
        ).strip()
        if docx_list:
            lines.append(f"  files={docx_list.splitlines()}")
        else:
            lines.append(
                "  count=0 (searched ~/Projects and ~/Documents for AI_ML*.docx; "
                "portfolio uses markdown under ~/Projects/pxl-registry/docs/)"
            )
    except Exception as exc:
        lines.append(f"  error={exc}")

    review_cmds = [
        ("F5_token_billing", "sed -n '/# F5 — xcelsior Tokens/,/^# F6/p' ~/Projects/pxl-registry/docs/AI_ML_FRONTIER_NEXT7_PLAN.md | head -80"),
        ("S2_xcelsior_rows", "grep -nE 'xcelsior|token|metering|embed|prefix|EAGLE|cached' ~/Projects/pxl-registry/docs/AI_ML_UPGRADE_PLAN.md 2>/dev/null | head -35"),
        ("model_paths", "grep -nE 'Qwen3-8B|bge-m3|b2|staging|/mnt/storage' ~/Projects/pxl-registry/docs/AI_ML_MODEL_UPGRADES_AND_DOWNLOADS.md 2>/dev/null | head -25"),
        ("checklist_scope", "grep -nE 'xcelsior|token|metering|excluded' ~/Projects/pxl-registry/docs/AI_ML_UPGRADE_IMPLEMENTATION_CHECKLIST.md 2>/dev/null | head -20"),
    ]
    review_path = SCRATCH / "mac-ai-ml-docs-review.txt"
    review_lines = ["=== Deep AI/ML doc content review (Mac) ===", f"host={MAC_HOST}", ""]
    for title, remote_cmd in review_cmds:
        review_lines.append(f"--- {title} ---")
        try:
            chunk = subprocess.check_output(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", MAC_HOST, remote_cmd],
                text=True,
                timeout=30,
                stderr=subprocess.STDOUT,
            )
            review_lines.append(chunk.strip() or "(empty)")
        except Exception as exc:
            review_lines.append(f"ERROR: {exc}")
        review_lines.append("")
    review_path.write_text("\n".join(review_lines) + "\n", encoding="utf-8")
    lines.append(f"deep_review={review_path}")

    out = SCRATCH / "mac-ai-ml-docs-crosscheck.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (SCRATCH / "mac-ai-ml-docs-section.txt").write_text(mac_section, encoding="utf-8")
    (SCRATCH / "master-index-probes.json").write_text(
        json.dumps(
            {
                "evidence_rows": evidence_rows,
                "mac_rows": mac_rows,
                "local_rows": local_rows,
                "verified": evidence.get("verified_rows"),
                "blocked": evidence.get("blocked_rows"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out)
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())