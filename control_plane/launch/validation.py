"""Side-effect-free validation of a canonical launch spec (§14.1 step 3).

These checks never mutate anything and never call an external service; they
inspect the canonical spec produced by :mod:`canonicalize`. Resource-scoped
checks that need a database read (volume ownership, region capacity) are the
service layer's job (B2.3) and are kept out of here so this stays pure and
trivially testable. A failed check is a structured :class:`Problem`, not an
exception, so the preview can report *all* problems at once rather than the
first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# SSH is reserved for the platform; users route extra services via
# ``exposed_ports`` and reach them through the gateway.
RESERVED_PORT = 22
MAX_GPUS = 64


@dataclass(frozen=True)
class Problem:
    code: str
    detail: str
    field: str = ""

    def as_dict(self) -> dict[str, str]:
        d = {"code": self.code, "detail": self.detail}
        if self.field:
            d["field"] = self.field
        return d


def validate_canonical_spec(spec: dict[str, Any]) -> list[Problem]:
    """Structural validation of a canonical spec. Returns all problems found."""
    problems: list[Problem] = []

    if not spec.get("name"):
        problems.append(Problem("name_required", "name must be non-empty", "name"))

    num_gpus = int(spec.get("num_gpus") or 0)
    if num_gpus < 1 or num_gpus > MAX_GPUS:
        problems.append(
            Problem("num_gpus_out_of_range", f"num_gpus must be 1..{MAX_GPUS}", "num_gpus")
        )

    vram = float(spec.get("vram_needed_gb") or 0)
    if vram < 0:
        problems.append(
            Problem("vram_negative", "vram_needed_gb must be >= 0", "vram_needed_gb")
        )

    ssh_port = int(spec.get("ssh_port") or 0)
    if ssh_port < 1 or ssh_port > 65535:
        problems.append(
            Problem("ssh_port_invalid", "ssh_port must be 1..65535", "ssh_port")
        )

    pricing_mode = str(spec.get("pricing_mode") or "")
    if pricing_mode not in ("on_demand", "spot", "reserved"):
        problems.append(
            Problem(
                "pricing_mode_invalid",
                "pricing_mode must be on_demand, spot, or reserved",
                "pricing_mode",
            )
        )

    for port in spec.get("exposed_ports") or []:
        if int(port) == RESERVED_PORT:
            problems.append(
                Problem(
                    "port_reserved",
                    "port 22 is reserved — use ssh_port",
                    "exposed_ports",
                )
            )
            break

    # The image, if present, must pass the same policy the REST path enforces.
    # validate_docker_image is pure (it parses and rejects, never pulls).
    image = str(spec.get("image") or "")
    if image:
        try:
            from security import validate_docker_image

            validate_docker_image(image)
        except Exception as exc:
            problems.append(Problem("image_invalid", str(exc), "image"))

    # git_repo must be a plain public https clone (no creds, no ssh/git/file).
    git_repo = str(spec.get("git_repo") or "")
    if git_repo:
        import re

        ok = re.fullmatch(
            r"https://[A-Za-z0-9._~-]+(?::\d+)?/[A-Za-z0-9._~\-/]+(?:\.git)?",
            git_repo,
        )
        if not ok or "@" in git_repo.split("://", 1)[-1]:
            problems.append(
                Problem(
                    "git_repo_invalid",
                    "git_repo must be a plain https:// URL with no credentials",
                    "git_repo",
                )
            )

    # init_script must be printable (no smuggled binary blobs).
    init_script = str(spec.get("init_script") or "")
    if init_script:
        if any(ord(c) < 32 and c not in "\t\n\r" for c in init_script):
            problems.append(
                Problem(
                    "init_script_control_chars",
                    "init_script contains disallowed control characters",
                    "init_script",
                )
            )

    return problems
