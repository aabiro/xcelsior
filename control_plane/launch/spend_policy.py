"""Standing-policy evaluation for a launch (§14.2, §15.4).

A launch always creates spend, so it always needs approval — the question is
*whose*. If the client has a standing ``mcp_client_policies`` row, the plan
is inside every ceiling and allowed capability, and the policy grants
auto-approval, the plan can be server-approved without bouncing a human
(``approval_mode = 'standing_policy'``). Otherwise a human must approve at
the dashboard (``approval_mode = 'human'``). A ``confirm:true`` from the
client is intent, never approval (§14.2) — it does not enter this decision.

Money is micro-CAD throughout, matching the quote and the ledger.

Note on the rolling spend ceilings: ``hourly_spend_max_micros`` and
``daily_spend_max_micros`` require live spend counters, which are the
distributed-limits job (B5.9). They are intentionally *not* evaluated here —
this module decides only what a spec+quote can show on its own. The service
layer combines this decision with the counters before auto-approving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    # Does the plan violate a hard capability/ceiling in the policy?
    allowed: bool
    # Which approval path the plan takes: 'standing_policy' | 'human'.
    approval_mode: str
    # Human-readable reasons the plan is blocked or cannot auto-approve.
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "approval_mode": self.approval_mode,
            "reasons": list(self.reasons),
        }


def _security_mode(spec: dict[str, Any]) -> str:
    return "encrypted" if spec.get("encrypted_workspace") else "standard"


def evaluate(
    spec: dict[str, Any],
    *,
    estimate_micros: int,
    policy: dict[str, Any] | None,
) -> PolicyDecision:
    """Evaluate a spec+estimate against an optional standing policy.

    Returns whether the plan is within the policy's hard limits and the
    approval mode it earns. With no policy the plan is not *blocked* (a human
    may still approve it) but it can never auto-approve.
    """
    if policy is None:
        return PolicyDecision(
            allowed=True,
            approval_mode="human",
            reasons=["no standing policy; requires human approval"],
        )

    reasons: list[str] = []

    per_action = policy.get("per_action_max_micros")
    if per_action is not None and int(estimate_micros) > int(per_action):
        reasons.append(
            f"estimate {estimate_micros} exceeds per-action ceiling {per_action}"
        )

    allowed_models = policy.get("allowed_gpu_models")
    gpu_model = str(spec.get("gpu_model") or "")
    if allowed_models and gpu_model and gpu_model not in allowed_models:
        reasons.append(f"gpu_model {gpu_model!r} not in allowed set")

    allowed_regions = policy.get("allowed_regions")
    region = str(spec.get("region") or "")
    if allowed_regions and region and region not in allowed_regions:
        reasons.append(f"region {region!r} not in allowed set")

    allowed_modes = policy.get("allowed_security_modes")
    mode = _security_mode(spec)
    if allowed_modes and mode not in allowed_modes:
        reasons.append(f"security mode {mode!r} not allowed")

    max_conc = policy.get("max_concurrency")
    num_gpus = int(spec.get("num_gpus") or 1)
    if max_conc is not None and num_gpus > int(max_conc):
        reasons.append(f"num_gpus {num_gpus} exceeds max_concurrency {max_conc}")

    within_limits = not reasons
    auto = bool(policy.get("auto_approve")) and within_limits
    return PolicyDecision(
        allowed=within_limits,
        approval_mode="standing_policy" if auto else "human",
        reasons=reasons,
    )
