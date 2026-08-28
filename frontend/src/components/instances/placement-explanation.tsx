"use client";

/**
 * Why this instance is queued, in a sentence — B6.5.
 *
 * §20.3 asks for a *"plain-language current reason ('Queued because no healthy
 * H100 with 80 GB is available in Ontario')"*. Everything needed to say that
 * already existed: `control_plane/scheduler/explain.py` builds a bounded
 * explanation, the scheduler persists it on the attempt, and
 * `/api/v1/instances/{job_id}/placement-explanation` serves it tenant-scoped.
 *
 * `explain_instance_placement` — a published MCP tool — has read it since it
 * shipped. The browser never called the route. So an agent could tell someone
 * why their instance was waiting, and the dashboard showed them a spinner.
 * That is the fourth surface this week where the agent saw more than the
 * person; see `tests/test_host_key_verification_parity.py` for the first.
 *
 * ## What it deliberately does not show
 *
 * The stored payload carries a per-host `rejections` map — "host is drained",
 * "host heartbeat/inventory is stale", keyed by host id, for hosts this job was
 * *not* placed on. That is other tenants' fleet state, and §20.3 says customers
 * see redacted infrastructure detail. This renders `rejection_summary` only:
 * how many hosts failed each constraint, never which.
 *
 * ## Why constraint labels fall back to the raw code
 *
 * The aggregate keeps codes, not messages — `FilterReason.message` is per host
 * and is exactly the infrastructure detail being withheld. So the labels here
 * are a small map that **annotates**: an unrecognised code renders as itself
 * rather than being skipped or re-spaced into a fake sentence. Same rule as the
 * payout-requirements glossary, and for the same reason — an incomplete map
 * should cost a nicer sentence, never a hidden constraint.
 */

import { Clock, MapPin, Cpu, Info } from "lucide-react";
import type { PlacementExplanationPayload } from "@/lib/api";

/**
 * Constraint codes from `control_plane/scheduler/filters.py`, as a phrase that
 * completes "N hosts …".
 */
const CONSTRAINT_PHRASES: Record<string, string> = {
  gpu_model_mismatch: "have a different GPU model",
  insufficient_gpus: "do not have enough free GPUs",
  insufficient_vram: "do not have enough free VRAM",
  host_not_admitted: "are not admitted to the fleet",
  host_not_ready: "are not currently ready",
  host_observation_stale: "have not reported in recently",
  region_mismatch: "are in a different region",
  price_above_max: "cost more than your maximum price",
  tier_mismatch: "are on a different service tier",
};

export function constraintPhrase(code: string): string {
  return CONSTRAINT_PHRASES[code] ?? code;
}

/** True when the phrase is a real sentence rather than the raw code. */
export function hasPhrase(code: string): boolean {
  return code in CONSTRAINT_PHRASES;
}

/**
 * The headline sentence, built from what was asked for.
 *
 * Deliberately describes the *request*, not the fleet: "no host currently
 * matches 2 × H100 with 80 GB in Ontario" is true and reveals nothing about
 * which hosts exist or what state they are in.
 */
export function summarySentence(payload: PlacementExplanationPayload): string {
  const r = payload.request ?? {};
  const parts: string[] = [];
  if (r.num_gpus && r.num_gpus > 1) parts.push(`${r.num_gpus} × ${r.gpu_model ?? "GPU"}`);
  else if (r.gpu_model) parts.push(r.gpu_model);
  else parts.push("a GPU");
  if (r.vram_needed_gb) parts.push(`with ${r.vram_needed_gb} GB`);
  const what = parts.join(" ");
  const where = r.region ? ` in ${r.region}` : "";
  return `No host is currently available matching ${what}${where}.`;
}

export interface PlacementExplanationProps {
  payload: PlacementExplanationPayload | null;
  /** `false` when the scheduler recorded nothing for this attempt. */
  explained: boolean;
}

export function PlacementExplanation({ payload, explained }: PlacementExplanationProps) {
  if (!explained || !payload) {
    return (
      <div
        data-testid="placement-explanation-absent"
        className="rounded-lg border border-border bg-surface/40 px-3 py-2.5"
      >
        <div className="flex items-start gap-2">
          <Info className="mt-0.5 h-4 w-4 shrink-0 text-text-muted" />
          <p className="text-xs text-text-muted">
            No placement explanation was recorded for this attempt. That is normal for
            instances launched before explanations were kept, and for ones that never
            reached scheduling.
          </p>
        </div>
      </div>
    );
  }

  // A chosen host is not a queue reason — say so and stop.
  if (payload.selected_host_id) {
    return (
      <div
        data-testid="placement-explanation-placed"
        className="rounded-lg border border-emerald/25 bg-emerald/[0.06] px-3 py-2.5"
      >
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 shrink-0 text-emerald" />
          <p className="text-xs text-emerald">
            Placed on a matching host
            {payload.hosts_eligible > 1
              ? ` — the best of ${payload.hosts_eligible} that qualified.`
              : "."}
          </p>
        </div>
      </div>
    );
  }

  const failed = payload.rejection_summary?.failed_constraints ?? {};
  // Most-common constraint first: that is the one worth changing.
  const ranked = Object.entries(failed).sort((a, b) => b[1] - a[1]);

  return (
    <div
      data-testid="placement-explanation-queued"
      className="space-y-2 rounded-lg border border-accent-gold/25 bg-accent-gold/[0.06] px-3 py-2.5"
    >
      <div className="flex items-start gap-2">
        <Clock className="mt-0.5 h-4 w-4 shrink-0 text-accent-gold" />
        <div className="min-w-0">
          <p data-testid="placement-summary" className="text-xs font-medium text-accent-gold">
            {summarySentence(payload)}
          </p>
          <p className="mt-0.5 text-[11px] text-text-muted">
            Checked {payload.hosts_considered} host
            {payload.hosts_considered === 1 ? "" : "s"}. This keeps retrying — nothing is
            lost while it waits.
          </p>
        </div>
      </div>

      {ranked.length > 0 && (
        <ul data-testid="placement-constraints" className="space-y-0.5 pl-6">
          {ranked.map(([code, count]) => (
            <li key={code} className="text-[11px] text-text-secondary">
              <span className="font-medium">{count}</span>{" "}
              {count === 1 ? "host " : "hosts "}
              {constraintPhrase(code)}
              {!hasPhrase(code) && (
                <span className="text-text-muted"> (unrecognised constraint)</span>
              )}
            </li>
          ))}
        </ul>
      )}

      {payload.request?.region && (
        <p className="flex items-center gap-1.5 pl-6 text-[11px] text-text-muted">
          <MapPin className="h-3 w-3" />
          Widening the region, or lowering the GPU or VRAM requirement, is usually the
          fastest way to get placed.
        </p>
      )}
    </div>
  );
}
