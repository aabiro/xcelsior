"use client";

import { useCallback, useEffect, useState } from "react";
import { AlertTriangle, Check, Loader2, ShieldCheck } from "lucide-react";
import {
  evaluatePlacement,
  type PlacementDecision,
  type PlacementPreferenceInput,
} from "@/lib/api";

/**
 * P5's frontend clause: *"placement preference as a first-class launch control,
 * showing the price/reliability trade-off it implies before launch."*
 *
 * The control plane has answered this since P5 — `POST /api/v1/placements/evaluate`
 * returns either the host a preference would pick (with `baseline_price`,
 * `chosen_price` and `premium_pct`) or a typed refusal carrying the number that
 * failed. `evaluate_placement_preference` put it on the tool surface. The launch
 * modal offered none of it, so a human could not state a reliability floor at
 * all, let alone see what it costs.
 *
 * **The refusal is the important half.** P5's gate: *"a placement preference that
 * cannot be satisfied refuses clearly rather than silently falling back to the
 * cheapest host — this is the failure mode that would quietly destroy trust."*
 * So a refusal renders as a refusal, with asked-vs-available side by side, and
 * never as an empty state that lets the launch proceed as though nothing was
 * asked.
 */

export interface PlacementSpec {
  gpu_model?: string;
  num_gpus?: number;
  vram_gb?: number;
  region?: string;
}

const TIERS = ["", "standard", "premium", "verified"] as const;

export function PlacementPreferenceControl({
  spec,
  onChange,
}: {
  spec: PlacementSpec;
  /** Reports the preference and whether it is currently satisfiable. */
  onChange?: (pref: PlacementPreferenceInput, decision: PlacementDecision | null) => void;
}) {
  const [minUptime, setMinUptime] = useState<string>("");
  const [minTier, setMinTier] = useState<string>("");
  const [requireVerified, setRequireVerified] = useState(false);
  const [maxPremium, setMaxPremium] = useState<string>("");
  const [decision, setDecision] = useState<PlacementDecision | null>(null);
  const [checking, setChecking] = useState(false);

  const stated =
    minUptime !== "" || minTier !== "" || requireVerified || maxPremium !== "";

  const evaluate = useCallback(async () => {
    const pref: PlacementPreferenceInput = {
      min_uptime_pct: minUptime === "" ? null : Number(minUptime),
      min_tier: minTier === "" ? null : minTier,
      require_verified: requireVerified,
      max_premium_pct: maxPremium === "" ? null : Number(maxPremium),
    };
    if (!stated) {
      setDecision(null);
      onChange?.(pref, null);
      return;
    }
    setChecking(true);
    try {
      const res = await evaluatePlacement(spec as Record<string, unknown>, pref);
      setDecision(res.preference);
      onChange?.(pref, res.preference);
    } catch {
      // An evaluation that cannot run must not read as "satisfiable" — the
      // whole point of this control is that silence is the failure mode.
      setDecision(null);
      onChange?.(pref, null);
    } finally {
      setChecking(false);
    }
  }, [minUptime, minTier, requireVerified, maxPremium, spec, stated, onChange]);

  useEffect(() => {
    const t = setTimeout(() => void evaluate(), 400);
    return () => clearTimeout(t);
  }, [evaluate]);

  return (
    <div className="rounded-lg border border-border p-4 space-y-3">
      <div className="flex items-center gap-2">
        <ShieldCheck className="w-4 h-4 text-text-muted" />
        <h4 className="text-sm font-medium">Placement preference</h4>
        <span className="text-xs text-text-muted">optional</span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="text-xs text-text-secondary">
          Minimum uptime %
          <input
            type="number" min={0} max={100} step={0.1}
            value={minUptime}
            onChange={(e) => setMinUptime(e.target.value)}
            placeholder="any"
            className="mt-1 w-full bg-surface border border-border rounded px-2 py-1.5 text-sm"
          />
        </label>
        <label className="text-xs text-text-secondary">
          Minimum tier
          <select
            value={minTier}
            onChange={(e) => setMinTier(e.target.value)}
            className="mt-1 w-full bg-surface border border-border rounded px-2 py-1.5 text-sm"
          >
            {TIERS.map((t) => (
              <option key={t || "any"} value={t}>{t || "any"}</option>
            ))}
          </select>
        </label>
        <label className="text-xs text-text-secondary">
          Max premium % over cheapest
          <input
            type="number" min={0} step={1}
            value={maxPremium}
            onChange={(e) => setMaxPremium(e.target.value)}
            placeholder="no cap"
            className="mt-1 w-full bg-surface border border-border rounded px-2 py-1.5 text-sm"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-text-secondary sm:mt-5">
          <input
            type="checkbox"
            checked={requireVerified}
            onChange={(e) => setRequireVerified(e.target.checked)}
          />
          Verified hosts only
        </label>
      </div>

      {checking && (
        <p className="text-xs text-text-muted flex items-center gap-1.5">
          <Loader2 className="w-3 h-3 animate-spin" /> Checking what this would place on…
        </p>
      )}

      {!checking && stated && decision?.refused && (
        <div className="rounded border border-amber-500/40 bg-amber-500/5 p-3 text-xs space-y-1">
          <p className="flex items-center gap-1.5 text-amber-400 font-medium">
            <AlertTriangle className="w-3.5 h-3.5" /> No host satisfies this right now
          </p>
          <p className="text-text-secondary">{decision.detail}</p>
          {decision.asked !== undefined && decision.best_available !== undefined && (
            <p className="text-text-muted">
              asked <span className="font-mono">{String(decision.asked)}</span> ·
              best available <span className="font-mono">{String(decision.best_available)}</span>
            </p>
          )}
          <p className="text-text-muted">
            Relax the constraint or launch without it — it will not quietly fall back.
          </p>
        </div>
      )}

      {!checking && stated && decision && !decision.refused && (
        <div className="rounded border border-emerald-500/30 bg-emerald-500/5 p-3 text-xs space-y-1">
          <p className="flex items-center gap-1.5 text-emerald-400 font-medium">
            <Check className="w-3.5 h-3.5" /> Satisfiable
          </p>
          <p className="text-text-secondary">
            ${Number(decision.chosen_price ?? 0).toFixed(2)}/hr on this preference
            {typeof decision.baseline_price === "number" && (
              <> · cheapest available ${decision.baseline_price.toFixed(2)}/hr</>
            )}
          </p>
          {typeof decision.premium_pct === "number" && decision.premium_pct > 0 && (
            <p className="text-text-muted">
              You pay <span className="font-mono">{decision.premium_pct}%</span> more for it.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
