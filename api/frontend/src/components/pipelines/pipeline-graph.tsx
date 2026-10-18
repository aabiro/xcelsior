"use client";

/**
 * P4's pipeline view: the graph, which stage is live, and one ceiling.
 *
 * Gate P4's frontend clause asks for *"a pipeline view showing the graph, which
 * stage is live, and one approval covering all of it — with the total committed
 * spend stated **before** approval, not after."*
 *
 * Before this, a pipeline reached the approval page as a generic plan: its
 * stages rendered as one JSON blob in the canonical-spec table. Every fact the
 * user needed in order to judge it was technically present and none of it was
 * legible — including `on_failure`, which `docs/pipeline-plan.md` §3.2 says is
 * part of what is being approved:
 *
 * > *A user approving a graph is approving its failure behaviour too — deciding
 * > it afterwards, at the moment something broke, is exactly when the decision
 * > is worst.*
 *
 * A JSON dump defeats that. Someone approving `train → evaluate → serve` has to
 * be able to see that stage 2 halts and stage 3 retries twice without reading a
 * serialised object.
 *
 * ## Why one component for before and after
 *
 * The pre-approval graph and the live run are the same graph. Rendering them
 * from two components is how they drift — the approval screen shows three
 * stages, the live view shows the stages that happen to have rows, and nobody
 * notices when one silently never materialised. `stages` here is the whole
 * declared graph in both modes; only the state varies.
 *
 * ## Why spend is shown against the ceiling and not against an estimate
 *
 * §3.3 enforces the ceiling *before each stage*, so a pipeline cannot outspend
 * it — "the number shown before approval is a ceiling, not an estimate". The
 * bar reads as progress toward a bound the platform enforces, not a forecast
 * that might be exceeded.
 */

import { CheckCircle2, XCircle, Loader2, Circle, MinusCircle } from "lucide-react";
import type { PipelineStage } from "@/lib/api";

const CAD = new Intl.NumberFormat("en-CA", { style: "currency", currency: "CAD" });

/** Micros → CAD. Integer micros are the money type; this only formats. */
function cad(micros: number): string {
  return CAD.format((Number(micros) || 0) / 1_000_000);
}

/**
 * The five states `ck_pipeline_stage_state` allows, and nothing else.
 *
 * Taken from the CHECK constraint rather than from a guess: a state the
 * database can store and this map does not know would render as a blank cell,
 * which is the one outcome worse than an ugly label. Unknown states fall back
 * to showing the raw value for that reason.
 */
const STATE_META: Record<string, { label: string; className: string; Icon: typeof Circle }> = {
  pending: { label: "Waiting", className: "text-text-muted", Icon: Circle },
  running: { label: "Running", className: "text-accent-cyan", Icon: Loader2 },
  succeeded: { label: "Done", className: "text-emerald", Icon: CheckCircle2 },
  failed: { label: "Failed", className: "text-accent-red", Icon: XCircle },
  skipped: { label: "Skipped", className: "text-text-muted", Icon: MinusCircle },
};

/** What each failure mode means, in the words someone approving it needs. */
function failureText(stage: Pick<PipelineStage, "on_failure" | "max_attempts">): string {
  switch (stage.on_failure) {
    case "halt":
      return "If this fails, the pipeline stops here";
    case "continue":
      return "If this fails, the pipeline carries on";
    case "retry":
      return `If this fails, retries up to ${stage.max_attempts}× then stops`;
    default:
      return `On failure: ${stage.on_failure}`;
  }
}

export interface PipelineGraphProps {
  /**
   * The whole declared graph, in order. Before approval these carry no state;
   * `state` defaults to `pending` so one rendering path serves both modes.
   */
  stages: Array<
    Pick<PipelineStage, "index" | "name" | "action_type" | "on_failure" | "max_attempts"> &
      Partial<Pick<PipelineStage, "state" | "attempt_count" | "failure_code" | "spent_micros">> & {
        estimate_micros?: number;
      }
  >;
  /** The ceiling the user agreed to, in micros. */
  approvedMaxMicros: number;
  /** Spend so far. Omitted before approval, when nothing has been spent. */
  spentMicros?: number;
  /** True once the plan is approved — switches the ceiling's wording. */
  approved?: boolean;
}

export function PipelineGraph({
  stages,
  approvedMaxMicros,
  spentMicros,
  approved = false,
}: PipelineGraphProps) {
  const live = typeof spentMicros === "number";
  const pct =
    approvedMaxMicros > 0 && live
      ? Math.min(100, ((spentMicros as number) / approvedMaxMicros) * 100)
      : 0;

  return (
    <div data-testid="pipeline-graph" className="space-y-4">
      <div className="rounded-xl border border-border/60 p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <p className="text-xs uppercase tracking-wide text-text-muted">
            {approved ? "Approved ceiling" : "Total you are approving"}
          </p>
          <p data-testid="pipeline-ceiling" className="text-xl font-semibold">
            {cad(approvedMaxMicros)}
          </p>
        </div>
        <p className="mt-1 text-xs text-text-secondary">
          {live ? (
            <>
              <span data-testid="pipeline-spent" className="font-medium">
                {cad(spentMicros as number)}
              </span>{" "}
              spent of this ceiling. A stage that would exceed it does not start.
            </>
          ) : (
            <>
              This covers all {stages.length} stage{stages.length === 1 ? "" : "s"} below. It is a
              ceiling, not an estimate — a stage that would exceed it does not start.
            </>
          )}
        </p>
        {live && (
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-border/60">
            <div
              data-testid="pipeline-spend-bar"
              className="h-full rounded-full bg-accent-cyan transition-[width]"
              style={{ width: `${pct}%` }}
              role="progressbar"
              aria-valuenow={Math.round(pct)}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Spend against the approved ceiling"
            />
          </div>
        )}
      </div>

      <ol className="space-y-2">
        {stages.map((stage, i) => {
          const state = stage.state ?? "pending";
          const meta = STATE_META[state] ?? {
            label: state,
            className: "text-text-muted",
            Icon: Circle,
          };
          const { Icon } = meta;
          const isLive = state === "running";
          return (
            <li
              key={`${stage.index}-${stage.name}`}
              data-testid={`pipeline-stage-${stage.index}`}
              data-state={state}
              className={`rounded-xl border p-3 ${
                isLive ? "border-accent-cyan/50 bg-accent-cyan/[0.06]" : "border-border/60"
              }`}
            >
              <div className="flex items-start gap-3">
                <Icon
                  className={`h-4 w-4 shrink-0 mt-0.5 ${meta.className} ${isLive ? "animate-spin" : ""}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
                    <span className="text-xs text-text-muted">{i + 1}.</span>
                    <span className="text-sm font-medium">{stage.name}</span>
                    <code className="text-[11px] font-mono text-text-muted">
                      {stage.action_type}
                    </code>
                    <span
                      data-testid={`pipeline-stage-${stage.index}-state`}
                      className={`ml-auto text-xs font-medium ${meta.className}`}
                    >
                      {meta.label}
                      {state === "running" && (stage.attempt_count ?? 0) > 1
                        ? ` · attempt ${stage.attempt_count}`
                        : ""}
                    </span>
                  </div>

                  <p
                    data-testid={`pipeline-stage-${stage.index}-failure`}
                    className="mt-1 text-[11px] text-text-secondary"
                  >
                    {failureText(stage)}
                  </p>

                  {stage.failure_code && (
                    <p className="mt-1 text-[11px] font-mono text-accent-red">
                      {stage.failure_code}
                    </p>
                  )}

                  {typeof stage.spent_micros === "number" && stage.spent_micros > 0 && (
                    <p className="mt-1 text-[11px] text-text-muted">
                      Spent {cad(stage.spent_micros)}
                    </p>
                  )}
                  {!live && typeof stage.estimate_micros === "number" && (
                    <p className="mt-1 text-[11px] text-text-muted">
                      Up to {cad(stage.estimate_micros)}
                    </p>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
