"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Clock3,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  approveLaunchPlan,
  fetchLaunchPlan,
  fetchPipeline,
  type LaunchPlanApprovalView,
  type PipelineView,
} from "@/lib/api";
import { PipelineGraph } from "@/components/pipelines/pipeline-graph";

const CAD = new Intl.NumberFormat("en-CA", {
  style: "currency",
  currency: "CAD",
  minimumFractionDigits: 2,
});

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (value && typeof value === "object") return JSON.stringify(value);
  return String(value ?? "—");
}

export default function LaunchPlanApprovalPage() {
  const params = useParams<{ planId: string }>();
  const planId = String(params.planId || "");
  const [data, setData] = useState<LaunchPlanApprovalView | null>(null);
  const [loading, setLoading] = useState(true);
  const [approving, setApproving] = useState(false);
  const [error, setError] = useState("");
  // Live stage state, once there is a run to have state. Kept separate from
  // `data` because the plan and the run answer different questions: the plan is
  // what was approved and never changes, the run is what is happening now.
  const [pipeline, setPipeline] = useState<PipelineView | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setData(await fetchLaunchPlan(planId));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to load this approval.");
    } finally {
      setLoading(false);
    }
  }, [planId]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll stage state while a pipeline is actually running.
  //
  // Only for `run_pipeline`, and it stops the moment the run reaches a terminal
  // state — a page left open on a finished pipeline should not keep a request
  // every three seconds going forever. A 404 here is the normal pre-execution
  // answer (the plan exists, the stages do not yet), so it clears rather than
  // raising: an approval screen must not show "error" for a pipeline whose only
  // fault is not having been started.
  const isPipeline = data?.plan.action_type === "run_pipeline";
  useEffect(() => {
    if (!isPipeline || !planId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const tick = async () => {
      try {
        const next = await fetchPipeline(planId);
        if (cancelled) return;
        setPipeline(next);
        if (next.finished) return;
      } catch {
        if (cancelled) return;
        setPipeline(null);
      }
      if (!cancelled) timer = setTimeout(() => void tick(), 3000);
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [isPipeline, planId]);

  const approve = async () => {
    if (!data) return;
    setApproving(true);
    setError("");
    try {
      setData(await approveLaunchPlan(planId, data.plan.version));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Approval failed.");
      await refresh();
    } finally {
      setApproving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center text-text-muted">
        <Loader2 className="mr-2 h-5 w-5 animate-spin" /> Loading server-bound plan…
      </div>
    );
  }

  if (!data) {
    return (
      <div className="mx-auto max-w-xl py-16">
        <div className="rounded-2xl border border-red-500/30 bg-red-500/5 p-6">
          <AlertTriangle className="mb-3 h-6 w-6 text-red-400" />
          <h1 className="text-xl font-semibold">Approval unavailable</h1>
          <p className="mt-2 text-sm text-text-secondary">{error || "This plan was not found."}</p>
          <Link
            href="/dashboard/mcp"
            className="mt-5 inline-flex h-10 items-center rounded-md border border-border px-4 text-sm font-medium"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />Back to MCP
          </Link>
        </div>
      </div>
    );
  }

  const { plan } = data;
  const isEviction = plan.action_type === "evict_host_workloads";
  // The declared graph, straight off the plan the user is approving — not off
  // the run. `canonical_spec` is the hash-bound copy, so what is rendered here
  // is exactly what editing would invalidate.
  const declaredStages = Array.isArray(
    (plan.canonical_spec as { stages?: unknown }).stages,
  )
    ? ((plan.canonical_spec as { stages: Array<Record<string, unknown>> }).stages.map(
        (stage, index) => ({
          index,
          name: String(stage.name ?? `stage ${index + 1}`),
          action_type: String(stage.action_type ?? "—"),
          on_failure: (stage.on_failure ?? "halt") as "halt" | "continue" | "retry",
          max_attempts: Number(stage.max_attempts ?? 1),
          estimate_micros: Number(stage.estimate_micros ?? 0),
        }),
      ))
    : [];
  // Stage rows once a run exists, otherwise the declared graph with no state.
  // The declared graph is the fallback rather than an empty list, so a pipeline
  // that has not started still shows what it will do.
  const graphStages =
    pipeline && pipeline.stages.length > 0 ? pipeline.stages : declaredStages;
  const approvable = ["quoted", "awaiting_approval"].includes(plan.status);
  const approved = plan.status === "approved" || plan.status === "succeeded";

  return (
    <div className="mx-auto max-w-3xl space-y-6 py-8">
      <Link
        href="/dashboard/mcp"
        className="inline-flex items-center text-sm text-text-muted hover:text-text-primary"
      >
        <ArrowLeft className="mr-2 h-4 w-4" /> Back to MCP
      </Link>

      <div className="rounded-2xl border border-border/70 bg-surface/50 p-6 shadow-xl shadow-black/10">
        <div className="flex items-start justify-between gap-4">
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm text-accent-cyan">
              <ShieldCheck className="h-4 w-4" /> Server-bound approval
            </div>
            <h1 className="text-2xl font-semibold">
              {isEviction
                ? "Review destructive host eviction"
                : isPipeline
                  ? "Review pipeline"
                  : "Review compute launch"}
            </h1>
            <p className="mt-2 text-sm text-text-secondary">
              Your agent prepared this plan but cannot approve it. Approval does not execute
              anything; the agent must execute this exact, versioned plan before it expires.
              {isEviction ? " Eviction removes running workloads and requires the dedicated hosts:evict scope." : ""}
            </p>
          </div>
          <span className="rounded-full border border-border px-3 py-1 text-xs uppercase tracking-wide">
            {plan.status.replaceAll("_", " ")}
          </span>
        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-3">
          {/* For a pipeline the ceiling belongs with the graph it covers —
              showing it here as well would state the same number twice, in two
              different framings, which invites the reader to wonder whether
              they are two different numbers. */}
          {!isPipeline && (
            <div className="rounded-xl border border-border/60 p-4">
              <p className="text-xs uppercase tracking-wide text-text-muted">Maximum authorized</p>
              <p className="mt-1 text-xl font-semibold">
                {CAD.format(Number(plan.estimate_micros || 0) / 1_000_000)}
              </p>
            </div>
          )}
          <div className="rounded-xl border border-border/60 p-4">
            <p className="text-xs uppercase tracking-wide text-text-muted">Expires</p>
            <p className="mt-1 flex items-center gap-2 text-sm">
              <Clock3 className="h-4 w-4" />
              {new Date(plan.expires_at).toLocaleString()}
            </p>
          </div>
          <div className="rounded-xl border border-border/60 p-4">
            <p className="text-xs uppercase tracking-wide text-text-muted">Plan version</p>
            <p className="mt-1 text-xl font-semibold">{plan.version}</p>
          </div>
        </div>

        {isPipeline && graphStages.length > 0 ? (
          <div className="mt-6">
            {/* Gate P4: the graph, which stage is live, and one ceiling
                covering all of it — instead of the stages serialised into a
                single cell of the table below, where `on_failure` was present
                and unreadable. */}
            <PipelineGraph
              stages={graphStages}
              approvedMaxMicros={
                pipeline?.approved_max_micros ?? Number(plan.estimate_micros || 0)
              }
              spentMicros={pipeline?.spent_micros}
              approved={approved}
            />
          </div>
        ) : (
          <div className="mt-6 overflow-hidden rounded-xl border border-border/60">
            <div className="border-b border-border/60 bg-background/30 px-4 py-3 text-sm font-medium">
              Canonical launch specification
            </div>
            <dl className="divide-y divide-border/50">
              {Object.entries(plan.canonical_spec).map(([key, value]) => (
                <div key={key} className="grid grid-cols-[minmax(9rem,1fr)_2fr] gap-4 px-4 py-3 text-sm">
                  <dt className="text-text-muted">{key.replaceAll("_", " ")}</dt>
                  <dd className="break-words font-mono text-xs">{displayValue(value)}</dd>
                </div>
              ))}
            </dl>
          </div>
        )}

        {error && (
          <p className="mt-4 rounded-lg border border-red-500/30 bg-red-500/5 p-3 text-sm text-red-300">
            {error}
          </p>
        )}

        <div className="mt-6 flex flex-wrap items-center gap-3">
          {approvable && (
            <Button onClick={approve} disabled={approving} className="gap-2">
              {approving ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
              Approve this exact plan
            </Button>
          )}
          {approved && (
            <div className="inline-flex items-center gap-2 text-sm text-emerald-400">
              <CheckCircle2 className="h-5 w-5" />
              Approved. Return to your agent to execute plan {plan.plan_id}.
            </div>
          )}
          {!approvable && !approved && (
            <p className="text-sm text-text-muted">
              This plan is terminal. Ask your agent to prepare a new preview.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
