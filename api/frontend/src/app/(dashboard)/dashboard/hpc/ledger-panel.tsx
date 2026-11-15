"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, CircleAlert, Coins, Loader2, ReceiptText, TriangleAlert } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import * as api from "@/lib/api";
import type { HpcBilling, HpcBillingDriftDay, HpcBillingSegment, HpcCluster } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  billingRuleLabel,
  formatAge,
  formatDay,
  formatDuration,
  formatGpuHours,
  formatMicros,
  formatTimestamp,
} from "./hpc-format";

const WINDOWS = [7, 14, 30, 90];

interface Props {
  clusters: HpcCluster[];
  now: number;
  /** Bumped by the page's refresh, so the ledger reloads with everything else. */
  refreshKey: number;
  onOpenJob: (xcelsiorJobId: string) => void;
}

export function LedgerPanel({ clusters, now, refreshKey, onOpenJob }: Props) {
  const [days, setDays] = useState(14);
  const [clusterId, setClusterId] = useState("");
  const [data, setData] = useState<HpcBilling | null>(null);
  const [segments, setSegments] = useState<HpcBillingSegment[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const request = useRef(0);

  const load = useCallback(async () => {
    const id = ++request.current;
    try {
      const res = await api.fetchHpcBilling({ days, clusterId: clusterId || undefined });
      if (id !== request.current) return;
      setData(res);
      setSegments(res.segments ?? []);
      setCursor(res.next_cursor ?? null);
      setError(null);
    } catch (err) {
      if (id !== request.current) return;
      setError(err instanceof Error ? err.message : "Could not load the billing ledger");
    } finally {
      if (id === request.current) setLoading(false);
    }
  }, [days, clusterId]);

  useEffect(() => { void load(); }, [load, refreshKey]);

  const loadMore = async () => {
    if (!cursor) return;
    setLoadingMore(true);
    try {
      const res = await api.fetchHpcBilling({ days, clusterId: clusterId || undefined, cursor });
      setSegments((prev) => {
        const seen = new Set(prev.map((s) => s.segment_id));
        return [...prev, ...(res.segments ?? []).filter((s) => !seen.has(s.segment_id))];
      });
      setCursor(res.next_cursor ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load more of the ledger");
    } finally {
      setLoadingMore(false);
    }
  };

  const clusterName = useMemo(() => {
    const names = new Map(clusters.map((c) => [c.cluster_id, c.display_name || c.cluster_id]));
    return (id: string) => names.get(id) ?? id;
  }, [clusters]);

  const totals = data?.totals;
  const tolerance = data?.tolerance_pct ?? 1;
  const drift = data?.drift ?? [];
  const over = drift.filter((d) => !d.within_tolerance);
  const worst = drift.reduce<number | null>(
    (max, d) => (d.drift_pct === null ? max : max === null ? d.drift_pct : Math.max(max, d.drift_pct)),
    null,
  );
  const nothingCharged = !!totals && totals.charged_segments === 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Window" className="flex gap-1 rounded-lg bg-surface-hover p-1">
          {WINDOWS.map((w) => (
            <button
              key={w}
              type="button"
              aria-pressed={days === w}
              onClick={() => setDays(w)}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                days === w ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary",
              )}
            >
              {w} days
            </button>
          ))}
        </div>
        <Select
          aria-label="Ledger cluster"
          value={clusterId}
          onChange={(e) => setClusterId(e.target.value)}
          className="h-9 w-auto min-w-40"
        >
          <option value="">All clusters</option>
          {clusters.map((c) => (
            <option key={c.cluster_id} value={c.cluster_id}>{c.display_name || c.cluster_id}</option>
          ))}
        </Select>
        {nothingCharged && (
          <Badge variant="info" className="ml-auto gap-1.5 text-xs">
            <Coins className="h-3 w-3" /> Shadow mode: nothing has been charged
          </Badge>
        )}
      </div>

      {error && (
        <Card className="border-accent-red/30 p-4">
          <p className="flex items-center gap-2 text-sm text-accent-red">
            <CircleAlert className="h-4 w-4 shrink-0" /> {error}
          </p>
        </Card>
      )}

      {loading ? (
        <div className="space-y-4" role="status" aria-label="Loading the billing ledger">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[0, 1, 2, 3].map((i) => <div key={i} className="h-24 animate-pulse rounded-xl border border-border bg-surface" />)}
          </div>
          <div className="h-56 animate-pulse rounded-xl border border-border bg-surface" />
        </div>
      ) : data && totals ? (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <Figure label="GPU hours billed" value={formatGpuHours(totals.gpu_seconds)} note={`${totals.jobs} job(s), ${totals.segments} run(s)`} />
            <Figure
              label="Shadow cost"
              value={formatMicros(totals.shadow_cost_micros)}
              note={totals.unpriced_segments > 0 ? `${totals.unpriced_segments} run(s) on an unpriced cluster` : "at each cluster's rate card"}
              tone={totals.unpriced_segments > 0 ? "warning" : undefined}
            />
            <Figure
              label="Charged"
              value={formatMicros(totals.charged_micros)}
              note={nothingCharged ? "settlement is off" : `${totals.charged_segments} run(s) settled`}
            />
            <Figure
              label="Worst daily drift"
              value={worst === null ? "—" : `${worst.toFixed(2)}%`}
              note={
                drift.length === 0
                  ? "no day compared yet"
                  : over.length === 0
                    ? `all ${drift.length} day(s) within ${tolerance}%`
                    : `${over.length} of ${drift.length} day(s) over ${tolerance}%`
              }
              tone={over.length > 0 ? "danger" : drift.length > 0 ? "good" : undefined}
            />
          </div>

          <Card className="space-y-4 p-5">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div>
                <h3 className="text-base font-semibold">Daily drift</h3>
                <p className="mt-0.5 text-sm text-text-muted">
                  The ledger&apos;s GPU seconds against the cluster&apos;s own accounting, day by day. Over {tolerance}% is a
                  finding.
                </p>
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-muted">
                <span className="flex items-center gap-1.5"><span aria-hidden className="h-2 w-2 rounded-sm bg-accent-cyan" /> Ledger</span>
                <span className="flex items-center gap-1.5"><span aria-hidden className="h-2 w-2 rounded-sm bg-accent-violet" /> sacct</span>
              </div>
            </div>
            {drift.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-text-muted">
                No day has been compared in this window. The drift report runs hourly for enabled clusters.
              </p>
            ) : (
              <>
                <DriftChart drift={drift} clusterName={clusterName} showCluster={!clusterId && clusters.length > 1} />
                <DriftTable drift={drift} clusterName={clusterName} now={now} />
              </>
            )}
          </Card>

          <Card className="space-y-4 p-5">
            <div>
              <h3 className="flex items-center gap-2 text-base font-semibold">
                <ReceiptText className="h-4 w-4 text-text-muted" /> Billed runs
              </h3>
              <p className="mt-0.5 text-sm text-text-muted">
                One row per accounting record of a settled job, newest first. A requeued job has one per run, each
                billed by the rule shown.
              </p>
            </div>
            {segments.length === 0 ? (
              <p className="rounded-lg border border-dashed border-border px-4 py-8 text-center text-sm text-text-muted">
                No run was billed in the last {days} days.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[820px] text-left text-sm">
                  <thead>
                    <tr className="border-b border-border text-[11px] uppercase tracking-wide text-text-muted">
                      <th scope="col" className="py-2 pr-4 font-medium">Job</th>
                      <th scope="col" className="py-2 pr-4 font-medium">Ended</th>
                      <th scope="col" className="py-2 pr-4 font-medium">Slurm state</th>
                      <th scope="col" className="py-2 pr-4 text-right font-medium">Elapsed</th>
                      <th scope="col" className="py-2 pr-4 text-right font-medium">Billed</th>
                      <th scope="col" className="py-2 pr-4 font-medium">Rule</th>
                      <th scope="col" className="py-2 pr-4 text-right font-medium">GPUs</th>
                      <th scope="col" className="py-2 text-right font-medium">Shadow cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {segments.map((s) => (
                      <tr key={s.segment_id} className="border-b border-border/60 align-top last:border-0 hover:bg-surface-hover/50">
                        <td className="py-2.5 pr-4">
                          <button
                            type="button"
                            onClick={() => onOpenJob(s.xcelsior_job_id)}
                            className="block max-w-48 truncate text-left font-mono text-xs text-text-primary underline-offset-2 hover:text-accent-cyan hover:underline"
                            title={s.xcelsior_job_id}
                          >
                            {s.xcelsior_job_id}
                          </button>
                          <p className="text-[11px] text-text-muted">
                            Slurm {s.slurm_job_id} · run {s.run_index + 1} · {clusterName(s.cluster_id)}
                          </p>
                        </td>
                        <td className="py-2.5 pr-4 text-xs tabular-nums text-text-secondary" title={formatTimestamp(s.ended_at)}>
                          {formatAge(s.ended_at, now)}
                        </td>
                        <td className="py-2.5 pr-4 font-mono text-xs text-text-secondary">{s.slurm_state}</td>
                        <td className="py-2.5 pr-4 text-right text-xs tabular-nums text-text-secondary">
                          {formatDuration(s.elapsed_seconds)}
                          {s.suspended_seconds > 0 && (
                            <span className="block text-[11px] text-text-muted">+{formatDuration(s.suspended_seconds)} suspended</span>
                          )}
                        </td>
                        <td className="py-2.5 pr-4 text-right text-xs font-medium tabular-nums">{formatDuration(s.billable_seconds)}</td>
                        <td className="py-2.5 pr-4 text-xs text-text-secondary">{billingRuleLabel(s.billing_rule)}</td>
                        <td className="py-2.5 pr-4 text-right text-xs tabular-nums text-text-secondary">{s.gpu_count}</td>
                        <td className="py-2.5 text-right">
                          <span className="font-mono text-xs tabular-nums">{formatMicros(s.shadow_cost_micros)}</span>
                          <span className={cn("block text-[11px]", s.charged ? "text-emerald" : "text-text-muted")}>
                            {s.charged ? "charged" : s.shadow_cost_micros === null ? "not priced" : "not charged"}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {cursor && (
              <div className="flex justify-center">
                <Button variant="outline" size="sm" onClick={() => void loadMore()} disabled={loadingMore}>
                  {loadingMore && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Load more
                </Button>
              </div>
            )}
          </Card>
        </>
      ) : null}
    </div>
  );
}

function Figure({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note: string;
  tone?: "good" | "warning" | "danger";
}) {
  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <p className="text-xs text-text-secondary">{label}</p>
      <p
        className={cn(
          "mt-1.5 font-mono text-2xl font-bold tracking-tight",
          tone === "danger" && "text-accent-red",
          tone === "good" && "text-emerald",
        )}
      >
        {value}
      </p>
      <p className={cn("mt-1 text-[11px]", tone === "warning" ? "text-accent-gold" : "text-text-muted")}>{note}</p>
    </div>
  );
}

/** Two bars a day, oldest on the left. Heights share one scale so days compare. */
function DriftChart({
  drift,
  clusterName,
  showCluster,
}: {
  drift: HpcBillingDriftDay[];
  clusterName: (id: string) => string;
  showCluster: boolean;
}) {
  const ordered = useMemo(
    () => [...drift].sort((a, b) => a.day.localeCompare(b.day) || a.cluster_id.localeCompare(b.cluster_id)),
    [drift],
  );
  const peak = Math.max(1, ...ordered.map((d) => Math.max(d.ledger_gpu_seconds, d.sacct_gpu_seconds)));
  return (
    <div className="overflow-x-auto">
      <ol
        className="flex h-44 min-w-full items-end gap-2 border-b border-border pb-px"
        style={{ width: ordered.length > 16 ? `${ordered.length * 3.25}rem` : undefined }}
        aria-label="GPU seconds per day: ledger against accounting"
      >
        {ordered.map((d) => {
          const label = `${formatDay(d.day)}${showCluster ? `, ${clusterName(d.cluster_id)}` : ""}: ledger ${d.ledger_gpu_seconds} s, sacct ${d.sacct_gpu_seconds} s, drift ${d.drift_pct === null ? "not computed" : `${d.drift_pct}%`}`;
          return (
            <li key={`${d.cluster_id}-${d.day}`} className="flex h-full min-w-10 flex-1 flex-col justify-end" title={label}>
              <span className="sr-only">{label}</span>
              <div aria-hidden className="flex h-[calc(100%-2.25rem)] items-end justify-center gap-1">
                <span
                  className="w-3 rounded-t bg-accent-cyan sm:w-4"
                  style={{ height: `${Math.max(2, (d.ledger_gpu_seconds / peak) * 100)}%` }}
                />
                <span
                  className="w-3 rounded-t bg-accent-violet sm:w-4"
                  style={{ height: `${Math.max(2, (d.sacct_gpu_seconds / peak) * 100)}%` }}
                />
              </div>
              <div aria-hidden className="mt-1 h-8 text-center">
                <p className={cn("text-[10px] font-medium tabular-nums", d.within_tolerance ? "text-text-muted" : "text-accent-red")}>
                  {d.drift_pct === null ? "—" : `${d.drift_pct.toFixed(1)}%`}
                </p>
                <p className="truncate text-[10px] text-text-muted">{formatDay(d.day)}</p>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function unmatchedSummary(unmatched: Record<string, unknown>): string[] {
  const out: string[] = [];
  for (const [key, value] of Object.entries(unmatched)) {
    if (Array.isArray(value)) {
      if (value.length > 0) out.push(`${key.replace(/_/g, " ")}: ${value.join(", ")}`);
    } else if (typeof value === "number") {
      if (value > 0) out.push(`${key.replace(/_/g, " ")}: ${value}`);
    } else if (value) {
      out.push(`${key.replace(/_/g, " ")}: ${String(value)}`);
    }
  }
  return out;
}

function DriftTable({
  drift,
  clusterName,
  now,
}: {
  drift: HpcBillingDriftDay[];
  clusterName: (id: string) => string;
  now: number;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-sm">
        <thead>
          <tr className="border-b border-border text-[11px] uppercase tracking-wide text-text-muted">
            <th scope="col" className="py-2 pr-4 font-medium">Day</th>
            <th scope="col" className="py-2 pr-4 font-medium">Cluster</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Ledger</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">sacct</th>
            <th scope="col" className="py-2 pr-4 text-right font-medium">Drift</th>
            <th scope="col" className="py-2 font-medium">Verdict</th>
          </tr>
        </thead>
        <tbody>
          {drift.map((d) => {
            const notes = unmatchedSummary(d.unmatched);
            return (
              <tr key={`${d.cluster_id}-${d.day}`} className="border-b border-border/60 align-top last:border-0">
                <td className="py-2.5 pr-4 text-xs font-medium">{formatDay(d.day)}</td>
                <td className="py-2.5 pr-4 text-xs text-text-secondary">{clusterName(d.cluster_id)}</td>
                <td className="py-2.5 pr-4 text-right font-mono text-xs tabular-nums">
                  {d.ledger_gpu_seconds.toLocaleString("en-CA")} s
                  <span className="block font-sans text-[11px] text-text-muted">{d.ledger_records} run(s)</span>
                </td>
                <td className="py-2.5 pr-4 text-right font-mono text-xs tabular-nums">
                  {d.sacct_gpu_seconds.toLocaleString("en-CA")} s
                  <span className="block font-sans text-[11px] text-text-muted">{d.sacct_records} run(s)</span>
                </td>
                <td className={cn("py-2.5 pr-4 text-right font-mono text-xs tabular-nums", !d.within_tolerance && "text-accent-red")}>
                  {d.drift_pct === null ? "—" : `${d.drift_pct.toFixed(2)}%`}
                </td>
                <td className="py-2.5 text-xs">
                  {d.within_tolerance ? (
                    <span className="inline-flex items-center gap-1 text-emerald">
                      <CheckCircle2 className="h-3.5 w-3.5" /> Agrees
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-accent-red">
                      <TriangleAlert className="h-3.5 w-3.5" /> Over tolerance
                    </span>
                  )}
                  {notes.map((note) => (
                    <span key={note} className="block break-words font-mono text-[11px] text-text-muted">{note}</span>
                  ))}
                  <span className="block text-[11px] text-text-muted" title={formatTimestamp(d.checked_at)}>
                    checked {formatAge(d.checked_at, now)}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
