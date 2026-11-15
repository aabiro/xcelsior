"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, Inbox, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select } from "@/components/ui/input";
import * as api from "@/lib/api";
import type { HpcCluster, HpcSubmission, HpcSubmissionScope, HpcSubmissionState } from "@/lib/api";
import { cn } from "@/lib/utils";
import { SUBMISSION_STATE_HELP, formatAge, formatTimestamp, submissionTone } from "./hpc-format";

const STATES: HpcSubmissionState[] = ["submitted", "pending", "submitting", "unknown", "rejected", "lost"];
const OPEN: HpcSubmissionState[] = ["pending", "submitting", "unknown"];

interface Props {
  clusters: HpcCluster[];
  now: number;
  refreshKey: number;
  onOpenJob: (xcelsiorJobId: string) => void;
}

/** The submit log: what was sent to sbatch, including what never became a job. */
export function SubmissionsPanel({ clusters, now, refreshKey, onOpenJob }: Props) {
  const [scope, setScope] = useState<HpcSubmissionScope>("all");
  const [clusterId, setClusterId] = useState("");
  const [rows, setRows] = useState<HpcSubmission[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [cursor, setCursor] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const request = useRef(0);

  const load = useCallback(async () => {
    const id = ++request.current;
    try {
      const res = await api.fetchHpcSubmissions({ state: scope, clusterId: clusterId || undefined });
      if (id !== request.current) return;
      setRows(res.submissions ?? []);
      setCounts(res.counts ?? {});
      setCursor(res.next_cursor ?? null);
      setError(null);
    } catch (err) {
      if (id !== request.current) return;
      setError(err instanceof Error ? err.message : "Could not load the submit log");
    } finally {
      if (id === request.current) setLoading(false);
    }
  }, [scope, clusterId]);

  useEffect(() => { void load(); }, [load, refreshKey]);

  const loadMore = async () => {
    if (!cursor) return;
    setLoadingMore(true);
    try {
      const res = await api.fetchHpcSubmissions({ state: scope, clusterId: clusterId || undefined, cursor });
      setRows((prev) => {
        const seen = new Set(prev.map((s) => s.submission_name));
        return [...prev, ...(res.submissions ?? []).filter((s) => !seen.has(s.submission_name))];
      });
      setCursor(res.next_cursor ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load more of the submit log");
    } finally {
      setLoadingMore(false);
    }
  };

  const clusterName = useMemo(() => {
    const names = new Map(clusters.map((c) => [c.cluster_id, c.display_name || c.cluster_id]));
    return (id: string) => names.get(id) ?? id;
  }, [clusters]);

  const total = STATES.reduce((n, s) => n + (counts[s] ?? 0), 0);
  const open = OPEN.reduce((n, s) => n + (counts[s] ?? 0), 0);
  const chips: { id: HpcSubmissionScope; label: string; count: number; alert?: boolean }[] = [
    { id: "all", label: "All", count: total },
    { id: "open", label: "Unresolved", count: open, alert: (counts.unknown ?? 0) > 0 },
    { id: "submitted", label: "Submitted", count: counts.submitted ?? 0 },
    { id: "rejected", label: "Rejected", count: counts.rejected ?? 0, alert: (counts.rejected ?? 0) > 0 },
    { id: "lost", label: "Lost", count: counts.lost ?? 0, alert: (counts.lost ?? 0) > 0 },
  ];

  return (
    <Card className="space-y-4 p-5">
      <div>
        <h3 className="text-base font-semibold">Submit log</h3>
        <p className="mt-0.5 text-sm text-text-muted">
          A row is written before any sbatch call, so a submission that was refused, timed out or lost is here even
          though it never became a job.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Submission state" className="flex flex-wrap gap-1.5">
          {chips.map((chip) => (
            <button
              key={chip.id}
              type="button"
              aria-pressed={scope === chip.id}
              onClick={() => setScope(chip.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors",
                scope === chip.id
                  ? "border-accent-cyan/50 bg-accent-cyan/15 text-text-primary"
                  : "border-border text-text-muted hover:text-text-primary",
              )}
            >
              {chip.label}
              <span
                className={cn(
                  "rounded-full px-1.5 text-[10px] tabular-nums",
                  chip.alert ? "bg-accent-gold/20 text-accent-gold" : "bg-surface-hover text-text-secondary",
                )}
              >
                {chip.count}
              </span>
            </button>
          ))}
        </div>
        <Select
          aria-label="Submit log cluster"
          value={clusterId}
          onChange={(e) => setClusterId(e.target.value)}
          className="ml-auto h-9 w-auto min-w-40"
        >
          <option value="">All clusters</option>
          {clusters.map((c) => (
            <option key={c.cluster_id} value={c.cluster_id}>{c.display_name || c.cluster_id}</option>
          ))}
        </Select>
      </div>

      {error && (
        <p className="flex items-center gap-2 text-sm text-accent-red">
          <CircleAlert className="h-4 w-4 shrink-0" /> {error}
        </p>
      )}

      {loading ? (
        <div className="space-y-2" role="status" aria-label="Loading the submit log">
          {[0, 1, 2].map((i) => <div key={i} className="h-12 animate-pulse rounded-lg bg-surface-hover" />)}
        </div>
      ) : rows.length === 0 ? (
        <div className="py-12 text-center">
          <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-2xl bg-surface-hover text-text-muted">
            <Inbox className="h-5 w-5" />
          </div>
          <p className="text-sm font-medium">{total === 0 ? "Nothing has been submitted" : "No submission in this view"}</p>
          <p className="mt-1 text-xs text-text-muted">
            {total === 0
              ? "The scheduler writes a row here when it places a job on a cluster."
              : "Pick another state or cluster."}
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead>
              <tr className="border-b border-border text-[11px] uppercase tracking-wide text-text-muted">
                <th scope="col" className="py-2 pr-4 font-medium">Submission</th>
                <th scope="col" className="py-2 pr-4 font-medium">Cluster</th>
                <th scope="col" className="py-2 pr-4 font-medium">State</th>
                <th scope="col" className="py-2 pr-4 font-medium">Slurm job</th>
                <th scope="col" className="py-2 pr-4 text-right font-medium">Calls</th>
                <th scope="col" className="py-2 font-medium">Created</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((s) => (
                <tr key={s.submission_name} className="border-b border-border/60 align-top last:border-0 hover:bg-surface-hover/50">
                  <td className="py-2.5 pr-4">
                    <p className="max-w-56 truncate font-medium" title={s.submission_name}>{s.submission_name}</p>
                    <button
                      type="button"
                      onClick={() => onOpenJob(s.xcelsior_job_id)}
                      disabled={!s.slurm_job_id}
                      title={s.slurm_job_id ? "Open the job" : "It never became a job"}
                      className="font-mono text-xs text-text-muted underline-offset-2 enabled:hover:text-accent-cyan enabled:hover:underline disabled:cursor-default"
                    >
                      {s.xcelsior_job_id}
                    </button>
                  </td>
                  <td className="py-2.5 pr-4 text-xs text-text-secondary">{clusterName(s.cluster_id)}</td>
                  <td className="py-2.5 pr-4">
                    <Badge variant={submissionTone(s.state)} className="text-xs" title={SUBMISSION_STATE_HELP[s.state]}>
                      {s.state}
                    </Badge>
                    {s.error_code && (
                      <p className="mt-1 font-mono text-[11px] text-accent-red">{s.error_code}</p>
                    )}
                    {s.error && (
                      <p className="mt-0.5 max-w-64 break-words text-[11px] text-text-muted">{s.error}</p>
                    )}
                  </td>
                  <td className="py-2.5 pr-4 font-mono text-xs text-text-secondary">{s.slurm_job_id ?? "—"}</td>
                  <td className="py-2.5 pr-4 text-right text-xs tabular-nums text-text-secondary">
                    {s.sbatch_calls} sbatch
                    {s.checks > 0 && <span className="block text-[11px] text-text-muted">{s.checks} lookup(s)</span>}
                  </td>
                  <td className="py-2.5 text-xs tabular-nums text-text-secondary" title={formatTimestamp(s.created_at)}>
                    {formatAge(s.created_at, now)}
                    {s.resolved_at && (
                      <span className="block text-[11px] text-text-muted" title={formatTimestamp(s.resolved_at)}>
                        resolved {formatAge(s.resolved_at, now)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {cursor && !loading && (
        <div className="flex justify-center">
          <Button variant="outline" size="sm" onClick={() => void loadMore()} disabled={loadingMore}>
            {loadingMore && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Load more
          </Button>
        </div>
      )}
    </Card>
  );
}
