"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CircleAlert, Copy, Cpu, Eye, Loader2, RefreshCw, Search, XCircle } from "lucide-react";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Dialog } from "@/components/ui/dialog";
import { Input, Select } from "@/components/ui/input";
import * as api from "@/lib/api";
import type { HpcCluster, HpcJob, HpcJobOutput, HpcJobScope } from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatAge, formatDuration, formatTimestamp, jobMatches, jobStateLabel, jobTone } from "./hpc-format";

const SCOPES: { id: HpcJobScope; label: string }[] = [
  { id: "open", label: "Open" },
  { id: "settled", label: "Settled" },
  { id: "all", label: "All" },
];

interface Props {
  jobs: HpcJob[];
  clusters: HpcCluster[];
  loading: boolean;
  error: string | null;
  now: number;
  scope: HpcJobScope;
  onScope: (scope: HpcJobScope) => void;
  clusterFilter: string;
  onClusterFilter: (clusterId: string) => void;
  hasMore: boolean;
  loadingMore: boolean;
  onLoadMore: () => void;
  onChanged: () => void;
}

function jobName(job: HpcJob): string {
  return job.submission_name || job.xcelsior_job_id;
}

function ProgressBar({ job }: { job: HpcJob }) {
  const { elapsed_seconds, time_limit_seconds, fraction } = job.scheduler_progress;
  if (elapsed_seconds === null) return <span className="text-xs text-text-muted">not started</span>;
  const percent = fraction === null ? null : Math.round(fraction * 100);
  return (
    <div className="min-w-28">
      <div
        className="h-1.5 overflow-hidden rounded-full bg-surface-hover"
        role="progressbar"
        aria-label="Elapsed against the time limit"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
      >
        <div
          className={cn("h-full rounded-full", percent !== null && percent >= 90 ? "bg-accent-gold" : "bg-accent-cyan")}
          style={{ width: `${percent ?? 0}%` }}
        />
      </div>
      <p className="mt-1 text-[11px] tabular-nums text-text-muted">
        {formatDuration(elapsed_seconds)}
        {time_limit_seconds ? ` of ${formatDuration(time_limit_seconds)}` : ""}
      </p>
    </div>
  );
}

export function JobsPanel(props: Props) {
  const { jobs, clusters, loading, error, now, scope, onScope, clusterFilter, onClusterFilter } = props;
  const [query, setQuery] = useState("");
  const [detail, setDetail] = useState<HpcJob | null>(null);
  const [cancelling, setCancelling] = useState<HpcJob | null>(null);
  const shown = useMemo(() => jobs.filter((job) => jobMatches(job, query)), [jobs, query]);

  const cancel = async (job: HpcJob) => {
    setCancelling(null);
    try {
      await api.cancelSlurmJob(job.slurm_job_id, job.cluster_id);
      // The reconciler is the only poller: the row changes on its next pass.
      toast.success(`Cancel sent for Slurm job ${job.slurm_job_id}. The state updates on the next poll.`);
      props.onChanged();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to cancel job");
    }
  };

  return (
    <Card className="space-y-4 p-5">
      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label="Job state" className="flex gap-1 rounded-lg bg-surface-hover p-1">
          {SCOPES.map((s) => (
            <button
              key={s.id}
              type="button"
              aria-pressed={scope === s.id}
              onClick={() => onScope(s.id)}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                scope === s.id ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary",
              )}
            >
              {s.label}
            </button>
          ))}
        </div>
        <Select
          aria-label="Cluster"
          value={clusterFilter}
          onChange={(e) => onClusterFilter(e.target.value)}
          className="h-9 w-auto min-w-40"
        >
          <option value="">All clusters</option>
          {clusters.map((c) => (
            <option key={c.cluster_id} value={c.cluster_id}>{c.display_name || c.cluster_id}</option>
          ))}
        </Select>
        <div className="relative min-w-48 flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-text-muted" />
          <Input
            aria-label="Filter loaded jobs"
            placeholder="Filter by job ID, Slurm ID or name"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-9 pl-9"
          />
        </div>
      </div>

      {error && (
        <p className="flex items-center gap-2 text-sm text-accent-red">
          <CircleAlert className="h-4 w-4 shrink-0" /> {error}
        </p>
      )}

      {loading ? (
        <div className="flex justify-center py-10" role="status" aria-label="Loading jobs">
          <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
        </div>
      ) : shown.length === 0 ? (
        <div className="py-10 text-center">
          <Cpu className="mx-auto mb-2 h-8 w-8 text-text-muted" />
          <p className="text-sm text-text-muted">
            {jobs.length === 0 ? `No ${scope === "all" ? "" : `${scope} `}jobs` : "No loaded job matches the filter"}
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm">
            <thead>
              <tr className="border-b border-border text-[11px] uppercase tracking-wide text-text-muted">
                <th scope="col" className="py-2 pr-4 font-medium">Job</th>
                <th scope="col" className="py-2 pr-4 font-medium">Cluster</th>
                <th scope="col" className="py-2 pr-4 font-medium">State</th>
                <th scope="col" className="py-2 pr-4 font-medium">Time used</th>
                <th scope="col" className="py-2 pr-4 font-medium">Observed</th>
                <th scope="col" className="py-2 text-right font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((job) => (
                <tr key={job.mapping_id} className="border-b border-border/60 align-top last:border-0">
                  <td className="py-3 pr-4">
                    <p className="max-w-56 truncate font-medium" title={jobName(job)}>{jobName(job)}</p>
                    <p className="font-mono text-xs text-text-muted">Slurm job {job.slurm_job_id}</p>
                  </td>
                  <td className="py-3 pr-4 text-text-secondary">{job.cluster_id || "unknown"}</td>
                  <td className="py-3 pr-4">
                    <Badge variant={jobTone(job)} className="text-xs">
                      {job.slurm_state ? `${jobStateLabel(job)} (${job.slurm_state})` : jobStateLabel(job)}
                    </Badge>
                    {job.state_reason && job.state_reason !== "None" && (
                      <p className="mt-1 max-w-48 truncate text-xs text-text-muted" title={job.state_reason}>
                        {job.state_reason}
                      </p>
                    )}
                    {job.awaiting_accounting_since && (
                      <p className="mt-1 text-xs text-accent-gold">awaiting accounting</p>
                    )}
                  </td>
                  <td className="py-3 pr-4"><ProgressBar job={job} /></td>
                  <td className="py-3 pr-4">
                    <span className="text-xs tabular-nums text-text-secondary" title={formatTimestamp(job.last_observed_at)}>
                      {formatAge(job.last_observed_at, now)}
                    </span>
                    {job.stale && <Badge variant="warning" className="ml-2 px-1.5 py-0 text-[10px]">stale</Badge>}
                  </td>
                  <td className="py-3">
                    <div className="flex justify-end gap-1">
                      <Button variant="ghost" size="icon" className="h-8 w-8" onClick={() => setDetail(job)} title="View job" aria-label={`View job ${jobName(job)}`}>
                        <Eye className="h-3.5 w-3.5" />
                      </Button>
                      {!job.settled && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-accent-red hover:text-accent-red"
                          onClick={() => setCancelling(job)}
                          title="Cancel job"
                          aria-label={`Cancel job ${jobName(job)}`}
                        >
                          <XCircle className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {props.hasMore && !loading && (
        <div className="flex justify-center">
          <Button variant="outline" size="sm" onClick={props.onLoadMore} disabled={props.loadingMore}>
            {props.loadingMore && <Loader2 className="h-3.5 w-3.5 animate-spin" />} Load more
          </Button>
        </div>
      )}

      <ConfirmDialog
        open={!!cancelling}
        title={cancelling ? `Cancel Slurm job ${cancelling.slurm_job_id}?` : ""}
        description={
          cancelling
            ? `This sends scancel to ${cancelling.cluster_id || "the cluster"} for ${jobName(cancelling)}. It cannot be undone.`
            : ""
        }
        confirmLabel="Cancel job"
        cancelLabel="Keep running"
        variant="danger"
        onConfirm={() => { if (cancelling) void cancel(cancelling); }}
        onCancel={() => setCancelling(null)}
      />

      {detail && <JobDetail job={detail} now={now} onClose={() => setDetail(null)} />}
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-wide text-text-muted">{label}</dt>
      <dd className="break-words text-sm text-text-primary">{children}</dd>
    </div>
  );
}

function JobDetail({ job: listed, now, onClose }: { job: HpcJob; now: number; onClose: () => void }) {
  const [job, setJob] = useState<HpcJob>(listed);
  const [stream, setStream] = useState<"out" | "err">("out");
  const [output, setOutput] = useState<HpcJobOutput | null>(null);
  const [outputError, setOutputError] = useState<string | null>(null);
  const [loadingOutput, setLoadingOutput] = useState(true);

  useEffect(() => {
    let live = true;
    // The single-job read returns the job's current submission, which for a
    // retried job may be a later attempt than the row that was clicked.
    api.fetchHpcJob(listed.xcelsior_job_id)
      .then((res) => { if (live && res.job.mapping_id === listed.mapping_id) setJob(res.job); })
      .catch(() => { /* the listed row is still shown */ });
    return () => { live = false; };
  }, [listed]);

  const loadOutput = useCallback(async () => {
    setLoadingOutput(true);
    try {
      const res = await api.fetchHpcJobOutput(listed.xcelsior_job_id, stream);
      setOutput(res.output);
      setOutputError(null);
    } catch (err) {
      setOutput(null);
      setOutputError(err instanceof Error ? err.message : "Could not load output");
    } finally {
      setLoadingOutput(false);
    }
  }, [listed.xcelsior_job_id, stream]);

  useEffect(() => { void loadOutput(); }, [loadOutput]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(output?.content ?? "");
      toast.success("Output copied");
    } catch {
      toast.error("Could not copy to the clipboard");
    }
  };

  const reported = job.reported_progress;

  return (
    <Dialog
      open
      onClose={onClose}
      title={jobName(job)}
      description={`Slurm job ${job.slurm_job_id} on ${job.cluster_id || "unknown cluster"}`}
      maxWidth="max-w-3xl"
    >
      <div className="space-y-5 pt-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={jobTone(job)} className="text-xs">
            {job.slurm_state ? `${jobStateLabel(job)} (${job.slurm_state})` : jobStateLabel(job)}
          </Badge>
          {job.stale && <Badge variant="warning" className="text-xs">stale</Badge>}
          {job.settled
            ? <Badge variant="outline" className="text-xs">settled</Badge>
            : <Badge variant="info" className="text-xs">open</Badge>}
          {job.awaiting_accounting_since && <Badge variant="warning" className="text-xs">awaiting accounting</Badge>}
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
          <Field label="Xcelsior job"><span className="font-mono text-xs">{job.xcelsior_job_id}</span></Field>
          <Field label="Attempt"><span className="font-mono text-xs">{job.attempt_id || "—"}</span></Field>
          <Field label="Desired state">{job.desired_state || "—"}</Field>
          <Field label="Exit code"><span className="font-mono text-xs">{job.exit_code || "—"}</span></Field>
          <Field label="Derived exit code"><span className="font-mono text-xs">{job.derived_exit_code || "—"}</span></Field>
          <Field label="Reason">{job.state_reason || "—"}</Field>
          <Field label="Submitted">{formatTimestamp(job.submitted_at)}</Field>
          <Field label="Started">{formatTimestamp(job.started_at)}</Field>
          <Field label="Ended">{formatTimestamp(job.ended_at)}</Field>
          <Field label="Settled">{formatTimestamp(job.settled_at)}</Field>
          <Field label="Last observed">{formatAge(job.last_observed_at, now)}</Field>
          <Field label="Time used">
            {formatDuration(job.scheduler_progress.elapsed_seconds)}
            {job.scheduler_progress.time_limit_seconds
              ? ` of ${formatDuration(job.scheduler_progress.time_limit_seconds)}`
              : ""}
          </Field>
        </dl>

        {job.missing_since && (
          <p className="rounded-lg border border-accent-red/30 bg-accent-red/10 px-3 py-2 text-xs text-text-primary">
            Missing from the controller and accounting since {formatTimestamp(job.missing_since)}.
          </p>
        )}

        {reported && (
          <div className="rounded-lg border border-border bg-background/40 px-3 py-2 text-xs">
            <p className="mb-1 font-semibold text-text-secondary">Progress reported by the job</p>
            {reported.valid ? (
              <p className="text-text-primary">
                {reported.fraction !== null && reported.fraction !== undefined
                  ? `${Math.round(reported.fraction * 100)}%`
                  : "No fraction"}
                {reported.message ? ` · ${reported.message}` : ""}
                {reported.stale ? " · stale" : ""}
              </p>
            ) : (
              <p className="text-accent-gold">{reported.error || "The progress file breaks the contract."}</p>
            )}
          </div>
        )}

        <div>
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div role="group" aria-label="Output stream" className="flex gap-1 rounded-lg bg-surface-hover p-1">
              {(["out", "err"] as const).map((s) => (
                <button
                  key={s}
                  type="button"
                  aria-pressed={stream === s}
                  onClick={() => setStream(s)}
                  className={cn(
                    "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                    stream === s ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary",
                  )}
                >
                  {s === "out" ? "stdout" : "stderr"}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-1">
              <Button variant="ghost" size="sm" onClick={() => void loadOutput()} disabled={loadingOutput}>
                <RefreshCw className={cn("h-3.5 w-3.5", loadingOutput && "animate-spin")} /> Reload
              </Button>
              <Button variant="ghost" size="sm" onClick={() => void copy()} disabled={!output?.content}>
                <Copy className="h-3.5 w-3.5" /> Copy
              </Button>
            </div>
          </div>

          {outputError ? (
            <p className="text-sm text-accent-red">{outputError}</p>
          ) : (
            <>
              <pre
                className="max-h-80 min-h-24 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-background p-3 font-mono text-[11px] text-text-secondary"
                data-testid="hpc-job-output"
              >
                {loadingOutput && !output ? "Loading…" : output?.content || "No output cached yet."}
              </pre>
              {output && (
                <p className="mt-1.5 text-[11px] text-text-muted">
                  {output.final
                    ? "Final copy."
                    : output.fetched_age_seconds === null
                      ? "Never fetched."
                      : `Fetched ${formatDuration(output.fetched_age_seconds)} ago.`}
                  {output.truncated ? " Showing the tail only." : ""}
                  {output.refresh_pending
                    ? ` A fresh copy is queued; reload after ${formatDuration(output.poll_interval_seconds)}.`
                    : ""}
                  {output.fetch_error_code ? ` Last fetch failed: ${output.fetch_error_code}.` : ""}
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </Dialog>
  );
}
