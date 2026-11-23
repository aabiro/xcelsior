"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Check, CircleAlert, Copy, Cpu, Loader2, RefreshCw, Search, XCircle } from "lucide-react";
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
import { ageSeconds, formatAge, formatDuration, formatTimestamp, jobMatches, jobStateLabel, jobTone } from "./hpc-format";

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
  /** The job in the address bar (`?job=`), opened on load and kept in sync. */
  openJobId: string | null;
  onOpenJob: (xcelsiorJobId: string | null) => void;
}

function jobName(job: HpcJob): string {
  return job.submission_name || job.xcelsior_job_id;
}

function StateBadge({ job }: { job: HpcJob }) {
  const label = jobStateLabel(job);
  // "running (RUNNING)" says the same thing twice; show Slurm's word only when it adds something.
  const slurm = job.slurm_state && job.slurm_state.toLowerCase() !== label.toLowerCase() ? job.slurm_state : null;
  return (
    <Badge variant={jobTone(job)} className="gap-1.5 text-xs capitalize">
      {label === "running" && <span aria-hidden className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald" />}
      {label}
      {slurm && <span className="font-mono text-[10px] normal-case opacity-80">{slurm}</span>}
    </Badge>
  );
}

function ProgressBar({ job }: { job: HpcJob }) {
  const { elapsed_seconds, time_limit_seconds, fraction } = job.scheduler_progress;
  if (elapsed_seconds === null) return <span className="text-xs text-text-muted">not started</span>;
  const percent = fraction === null ? null : Math.min(100, Math.round(fraction * 100));
  const near = percent !== null && percent >= 90 && !job.settled;
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
          className={cn("h-full rounded-full", near ? "bg-accent-gold" : job.settled ? "bg-text-muted" : "bg-accent-cyan")}
          style={{ width: `${percent ?? 0}%` }}
        />
      </div>
      <p className="mt-1 text-[11px] tabular-nums text-text-muted">
        {formatDuration(elapsed_seconds)}
        {time_limit_seconds ? ` of ${formatDuration(time_limit_seconds)}` : ""}
        {near && <span className="text-accent-gold"> · near limit</span>}
      </p>
    </div>
  );
}

export function JobsPanel(props: Props) {
  const { jobs, clusters, loading, error, now, scope, onScope, clusterFilter, onClusterFilter, openJobId, onOpenJob } = props;
  const [query, setQuery] = useState("");
  const [cancelling, setCancelling] = useState<HpcJob | null>(null);
  const shown = useMemo(() => jobs.filter((job) => jobMatches(job, query)), [jobs, query]);
  const clusterName = useMemo(() => {
    const names = new Map(clusters.map((c) => [c.cluster_id, c.display_name || c.cluster_id]));
    return (id: string | null) => (id ? names.get(id) ?? id : "unknown");
  }, [clusters]);

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

  const listed = openJobId ? jobs.find((j) => j.xcelsior_job_id === openJobId) ?? null : null;

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
          <Input autoComplete="off" autoCapitalize="none" autoCorrect="off" spellCheck={false}
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
        <div className="space-y-2" role="status" aria-label="Loading jobs">
          {[0, 1, 2].map((i) => <div key={i} className="h-14 animate-pulse rounded-lg bg-surface-hover" />)}
        </div>
      ) : shown.length === 0 ? (
        <div className="py-12 text-center">
          <div className="mx-auto mb-3 flex h-11 w-11 items-center justify-center rounded-2xl bg-surface-hover text-text-muted">
            <Cpu className="h-5 w-5" />
          </div>
          <p className="text-sm font-medium">
            {jobs.length === 0 ? `No ${scope === "all" ? "" : `${scope} `}jobs` : "No loaded job matches the filter"}
          </p>
          <p className="mt-1 text-xs text-text-muted">
            {jobs.length === 0
              ? clusterFilter
                ? `Nothing on ${clusterName(clusterFilter)} in this view.`
                : "Jobs appear here once the reconciler has seen them."
              : "Clear the filter, or load more."}
          </p>
        </div>
      ) : (
        <>
          {/* A table from md up; one card per job on a phone, where six columns don't fit. */}
          <div className="hidden overflow-x-auto md:block">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border text-[11px] uppercase tracking-wide text-text-muted">
                  <th scope="col" className="py-2 pr-4 font-medium">Job</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Cluster</th>
                  <th scope="col" className="py-2 pr-4 font-medium">State</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Time used</th>
                  <th scope="col" className="py-2 pr-4 font-medium">Observed</th>
                  <th scope="col" className="py-2 text-right font-medium"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {shown.map((job) => (
                  <tr
                    key={job.mapping_id}
                    className="group border-b border-border/60 align-top transition-colors last:border-0 hover:bg-surface-hover/50"
                  >
                    <td className="py-3 pr-4">
                      <button
                        type="button"
                        onClick={() => onOpenJob(job.xcelsior_job_id)}
                        aria-label={`View job ${jobName(job)}`}
                        className="block max-w-64 truncate text-left font-medium text-text-primary underline-offset-2 hover:text-accent-cyan hover:underline"
                        title={jobName(job)}
                      >
                        {jobName(job)}
                      </button>
                      <p className="font-mono text-xs text-text-muted">Slurm job {job.slurm_job_id}</p>
                    </td>
                    <td className="py-3 pr-4 text-text-secondary">{clusterName(job.cluster_id)}</td>
                    <td className="py-3 pr-4">
                      <StateBadge job={job} />
                      <JobNotes job={job} />
                    </td>
                    <td className="py-3 pr-4"><ProgressBar job={job} /></td>
                    <td className="py-3 pr-4">
                      <span className="text-xs tabular-nums text-text-secondary" title={formatTimestamp(job.last_observed_at)}>
                        {formatAge(job.last_observed_at, now)}
                      </span>
                      {job.stale && <Badge variant="warning" className="ml-2 px-1.5 py-0 text-[10px]">stale</Badge>}
                    </td>
                    <td className="py-3 text-right">
                      {!job.settled && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-text-muted hover:text-accent-red"
                          onClick={() => setCancelling(job)}
                          title="Cancel job"
                          aria-label={`Cancel job ${jobName(job)}`}
                        >
                          <XCircle className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <ul className="space-y-3 md:hidden">
            {shown.map((job) => (
              <li key={job.mapping_id} className="rounded-lg border border-border bg-background/40 p-3">
                <div className="flex items-start justify-between gap-2">
                  <button
                    type="button"
                    onClick={() => onOpenJob(job.xcelsior_job_id)}
                    aria-label={`View job ${jobName(job)}`}
                    className="min-w-0 text-left"
                  >
                    <span className="block truncate text-sm font-medium text-text-primary">{jobName(job)}</span>
                    <span className="block font-mono text-xs text-text-muted">
                      Slurm job {job.slurm_job_id} · {clusterName(job.cluster_id)}
                    </span>
                  </button>
                  <StateBadge job={job} />
                </div>
                <JobNotes job={job} />
                <div className="mt-3 flex items-end justify-between gap-3">
                  <ProgressBar job={job} />
                  <span className="shrink-0 text-[11px] tabular-nums text-text-muted">
                    seen {formatAge(job.last_observed_at, now)}
                    {job.stale && <span className="text-accent-gold"> · stale</span>}
                  </span>
                </div>
                {!job.settled && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-3 w-full text-accent-red"
                    onClick={() => setCancelling(job)}
                    aria-label={`Cancel job ${jobName(job)}`}
                  >
                    <XCircle className="h-3.5 w-3.5" /> Cancel job
                  </Button>
                )}
              </li>
            ))}
          </ul>
        </>
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

      {openJobId && (
        <JobDetail
          key={openJobId}
          jobId={openJobId}
          listed={listed}
          clusterName={clusterName}
          now={now}
          onClose={() => onOpenJob(null)}
        />
      )}
    </Card>
  );
}

function JobNotes({ job }: { job: HpcJob }) {
  const reason = job.state_reason && job.state_reason !== "None" ? job.state_reason : null;
  if (!reason && !job.awaiting_accounting_since && !job.missing_since) return null;
  return (
    <div className="mt-1 space-y-0.5 text-xs">
      {reason && <p className="max-w-56 truncate text-text-muted" title={reason}>{reason}</p>}
      {job.awaiting_accounting_since && <p className="text-accent-gold">awaiting accounting</p>}
      {job.missing_since && <p className="text-accent-red">missing from the cluster</p>}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-wide text-text-muted">{label}</dt>
      <dd className="mt-0.5 break-words text-sm text-text-primary">{children}</dd>
    </div>
  );
}

function CopyId({ value, label }: { value: string; label: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Could not copy to the clipboard");
    }
  };
  return (
    <button
      type="button"
      onClick={() => void copy()}
      aria-label={`Copy ${label}`}
      className="inline-flex max-w-full items-center gap-1.5 rounded font-mono text-xs text-text-primary hover:text-accent-cyan"
    >
      <span className="truncate">{value}</span>
      {copied ? <Check className="h-3 w-3 shrink-0 text-emerald" /> : <Copy className="h-3 w-3 shrink-0 text-text-muted" />}
    </button>
  );
}

/** Submitted → Started → Ended → Settled, with how long each gap took. */
function Timeline({ job, now }: { job: HpcJob; now: number }) {
  const gap = (from: string | null, to: string | null) => {
    const a = from ? Date.parse(from) : NaN;
    const b = to ? Date.parse(to) : NaN;
    return Number.isNaN(a) || Number.isNaN(b) ? null : Math.max(0, Math.round((b - a) / 1000));
  };
  const queued = gap(job.submitted_at, job.started_at) ?? (job.started_at ? null : ageSeconds(job.submitted_at, now));
  const ran = job.scheduler_progress.elapsed_seconds;
  const steps = [
    { label: "Submitted", at: job.submitted_at, note: null as string | null },
    {
      label: "Started",
      at: job.started_at,
      note: queued === null ? null : `${job.started_at ? "queued" : "queued so far"} ${formatDuration(queued)}`,
    },
    {
      label: "Ended",
      at: job.ended_at,
      note: ran === null ? null : `${job.ended_at ? "ran" : "running"} ${formatDuration(ran)}`,
    },
    {
      label: "Settled",
      at: job.settled_at,
      note: job.awaiting_accounting_since && !job.settled ? "waiting on accounting" : null,
    },
  ];
  return (
    <ol className="grid grid-cols-2 gap-x-3 gap-y-4 sm:grid-cols-4" aria-label="Job timeline">
      {steps.map((step, i) => {
        const done = !!step.at;
        const current = !done && (i === 0 || !!steps[i - 1].at);
        return (
          <li key={step.label} className="relative">
            <div className="flex items-center gap-2">
              <span
                aria-hidden
                className={cn(
                  "h-2.5 w-2.5 shrink-0 rounded-full border",
                  done
                    ? "border-emerald bg-emerald"
                    : current
                      ? "animate-pulse border-accent-cyan bg-accent-cyan/40"
                      : "border-border bg-transparent",
                )}
              />
              <span aria-hidden className={cn("h-px flex-1", done ? "bg-emerald/40" : "bg-border")} />
            </div>
            <p className={cn("mt-2 text-xs font-medium", done ? "text-text-primary" : "text-text-muted")}>{step.label}</p>
            <p className="text-[11px] tabular-nums text-text-muted">{done ? formatTimestamp(step.at) : current ? "in progress" : "—"}</p>
            {step.note && <p className="text-[11px] text-text-secondary">{step.note}</p>}
          </li>
        );
      })}
    </ol>
  );
}

function JobDetail({
  jobId,
  listed,
  clusterName,
  now,
  onClose,
}: {
  jobId: string;
  listed: HpcJob | null;
  clusterName: (id: string | null) => string;
  now: number;
  onClose: () => void;
}) {
  const [job, setJob] = useState<HpcJob | null>(listed);
  const [jobError, setJobError] = useState<string | null>(null);
  const [stream, setStream] = useState<"out" | "err">("out");
  const [output, setOutput] = useState<HpcJobOutput | null>(null);
  const [outputError, setOutputError] = useState<string | null>(null);
  const [loadingOutput, setLoadingOutput] = useState(true);

  useEffect(() => {
    let live = true;
    // The single-job read returns the job's current submission, which for a
    // retried job may be a later attempt than the row that was clicked.
    api.fetchHpcJob(jobId)
      .then((res) => {
        if (!live) return;
        if (!listed || res.job.mapping_id === listed.mapping_id) setJob(res.job);
      })
      .catch((err) => {
        // A listed row is still shown; a deep link to a job that isn't there says so.
        if (live && !listed) setJobError(err instanceof Error ? err.message : "Job not found");
      });
    return () => { live = false; };
  }, [jobId, listed]);

  const loadOutput = useCallback(async () => {
    setLoadingOutput(true);
    try {
      const res = await api.fetchHpcJobOutput(jobId, stream);
      setOutput(res.output);
      setOutputError(null);
    } catch (err) {
      setOutput(null);
      setOutputError(err instanceof Error ? err.message : "Could not load output");
    } finally {
      setLoadingOutput(false);
    }
  }, [jobId, stream]);

  useEffect(() => { void loadOutput(); }, [loadOutput]);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(output?.content ?? "");
      toast.success("Output copied");
    } catch {
      toast.error("Could not copy to the clipboard");
    }
  };

  if (!job) {
    return (
      <Dialog open onClose={onClose} title={jobId} description="Slurm job" maxWidth="max-w-3xl">
        <div className="py-10 text-center text-sm" role="status">
          {jobError ? (
            <span className="text-accent-red">{jobError}</span>
          ) : (
            <span className="inline-flex items-center gap-2 text-text-muted">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading job…
            </span>
          )}
        </div>
      </Dialog>
    );
  }

  const reported = job.reported_progress;
  const reportedPercent =
    reported?.valid && reported.fraction !== null && reported.fraction !== undefined
      ? Math.min(100, Math.round(reported.fraction * 100))
      : null;

  return (
    <Dialog
      open
      onClose={onClose}
      title={jobName(job)}
      description={`Slurm job ${job.slurm_job_id} on ${clusterName(job.cluster_id)}`}
      maxWidth="max-w-3xl"
    >
      <div className="space-y-6 pt-4">
        <div className="flex flex-wrap items-center gap-2">
          <StateBadge job={job} />
          {job.settled
            ? <Badge variant="outline" className="text-xs">settled</Badge>
            : <Badge variant="info" className="text-xs">open</Badge>}
          {job.stale && <Badge variant="warning" className="text-xs">stale</Badge>}
          {job.awaiting_accounting_since && <Badge variant="warning" className="text-xs">awaiting accounting</Badge>}
          <span className="ml-auto text-xs text-text-muted" title={formatTimestamp(job.last_observed_at)}>
            observed {formatAge(job.last_observed_at, now)}
          </span>
        </div>

        {job.missing_since && (
          <p className="rounded-lg border border-accent-red/30 bg-accent-red/10 px-3 py-2 text-xs text-text-primary">
            Missing from the controller and accounting since {formatTimestamp(job.missing_since)}.
          </p>
        )}

        <Timeline job={job} now={now} />

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-border bg-background/40 p-3">
            <p className="mb-2 text-xs font-semibold text-text-secondary">Time used against the limit</p>
            <ProgressBar job={job} />
          </div>
          <div className="rounded-lg border border-border bg-background/40 p-3">
            <p className="mb-2 text-xs font-semibold text-text-secondary">Progress reported by the job</p>
            {!reported ? (
              <p className="text-xs text-text-muted">The job does not write a progress file.</p>
            ) : reported.valid ? (
              <>
                <div
                  className="h-1.5 overflow-hidden rounded-full bg-surface-hover"
                  role="progressbar"
                  aria-label="Progress reported by the job"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={reportedPercent ?? undefined}
                >
                  <div className="h-full rounded-full bg-accent-violet" style={{ width: `${reportedPercent ?? 0}%` }} />
                </div>
                <p className="mt-1 text-[11px] text-text-muted">
                  {reportedPercent !== null ? `${reportedPercent}%` : "No fraction"}
                  {reported.message ? ` · ${reported.message}` : ""}
                  {reported.stale ? " · stale" : ""}
                </p>
              </>
            ) : (
              <p className="text-xs text-accent-gold">{reported.error || "The progress file breaks the contract."}</p>
            )}
          </div>
        </div>

        <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
          <Field label="Xcelsior job"><CopyId value={job.xcelsior_job_id} label="Xcelsior job ID" /></Field>
          <Field label="Slurm job"><CopyId value={job.slurm_job_id} label="Slurm job ID" /></Field>
          <Field label="Attempt">
            {job.attempt_id ? <CopyId value={job.attempt_id} label="attempt ID" /> : "—"}
          </Field>
          <Field label="Desired state">{job.desired_state || "—"}</Field>
          <Field label="Exit code"><span className="font-mono text-xs">{job.exit_code || "—"}</span></Field>
          <Field label="Derived exit code"><span className="font-mono text-xs">{job.derived_exit_code || "—"}</span></Field>
          <Field label="Reason">{job.state_reason && job.state_reason !== "None" ? job.state_reason : "—"}</Field>
        </dl>

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
                    "rounded-md px-3 py-1 font-mono text-xs font-medium transition-colors",
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
                className="max-h-80 min-h-24 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-border bg-background p-3 font-mono text-[11px] leading-relaxed text-text-secondary"
                data-testid="hpc-job-output"
                tabIndex={0}
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
