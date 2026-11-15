import type { HpcCluster, HpcJob } from "@/lib/api";

/** A health signal older than this many poll intervals counts as late. */
export const LATE_INTERVALS = 3;

export type ClusterStatus = "healthy" | "degraded" | "backing_off" | "never_polled" | "disabled";

export const CLUSTER_STATUS_LABEL: Record<ClusterStatus, string> = {
  healthy: "Healthy",
  degraded: "Degraded",
  backing_off: "Backing off",
  never_polled: "Never polled",
  disabled: "Disabled",
};

function ms(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const value = Date.parse(iso);
  return Number.isNaN(value) ? null : value;
}

/** Seconds since `iso`, or null when it is missing or unparseable. */
export function ageSeconds(iso: string | null | undefined, now: number): number | null {
  const at = ms(iso);
  return at === null ? null : Math.max(0, Math.round((now - at) / 1000));
}

/** 42 → "42s", 3725 → "1h 2m", 200000 → "2d 7h". */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

export function formatAge(iso: string | null | undefined, now: number): string {
  const age = ageSeconds(iso, now);
  return age === null ? "never" : `${formatDuration(age)} ago`;
}

export function formatTimestamp(iso: string | null | undefined): string {
  const at = ms(iso);
  return at === null ? "—" : new Date(at).toLocaleString();
}

/** Whether one of a cluster's health signals is missing or late. */
export function signalLate(iso: string | null, cluster: HpcCluster, now: number): boolean {
  const age = ageSeconds(iso, now);
  return age === null || age > LATE_INTERVALS * cluster.poll_interval_seconds;
}

export function clusterStatus(cluster: HpcCluster, now: number): ClusterStatus {
  if (!cluster.enabled) return "disabled";
  const health = cluster.health;
  const backingOff = ms(health.backing_off_until);
  if (backingOff !== null && backingOff > now) return "backing_off";
  if (!health.last_poll_at) return "never_polled";
  if (
    health.consecutive_failures > 0 ||
    health.last_error_code ||
    signalLate(health.transport_ok_at, cluster, now) ||
    signalLate(health.controller_ok_at, cluster, now) ||
    signalLate(health.accounting_ok_at, cluster, now)
  ) {
    return "degraded";
  }
  return "healthy";
}

/** The label a job row shows: the verdict once settled, else what was observed. */
export function jobStateLabel(job: HpcJob): string {
  if (job.settled && job.verdict && job.verdict !== "unknown") return job.verdict;
  return job.observed_state || "unknown";
}

export type JobTone = "running" | "queued" | "completed" | "failed" | "cancelled" | "warning" | "outline";

export function jobTone(job: HpcJob): JobTone {
  switch (jobStateLabel(job)) {
    case "running":
      return "running";
    case "pending":
      return "queued";
    case "succeeded":
    case "completed":
      return "completed";
    case "failed":
    case "preempted":
      return "failed";
    case "cancelled":
      return "cancelled";
    case "held":
    case "suspended":
      return "warning";
    default:
      return "outline";
  }
}

export interface HpcSummary {
  clustersEnabled: number;
  clustersHealthy: number;
  open: number;
  running: number;
  pending: number;
  stale: number;
  awaitingAccounting: number;
  orphans: number;
  /** False when the counts come from the loaded page and more jobs may exist. */
  exact: boolean;
}

type OpenJobKey = "total" | "running" | "pending" | "stale" | "awaiting_accounting";

/** Counts come from the registry when it sends them, else from the loaded page of open jobs. */
export function summarise(clusters: HpcCluster[], openJobs: HpcJob[], now: number): HpcSummary {
  const enabled = clusters.filter((c) => c.enabled);
  // The registry counts every open job; the loaded page stops at 200.
  const exact = clusters.length > 0 && clusters.every((c) => c.open_jobs);
  const total = (key: OpenJobKey) => clusters.reduce((n, c) => n + (c.open_jobs?.[key] ?? 0), 0);
  return {
    clustersEnabled: enabled.length,
    clustersHealthy: enabled.filter((c) => clusterStatus(c, now) === "healthy").length,
    open: exact ? total("total") : openJobs.length,
    running: exact ? total("running") : openJobs.filter((j) => j.observed_state === "running").length,
    pending: exact ? total("pending") : openJobs.filter((j) => j.observed_state === "pending").length,
    stale: exact ? total("stale") : openJobs.filter((j) => j.stale).length,
    awaitingAccounting: exact
      ? total("awaiting_accounting")
      : openJobs.filter((j) => j.awaiting_accounting_since).length,
    orphans: clusters.reduce((n, c) => n + c.orphan_slurm_job_ids.length, 0),
    exact,
  };
}

export type HpcTab = "fleet" | "jobs" | "ledger" | "submissions" | "submit";

export interface AttentionItem {
  /** `critical`: something is not being watched. `warning`: look when you can. */
  severity: "critical" | "warning";
  text: string;
  /** Where the operator can act on it. */
  tab: HpcTab;
  clusterId?: string;
}

/** Things an operator should look at, most urgent first. */
export function attentionItems(clusters: HpcCluster[], openJobs: HpcJob[], now: number): AttentionItem[] {
  const items: AttentionItem[] = [];
  for (const cluster of clusters) {
    const status = clusterStatus(cluster, now);
    const name = cluster.display_name || cluster.cluster_id;
    const code = cluster.health.last_error_code ? ` (${cluster.health.last_error_code})` : "";
    const at = { tab: "fleet" as const, clusterId: cluster.cluster_id };
    if (status === "backing_off") {
      items.push({
        severity: "critical",
        text: `${name} is backing off after ${cluster.health.consecutive_failures} failed passes${code}`,
        ...at,
      });
    } else if (status === "degraded") {
      items.push({ severity: "warning", text: `${name} is degraded${code}`, ...at });
    } else if (status === "never_polled") {
      items.push({ severity: "warning", text: `${name} is enabled but has never been polled`, ...at });
    }
    if (cluster.orphan_slurm_job_ids.length > 0) {
      items.push({
        severity: "warning",
        text: `${name} has ${cluster.orphan_slurm_job_ids.length} Slurm job(s) with no Xcelsior mapping`,
        ...at,
      });
    }
  }
  const missing = openJobs.filter((j) => j.missing_since).length;
  if (missing > 0) {
    items.push({
      severity: "critical",
      text: `${missing} open job(s) are missing from both the controller and accounting`,
      tab: "jobs",
    });
  }
  const stale = openJobs.filter((j) => j.stale).length;
  if (stale > 0) {
    items.push({
      severity: "warning",
      text: `${stale} open job(s) have not been confirmed for several poll intervals`,
      tab: "jobs",
    });
  }
  // Stable: critical first, each group in the order it was found.
  return [...items.filter((i) => i.severity === "critical"), ...items.filter((i) => i.severity === "warning")];
}

/** A job matches when the query appears in any of its ids or its name. */
export function jobMatches(job: HpcJob, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [job.xcelsior_job_id, job.slurm_job_id, job.submission_name, job.cluster_id, job.attempt_id]
    .some((value) => (value || "").toLowerCase().includes(q));
}

// ── Money, GPU time and the billing ledger ───────────────────────────

/** 389083 → "$0.3891". Under a dollar keeps four places, so a short run isn't "$0.00". */
export function formatMicros(micros: number | null | undefined): string {
  if (micros === null || micros === undefined || !Number.isFinite(micros)) return "—";
  const dollars = micros / 1_000_000;
  const places = dollars !== 0 && Math.abs(dollars) < 1 ? 4 : 2;
  return `$${dollars.toLocaleString("en-CA", { minimumFractionDigits: places, maximumFractionDigits: places })}`;
}

/** GPU seconds as hours: 4002 → "1.11". */
export function formatGpuHours(gpuSeconds: number | null | undefined): string {
  if (gpuSeconds === null || gpuSeconds === undefined || !Number.isFinite(gpuSeconds)) return "—";
  const hours = gpuSeconds / 3600;
  const places = hours < 10 ? 2 : 1;
  return hours.toLocaleString("en-CA", { minimumFractionDigits: places, maximumFractionDigits: places });
}

/** "$0.35 per GPU hour · RTX 3060", or null when the cluster is not priced. */
export function rateLabel(cluster: HpcCluster): string | null {
  const card = cluster.rate_card;
  if (!card || card.gpu_hour_micros === undefined) return null;
  const rate = (card.gpu_hour_micros / 1_000_000).toLocaleString("en-CA", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 4,
  });
  return `$${rate} per GPU hour${card.gpu_model ? ` · ${card.gpu_model}` : ""}`;
}

/** The written billing rules (workers/slurm_billing.py), in an operator's words. */
const RULE_LABEL: Record<string, string> = {
  never_started: "Never started",
  ran_to_end: "Ran to its end",
  timeout_capped_to_limit: "Timed out, capped at the limit",
  preempted_to_preemption: "Billed to the preemption",
  requeued_run: "Requeued run",
  suspended_gres_held: "suspended time billed (GPUs held)",
  suspended_gres_released: "suspended time free (GPUs released)",
};

/** `ran_to_end+suspended_gres_held` → "Ran to its end, suspended time billed (GPUs held)". */
export function billingRuleLabel(rule: string): string {
  return rule
    .split("+")
    .map((part) => RULE_LABEL[part] ?? part.replace(/_/g, " "))
    .join(", ");
}

/** "2026-11-12" → "Nov 12", read as a calendar day, not a local instant. */
export function formatDay(day: string): string {
  const at = Date.parse(`${day.slice(0, 10)}T12:00:00Z`);
  if (Number.isNaN(at)) return day;
  return new Date(at).toLocaleDateString("en-CA", { month: "short", day: "numeric", timeZone: "UTC" });
}

export type SubmissionTone = "queued" | "info" | "completed" | "warning" | "failed";

/** `unknown` is amber, not red: the job is still being looked for by name. */
export function submissionTone(state: string): SubmissionTone {
  switch (state) {
    case "submitted":
      return "completed";
    case "pending":
      return "queued";
    case "submitting":
      return "info";
    case "unknown":
      return "warning";
    default:
      return "failed";
  }
}

export const SUBMISSION_STATE_HELP: Record<string, string> = {
  pending: "Written, not yet sent to sbatch.",
  submitting: "sbatch is being called.",
  submitted: "Slurm accepted it and the job is mapped.",
  unknown: "The call timed out. Slurm may hold the job; it is being looked for by name.",
  rejected: "sbatch refused it. It is never retried.",
  lost: "Looked for by name twice and not found.",
};
