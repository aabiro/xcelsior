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
}

/** `openJobs` is the loaded page of open jobs, so the counts are a floor when more exist. */
export function summarise(clusters: HpcCluster[], openJobs: HpcJob[], now: number): HpcSummary {
  const enabled = clusters.filter((c) => c.enabled);
  return {
    clustersEnabled: enabled.length,
    clustersHealthy: enabled.filter((c) => clusterStatus(c, now) === "healthy").length,
    open: openJobs.length,
    running: openJobs.filter((j) => j.observed_state === "running").length,
    pending: openJobs.filter((j) => j.observed_state === "pending").length,
    stale: openJobs.filter((j) => j.stale).length,
    awaitingAccounting: openJobs.filter((j) => j.awaiting_accounting_since).length,
    orphans: clusters.reduce((n, c) => n + c.orphan_slurm_job_ids.length, 0),
  };
}

/** Things an operator should look at, most urgent first. */
export function attentionItems(clusters: HpcCluster[], openJobs: HpcJob[], now: number): string[] {
  const items: string[] = [];
  for (const cluster of clusters) {
    const status = clusterStatus(cluster, now);
    const name = cluster.display_name || cluster.cluster_id;
    if (status === "backing_off") {
      items.push(
        `${name} is backing off after ${cluster.health.consecutive_failures} failed passes` +
          (cluster.health.last_error_code ? ` (${cluster.health.last_error_code})` : ""),
      );
    } else if (status === "degraded") {
      items.push(
        `${name} is degraded` +
          (cluster.health.last_error_code ? ` (${cluster.health.last_error_code})` : ""),
      );
    } else if (status === "never_polled") {
      items.push(`${name} is enabled but has never been polled`);
    }
    if (cluster.orphan_slurm_job_ids.length > 0) {
      items.push(
        `${name} has ${cluster.orphan_slurm_job_ids.length} Slurm job(s) with no Xcelsior mapping`,
      );
    }
  }
  const stale = openJobs.filter((j) => j.stale).length;
  if (stale > 0) items.push(`${stale} open job(s) have not been confirmed for several poll intervals`);
  const missing = openJobs.filter((j) => j.missing_since).length;
  if (missing > 0) items.push(`${missing} open job(s) are missing from both the controller and accounting`);
  return items;
}

/** A job matches when the query appears in any of its ids or its name. */
export function jobMatches(job: HpcJob, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return [job.xcelsior_job_id, job.slurm_job_id, job.submission_name, job.cluster_id, job.attempt_id]
    .some((value) => (value || "").toLowerCase().includes(q));
}
