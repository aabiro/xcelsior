"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, AlertTriangle, Minus, Clock, CircleHelp } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export type ConvergenceState = "converged" | "diverged" | "terminal" | "stopped" | "pending" | "unknown";
export type ObservationStatus = "loading" | "ready" | "unavailable";

export interface DesiredObservedBadgeProps {
  jobStatus: string;
  observationStatus?: ObservationStatus;
  activeAttemptId?: string | null;
  activeAttemptStatus?: string | null;
  leaseAttemptId?: string | null;
  leaseStatus?: string | null;
  leaseExpiresAt?: string | null;
  leaseClaimDeadline?: string | null;
}

function leaseDeadline(props: DesiredObservedBadgeProps): number {
  const value = props.leaseStatus === "offered" ? props.leaseClaimDeadline : props.leaseExpiresAt;
  return value ? Date.parse(value) : NaN;
}

export function computeConvergence(props: DesiredObservedBadgeProps, now = Date.now()): ConvergenceState {
  const { jobStatus, activeAttemptStatus, activeAttemptId, leaseAttemptId, leaseStatus } = props;
  if (props.observationStatus && props.observationStatus !== "ready") return "unknown";

  const hasLease = leaseStatus === "active" || leaseStatus === "offered";
  const deadline = leaseDeadline(props);
  if (hasLease && !Number.isFinite(deadline)) return "unknown";
  const expired = hasLease && deadline <= now;
  const endedAttempt = ["succeeded", "failed", "cancelled", "stopped", "abandoned", "expired"].includes(activeAttemptStatus ?? "");
  const executingAttempt = Boolean(activeAttemptStatus) && !endedAttempt;

  const terminal = ["completed", "failed", "cancelled", "terminated"].includes(jobStatus);
  const stopped = ["stopped", "paused", "user_paused", "paused_low_balance"].includes(jobStatus);
  if (terminal || stopped) {
    // A terminal job label must not hide a worker still reporting execution.
    if ((hasLease && !expired) || executingAttempt) return "diverged";
    return stopped ? "stopped" : "terminal";
  }
  if (expired || leaseStatus === "expired" || leaseStatus === "revoked") return "diverged";
  if (hasLease && activeAttemptId && leaseAttemptId && activeAttemptId !== leaseAttemptId) return "diverged";

  // Retrying a finished attempt can legitimately leave the old attempt in the
  // timeline while the next one is being scheduled.
  if (["queued", "restarting", "preempted", "stopping"].includes(jobStatus)) return "pending";
  if (["starting", "assigned", "leased"].includes(jobStatus)) return endedAttempt ? "diverged" : "pending";
  if (jobStatus !== "running") return "unknown";
  if (leaseStatus !== "active") return "diverged";
  if (!activeAttemptId || !leaseAttemptId || !activeAttemptStatus) return "unknown";
  return activeAttemptStatus === "running" ? "converged" : "diverged";
}

export function DesiredObservedBadge(props: DesiredObservedBadgeProps) {
  const [clockTick, setClockTick] = useState(0);
  const deadline = leaseDeadline(props);
  useEffect(() => {
    if (!["active", "offered"].includes(props.leaseStatus ?? "") || !Number.isFinite(deadline)) return;
    const delay = deadline - Date.now();
    if (delay <= 0) return;
    const timer = setTimeout(() => setClockTick((tick) => tick + 1), Math.min(delay, 2_147_483_647));
    return () => clearTimeout(timer);
  }, [deadline, props.leaseStatus, clockTick]);

  const state = computeConvergence(props);
  if (state === "unknown") {
    return (
      <Badge variant="outline" className="text-text-muted" title="Current attempt and lease observations are not available to verify synchronization.">
        <CircleHelp className="w-3 h-3 mr-1" />
        {props.observationStatus === "loading" ? "Checking" : "Unavailable"}
      </Badge>
    );
  }
  if (state === "converged") {
    return (
      <Badge variant="outline" className="text-green-500 border-green-500/30 bg-green-500/10" title="The current attempt reports running and holds an unexpired active lease.">
        <CheckCircle2 className="w-3 h-3 mr-1" />
        Converged
      </Badge>
    );
  }
  if (state === "diverged") {
    return (
      <Badge variant="outline" className="text-amber-500 border-amber-500/30 bg-amber-500/10" title={`Observed execution does not match the instance state. Job: ${props.jobStatus}, Attempt: ${props.activeAttemptStatus || "none"}, Lease: ${props.leaseStatus || "none"}. The attempt identity and lease expiry must also match.`}>
        <AlertTriangle className="w-3 h-3 mr-1" />
        Diverged
      </Badge>
    );
  }
  if (state === "pending") {
    return (
      <Badge variant="outline" className="text-blue-500 border-blue-500/30 bg-blue-500/10" title="An instance lifecycle transition is in progress.">
        <Clock className="w-3 h-3 mr-1" />
        Pending
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="text-gray-500 border-gray-500/30 bg-gray-500/10" title={state === "stopped" ? "Execution is stopped; this instance may be resumed." : "Instance has reached a terminal state."}>
      <Minus className="w-3 h-3 mr-1" />
      {state === "stopped" ? "Stopped" : "Terminal"}
    </Badge>
  );
}
