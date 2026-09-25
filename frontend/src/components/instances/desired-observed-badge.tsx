"use client";

/**
 * B6.5: Desired-versus-observed status badge.
 *
 * §20.3 asks the instance detail to display whether the control plane's
 * desired state (the job status and active attempt) matches what the worker
 * last reported. When they agree, the badge is green ("Converged"). When
 * they disagree, it's amber ("Diverged") with a tooltip explaining the
 * mismatch.
 *
 * The comparison is intentionally simple: if the job has an active attempt
 * that is "running" and the lease is "active", state is converged. Otherwise
 * if the job is in a terminal state, there's nothing to converge. Any other
 * mismatch is divergence.
 */

import { CheckCircle2, AlertTriangle, Minus, Clock } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export type ConvergenceState = "converged" | "diverged" | "terminal" | "pending";

export interface DesiredObservedBadgeProps {
  jobStatus: string;
  activeAttemptStatus?: string | null;
  leaseStatus?: string | null;
}

export function computeConvergence(props: DesiredObservedBadgeProps): ConvergenceState {
  const { jobStatus, activeAttemptStatus, leaseStatus } = props;
  const terminal = ["completed", "failed", "cancelled", "terminated"].includes(jobStatus);
  if (terminal) return "terminal";
  if (jobStatus === "queued" && !activeAttemptStatus) return "pending";
  if (jobStatus === "running" && activeAttemptStatus === "running" && leaseStatus === "active") return "converged";
  if (jobStatus === "starting" || jobStatus === "assigned" || jobStatus === "leased") return "pending";
  return "diverged";
}

export function DesiredObservedBadge(props: DesiredObservedBadgeProps) {
  const state = computeConvergence(props);
  
  if (state === "converged") {
    return (
      <Badge variant="outline" className="text-green-500 border-green-500/30 bg-green-500/10" title="Control plane desired state matches worker reported state.">
        <CheckCircle2 className="w-3 h-3 mr-1" />
        Converged
      </Badge>
    );
  }
  
  if (state === "diverged") {
    return (
      <Badge variant="outline" className="text-amber-500 border-amber-500/30 bg-amber-500/10" title={`Control plane expects state to be converged, but observed state does not match. Job: ${props.jobStatus}, Attempt: ${props.activeAttemptStatus || "none"}, Lease: ${props.leaseStatus || "none"}`}>
        <AlertTriangle className="w-3 h-3 mr-1" />
        Diverged
      </Badge>
    );
  }
  
  if (state === "pending") {
    return (
      <Badge variant="outline" className="text-blue-500 border-blue-500/30 bg-blue-500/10" title="Waiting for worker to converge with desired state.">
        <Clock className="w-3 h-3 mr-1" />
        Pending
      </Badge>
    );
  }
  
  return (
    <Badge variant="outline" className="text-gray-500 border-gray-500/30 bg-gray-500/10" title="Instance has reached a terminal state.">
      <Minus className="w-3 h-3 mr-1" />
      Terminal
    </Badge>
  );
}
