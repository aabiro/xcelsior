"use client";

/**
 * The scheduling attempts behind an instance, and its current lease — B6.5.
 *
 * §20.3 asks the instance detail to show *"phase and conditions, attempt
 * timeline, selected host/GPU aliases and score, lease health"*. Both routes
 * existed and were read only by MCP tools — `get_instance_timeline` and
 * `get_active_lease` — while the browser called neither. The dashboard's
 * timeline showed five fixed status pills; this shows what actually happened,
 * including the attempts that failed before the one that worked.
 *
 * ## What is left out, and why it matters
 *
 * `/api/v1/instances/{job_id}/timeline` returns `placement_explanation` on each
 * attempt, and that payload contains the per-host rejections map — "host is
 * drained", keyed by host id, for hosts this job was not placed on. That is the
 * data `PlacementExplanation` deliberately does not render. Dumping it into a
 * timeline row would undo that redaction through a second door, so
 * `InstanceAttempt` does not even type the field.
 *
 * The lease is different: `/active-lease` already substitutes a SHA-derived
 * `host_alias` server-side and returns no credential, so it renders as sent.
 */

import { CheckCircle2, XCircle, Circle } from "lucide-react";
import type { InstanceAttempt, ActiveLease } from "@/lib/api";

const STEPS = [
  { key: "reserved_at", label: "Reserved" },
  { key: "command_created_at", label: "Command sent" },
  { key: "lease_claimed_at", label: "Lease claimed" },
  { key: "started_at", label: "Started" },
  { key: "ended_at", label: "Ended" },
] as const;

function when(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleTimeString();
}

/** Terminal-failure statuses render red; everything else is progress. */
function isFailure(status: string): boolean {
  return /fail|error|expired|lost|abandon/i.test(status);
}

export function AttemptTimeline({
  attempts,
  lease,
}: {
  attempts: InstanceAttempt[];
  lease: ActiveLease | null;
}) {
  if (attempts.length === 0 && !lease) {
    return (
      <p data-testid="attempt-timeline-empty" className="text-xs text-text-muted">
        No scheduling attempts have been recorded yet.
      </p>
    );
  }

  return (
    <div data-testid="attempt-timeline" className="space-y-3">
      {lease && (
        <div
          data-testid="active-lease"
          className="rounded-lg border border-ice-blue/25 bg-ice-blue/[0.06] px-3 py-2"
        >
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1 text-xs">
            <span className="font-medium text-ice-blue">Lease {lease.status}</span>
            {/* An alias, not a host id — the route substitutes it. */}
            <code className="font-mono text-[11px] text-text-secondary">{lease.host_alias}</code>
            {lease.expires_at && (
              <span className="text-[11px] text-text-muted">
                renews until {when(lease.expires_at)}
              </span>
            )}
          </div>
        </div>
      )}

      <ol className="space-y-2">
        {attempts.map((attempt) => {
          const failed = isFailure(attempt.status);
          return (
            <li
              key={attempt.attempt_id}
              data-testid={`attempt-${attempt.attempt_number}`}
              data-status={attempt.status}
              className={`rounded-lg border p-3 ${
                failed ? "border-accent-red/30 bg-accent-red/[0.04]" : "border-border/60"
              }`}
            >
              <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                <span className="text-xs font-medium">Attempt {attempt.attempt_number}</span>
                <span
                  className={`text-xs ${failed ? "text-accent-red" : "text-text-secondary"}`}
                >
                  {attempt.status}
                </span>
                {attempt.placement_score !== null && (
                  <span className="text-[11px] text-text-muted">
                    score {attempt.placement_score}
                  </span>
                )}
                {attempt.failure_code && (
                  <code
                    data-testid={`attempt-${attempt.attempt_number}-failure`}
                    className="ml-auto font-mono text-[10px] text-accent-red"
                  >
                    {attempt.failure_code}
                  </code>
                )}
              </div>

              <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
                {STEPS.map((step) => {
                  const value = attempt[step.key];
                  const reached = Boolean(value);
                  const Icon = reached
                    ? failed && step.key === "ended_at"
                      ? XCircle
                      : CheckCircle2
                    : Circle;
                  return (
                    <span
                      key={step.key}
                      className="flex items-center gap-1 text-[11px]"
                      title={`${step.label}: ${when(value)}`}
                    >
                      <Icon
                        className={`h-3 w-3 shrink-0 ${
                          !reached
                            ? "text-text-muted/50"
                            : failed && step.key === "ended_at"
                              ? "text-accent-red"
                              : "text-emerald"
                        }`}
                      />
                      <span className={reached ? "text-text-secondary" : "text-text-muted/60"}>
                        {step.label}
                      </span>
                      {reached && <span className="text-text-muted">{when(value)}</span>}
                    </span>
                  );
                })}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
