import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { computeConvergence, DesiredObservedBadge } from "@/components/instances/desired-observed-badge";

const healthy = {
  jobStatus: "running",
  activeAttemptStatus: "running",
  activeAttemptId: "current-attempt",
  leaseStatus: "active",
  leaseAttemptId: "current-attempt",
  leaseExpiresAt: "2099-01-01T00:00:00Z",
};

describe("state synchronization uses current observations", () => {
  afterEach(() => vi.useRealTimers());

  it("does not combine a running old attempt with another attempt's lease", () => {
    expect(computeConvergence({ ...healthy, leaseAttemptId: "old-attempt" })).toBe("diverged");
  });

  it("does not call an expired active lease converged", () => {
    expect(computeConvergence({ ...healthy, leaseExpiresAt: "2000-01-01T00:00:00Z" })).toBe("diverged");
  });

  it.each([null, "not-a-date"])("reports unknown when lease expiry is unavailable: %s", (expiry) => {
    expect(computeConvergence({ ...healthy, leaseExpiresAt: expiry })).toBe("unknown");
  });

  it.each(["loading", "unavailable"] as const)("does not infer divergence or health from %s observations", (observationStatus) => {
    expect(computeConvergence({ ...healthy, observationStatus })).toBe("unknown");
  });

  it("requires a recorded attempt matching the lease", () => {
    expect(computeConvergence({ ...healthy, activeAttemptId: null })).toBe("unknown");
  });

  it("shows a normally stopped instance as stopped, not broken or terminal", () => {
    expect(computeConvergence({ jobStatus: "stopped" })).toBe("stopped");
  });

  it.each(["stopping", "restarting", "preempted"])("recognizes %s as an in-progress transition", (jobStatus) => {
    expect(computeConvergence({ jobStatus })).toBe("pending");
  });

  it("does not hide continuing execution behind a completed job status", () => {
    expect(computeConvergence({ ...healthy, jobStatus: "completed" })).toBe("diverged");
  });

  it("rejects an offered lease after its claim deadline", () => {
    expect(computeConvergence({
      ...healthy, jobStatus: "assigned", activeAttemptStatus: "reserved",
      leaseStatus: "offered", leaseClaimDeadline: "2000-01-01T00:00:00Z",
    })).toBe("diverged");
  });

  it("updates lease health at expiry even when no parent render arrives", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-09-28T12:00:00Z"));
    render(<DesiredObservedBadge {...healthy} leaseExpiresAt="2026-09-28T12:00:02Z" />);
    expect(screen.getByText("Converged")).toBeInTheDocument();
    act(() => vi.advanceTimersByTime(2000));
    expect(screen.getByText("Diverged")).toBeInTheDocument();
  });
});
