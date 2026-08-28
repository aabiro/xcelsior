import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

/**
 * B6.5's other half: *"attempt timeline … lease health"*.
 *
 * `/api/v1/instances/{id}/timeline` and `/active-lease` existed and were read
 * only by `get_instance_timeline` and `get_active_lease` — both MCP tools. The
 * browser called neither, so the dashboard's five status pills were the whole
 * story and a first attempt that failed before the successful one was invisible.
 *
 * The property worth guarding is not the layout. The timeline route returns
 * `placement_explanation` on every attempt, and that payload carries the
 * per-host rejections map — other tenants' fleet state, the exact thing
 * `PlacementExplanation` refuses to render. A timeline row that dumped it would
 * undo that redaction through a second door.
 */

import { AttemptTimeline } from "@/components/instances/attempt-timeline";
import type { InstanceAttempt, ActiveLease } from "@/lib/api";

const ATTEMPT = (over: Partial<InstanceAttempt> = {}): InstanceAttempt => ({
  attempt_id: "a1",
  attempt_number: 1,
  status: "succeeded",
  host_id: "host-1",
  placement_score: 92,
  failure_code: null,
  reserved_at: "2026-08-28T10:00:00Z",
  command_created_at: "2026-08-28T10:00:05Z",
  lease_claimed_at: "2026-08-28T10:00:09Z",
  started_at: "2026-08-28T10:00:20Z",
  ended_at: null,
  trace_id: "t1",
  ...over,
});

const LEASE: ActiveLease = {
  lease_id: "l1",
  attempt_id: "a1",
  status: "active",
  host_alias: "host-4f2a9c1b3d",
  offered_at: "2026-08-28T10:00:06Z",
  claim_deadline: "2026-08-28T10:00:36Z",
  claimed_at: "2026-08-28T10:00:09Z",
  last_renewed_at: "2026-08-28T10:05:00Z",
  expires_at: "2026-08-28T10:10:00Z",
};

describe("attempts", () => {
  it("shows every attempt, not just the last", () => {
    render(
      <AttemptTimeline
        attempts={[
          ATTEMPT({ attempt_id: "a1", attempt_number: 1, status: "failed", failure_code: "lease_expired" }),
          ATTEMPT({ attempt_id: "a2", attempt_number: 2 }),
        ]}
        lease={null}
      />,
    );
    // The failed first attempt is exactly what someone is looking for.
    expect(screen.getByTestId("attempt-1")).toHaveAttribute("data-status", "failed");
    expect(screen.getByTestId("attempt-2")).toHaveAttribute("data-status", "succeeded");
  });

  it("surfaces the failure code rather than only a colour", () => {
    render(
      <AttemptTimeline
        attempts={[ATTEMPT({ status: "failed", failure_code: "lease_expired" })]}
        lease={null}
      />,
    );
    expect(screen.getByTestId("attempt-1-failure")).toHaveTextContent("lease_expired");
  });

  it("marks steps that have not happened yet as unreached", () => {
    const { container } = render(
      <AttemptTimeline attempts={[ATTEMPT({ started_at: null, ended_at: null })]} lease={null} />,
    );
    // "Started" and "Ended" have no timestamp; the labels still render so the
    // shape of the process is visible rather than truncated.
    expect(container.textContent).toContain("Started");
    expect(container.textContent).toContain("Ended");
  });
});

describe("lease", () => {
  it("renders the alias the route substitutes, never a host id", () => {
    render(<AttemptTimeline attempts={[]} lease={LEASE} />);
    expect(screen.getByTestId("active-lease")).toHaveTextContent("host-4f2a9c1b3d");
  });

  it("is absent when there is no active lease", () => {
    render(<AttemptTimeline attempts={[ATTEMPT()]} lease={null} />);
    expect(screen.queryByTestId("active-lease")).toBeNull();
  });
});

describe("redaction", () => {
  it("never renders a placement explanation blob smuggled onto an attempt", () => {
    // The route really does include this; the type omits it and the component
    // must ignore it even when it arrives.
    const withBlob = {
      ...ATTEMPT(),
      placement_explanation: {
        rejections: { "host-secret-9": [{ code: "host_not_ready", message: "host status is drained" }] },
      },
    } as unknown as InstanceAttempt;
    const { container } = render(<AttemptTimeline attempts={[withBlob]} lease={null} />);
    const text = container.textContent ?? "";
    expect(text).not.toContain("host-secret-9");
    expect(text).not.toContain("drained");
  });
});

describe("empty", () => {
  it("says nothing has been scheduled rather than rendering a blank card", () => {
    render(<AttemptTimeline attempts={[]} lease={null} />);
    expect(screen.getByTestId("attempt-timeline-empty")).toBeInTheDocument();
  });
});
