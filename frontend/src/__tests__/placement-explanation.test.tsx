import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

/**
 * B6.5: *"plain-language current reason ('Queued because no healthy H100 with
 * 80 GB is available in Ontario')"*.
 *
 * Two properties carry this component, and neither is about wording.
 *
 * **It must not leak fleet state.** The stored explanation carries a per-host
 * `rejections` map — "host is drained", "heartbeat is stale", keyed by host id,
 * for hosts this job was *not* placed on. §20.3 says customers see redacted
 * infrastructure detail, so only `rejection_summary` counts may be rendered.
 * A test that merely checks the sentence reads nicely would not notice a
 * regression that started printing host ids.
 *
 * **`explained: false` is a real state.** An older attempt recorded nothing.
 * That must read as "no explanation was recorded", never as an empty reason,
 * and never as a spinner that resolves to nothing.
 */

import {
  PlacementExplanation,
  summarySentence,
  constraintPhrase,
} from "@/components/instances/placement-explanation";
import type { PlacementExplanationPayload } from "@/lib/api";

const QUEUED: PlacementExplanationPayload = {
  explain_version: "explain/v1",
  request: { gpu_model: "H100", num_gpus: 1, vram_needed_gb: 80, region: "Ontario" },
  hosts_considered: 14,
  hosts_eligible: 0,
  hosts_rejected: 14,
  rejection_summary: {
    policy_version: "filters/v1",
    hosts_evaluated: 14,
    failed_constraints: { gpu_model_mismatch: 11, insufficient_vram: 2, host_not_ready: 1 },
  },
  queue_reason_code: "no_eligible_host",
};

describe("while queued", () => {
  it("names what was asked for, in a sentence", () => {
    render(<PlacementExplanation payload={QUEUED} explained />);
    const text = screen.getByTestId("placement-summary").textContent ?? "";
    expect(text).toContain("H100");
    expect(text).toContain("80 GB");
    expect(text).toContain("Ontario");
  });

  it("ranks constraints by how many hosts failed them", () => {
    render(<PlacementExplanation payload={QUEUED} explained />);
    const items = screen.getByTestId("placement-constraints").textContent ?? "";
    // The most common failure is the one worth acting on, so it leads.
    expect(items.indexOf("different GPU model")).toBeLessThan(items.indexOf("free VRAM"));
  });

  it("reassures that waiting is not losing", () => {
    render(<PlacementExplanation payload={QUEUED} explained />);
    expect(screen.getByText(/keeps retrying/i)).toBeInTheDocument();
  });

  it("never renders a host id or per-host state", () => {
    // The dangerous regression: rendering `rejections` instead of the summary.
    const withHosts = {
      ...QUEUED,
      rejections: {
        "host-abc123": [{ code: "host_not_ready", message: "host status is drained", details: {} }],
      },
    } as unknown as PlacementExplanationPayload;
    const { container } = render(<PlacementExplanation payload={withHosts} explained />);
    const text = container.textContent ?? "";
    expect(text).not.toContain("host-abc123");
    expect(text).not.toContain("drained");
  });
});

describe("when a host was chosen", () => {
  it("does not present a queue reason", () => {
    const placed: PlacementExplanationPayload = {
      ...QUEUED,
      hosts_eligible: 3,
      selected_host_id: "host-xyz",
      queue_reason_code: undefined,
    };
    render(<PlacementExplanation payload={placed} explained />);
    expect(screen.getByTestId("placement-explanation-placed")).toBeInTheDocument();
    expect(screen.queryByTestId("placement-explanation-queued")).toBeNull();
  });

  it("does not name the host it chose", () => {
    const placed: PlacementExplanationPayload = { ...QUEUED, selected_host_id: "host-xyz" };
    const { container } = render(<PlacementExplanation payload={placed} explained />);
    expect(container.textContent).not.toContain("host-xyz");
  });
});

describe("when nothing was recorded", () => {
  it.each([
    ["explained false", QUEUED, false],
    ["null payload", null, true],
  ])("says so plainly for %s", (_label, payload, explained) => {
    render(
      <PlacementExplanation
        payload={payload as PlacementExplanationPayload | null}
        explained={explained}
      />,
    );
    expect(screen.getByTestId("placement-explanation-absent")).toBeInTheDocument();
    expect(screen.queryByTestId("placement-explanation-queued")).toBeNull();
  });
});

describe("constraint vocabulary", () => {
  it("phrases the codes it knows", () => {
    expect(constraintPhrase("gpu_model_mismatch")).toMatch(/different GPU model/i);
  });

  it("shows an unknown code rather than dropping it", () => {
    // Annotate, never gate: a constraint the map has not learned still has to
    // appear, or the count of reasons silently disagrees with the fleet.
    expect(constraintPhrase("some_new_filter")).toBe("some_new_filter");
    const odd = {
      ...QUEUED,
      rejection_summary: { ...QUEUED.rejection_summary, failed_constraints: { some_new_filter: 4 } },
    };
    render(<PlacementExplanation payload={odd} explained />);
    expect(screen.getByTestId("placement-constraints")).toHaveTextContent("some_new_filter");
  });
});

describe("the sentence itself", () => {
  it("pluralises a multi-GPU request", () => {
    expect(summarySentence({ ...QUEUED, request: { gpu_model: "A100", num_gpus: 4 } })).toContain(
      "4 × A100",
    );
  });

  it("survives a request with nothing in it", () => {
    const bare = { ...QUEUED, request: {} };
    expect(summarySentence(bare)).toContain("a GPU");
  });
});
