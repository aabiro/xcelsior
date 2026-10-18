import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

/**
 * Gate P5: *"a placement preference that cannot be satisfied **refuses clearly**
 * rather than silently falling back to the cheapest host. This is the failure
 * mode that would quietly destroy trust."*
 *
 * The control plane already answers this properly — `POST /api/v1/placements/evaluate`
 * returns a **200** carrying a typed refusal with the number that failed, on the
 * reasoning that an HTTP error would claim the request was wrong when the
 * request was fine and the fleet could not satisfy it.
 *
 * That design only pays off if the UI renders the refusal *as* a refusal. The
 * quiet failure this guards is the tempting one: treat `refused: true` as "no
 * result", show nothing, and let the launch proceed as though no constraint had
 * been asked for. The user then gets the cheapest host and believes they got a
 * verified one.
 */

const evaluatePlacement = vi.fn();
vi.mock("@/lib/api", () => ({ evaluatePlacement: (...a: unknown[]) => evaluatePlacement(...a) }));

import { PlacementPreferenceControl } from "@/components/instances/placement-preference-control";

const SPEC = { gpu_model: "RTX 4090", num_gpus: 1 };

describe("placement preference refuses clearly", () => {
  beforeEach(() => {
    evaluatePlacement.mockReset();
  });

  it("shows the refusal, with what was asked beside what is available", async () => {
    evaluatePlacement.mockResolvedValue({
      ok: true,
      preference: {
        refused: true,
        code: "uptime_unsatisfiable",
        detail: "No host meets the requested uptime.",
        asked: 99.5,
        best_available: 99.1,
      },
    });

    render(<PlacementPreferenceControl spec={SPEC} />);
    // Stating a constraint is what triggers an evaluation.
    // `fireEvent.change`, not a raw input event: React tracks a controlled
    // input's value through its own setter, so dispatching `input` by hand
    // updates the DOM node and never the component.
    fireEvent.change(screen.getByPlaceholderText("any"), { target: { value: "99.5" } });

    await waitFor(() => expect(evaluatePlacement).toHaveBeenCalled(), { timeout: 3000 });
    await waitFor(() =>
      expect(screen.getByText(/No host satisfies this right now/i)).toBeTruthy(),
    );
    // The numbers, not just the verdict: "no host matched" is not actionable.
    expect(screen.getByText(/99\.5/)).toBeTruthy();
    expect(screen.getByText(/99\.1/)).toBeTruthy();
    // And it must say the launch will not quietly proceed on other terms.
    expect(screen.getByText(/will not quietly fall back/i)).toBeTruthy();
  });

  it("does not report satisfiable when the evaluation itself fails", async () => {
    // Silence is the failure mode this control exists to remove, so an
    // evaluation that errors must not render the green "Satisfiable" state.
    evaluatePlacement.mockRejectedValue(new Error("network"));

    render(<PlacementPreferenceControl spec={SPEC} />);
    fireEvent.change(screen.getByPlaceholderText("any"), { target: { value: "99.9" } });

    await waitFor(() => expect(evaluatePlacement).toHaveBeenCalled(), { timeout: 3000 });
    await waitFor(() => expect(screen.queryByText(/Satisfiable/i)).toBeNull());
  });

  it("evaluates nothing until a constraint is actually stated", async () => {
    // An empty preference is not a request; firing an evaluation for it would
    // put a verdict on screen the user never asked for.
    render(<PlacementPreferenceControl spec={SPEC} />);
    await new Promise((r) => setTimeout(r, 600));
    expect(evaluatePlacement).not.toHaveBeenCalled();
  });
});
