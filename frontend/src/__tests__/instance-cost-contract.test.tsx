import { act, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CostMeterCard } from "@/components/instances/cost-meter-card";

describe("instance cost API contract", () => {
  afterEach(() => vi.useRealTimers());

  it("uses the CAD rate and estimate supplied by the instance API", () => {
    render(<CostMeterCard instance={{ rate_per_hour_cad: 2.5, cost_cad: 7.25 }} />);
    expect(screen.getByText("2.50")).toBeInTheDocument();
    expect(screen.getByText("7.25")).toBeInTheDocument();
    expect(screen.queryByText("Total Billed Cost")).not.toBeInTheDocument();
  });

  it("distinguishes unknown amounts from a real zero", () => {
    const { rerender } = render(<CostMeterCard instance={{}} />);
    expect(screen.getAllByText("Unavailable")).toHaveLength(2);
    expect(screen.queryByText("0.00")).not.toBeInTheDocument();
    rerender(<CostMeterCard instance={{ rate_per_hour_cad: 0, cost_cad: 0 }} />);
    expect(screen.getAllByText("0.00")).toHaveLength(2);
    expect(screen.queryByText("Unavailable")).not.toBeInTheDocument();
  });

  it("does not invent additional charges between server updates", () => {
    vi.useFakeTimers();
    render(<CostMeterCard instance={{ rate_per_hour_cad: 2, cost_cad: 4.75 }} />);
    act(() => vi.advanceTimersByTime(3600_000));
    expect(screen.getByText("4.75")).toBeInTheDocument();
  });

  it("replaces the estimate when the API refreshes and clears stale values", () => {
    const { rerender } = render(<CostMeterCard instance={{ rate_per_hour_cad: 2, cost_cad: 4.75 }} />);
    rerender(<CostMeterCard instance={{ rate_per_hour_cad: 2, cost_cad: 5.5, rate_is_estimate: true }} />);
    expect(screen.getByText("5.50")).toBeInTheDocument();
    expect(screen.getByText("Est. Hourly Rate")).toBeInTheDocument();
    rerender(<CostMeterCard instance={{}} />);
    expect(screen.queryByText("5.50")).not.toBeInTheDocument();
    expect(screen.getAllByText("Unavailable")).toHaveLength(2);
  });

  it("never displays nonfinite amounts as a price", () => {
    render(<CostMeterCard instance={{ rate_per_hour_cad: Infinity, cost_cad: NaN }} />);
    expect(screen.getAllByText("Unavailable")).toHaveLength(2);
  });
});
