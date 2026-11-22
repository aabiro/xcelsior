import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ReceiptCard } from "@/components/instances/receipt-card";
import { CostMeterCard } from "@/components/instances/cost-meter-card";
import type { RunReceipt } from "@/lib/api";
import { formatDuration, formatMicros } from "@/lib/format-units";

function receipt(overrides: Partial<RunReceipt> = {}): RunReceipt {
  return {
    receipt_version: 1,
    job_id: "abcd1234ef567890",
    name: "demo",
    status: "completed",
    final: true,
    currency: "CAD",
    gpu_model: "RTX 3060",
    gpu_count: 1,
    pricing_mode: "on_demand",
    image: "python:3.11",
    launch_template_id: "pytorch",
    saved_image_id: null,
    started_at: 1_700_000_000,
    ended_at: 1_700_000_090,
    billed_seconds: 90.02,
    rate_micros_per_hour: 300_000,
    total_micros: 7_502,
    cycles_charged_micros: 7_502,
    wallet_debit_micros: 7_502,
    wallet_debit_count: 1,
    outstanding_micros: 0,
    credits_micros: 0,
    reconciled: true,
    hourly_rounded_micros: 300_000,
    hourly_rounded_label: "if billed in whole hours at the same rate",
    rounding: "exact",
    host_verification_at_start: {
      state: "verified",
      freshness: "verified",
      verification_id: "v-1",
      checked_at: 1_699_990_000,
      source: "placement",
    },
    attempts: [],
    ...overrides,
  };
}

describe("formatters", () => {
  it("renders durations the way people read them", () => {
    expect(formatDuration(42)).toBe("42s");
    expect(formatDuration(90)).toBe("1m 30s");
    expect(formatDuration(65)).toBe("1m 05s");
    expect(formatDuration(7380)).toBe("2h 03m");
    expect(formatDuration(0)).toBe("0s");
    expect(formatDuration(undefined)).toBe("—");
  });

  it("keeps every micro on a receipt", () => {
    expect(formatMicros(7_502, { exact: true })).toBe("$0.007502");
    expect(formatMicros(90_000, { exact: true })).toBe("$0.09");
    expect(formatMicros(389_083)).toBe("$0.3891");
    expect(formatMicros(12_500_000)).toBe("$12.50");
  });
});

describe("ReceiptCard", () => {
  it("shows the billed seconds, the locked rate and the exact total", () => {
    render(<ReceiptCard receipt={receipt()} />);
    expect(screen.getByText("1m 30s")).toBeInTheDocument();
    expect(screen.getByText("$0.3000/hr")).toBeInTheDocument();
    expect(screen.getByTestId("receipt-total")).toHaveTextContent("$0.007502");
    expect(screen.getByTestId("receipt-reconciled")).toBeInTheDocument();
    // The comparison is labelled, and it is not the charge.
    expect(screen.getByTestId("receipt-hourly")).toHaveTextContent("$0.30");
    expect(screen.queryByText(/CAD/)).not.toBeInTheDocument();
  });

  it("says what is still owed when the wallet did not cover the run", () => {
    render(
      <ReceiptCard
        receipt={receipt({ reconciled: false, wallet_debit_micros: 5_000, outstanding_micros: 2_502 })}
      />,
    );
    expect(screen.getByTestId("receipt-unreconciled")).toBeInTheDocument();
    expect(screen.queryByTestId("receipt-reconciled")).not.toBeInTheDocument();
  });
});

describe("CostMeterCard labels", () => {
  it("calls a billed figure billed and a live one an estimate", () => {
    const { rerender } = render(<CostMeterCard instance={{ cost_cad: 0.09, cost_is_estimate: false }} />);
    expect(screen.getByText("Billed Compute")).toBeInTheDocument();
    rerender(<CostMeterCard instance={{ cost_cad: 0.09, cost_is_estimate: true }} />);
    expect(screen.getByText("Est. Compute Cost")).toBeInTheDocument();
  });
});
