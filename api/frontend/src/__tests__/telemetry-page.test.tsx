import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({ fetchTelemetry: vi.fn() }));
vi.mock("@/lib/use-api", () => ({useApi: () => api}));
vi.mock("@/lib/locale", () => ({useLocale: () => ({t: (key: string) => key})}));

import TelemetryPage from "@/app/(dashboard)/dashboard/telemetry/page";

describe("telemetry response and missing measurements", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // jsdom has no layout observer. Browser acceptance checks actual charts;
    // these cases exercise the response contract and rendered metric values.
    vi.stubGlobal("ResizeObserver", class {
      observe() {}
      unobserve() {}
      disconnect() {}
    });
  });
  afterEach(() => vi.unstubAllGlobals());

  it("renders observed nested values from the real hosts envelope", async () => {
    api.fetchTelemetry.mockResolvedValue({ok:true, count:1, hosts:{"owned-host":{timestamp:100, received_at:101, stale:false, metrics:{utilization:37, temp:56, memory_used_gb:3.5, memory_total_gb:24, power_draw_w:108}}}});
    render(<TelemetryPage />);
    expect((await screen.findAllByText("37%")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("3.5 / 24GB").length).toBeGreaterThan(0);
    expect(screen.getAllByText("108W").length).toBeGreaterThan(0);
  });

  it("does not display missing sensors as zero", async () => {
    api.fetchTelemetry.mockResolvedValue({ok:true, count:1, hosts:{"owned-host":{timestamp:100, received_at:101, stale:false, metrics:{}}}});
    render(<TelemetryPage />);
    expect((await screen.findAllByText("Unavailable")).length).toBeGreaterThanOrEqual(4);
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
    expect(screen.queryByText("0°C")).not.toBeInTheDocument();
  });

  it("an outage is an error rather than an empty fleet", async () => {
    api.fetchTelemetry.mockRejectedValue(new Error("storage unavailable"));
    render(<TelemetryPage />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Telemetry could not be loaded");
    expect(screen.queryByText(/No live telemetry/)).not.toBeInTheDocument();
    expect(screen.queryByText("0%")).not.toBeInTheDocument();
  });
});
