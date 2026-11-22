import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    locale: "en",
    t: (key: string, params?: Record<string, unknown>) =>
      params ? `${key} ${JSON.stringify(params)}` : key,
  }),
}));
vi.mock("@/lib/recharts", () => ({
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CartesianGrid: () => null,
}));

const { FakeApiError } = vi.hoisted(() => ({
  FakeApiError: class FakeApiError extends Error {
    status: number;
    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
}));
const latest = vi.fn();
const history = vi.fn();
vi.mock("@/lib/api", () => ({
  ApiError: FakeApiError,
  fetchInstanceTelemetry: (...a: unknown[]) => latest(...a),
  fetchInstanceTelemetryHistory: (...a: unknown[]) => history(...a),
}));

import { GpuTelemetryCard } from "@/components/instances/gpu-telemetry-card";

const now = () => Date.now() / 1000;
const gpu = (over: Record<string, unknown> = {}) => ({
  index: 0,
  gpu_model: "RTX 4090",
  utilization: 87,
  memory_util: 70,
  memory_used_gb: 18.2,
  memory_total_gb: 24,
  temp: 61,
  power_draw_w: 210,
  power_limit_w: 450,
  received_at: now() - 5,
  stale: false,
  ...over,
});
const series = (values: number[]) => ({
  interval_seconds: 30,
  fields: ["utilization", "memory_used_gb", "temp", "power_draw_w"],
  gpus: [{
    index: 0,
    gpu_model: "RTX 4090",
    points: values.map((u, i) => ({ t: now() - (values.length - i) * 30, utilization: u, memory_used_gb: 10 + i, temp: 60, power_draw_w: 200 + i })),
  }],
});

describe("GPU telemetry card", () => {
  beforeEach(() => {
    latest.mockReset();
    history.mockReset();
  });

  it("shows a running instance's live readings and its utilization over time", async () => {
    latest.mockResolvedValue({ ok: true, job_id: "j1", host_id: "h", stale: false, telemetry: { gpus: [gpu()] } });
    history.mockResolvedValue({ ok: true, job_id: "j1", history: series([20, 40, 60]) });
    render(<GpuTelemetryCard jobId="j1" live />);
    const tiles = await screen.findByTestId("gpu-telemetry-tiles");
    expect(tiles.textContent).toContain("87");
    expect(tiles.textContent).toContain("18.2");
    expect(tiles.textContent).toContain("/ 24 GB");
    expect(tiles.textContent).toContain("61");
    expect(screen.getByTestId("area-chart")).toBeTruthy();
    expect(screen.getByTestId("gpu-telemetry-summary").textContent).toContain('"avg":"40","peak":"60"');
    expect(screen.getByTestId("gpu-telemetry-age").textContent).toContain("last_reading_s");
  });

  it("summarises a finished run instead of showing stale live numbers", async () => {
    latest.mockResolvedValue({ ok: true, job_id: "j1", host_id: "h", stale: true, telemetry: { gpus: [gpu({ stale: true })] } });
    history.mockResolvedValue({ ok: true, job_id: "j1", history: series([10, 90]) });
    render(<GpuTelemetryCard jobId="j1" live={false} />);
    const tiles = await screen.findByTestId("gpu-telemetry-tiles");
    expect(tiles.textContent).toContain("dash.gpu_telemetry.avg_util");
    expect(tiles.textContent).toContain("dash.gpu_telemetry.peak_power");
    expect(tiles.textContent).not.toContain("dash.gpu_telemetry.temperature");
    expect(screen.queryByTestId("gpu-telemetry-age")).toBeNull();
  });

  it("says readings start once there is a GPU", async () => {
    latest.mockResolvedValue({ ok: true, job_id: "j1", host_id: null, stale: true, telemetry: null, reason: "no_gpu" });
    history.mockResolvedValue({ ok: true, job_id: "j1", history: null });
    render(<GpuTelemetryCard jobId="j1" live />);
    expect((await screen.findByTestId("gpu-telemetry-empty")).textContent).toBe("dash.gpu_telemetry.waiting_gpu");
  });

  it("renders nothing for a finished run that recorded nothing", async () => {
    latest.mockResolvedValue({ ok: true, job_id: "j1", host_id: "h", stale: true, telemetry: null, reason: "no_readings" });
    history.mockResolvedValue({ ok: true, job_id: "j1", history: { interval_seconds: 30, fields: [], gpus: [{ index: 0, gpu_model: "", points: [] }] } });
    const { container } = render(<GpuTelemetryCard jobId="j1" live={false} />);
    await waitFor(() => expect(latest).toHaveBeenCalled());
    await waitFor(() => expect(container.querySelector('[aria-busy="true"]')).toBeNull());
    expect(screen.queryByTestId("gpu-telemetry-card")).toBeNull();
  });

  it("renders nothing when the instance is not the viewer's", async () => {
    latest.mockRejectedValue(new FakeApiError(404, "not found"));
    history.mockRejectedValue(new FakeApiError(404, "not found"));
    render(<GpuTelemetryCard jobId="j1" live />);
    await waitFor(() => expect(screen.queryByTestId("gpu-telemetry-card")).toBeNull());
  });
});
