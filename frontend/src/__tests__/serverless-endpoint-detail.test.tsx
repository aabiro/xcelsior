import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { mockEndpoint } from "./serverless-test-helpers";

const apiMocks = vi.hoisted(() => ({
  getServerlessEndpoint: vi.fn(),
  getServerlessEndpointMetrics: vi.fn(),
  listServerlessWorkers: vi.fn(),
  listServerlessJobs: vi.fn(),
  listServerlessKeys: vi.fn(),
  createServerlessKey: vi.fn(),
  revokeServerlessKey: vi.fn(),
  deleteServerlessEndpoint: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);
vi.mock("@/hooks/useEventStream", () => ({ useEventStream: () => ({ status: "disconnected" }) }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));
vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en" }),
}));
vi.mock("framer-motion", async () => {
  const ReactModule = await import("react");
  const div = ReactModule.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => ReactModule.createElement("div", { ...props, ref }, children),
  );
  return { FadeIn: ({ children }: { children: React.ReactNode }) => children, motion: { div } };
});
vi.mock("@/lib/recharts", () => ({
  AreaChart: ({ children }: { children: React.ReactNode }) => <div data-testid="area-chart">{children}</div>,
  Area: () => null,
  XAxis: () => null,
  YAxis: () => null,
  Tooltip: () => null,
  ResponsiveContainer: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  CartesianGrid: () => null,
}));

import { EndpointDetail } from "@/features/serverless/endpoint-detail";

const metrics = {
  endpoint_id: mockEndpoint.endpoint_id,
  window_sec: 3600,
  total_requests: 42,
  window_requests: 5,
  jobs_completed: 4,
  jobs_failed: 1,
  jobs_cancelled: 0,
  success_rate: 0.8,
  error_rate: 0.2,
  queue_depth: 0,
  avg_queue_ms: 100,
  avg_execution_ms: 500,
  avg_gpu_seconds: 10,
  total_gpu_seconds: 120,
  tokens_per_sec: 12,
  total_output_tokens: 1000,
  total_cost_cad: 2.5,
  active_workers: 1,
  idle_workers: 0,
  busy_workers: 0,
};

describe("EndpointDetail", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMocks.getServerlessEndpoint.mockResolvedValue({ endpoint: mockEndpoint });
    apiMocks.getServerlessEndpointMetrics.mockResolvedValue({ metrics });
    apiMocks.listServerlessWorkers.mockResolvedValue({ workers: [] });
    apiMocks.listServerlessJobs.mockResolvedValue({ jobs: [] });
    apiMocks.listServerlessKeys.mockResolvedValue({ keys: [] });
  });

  it("renders overview metrics for readers", async () => {
    render(<EndpointDetail endpointId={mockEndpoint.endpoint_id} canWrite={false} />);
    expect(await screen.findByText("test-llama", {}, { timeout: 10_000 })).toBeInTheDocument();
    expect(screen.getByText("dash.serverless.metric_requests")).toBeInTheDocument();
    expect(screen.queryByText("dash.serverless.key_create")).not.toBeInTheDocument();
  }, 15_000);

  it("shows keys panel and hides create control for viewers", async () => {
    render(<EndpointDetail endpointId={mockEndpoint.endpoint_id} canWrite={false} />);
    await waitFor(() => expect(screen.getByText("test-llama")).toBeInTheDocument());

    fireEvent.click(screen.getByText("dash.serverless.tab_keys"));
    await waitFor(() => {
      expect(screen.getByText("dash.serverless.keys_empty")).toBeInTheDocument();
    });
    expect(screen.queryByText("dash.serverless.key_create")).not.toBeInTheDocument();
  });

  it("allows writers to open try-it tab", async () => {
    render(<EndpointDetail endpointId={mockEndpoint.endpoint_id} canWrite />);
    await waitFor(() => expect(screen.getByText("test-llama")).toBeInTheDocument());

    fireEvent.click(screen.getByText("dash.serverless.tab_tryit"));
    await waitFor(() => {
      expect(screen.getByText("dash.serverless.try_chat")).toBeInTheDocument();
    });
  });
});