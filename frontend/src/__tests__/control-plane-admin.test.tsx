import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import ControlPlaneAdminPage from "../app/(dashboard)/dashboard/admin/control-plane/page";

const mockApiFetch = vi.fn();

vi.mock("@/lib/api", () => ({
  apiFetch: (...args: any[]) => mockApiFetch(...args),
}));

vi.mock("@/lib/auth", () => ({
  useAuth: () => ({
    user: { user_id: "u-admin", role: "admin" },
  }),
}));

describe("ControlPlaneAdminPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockApiFetch.mockImplementation(async (url: string) => {
      if (url.includes("/findings")) {
        return {
          findings: [
            {
              finding_id: "f-12345678",
              resource_type: "attempt",
              resource_id: "att-123",
              finding_type: "billing_orphaned_meter",
              severity: "warning",
              summary: "usage meter for attempt att-123 is open",
              created_at: new Date().toISOString(),
              desired: { status: "closed" },
              observed: { status: "open" },
              action_taken: "Closing meter",
              action_result: { success: true }
            },
          ],
        };
      }
      if (url.includes("/jobs")) {
        return {
          jobs: [
            {
              job_id: "job-queued-1",
              status: "queued",
              submitted_at: Date.now() / 1000 - 120,
              queue_reason: "RESOURCES_UNAVAILABLE",
              queue_reason_detail: "Waiting for RTX 4090 capacity",
              gpu_model: "RTX 4090",
              attempts: [],
            },
            {
              job_id: "job-running-1",
              status: "running",
              submitted_at: Date.now() / 1000 - 3600,
              active_attempt_id: "att-run-1",
              gpu_model: "RTX 4090",
              attempts: [
                {
                  attempt_id: "att-run-1",
                  status: "running",
                  host_id: "host-1",
                  fencing_token: 10,
                  lease_claimed_at: new Date().toISOString(),
                },
              ],
            },
          ],
        };
      }
      if (url.includes("/hosts")) {
        return {
          hosts: [
            {
              host_id: "host-1",
              status: "active",
              gpu_model: "NVIDIA RTX 4090",
              vram_gb: 24,
              allocated_vram_gb: 16,
              agent_version: "2.1.4",
              instance_count: 1,
            },
            {
              host_id: "host-2",
              status: "draining",
              gpu_model: "NVIDIA A100",
              vram_gb: 80,
              allocated_vram_gb: 0,
              agent_version: "2.1.4",
              instance_count: 0,
            },
          ],
        };
      }
      if (url.includes("/scheduled-tasks")) {
        return {
          tasks: [
            {
              task_name: "reconciler",
              interval_seconds: 30,
              last_status: "succeeded",
              last_run_at: new Date().toISOString(),
              enabled: true,
            },
          ],
        };
      }
      if (url.includes("/tool-audit")) {
        return {
          ok: true,
          audits: [
            {
              audit_id: "aud-1",
              tool_name: "check_health",
              outcome: "success",
              latency_ms: 45,
              occurred_at: new Date().toISOString(),
              transport: "sse",
              api_route: "/api/health",
              api_status: 200,
            }
          ]
        };
      }
      if (url.includes("/activation-funnel")) {
        return {
          ok: true,
          funnel: {
            stages: [
              { name: "Visited", count: 100 },
              { name: "Signed Up", count: 50 },
            ]
          }
        };
      }
      if (url.includes("/oauth/clients")) {
        return {
          ok: true,
          clients: [
            {
              client_id: "client-abc",
              display_name: "Test Client",
              scopes: ["read", "write"],
              created_at: new Date().toISOString(),
              last_used_at: new Date().toISOString(),
            }
          ]
        };
      }
      return {};
    });
  });

  it("renders live scheduler queue telemetry and demand", async () => {
    render(<ControlPlaneAdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Live Scheduler Queue Telemetry & Demand")).toBeInTheDocument();
    });

    // Check queue metrics
    expect(screen.getByText("Queue Depth")).toBeInTheDocument();
    expect(screen.getByText("Active Leases")).toBeInTheDocument();
    expect(screen.getByText("Oldest In Queue")).toBeInTheDocument();
    expect(screen.getByText("Throughput Settled")).toBeInTheDocument();

    // Check hardware demand & bottlenecks
    expect(screen.getByText("Hardware Demand in Queue")).toBeInTheDocument();
    expect(screen.getByText("Placement Bottlenecks")).toBeInTheDocument();
    expect(screen.getByText("RESOURCES_UNAVAILABLE:")).toBeInTheDocument();
  });

  it("renders cluster GPU capacity & fleet allocation matrix on hosts tab", async () => {
    render(<ControlPlaneAdminPage />);

    await waitFor(() => {
      expect(screen.getByText("Host Drains & Capacity")).toBeInTheDocument();
    });

    // Switch to hosts tab
    fireEvent.click(screen.getByText("Host Drains & Capacity"));

    await waitFor(() => {
      expect(screen.getByText("Cluster GPU Capacity & Fleet Allocation Matrix")).toBeInTheDocument();
    });

    expect(screen.getByText("Fleet GPU Memory Utilization")).toBeInTheDocument();
    expect(screen.getByText("GPU Model Inventory")).toBeInTheDocument();
    expect(screen.getAllByText(/NVIDIA RTX 4090/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/NVIDIA A100/i).length).toBeGreaterThanOrEqual(1);
  });

  it("filters findings by severity", async () => {
    render(<ControlPlaneAdminPage />);
    
    await waitFor(() => {
      expect(screen.getByText("Reconciler Findings")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Reconciler Findings"));

    await waitFor(() => {
      expect(screen.getByText("usage meter for attempt att-123 is open")).toBeInTheDocument();
    });

    // filter to critical
    const severitySelect = screen.getAllByRole("combobox")[2];
    fireEvent.change(severitySelect, { target: { value: "critical" } });

    expect(screen.queryByText("usage meter for attempt att-123 is open")).not.toBeInTheDocument();
  });

  it("toggles desired/observed diff inspector", async () => {
    render(<ControlPlaneAdminPage />);
    
    await waitFor(() => {
      expect(screen.getByText("Reconciler Findings")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("Reconciler Findings"));

    await waitFor(() => {
      expect(screen.getByText("View Desired/Observed Diff")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("View Desired/Observed Diff"));

    expect(screen.getByText("Desired State")).toBeInTheDocument();
    expect(screen.getByText("Observed State")).toBeInTheDocument();
    expect(screen.getByText("Automated Actions")).toBeInTheDocument();
  });

  it("displays MCP tool audit table on MCP tab", async () => {
    render(<ControlPlaneAdminPage />);
    
    await waitFor(() => {
      expect(screen.getByText("MCP Activity")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("MCP Activity"));

    await waitFor(() => {
      expect(screen.getByText("MCP Operations & Connected Clients")).toBeInTheDocument();
    });

    expect(screen.getByText("check_health")).toBeInTheDocument();
    expect(screen.getByText("Activation Funnel")).toBeInTheDocument();
    expect(screen.getByText("Visited")).toBeInTheDocument();
  });

  it("displays connected OAuth clients with revoke", async () => {
    render(<ControlPlaneAdminPage />);
    
    await waitFor(() => {
      expect(screen.getByText("MCP Activity")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText("MCP Activity"));

    await waitFor(() => {
      expect(screen.getByText("Test Client")).toBeInTheDocument();
    });

    const revokeBtn = screen.getByText("Revoke");
    fireEvent.click(revokeBtn);

    expect(mockApiFetch).toHaveBeenCalledWith("/api/oauth/clients/client-abc", expect.objectContaining({ method: "DELETE" }));
  });
});
