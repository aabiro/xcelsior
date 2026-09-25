import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import React from "react";
import { computeConvergence, DesiredObservedBadge } from "@/components/instances/desired-observed-badge";
import { CostMeterCard } from "@/components/instances/cost-meter-card";
import InstanceDetailPage from "@/app/(dashboard)/dashboard/instances/[id]/page";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    apiFetch: vi.fn(),
    fetchInstance: vi.fn(),
    fetchInstanceTimeline: vi.fn(),
    fetchActiveLease: vi.fn(),
    fetchPlacementExplanation: vi.fn(),
    fetchInstanceLogs: vi.fn().mockResolvedValue({ logs: [] }),
    createInstanceLogStream: vi.fn().mockReturnValue({
      addEventListener: vi.fn(),
      close: vi.fn(),
    }),
    ApiError: class ApiError extends Error {
      status: number;
      constructor(msg: string, status: number) { super(msg); this.status = status; }
    }
  };
});

vi.mock("@/lib/auth", () => ({
  useAuth: () => ({ user: { user_id: "u-1", role: "user" } }),
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key }),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "j-123" }),
  useRouter: () => ({ push: vi.fn() }),
}));

describe("B6.5: Instance Control Plane Enhancements", () => {
  describe("computeConvergence", () => {
    it("returns terminal for terminal states", () => {
      expect(computeConvergence({ jobStatus: "completed" })).toBe("terminal");
      expect(computeConvergence({ jobStatus: "failed" })).toBe("terminal");
      expect(computeConvergence({ jobStatus: "cancelled" })).toBe("terminal");
    });

    it("returns pending for queued jobs without an active attempt", () => {
      expect(computeConvergence({ jobStatus: "queued", activeAttemptStatus: null })).toBe("pending");
      expect(computeConvergence({ jobStatus: "queued", activeAttemptStatus: undefined })).toBe("pending");
    });

    it("returns converged when running and active", () => {
      expect(
        computeConvergence({
          jobStatus: "running",
          activeAttemptStatus: "running",
          leaseStatus: "active",
        })
      ).toBe("converged");
    });

    it("returns pending when starting, assigned, or leased", () => {
      expect(computeConvergence({ jobStatus: "starting" })).toBe("pending");
      expect(computeConvergence({ jobStatus: "assigned" })).toBe("pending");
      expect(computeConvergence({ jobStatus: "leased" })).toBe("pending");
    });

    it("returns diverged for other mismatches", () => {
      expect(
        computeConvergence({
          jobStatus: "running",
          activeAttemptStatus: "starting",
          leaseStatus: "active",
        })
      ).toBe("diverged");
      
      expect(
        computeConvergence({
          jobStatus: "running",
          activeAttemptStatus: "running",
          leaseStatus: "expired",
        })
      ).toBe("diverged");
    });
  });

  describe("DesiredObservedBadge", () => {
    it("renders Converged with green styling", () => {
      render(
        <DesiredObservedBadge
          jobStatus="running"
          activeAttemptStatus="running"
          leaseStatus="active"
        />
      );
      expect(screen.getByText("Converged")).toBeInTheDocument();
      // the class includes green
      expect(screen.getByText("Converged").className).toMatch(/text-green/);
    });

    it("renders Diverged with amber styling", () => {
      render(
        <DesiredObservedBadge
          jobStatus="running"
          activeAttemptStatus="failed"
          leaseStatus="active"
        />
      );
      expect(screen.getByText("Diverged")).toBeInTheDocument();
      expect(screen.getByText("Diverged").className).toMatch(/text-amber/);
    });

    it("renders Pending with blue styling", () => {
      render(
        <DesiredObservedBadge
          jobStatus="queued"
        />
      );
      expect(screen.getByText("Pending")).toBeInTheDocument();
      expect(screen.getByText("Pending").className).toMatch(/text-blue/);
    });
  });

  describe("CostMeterCard", () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it("renders rate and estimates cost based on uptime, without NaN", () => {
      // 1 hour ago
      const startedAt = Math.floor(Date.now() / 1000) - 3600;
      
      render(
        <CostMeterCard
          instance={{
            status: "running",
            rate_per_hour: 2.5,
            started_at: startedAt,
            total_cost: null,
          }}
        />
      );
      
      expect(screen.getByText("2.5")).toBeInTheDocument();
      // Should show 2.50 (1 hour * 2.5)
      expect(screen.getByText("2.50")).toBeInTheDocument();
    });

    it("shows 0.00 for missing rates", () => {
      render(<CostMeterCard instance={{}} />);
      const elements = screen.getAllByText("0.00");
      expect(elements.length).toBeGreaterThan(0);
      expect(screen.queryByText("NaN")).not.toBeInTheDocument();
    });
  });

  describe("Integration: Reconcile Button", () => {
    let apiFetchMock: any;
    
    beforeEach(async () => {
      const api = await import("@/lib/api");
      apiFetchMock = api.apiFetch;
      apiFetchMock.mockResolvedValue({ ok: true });
      
      (api.fetchInstance as any).mockResolvedValue({
        instance: {
          job_id: "j-123",
          status: "running",
          rate_per_hour: 1.5,
          host_id: "h-123"
        }
      });
      (api.fetchInstanceTimeline as any).mockResolvedValue({ ok: true, attempts: [] });
      (api.fetchActiveLease as any).mockResolvedValue({ ok: true, lease: null });
    });

    it("renders reconcile button and calls API on click", async () => {
      render(<InstanceDetailPage />);
      
      // Wait for load
      await waitFor(() => {
        expect(screen.getByText("Reconcile")).toBeInTheDocument();
      });
      
      fireEvent.click(screen.getByText("Reconcile"));
      
      await waitFor(() => {
        expect(apiFetchMock).toHaveBeenCalledWith("/api/v1/instances/j-123/reconcile", {
          method: "POST"
        });
      });
    });
    
    it("does not render reconcile button for terminal states", async () => {
      const api = await import("@/lib/api");
      (api.fetchInstance as any).mockResolvedValue({
        instance: {
          job_id: "j-123",
          status: "completed",
        }
      });
      
      render(<InstanceDetailPage />);
      
      await waitFor(() => {
        expect(screen.queryByText("Reconcile")).not.toBeInTheDocument();
      });
    });
  });
});
