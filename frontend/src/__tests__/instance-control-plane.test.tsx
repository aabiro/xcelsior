import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, within, act } from "@testing-library/react";
import React from "react";
import { computeConvergence, DesiredObservedBadge } from "@/components/instances/desired-observed-badge";
import { CostMeterCard } from "@/components/instances/cost-meter-card";
import InstanceDetailPage from "@/app/(dashboard)/dashboard/instances/[id]/page";
import * as api from "@/lib/api";

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

vi.mock("@/components/terminal/WebTerminal", () => ({ WebTerminal: () => null }));
vi.mock("@/hooks/useInstanceWebSocket", () => ({
  useInstanceWebSocket: () => ({ connected: false, reconnecting: false, error: null }),
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
          activeAttemptId: "current",
          leaseAttemptId: "current",
          leaseExpiresAt: "2099-01-01T00:00:00Z",
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
          activeAttemptId: "current",
          leaseAttemptId: "current",
          leaseExpiresAt: "2099-01-01T00:00:00Z",
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
          activeAttemptId="current"
          leaseAttemptId="current"
          leaseExpiresAt="2099-01-01T00:00:00Z"
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
          activeAttemptId="current"
          leaseAttemptId="current"
          leaseExpiresAt="2099-01-01T00:00:00Z"
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

    it("renders the server-provided CAD rate and compute estimate", () => {
      render(<CostMeterCard instance={{ rate_per_hour_cad: 2.5, cost_cad: 3.75 }} />);
      expect(screen.getByText("2.50")).toBeInTheDocument();
      expect(screen.getByText("3.75")).toBeInTheDocument();
    });

    it("shows unavailable for missing rates", () => {
      render(<CostMeterCard instance={{}} />);
      expect(screen.getAllByText("Unavailable")).toHaveLength(2);
      expect(screen.queryByText("NaN")).not.toBeInTheDocument();
    });
  });

  describe("Integration: Reconcile Button", () => {
    let apiFetchMock: any;
    
    beforeEach(async () => {
      vi.clearAllMocks();
      const api = await import("@/lib/api");
      apiFetchMock = api.apiFetch;
      apiFetchMock.mockResolvedValue({ ok: true });
      
      (api.fetchInstance as any).mockResolvedValue({
        instance: {
          job_id: "j-123",
          status: "running",
          rate_per_hour_cad: 1.5,
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
      await screen.findByText("Terminal");
      await waitFor(() => {
        expect(screen.queryByText("Reconcile")).not.toBeInTheDocument();
      });
    });

    const synchronization = () => within(screen.getByRole("status", { name: "Instance synchronization" }));
    const runningAttempt = {
      attempt_id: "current", attempt_number: 2, status: "running", host_id: null,
      placement_score: null, failure_code: null, reserved_at: null, command_created_at: null,
      lease_claimed_at: null, started_at: null, ended_at: null, trace_id: null,
    };
    const currentLease = {
      lease_id: "lease-current", attempt_id: "current", status: "active", host_alias: "host-a",
      offered_at: null, claim_deadline: null, claimed_at: null, last_renewed_at: null,
      expires_at: "2099-01-01T00:00:00Z",
    };

    it("does not let an old running attempt mask failure of the current attempt", async () => {
      vi.mocked(api.fetchInstanceTimeline).mockResolvedValue({ ok: true, job_id: "j-123", attempts: [
        { ...runningAttempt, attempt_id: "old", attempt_number: 1 },
        { ...runningAttempt, status: "failed" },
      ] });
      vi.mocked(api.fetchActiveLease).mockResolvedValue({ ok: true, job_id: "j-123", lease: currentLease });
      render(<InstanceDetailPage />);
      await screen.findByText("State Synchronization");
      await waitFor(() => expect(synchronization().getByText("Diverged")).toBeInTheDocument());
    });

    it("shows unavailable when observation requests fail", async () => {
      vi.mocked(api.fetchInstanceTimeline).mockRejectedValue(new Error("unavailable"));
      vi.mocked(api.fetchActiveLease).mockRejectedValue(new Error("unavailable"));
      render(<InstanceDetailPage />);
      await screen.findByText("State Synchronization");
      await waitFor(() => expect(synchronization().getByText("Unavailable")).toBeInTheDocument());
      expect(synchronization().queryByText("Diverged")).not.toBeInTheDocument();
    });

    it("shows checking until both observation requests finish", async () => {
      let finish!: (value: Awaited<ReturnType<typeof api.fetchActiveLease>>) => void;
      vi.mocked(api.fetchInstanceTimeline).mockResolvedValue({ ok: true, job_id: "j-123", attempts: [runningAttempt] });
      vi.mocked(api.fetchActiveLease).mockReturnValue(new Promise((resolve) => { finish = resolve; }));
      render(<InstanceDetailPage />);
      await screen.findByText("State Synchronization");
      expect(synchronization().getByText("Checking")).toBeInTheDocument();
      await act(async () => { finish({ ok: true, job_id: "j-123", lease: currentLease }); });
      expect(synchronization().getByText("Converged")).toBeInTheDocument();
    });

    it("refreshes lease renewals even when job status and updated_at stay unchanged", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        vi.mocked(api.fetchInstance).mockImplementation(async () => ({
          ok: true,
          instance: { job_id: "j-123", status: "running", gpu_model: "A100", docker_image: "test", submitted_at: 1, updated_at: 1 },
        }));
        vi.mocked(api.fetchInstanceTimeline).mockResolvedValue({ ok: true, job_id: "j-123", attempts: [runningAttempt] });
        vi.mocked(api.fetchActiveLease).mockResolvedValue({ ok: true, job_id: "j-123", lease: currentLease });
        render(<InstanceDetailPage />);
        await screen.findByText("Converged");
        await act(async () => { await vi.advanceTimersByTimeAsync(5100); });
        const reads = vi.mocked(api.fetchActiveLease).mock.calls.length;
        await act(async () => { await vi.advanceTimersByTimeAsync(5100); });
        expect(vi.mocked(api.fetchActiveLease).mock.calls.length).toBeGreaterThan(reads);
      } finally {
        vi.useRealTimers();
      }
    });
  });
});
