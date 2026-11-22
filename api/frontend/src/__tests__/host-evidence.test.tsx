import { render as rtlRender, screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "@/lib/locale";
import {
  EvidenceSummary,
  HostEvidenceCard,
  VerificationChip,
} from "@/components/hosts/host-evidence";
import type { HostEvidence, HostEvidenceBadge } from "@/lib/api";
import { formatEpochDate } from "@/lib/format-units";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn() }),
  usePathname: () => "/dashboard",
}));

function render(ui: ReactElement) {
  return rtlRender(ui, { wrapper: LocaleProvider });
}

const RAN_AT = 1_763_800_000; // 2025-11-22, in seconds

function badge(overrides: Partial<HostEvidenceBadge> = {}): HostEvidenceBadge {
  return {
    verification: "verified",
    last_check_at: RAN_AT,
    next_check_at: RAN_AT + 86_400,
    benchmark: {
      ran_at: RAN_AT,
      bench_version: "1.0.0",
      verdict: "pass",
      fp16_tflops: 25.4,
      memory_bandwidth_gbps: 330,
      allocatable_vram_gb: 11.2,
      temp_peak_c: 71,
      thermal_throttle_share: 0,
    },
    reputation_tier: "gold",
    reputation_score: 480,
    job_success_rate: 0.75,
    ...overrides,
  };
}

describe("formatEpochDate", () => {
  it("reads Unix seconds, not milliseconds", () => {
    // The trust page passed seconds to `new Date()` and showed January 1970.
    expect(formatEpochDate(RAN_AT, "en-CA")).not.toContain("1970");
    expect(formatEpochDate(RAN_AT, "en-CA")).toContain("2025");
    expect(formatEpochDate(null)).toBe("—");
    expect(formatEpochDate(0)).toBe("—");
  });
});

describe("VerificationChip", () => {
  it.each([
    ["verified", "Verified"],
    ["overdue", "Check overdue"],
    ["deverified", "Deverified"],
    [undefined, "Unverified"],
    ["something-new", "Unverified"],
  ])("labels %s as %s", (state, label) => {
    render(<VerificationChip state={state} />);
    expect(screen.getByTestId("verification-chip").textContent).toBe(label);
  });
});

describe("EvidenceSummary", () => {
  it("shows the benchmark date, verdict, numbers and standing", () => {
    render(<EvidenceSummary badge={badge()} />);
    const text = screen.getByTestId("evidence-summary").textContent ?? "";
    expect(text).toContain("Benchmarked");
    expect(text).toContain("Pass");
    expect(text).toContain("25.4 TFLOPS");
    expect(text).toContain("71°C");
    expect(text).toContain("gold");
    expect(text).toContain("75% job success (30 days)");
  });

  it("says plainly when there is nothing to show", () => {
    render(<EvidenceSummary badge={badge({ benchmark: null, job_success_rate: null, reputation_tier: null })} />);
    const text = screen.getByTestId("evidence-summary").textContent ?? "";
    expect(text).toContain("No benchmark yet");
    expect(text).toContain("No finished runs in 30 days");
  });
});

describe("HostEvidenceCard", () => {
  it("names each fault a failed benchmark found", () => {
    const evidence: HostEvidence = {
      host_id: "h",
      verification: {
        state: "deverified",
        freshness: "deverified",
        overdue: false,
        last_verified_at: null,
        last_check_at: RAN_AT,
        next_check_at: RAN_AT + 3600,
      },
      benchmark: {
        ran_at: RAN_AT,
        bench_version: "1.0.0",
        verdict: "fail",
        faults: ["thermal_throttling", "allocatable_vram_below_listing"],
        gpu_model: "NVIDIA GeForce RTX 3060",
        fp16_tflops: 19,
        bf16_tflops: null,
        memory_bandwidth_gbps: 300,
        allocatable_vram_gb: 9,
        pcie_h2d_gbps: 12,
        pcie_d2h_gbps: 13,
        sustained: {
          duration_s: 300,
          sm_clock_p50_mhz: 1500,
          sm_clock_min_mhz: 1200,
          temp_peak_c: 86,
          thermal_throttle_share: 0.2,
          power_fault_share: 0,
          power_cap_share: 0,
        },
        driver_version: "580",
        cuda_version: "12.4",
      },
      benchmark_history: [{ ran_at: RAN_AT - 864_000, bench_version: "1.0.0", verdict: "pass" }],
      reputation: { tier: null, score: null },
      jobs: { window_days: 30, runs_ended: 2, host_fault_failures: 2, success_rate: 0 },
    };
    render(<HostEvidenceCard evidence={evidence} />);
    const text = screen.getByTestId("host-evidence-card").textContent ?? "";
    expect(text).toContain("Fail");
    expect(text).toContain("Thermal throttling");
    expect(text).toContain("Less VRAM than listed");
    expect(text).toContain("20.0%");
    expect(text).toContain("2 of 2 runs ended by a host fault");
    expect(text).toContain("Earlier runs:");
  });
});
