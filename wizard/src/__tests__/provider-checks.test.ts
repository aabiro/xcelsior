// Tests for provider-checks.ts — pure functions: runVerificationChecks, buildVerificationReport

import { describe, it, expect } from "vitest";
import {
    runVerificationChecks,
    buildVerificationReport,
    MINIMUM_VERSIONS,
    VERIFICATION_THRESHOLDS,
    type GpuInfo,
    type BenchmarkResult,
    type NetworkBenchResult,
    type VersionCheck,
} from "../provider-checks.js";

// ── Fixtures ─────────────────────────────────────────────────────────

function makeGpu(overrides: Partial<GpuInfo> = {}): GpuInfo {
    return {
        gpu_model: "NVIDIA RTX 4090",
        total_vram_gb: 24,
        free_vram_gb: 22,
        serial: "SN-123",
        uuid: "GPU-abc-123-def-456",
        pci_bus_id: "00:01.0",
        driver_version: "555.42.02",
        compute_capability: "8.9",
        ...overrides,
    };
}

function makeBench(overrides: Partial<BenchmarkResult> = {}): BenchmarkResult {
    return {
        tflops: 82.5,
        pcie_bandwidth_gbps: 14.2,
        pcie_h2d_gbps: 13.1,
        pcie_d2h_gbps: 15.3,
        gpu_temp_celsius: 72,
        gpu_temp_avg_celsius: 68,
        gpu_temp_samples: 10,
        gpu_model: "NVIDIA RTX 4090",
        total_vram_gb: 24,
        compute_capability: "8.9",
        cuda_version: "12.4",
        driver_version: "555.42.02",
        xcu_score: 950,
        elapsed_s: 12.5,
        ...overrides,
    };
}

function makeNetwork(overrides: Partial<NetworkBenchResult> = {}): NetworkBenchResult {
    return {
        latency_avg_ms: 15,
        latency_min_ms: 10,
        latency_max_ms: 25,
        jitter_ms: 5,
        packet_loss_pct: 0,
        throughput_mbps: 500,
        ...overrides,
    };
}

function makeVersions(allPass = true): VersionCheck[] {
    return Object.entries(MINIMUM_VERSIONS).map(([component, minimum]) => ({
        component,
        version: allPass ? minimum : null,
        minimum,
        passed: allPass,
    }));
}

// ── Tests ────────────────────────────────────────────────────────────

describe("runVerificationChecks", () => {
    it("all checks pass with good hardware", () => {
        const checks = runVerificationChecks(makeGpu(), makeBench(), makeNetwork(), makeVersions());
        expect(checks.length).toBeGreaterThanOrEqual(7);
        expect(checks.every((c) => c.passed)).toBe(true);
    });

    it("fails GPU Identity when model is empty", () => {
        const checks = runVerificationChecks(
            makeGpu({ gpu_model: "", uuid: "" }),
            makeBench(),
            makeNetwork(),
            makeVersions(),
        );
        const identity = checks.find((c) => c.name === "GPU Identity");
        expect(identity?.passed).toBe(false);
    });

    it("fails CUDA Readiness when compute capability is too low", () => {
        const checks = runVerificationChecks(
            makeGpu({ compute_capability: "5.0" }),
            makeBench({ compute_capability: "5.0" }),
            makeNetwork(),
            makeVersions(),
        );
        const cuda = checks.find((c) => c.name === "CUDA Readiness");
        expect(cuda?.passed).toBe(false);
    });

    it("fails PCIe Bandwidth when below threshold", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench({ pcie_bandwidth_gbps: 4.0 }),
            makeNetwork(),
            makeVersions(),
        );
        const pcie = checks.find((c) => c.name === "PCIe Bandwidth");
        expect(pcie?.passed).toBe(false);
    });

    it("thermal passes when sensor unavailable (temp = 0)", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench({ gpu_temp_celsius: 0 }),
            makeNetwork(),
            makeVersions(),
        );
        const thermal = checks.find((c) => c.name === "Thermal Stability");
        expect(thermal?.passed).toBe(true);
        expect(thermal?.detail).toContain("unavailable");
    });

    it("thermal fails when temp exceeds threshold", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench({ gpu_temp_celsius: 95 }),
            makeNetwork(),
            makeVersions(),
        );
        const thermal = checks.find((c) => c.name === "Thermal Stability");
        expect(thermal?.passed).toBe(false);
    });

    it("fails Network Quality on high packet loss", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench(),
            makeNetwork({ packet_loss_pct: 5 }),
            makeVersions(),
        );
        const net = checks.find((c) => c.name === "Network Quality");
        expect(net?.passed).toBe(false);
    });

    it("fails Network Quality on high jitter", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench(),
            makeNetwork({ jitter_ms: 100 }),
            makeVersions(),
        );
        const net = checks.find((c) => c.name === "Network Quality");
        expect(net?.passed).toBe(false);
    });

    it("fails Network Quality on low throughput", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench(),
            makeNetwork({ throughput_mbps: 50 }),
            makeVersions(),
        );
        const net = checks.find((c) => c.name === "Network Quality");
        expect(net?.passed).toBe(false);
    });

    it("fails Memory Integrity when VRAM mismatch", () => {
        const checks = runVerificationChecks(
            makeGpu({ total_vram_gb: 24 }),
            makeBench({ total_vram_gb: 20 }),
            makeNetwork(),
            makeVersions(),
        );
        const mem = checks.find((c) => c.name === "Memory Integrity");
        expect(mem?.passed).toBe(false);
    });

    it("fails Security Posture when versions fail", () => {
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench(),
            makeNetwork(),
            makeVersions(false),
        );
        const sec = checks.find((c) => c.name === "Security Posture");
        expect(sec?.passed).toBe(false);
    });

    it("network passes at exact threshold boundary", () => {
        const T = VERIFICATION_THRESHOLDS;
        const checks = runVerificationChecks(
            makeGpu(),
            makeBench(),
            makeNetwork({
                packet_loss_pct: T.max_network_loss_pct,
                jitter_ms: T.max_network_jitter_ms,
                throughput_mbps: T.min_network_throughput_mbps,
            }),
            makeVersions(),
        );
        const net = checks.find((c) => c.name === "Network Quality");
        expect(net?.passed).toBe(true);
    });
});

describe("buildVerificationReport", () => {
    it("returns structured report with allPassed true", () => {
        const report = buildVerificationReport(makeGpu(), makeBench(), makeNetwork(), makeVersions());
        expect(report.allPassed).toBe(true);
        expect(report.checks.length).toBeGreaterThanOrEqual(7);
        expect(report.gpu_fingerprint).toContain("NVIDIA RTX 4090");
        expect(report.gpu_fingerprint).toContain(makeGpu().uuid);
        expect(report.benchmark).toBeDefined();
        expect(report.network).toBeDefined();
        expect(report.versions).toBeDefined();
    });

    it("returns allPassed false when any check fails", () => {
        const report = buildVerificationReport(
            makeGpu({ gpu_model: "" }),
            makeBench(),
            makeNetwork(),
            makeVersions(),
        );
        expect(report.allPassed).toBe(false);
    });

    it("includes zeroed network data when network bench was skipped", () => {
        const zeroNet = makeNetwork({ latency_avg_ms: 0, jitter_ms: 0, packet_loss_pct: 0, throughput_mbps: 0 });
        const report = buildVerificationReport(makeGpu(), makeBench(), zeroNet, makeVersions());
        // Network check should fail on throughput
        const net = report.checks.find((c) => c.name === "Network Quality");
        expect(net?.passed).toBe(false);
        expect(report.allPassed).toBe(false);
    });
});

describe("constants", () => {
    it("MINIMUM_VERSIONS has required components", () => {
        expect(MINIMUM_VERSIONS.runc).toBeDefined();
        expect(MINIMUM_VERSIONS.nvidia_toolkit).toBeDefined();
        expect(MINIMUM_VERSIONS.nvidia_driver).toBeDefined();
        expect(MINIMUM_VERSIONS.docker).toBeDefined();
    });

    it("VERIFICATION_THRESHOLDS has expected fields", () => {
        expect(VERIFICATION_THRESHOLDS.min_pcie_bandwidth_gbps).toBeGreaterThan(0);
        expect(VERIFICATION_THRESHOLDS.max_gpu_temp_celsius).toBeGreaterThan(0);
        expect(VERIFICATION_THRESHOLDS.min_network_throughput_mbps).toBeGreaterThan(0);
    });
});
