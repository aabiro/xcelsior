// Tests that checkDocker / checkVersions stream each item via onItem as it
// resolves, in order, and that the streamed items equal the returned array.

import { describe, it, expect } from "vitest";
import { checkVersions, parseBenchmarkOutput } from "../provider-checks.js";

describe("checkVersions streaming", () => {
    it("calls onItem once per component, in order, matching the returned array", async () => {
        const streamed: string[] = [];
        const results = await checkVersions((v) => streamed.push(v.component));
        // 4 components: runc, docker, nvidia_driver, nvidia_toolkit
        expect(results.map((r) => r.component)).toEqual([
            "runc", "docker", "nvidia_driver", "nvidia_toolkit",
        ]);
        // streamed order must match the returned order exactly
        expect(streamed).toEqual(results.map((r) => r.component));
    });

    it("works without a callback (back-compatible)", async () => {
        const results = await checkVersions();
        expect(results.length).toBe(4);
    });

    it("streams the same object instances it returns", async () => {
        const streamed: unknown[] = [];
        const results = await checkVersions((v) => streamed.push(v));
        expect(streamed).toEqual(results);
    });
});

describe("parseBenchmarkOutput", () => {
    it("ignores @progress lines and parses the final JSON", () => {
        const stdout = [
            "@progress Initializing CUDA on H100",
            "@progress FP16 matmul…",
            JSON.stringify({ tflops: 312, pcie_bandwidth_gbps: 24, gpu_model: "H100", total_vram_gb: 80 }),
        ].join("\n");
        const r = parseBenchmarkOutput(stdout);
        expect(r.tflops).toBe(312);
        expect(r.gpu_model).toBe("H100");
        expect(r.xcu_score).toBe(31.2); // 312 / 10
    });

    it("returns an error result for a torch-missing payload", () => {
        const r = parseBenchmarkOutput(JSON.stringify({ error: "no_torch" }));
        expect(r.error).toBe("no_torch");
        expect(r.xcu_score).toBe(0);
    });

    it("handles no output gracefully", () => {
        const r = parseBenchmarkOutput("@progress only\n", "boom");
        expect(r.error).toContain("boom");
    });

    it("handles unparseable output", () => {
        const r = parseBenchmarkOutput("not json at all");
        expect(r.error).toMatch(/unparseable/i);
    });
});
