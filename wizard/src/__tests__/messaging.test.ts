// Tests for messaging.ts — honest, named failure summaries (Part F).

import { describe, it, expect } from "vitest";
import { summarizeFailure } from "../messaging.js";
import type { CheckResult } from "../checks.js";

const ok = (name: string): CheckResult => ({ name, ok: true, detail: "" });
const bad = (name: string): CheckResult => ({ name, ok: false, detail: "" });

describe("summarizeFailure", () => {
    it("names a driver failure and the self-correcting intent", () => {
        expect(summarizeFailure([bad("nvidia_driver")])).toMatch(/driver.*fix/i);
    });

    it("names a docker failure", () => {
        expect(summarizeFailure([bad("Docker")]).toLowerCase()).toContain("docker");
    });

    it("names a benchmark/matmul failure", () => {
        expect(summarizeFailure([bad("FP16 Matmul")]).toLowerCase()).toContain("benchmark");
    });

    it("names a wallet failure", () => {
        expect(summarizeFailure([bad("Wallet Balance")]).toLowerCase()).toContain("funds");
    });

    it("names every failed check instead of a leftover count", () => {
        const msg = summarizeFailure([
            bad("nvidia_driver"),
            { name: "runc", ok: false, detail: "old", remediation: "Upgrade runc to ≥1.1.12 via your distro packages or containerd" },
            bad("docker"),
        ]);
        expect(msg).not.toMatch(/\+\d+ more/);
        expect(msg.toLowerCase()).toContain("driver");
        expect(msg.toLowerCase()).toContain("runc");
        expect(msg.toLowerCase()).toContain("docker");
        expect(msg).toContain("Upgrade runc");
    });

    it("uses each failed item, not only the first", () => {
        const msg = summarizeFailure([ok("nvidia_driver"), bad("runc"), bad("docker")]);
        expect(msg.toLowerCase()).toContain("runc");
        expect(msg.toLowerCase()).toContain("docker");
        expect(msg).not.toMatch(/\+\d+ more/);
    });

    it("falls back to the item name for an unknown check", () => {
        expect(summarizeFailure([bad("Quantum Flux Capacitor")])).toContain("Quantum Flux Capacitor");
    });

    it("reports success when nothing failed", () => {
        expect(summarizeFailure([ok("Docker"), ok("runc")])).toBe("All checks passed!");
    });
});
