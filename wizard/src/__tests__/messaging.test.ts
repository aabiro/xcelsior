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

    it("appends (+N more) for multiple failures", () => {
        const msg = summarizeFailure([bad("nvidia_driver"), bad("runc"), bad("docker")]);
        expect(msg).toContain("(+2 more)");
    });

    it("uses the primary (first) failed item", () => {
        // first failure is runc → message should mention runc, not docker
        const msg = summarizeFailure([ok("nvidia_driver"), bad("runc"), bad("docker")]);
        expect(msg.toLowerCase()).toContain("runc");
        expect(msg).toContain("(+1 more)");
    });

    it("falls back to the item name for an unknown check", () => {
        expect(summarizeFailure([bad("Quantum Flux Capacitor")])).toContain("Quantum Flux Capacitor");
    });

    it("reports success when nothing failed", () => {
        expect(summarizeFailure([ok("Docker"), ok("runc")])).toBe("All checks passed!");
    });
});
