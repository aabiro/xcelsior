// Tests for preflight.ts — verdict classification, response parsing, AI health.

import { describe, it, expect } from "vitest";
import {
    classifyVerdict, parseStatusResponse, findService, aiHealthyFromReport,
    type ServiceStatus, type StatusReport,
} from "../preflight.js";

const svc = (name: string, state: ServiceStatus["state"], required = false): ServiceStatus => ({
    name, state, detail: "", required,
});

describe("classifyVerdict", () => {
    it("operational when all services are operational", () => {
        expect(classifyVerdict([svc("API", "operational", true), svc("DB", "operational", true)]))
            .toBe("operational");
    });

    it("degraded when a non-required service is down", () => {
        expect(classifyVerdict([svc("API", "operational", true), svc("AI", "down", false)]))
            .toBe("degraded");
    });

    it("degraded when any service is degraded", () => {
        expect(classifyVerdict([svc("API", "operational", true), svc("Billing", "degraded", false)]))
            .toBe("degraded");
    });

    it("blocked when a required service is down", () => {
        expect(classifyVerdict([svc("API", "operational", true), svc("Database", "down", true)]))
            .toBe("blocked");
    });
});

describe("parseStatusResponse", () => {
    it("parses a well-formed backend response", () => {
        const parsed = parseStatusResponse({
            ok: true,
            verdict: "operational",
            services: [
                { name: "API", state: "operational", detail: "reachable", required: true },
                { name: "AI (Hexara)", state: "degraded", detail: "no key", required: false },
            ],
        });
        expect(parsed).not.toBeNull();
        expect(parsed!.services).toHaveLength(2);
        // verdict is recomputed from services → degraded (AI degraded)
        expect(parsed!.verdict).toBe("degraded");
    });

    it("recomputes verdict from services even if server field disagrees", () => {
        const parsed = parseStatusResponse({
            verdict: "operational", // server lies
            services: [{ name: "DB", state: "down", required: true }],
        });
        expect(parsed!.verdict).toBe("blocked");
    });

    it("coerces unknown states to 'unknown'", () => {
        const parsed = parseStatusResponse({
            services: [{ name: "X", state: "weird", required: false }],
        });
        expect(parsed!.services[0].state).toBe("unknown");
    });

    it("returns null for missing services array", () => {
        expect(parseStatusResponse({ ok: true })).toBeNull();
    });

    it("returns null for empty services", () => {
        expect(parseStatusResponse({ services: [] })).toBeNull();
    });

    it("returns null for non-object input", () => {
        expect(parseStatusResponse(null)).toBeNull();
        expect(parseStatusResponse("nope")).toBeNull();
    });

    it("drops malformed service entries but keeps valid ones", () => {
        const parsed = parseStatusResponse({
            services: [{ name: "API", state: "operational", required: true }, { bad: 1 }, 42],
        });
        expect(parsed!.services).toHaveLength(1);
    });
});

describe("findService / aiHealthyFromReport", () => {
    const report: StatusReport = {
        verdict: "degraded",
        fallback: false,
        services: [
            svc("API", "operational", true),
            { name: "AI (Hexara)", state: "degraded", detail: "no key", required: false },
        ],
    };

    it("matches a service by name prefix", () => {
        expect(findService(report, "ai")?.name).toBe("AI (Hexara)");
        expect(findService(report, "API")?.name).toBe("API");
    });

    it("aiHealthyFromReport false when AI degraded", () => {
        expect(aiHealthyFromReport(report)).toBe(false);
    });

    it("aiHealthyFromReport true when AI operational", () => {
        const ok: StatusReport = { ...report, services: [svc("AI (Hexara)", "operational")] };
        expect(aiHealthyFromReport(ok)).toBe(true);
    });

    it("aiHealthyFromReport defaults to true when no AI signal", () => {
        const noAi: StatusReport = { ...report, services: [svc("API", "operational", true)] };
        expect(aiHealthyFromReport(noAi)).toBe(true);
        expect(aiHealthyFromReport(null)).toBe(true);
    });
});
