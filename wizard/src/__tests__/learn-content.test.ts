// Tests for learn-content.ts — slide data integrity and helpers.

import { describe, it, expect } from "vitest";
import { LEARN_SLIDES, nextSlideIndex, slidesForMode, timeHintForStep } from "../learn-content.js";

describe("LEARN_SLIDES", () => {
    it("has several slides, each with a heading and body", () => {
        expect(LEARN_SLIDES.length).toBeGreaterThanOrEqual(5);
        for (const s of LEARN_SLIDES) {
            expect(s.heading.length).toBeGreaterThan(0);
            expect(s.lines.length).toBeGreaterThan(0);
            expect(["learn", "tips"]).toContain(s.mode);
        }
    });

    it("precomputes non-empty charts where present", () => {
        const withCharts = LEARN_SLIDES.filter((s) => s.chart);
        expect(withCharts.length).toBeGreaterThan(0);
        for (const s of withCharts) {
            expect(s.chart!.length).toBeGreaterThan(0);
            expect(s.chart!.join("")).not.toContain("undefined");
            expect(s.chart!.join("")).not.toContain("NaN");
        }
    });
});

describe("nextSlideIndex", () => {
    it("wraps around", () => {
        expect(nextSlideIndex(0, 3)).toBe(1);
        expect(nextSlideIndex(2, 3)).toBe(0);
    });
    it("is safe for zero/negative totals", () => {
        expect(nextSlideIndex(0, 0)).toBe(0);
    });
});

describe("slidesForMode", () => {
    it("learn mode excludes tips slides", () => {
        expect(slidesForMode("learn").every((s) => s.mode === "learn")).toBe(true);
    });
    it("tips mode includes everything", () => {
        expect(slidesForMode("tips").length).toBe(LEARN_SLIDES.length);
    });
});

describe("timeHintForStep", () => {
    it("gives a ~60s expectation for the benchmark", () => {
        expect(timeHintForStep("benchmark")).toContain("60s");
    });
    it("has hints for the long provider/renter steps", () => {
        for (const id of ["verification", "host-register", "launch-instance", "browse-gpus", "version-check"]) {
            expect(timeHintForStep(id)).toBeTruthy();
        }
    });
    it("returns null for steps without a meaningful wait", () => {
        expect(timeHintForStep("mode")).toBeNull();
        expect(timeHintForStep("pricing")).toBeNull();
    });
});
