// Tests for ascii-viz.ts — pure chart renderers. Geometry & edge cases.

import { describe, it, expect } from "vitest";
import {
    sparkline, barChart, lineChart, funnel, pipeline,
    trimNumber, compactNumber,
} from "../ascii-viz.js";

describe("sparkline", () => {
    it("returns empty string for empty input", () => {
        expect(sparkline([])).toBe("");
    });

    it("renders one tick per value", () => {
        expect(sparkline([1, 2, 3, 4]).length).toBe(4);
    });

    it("maps min to lowest and max to highest tick", () => {
        const out = sparkline([0, 100]);
        expect(out[0]).toBe("▁");
        expect(out[1]).toBe("█");
    });

    it("renders a flat series as a constant mid baseline (no NaN)", () => {
        const out = sparkline([5, 5, 5]);
        expect(out).toBe("▄▄▄");
        expect(out).not.toContain("undefined");
    });

    it("tolerates NaN / Infinity by coercing", () => {
        const out = sparkline([NaN, Infinity, 1, 2]);
        expect(out.length).toBe(4);
        expect(out).not.toContain("undefined");
    });

    it("rises monotonically for an increasing series", () => {
        const out = sparkline([1, 2, 3, 4, 5, 6, 7, 8]);
        // each tick index should be non-decreasing
        const idx = [...out].map((c) => "▁▂▃▄▅▆▇█".indexOf(c));
        for (let i = 1; i < idx.length; i++) expect(idx[i]).toBeGreaterThanOrEqual(idx[i - 1]);
    });
});

describe("barChart", () => {
    it("returns empty array for no items", () => {
        expect(barChart([])).toEqual([]);
    });

    it("renders a full bar for the max item and aligns labels", () => {
        const lines = barChart([
            { label: "a", value: 100 },
            { label: "bb", value: 50 },
        ], { width: 10 });
        expect(lines).toHaveLength(2);
        expect(lines[0]).toContain("██████████"); // 100% → 10 filled
        expect(lines[0]).toContain("100%");
        expect(lines[1]).toContain("50%");
        // labels are padded to equal width → bars start at same column
        const barCol0 = lines[0].indexOf("█");
        const barCol1 = lines[1].indexOf("█");
        expect(barCol0).toBe(barCol1);
    });

    it("clamps values above max and never overflows width", () => {
        const lines = barChart([{ label: "x", value: 200, max: 100 }], { width: 8 });
        const filled = (lines[0].match(/█/g) || []).length;
        expect(filled).toBe(8);
    });

    it("handles zero max without dividing by zero", () => {
        const lines = barChart([{ label: "x", value: 0 }], { width: 8 });
        expect(lines[0]).toContain("░");
        expect(lines[0]).toContain("0%");
    });

    it("supports value mode with a unit", () => {
        const lines = barChart([{ label: "vram", value: 24, max: 80 }], {
            width: 10, valueMode: "value", unit: " GB",
        });
        expect(lines[0]).toContain("24 GB");
    });

    it("truncates over-long labels with an ellipsis", () => {
        const lines = barChart([{ label: "a-very-long-label-indeed", value: 1 }], { labelWidth: 8 });
        expect(lines[0]).toContain("…");
    });
});

describe("lineChart", () => {
    it("returns `height` rows", () => {
        expect(lineChart([1, 2, 3], { height: 5, width: 10 })).toHaveLength(5);
    });

    it("returns blank rows for empty input", () => {
        const rows = lineChart([], { height: 4 });
        expect(rows).toHaveLength(4);
        expect(rows.every((r) => r === "")).toBe(true);
    });

    it("places the max near the top row and min near the bottom", () => {
        const rows = lineChart([0, 100], { height: 4, width: 8, axis: false });
        // top row should contain a point (the 100), bottom row should contain a point (the 0)
        expect(rows[0]).toContain("•");
        expect(rows[rows.length - 1]).toContain("•");
    });

    it("includes axis labels when axis is on", () => {
        const rows = lineChart([0, 9575], { height: 4, width: 8, axis: true });
        expect(rows[0]).toContain("┤");
        expect(rows.join("\n")).toContain("9.6k");
    });
});

describe("funnel", () => {
    it("computes drop between stages", () => {
        const lines = funnel([
            { label: "app_launched", value: 1200 },
            { label: "ride_requested", value: 864 },
        ], 20).join("\n");
        expect(lines).toContain("app_launched");
        expect(lines).toContain("100%");
        expect(lines).toContain("72%");
        expect(lines).toContain("336"); // 1200 - 864 drop
    });

    it("returns empty for no stages", () => {
        expect(funnel([])).toEqual([]);
    });
});

describe("pipeline", () => {
    it("joins stages with arrows and marks the active one", () => {
        const out = pipeline(["submit", "schedule", "run", "settle"], 2);
        expect(out).toContain("→");
        expect(out).toContain("◆ run");
        expect(out).toContain("◇ submit");
    });
});

describe("number formatting", () => {
    it("trimNumber adds thousands separators for integers", () => {
        expect(trimNumber(9575)).toBe("9,575");
        expect(trimNumber(3.5)).toBe("3.5");
    });
    it("compactNumber abbreviates", () => {
        expect(compactNumber(9575)).toBe("9.6k");
        expect(compactNumber(1_200_000)).toBe("1.2M");
        expect(compactNumber(42)).toBe("42");
    });
});
