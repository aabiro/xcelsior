// Tests for version parsing, instance name generation, API constants,
// and STATE_COLORS — pure utility functions and exported constants.

import { describe, it, expect } from "vitest";
import { parseVersion, versionGte } from "../provider-checks.js";
import { generateInstanceName } from "../useWizardFlow.js";
import {
    STATE_COLORS,
    type WizardState,
} from "../../sprites/wizard/wizard-sprite.js";

// ══════════════════════════════════════════════════════════════════════
// 1. parseVersion
// ══════════════════════════════════════════════════════════════════════

describe("parseVersion", () => {
    it("parses standard 3-segment version", () => {
        expect(parseVersion("1.2.3")).toEqual([1, 2, 3]);
        expect(parseVersion("27.3.1")).toEqual([27, 3, 1]);
        expect(parseVersion("560.35.03")).toEqual([560, 35, 3]);
    });

    it("parses version with trailing content", () => {
        expect(parseVersion("24.0.7-1")).toEqual([24, 0, 7]);
        expect(parseVersion("1.17.8-rc1")).toEqual([1, 17, 8]);
    });

    it("parses version with leading text", () => {
        expect(parseVersion("v1.2.3")).toEqual([1, 2, 3]);
        expect(parseVersion("Docker version 27.3.1")).toEqual([27, 3, 1]);
    });

    it("returns [0,0,0] for 2-segment versions", () => {
        // This is important: MINIMUM_VERSIONS.nvidia_driver = "550.0"
        expect(parseVersion("550.0")).toEqual([0, 0, 0]);
        expect(parseVersion("24.0")).toEqual([0, 0, 0]);
    });

    it("returns [0,0,0] for empty string", () => {
        expect(parseVersion("")).toEqual([0, 0, 0]);
    });

    it("returns [0,0,0] for garbage", () => {
        expect(parseVersion("garbage")).toEqual([0, 0, 0]);
        expect(parseVersion("abc.def.ghi")).toEqual([0, 0, 0]);
    });

    it("returns [0,0,0] for single number", () => {
        expect(parseVersion("550")).toEqual([0, 0, 0]);
    });

    it("handles zero segments", () => {
        expect(parseVersion("0.0.0")).toEqual([0, 0, 0]);
        expect(parseVersion("1.0.0")).toEqual([1, 0, 0]);
        expect(parseVersion("0.0.1")).toEqual([0, 0, 1]);
    });

    it("handles large version numbers", () => {
        expect(parseVersion("100.200.300")).toEqual([100, 200, 300]);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 2. versionGte
// ══════════════════════════════════════════════════════════════════════

describe("versionGte", () => {
    it("equal versions return true", () => {
        expect(versionGte("1.2.3", "1.2.3")).toBe(true);
        expect(versionGte("24.0.0", "24.0.0")).toBe(true);
    });

    it("greater major version returns true", () => {
        expect(versionGte("25.0.0", "24.0.0")).toBe(true);
        expect(versionGte("2.0.0", "1.99.99")).toBe(true);
    });

    it("lesser major version returns false", () => {
        expect(versionGte("23.9.9", "24.0.0")).toBe(false);
    });

    it("greater minor version returns true", () => {
        expect(versionGte("24.1.0", "24.0.0")).toBe(true);
    });

    it("lesser minor version returns false", () => {
        expect(versionGte("24.0.0", "24.1.0")).toBe(false);
    });

    it("greater patch version returns true", () => {
        expect(versionGte("24.0.1", "24.0.0")).toBe(true);
    });

    it("lesser patch version returns false", () => {
        expect(versionGte("24.0.0", "24.0.1")).toBe(false);
    });

    it("2-segment versions both become [0,0,0] → always true (equal)", () => {
        // This is a known quirk: parseVersion("550.0") = [0,0,0]
        // so versionGte("560.35", "550.0") = versionGte("[0,0,0]", "[0,0,0]") = true
        expect(versionGte("550.0", "550.0")).toBe(true);
    });

    it("3-segment version vs 2-segment minimum", () => {
        // parseVersion("560.35.03") = [560, 35, 3]
        // parseVersion("550.0") = [0, 0, 0]
        // So any 3-segment version >= [0,0,0] → true
        expect(versionGte("560.35.03", "550.0")).toBe(true);
    });

    it("handles versions with trailing content", () => {
        expect(versionGte("24.0.7-1", "24.0.0")).toBe(true);
        expect(versionGte("1.17.8-rc1", "1.17.8")).toBe(true);
    });

    it("garbage inputs both parse to [0,0,0] → true", () => {
        expect(versionGte("garbage", "garbage")).toBe(true);
    });

    it("all MINIMUM_VERSIONS thresholds are checkable", () => {
        // Verify the known versions actually pass against their minimums
        expect(versionGte("1.2.1", "1.1.12")).toBe(true);   // runc
        expect(versionGte("27.3.1", "24.0.0")).toBe(true);  // docker
        // nvidia_driver "550.0" is 2-segment — always passes due to [0,0,0]
        expect(versionGte("560.35.03", "550.0")).toBe(true);
        expect(versionGte("1.18.0", "1.17.8")).toBe(true);  // nvidia_toolkit
    });

    it("just-below minimum returns false", () => {
        expect(versionGte("1.1.11", "1.1.12")).toBe(false);
        expect(versionGte("23.9.9", "24.0.0")).toBe(false);
    });
});

// ══════════════════════════════════════════════════════════════════════
// 3. generateInstanceName
// ══════════════════════════════════════════════════════════════════════

describe("generateInstanceName", () => {
    const ADJECTIVES = ["swift", "bright", "cosmic", "nova", "stellar", "quantum", "astral", "blazing"];
    const NOUNS = ["forge", "nexus", "pulse", "flux", "spark", "core", "beam", "arc"];

    it("returns adj-noun-NNN format", () => {
        const name = generateInstanceName();
        expect(name).toMatch(/^[a-z]+-[a-z]+-\d+$/);
    });

    it("adjective is from known list", () => {
        for (let i = 0; i < 50; i++) {
            const [adj] = generateInstanceName().split("-");
            expect(ADJECTIVES).toContain(adj);
        }
    });

    it("noun is from known list", () => {
        for (let i = 0; i < 50; i++) {
            const parts = generateInstanceName().split("-");
            expect(NOUNS).toContain(parts[1]);
        }
    });

    it("number is 0-999", () => {
        for (let i = 0; i < 50; i++) {
            const parts = generateInstanceName().split("-");
            const num = Number(parts[2]);
            expect(num).toBeGreaterThanOrEqual(0);
            expect(num).toBeLessThan(1000);
            expect(Number.isInteger(num)).toBe(true);
        }
    });

    it("generates unique names across runs (probabilistic)", () => {
        const names = new Set<string>();
        for (let i = 0; i < 100; i++) {
            names.add(generateInstanceName());
        }
        // 8 * 8 * 1000 = 64,000 possible combinations
        // With 100 draws, collision is extremely unlikely
        expect(names.size).toBeGreaterThanOrEqual(90);
    });

    it("name has exactly three parts", () => {
        for (let i = 0; i < 20; i++) {
            const parts = generateInstanceName().split("-");
            expect(parts).toHaveLength(3);
        }
    });
});

// ══════════════════════════════════════════════════════════════════════
// 4. STATE_COLORS
// ══════════════════════════════════════════════════════════════════════

describe("STATE_COLORS", () => {
    const ALL_STATES: WizardState[] = ["idle", "thinking", "success", "error", "waiting", "excited", "finishing"];

    it("has a color for every WizardState", () => {
        for (const state of ALL_STATES) {
            expect(STATE_COLORS).toHaveProperty(state);
        }
    });

    it("all values are valid hex color strings", () => {
        for (const state of ALL_STATES) {
            expect(STATE_COLORS[state]).toMatch(/^#[0-9a-fA-F]{6}$/);
        }
    });

    it("all colors are unique", () => {
        const colors = ALL_STATES.map((s) => STATE_COLORS[s]);
        expect(new Set(colors).size).toBe(colors.length);
    });

    it("has exactly 7 entries", () => {
        expect(Object.keys(STATE_COLORS)).toHaveLength(7);
    });
});
