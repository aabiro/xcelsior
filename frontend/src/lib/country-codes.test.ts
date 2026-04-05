import { describe, expect, it } from "vitest";
import { COUNTRY_CODES } from "@/lib/country-codes";

describe("COUNTRY_CODES", () => {
    it("is a non-empty array", () => {
        expect(Array.isArray(COUNTRY_CODES)).toBe(true);
        expect(COUNTRY_CODES.length).toBeGreaterThan(50);
    });

    it("every entry has code, flag, and name", () => {
        for (const entry of COUNTRY_CODES) {
            expect(entry.code).toBeTruthy();
            expect(entry.flag).toBeTruthy();
            expect(entry.name).toBeTruthy();
        }
    });

    it("contains Canada with +1 code", () => {
        const canada = COUNTRY_CODES.find((c) => c.name === "Canada");
        expect(canada).toBeDefined();
        expect(canada!.code).toBe("+1");
    });

    it("has no duplicate entries by name", () => {
        const names = COUNTRY_CODES.map((c) => c.name);
        expect(new Set(names).size).toBe(names.length);
    });
});
