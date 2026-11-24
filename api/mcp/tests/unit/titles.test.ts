import { describe, expect, it } from "vitest";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import { TOOL_TITLES } from "../../src/tools/titles.js";

/**
 * Titles are what a user reads in Claude's confirmation prompt ("Allow
 * Terminate instance?"), and the Connectors Directory requires one on every
 * tool. Keep them short, distinct and plain.
 */
describe("tool titles", () => {
  const entries = Object.entries(TOOL_TITLES);

  it("covers exactly the tools that have contracts", () => {
    expect(Object.keys(TOOL_TITLES).sort()).toEqual(Object.keys(TOOL_CONTRACTS).sort());
  });

  it("is sentence case, at most 40 characters, with no trailing punctuation", () => {
    const bad = entries.filter(
      ([, t]) => t.length > 40 || !/^[A-Z]/.test(t) || /[.!?:;,]$/.test(t) || t !== t.trim(),
    );
    expect(bad).toEqual([]);
  });

  it("gives no two tools the same title", () => {
    const seen = new Map<string, string>();
    const dupes: string[] = [];
    for (const [name, title] of entries) {
      const key = title.toLowerCase();
      if (seen.has(key)) dupes.push(`${seen.get(key)} / ${name}: ${title}`);
      seen.set(key, name);
    }
    expect(dupes).toEqual([]);
  });
});
