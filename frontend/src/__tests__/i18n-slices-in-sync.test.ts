import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

// en-public/en-dashboard (and fr-*) are generated from en.ts/fr.ts by
// scripts/split-i18n.mjs, and the build regenerates them. An edit made only
// to a slice is silently reverted on deploy (copy fixed on 2026-10-12 came back
// that way). This fails while a slice disagrees with its source.

const DIR = join(__dirname, "..", "lib", "i18n");
const ENTRY = /^\s+"([^"]+)":\s*("(?:\\.|[^"\\])*"|`[\s\S]*?`)/gm;

function entries(file: string): Map<string, string> {
  const out = new Map<string, string>();
  const src = readFileSync(join(DIR, file), "utf8");
  for (let m; (m = ENTRY.exec(src)); ) out.set(m[1], m[2]);
  return out;
}

describe.each(["en", "fr"])("%s slices", (locale) => {
  it("match the source dictionary exactly", () => {
    const source = entries(`${locale}.ts`);
    const slices = new Map([...entries(`${locale}-public.ts`), ...entries(`${locale}-dashboard.ts`)]);
    const drifted = [...slices].filter(([key, value]) => source.get(key) !== value).map(([key]) => key);
    const missing = [...source.keys()].filter((key) => !slices.has(key));
    expect({ drifted, missing }, "run: node scripts/split-i18n.mjs").toEqual({ drifted: [], missing: [] });
  });
});
