import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

// `t(key)` falls back to the key itself, so a key missing from the
// dictionaries renders as text like "dash.settings.profile_saved". Eleven
// such keys sat in Settings: the email-change form's placeholders and buttons,
// and the toast for a saved name (owner review, 2026-11-22). This fails while
// any literal key used in the app is missing from en.ts or fr.ts.
// Keys built at runtime (template literals) are not covered.

const SRC = join(__dirname, "..");
const I18N = join(SRC, "lib", "i18n");
const KEY = /^\s+"([^"]+)":/gm;
const CALL = /\bt\(\s*"([a-z][a-zA-Z0-9_.-]*\.[a-zA-Z0-9_.-]+)"/g;

function keys(file: string): Set<string> {
  const src = readFileSync(join(I18N, file), "utf8");
  const out = new Set<string>();
  for (let m; (m = KEY.exec(src)); ) out.add(m[1]);
  return out;
}

function sources(dir: string, acc: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) {
      if (name === "__tests__" || name === "i18n" || name === "node_modules") continue;
      sources(path, acc);
    } else if (/\.(ts|tsx)$/.test(name) && !/\.test\.(ts|tsx)$/.test(name)) {
      acc.push(path);
    }
  }
  return acc;
}

describe("i18n keys", () => {
  const en = keys("en.ts");
  const fr = keys("fr.ts");
  const used = new Map<string, string>();
  for (const file of sources(SRC)) {
    const text = readFileSync(file, "utf8");
    for (let m; (m = CALL.exec(text)); ) if (!used.has(m[1])) used.set(m[1], relative(SRC, file));
  }

  it("finds the calls it checks", () => {
    expect(used.size).toBeGreaterThan(500);
  });

  it.each([
    ["en", en],
    ["fr", fr],
  ] as const)("every literal key is in %s.ts", (_locale, dictionary) => {
    const missing = [...used].filter(([key]) => !dictionary.has(key)).map(([key, file]) => `${key} (${file})`);
    expect(missing).toEqual([]);
  });
});
