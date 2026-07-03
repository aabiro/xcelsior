#!/usr/bin/env node
/**
 * Split monolithic i18n dictionaries into public (marketing/auth) and dashboard slices.
 * Run after editing en.ts / fr.ts: node scripts/split-i18n.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "src", "lib", "i18n");
const DASHBOARD_PREFIXES = ["dash.", "gear.", "ai."];

function isDashboardKey(key) {
  return DASHBOARD_PREFIXES.some((p) => key.startsWith(p));
}

function parseEntries(source) {
  const entries = [];
  const re = /^\s+"([^"]+)":\s*("(?:\\.|[^"\\])*"|`[\s\S]*?`)/gm;
  let m;
  while ((m = re.exec(source)) !== null) {
    entries.push({ key: m[1], value: m[2] });
  }
  return entries;
}

function renderSlice(locale, slice, entries) {
  const header =
    slice === "public"
      ? `/* Auto-generated from ${locale}.ts - public/marketing/auth keys. Do not edit directly. */`
      : `/* Auto-generated from ${locale}.ts - dashboard-only keys. Do not edit directly. */`;
  const body = entries
    .map(({ key, value }) => `  "${key}": ${value},`)
    .join("\n");
  return `${header}\n\nconst ${locale}_${slice}: Record<string, string> = {\n${body}\n};\n\nexport default ${locale}_${slice};\n`;
}

function splitLocale(locale) {
  const source = readFileSync(join(ROOT, `${locale}.ts`), "utf8");
  const entries = parseEntries(source);
  const pub = [];
  const dash = [];
  for (const entry of entries) {
    (isDashboardKey(entry.key) ? dash : pub).push(entry);
  }
  writeFileSync(join(ROOT, `${locale}-public.ts`), renderSlice(locale, "public", pub));
  writeFileSync(join(ROOT, `${locale}-dashboard.ts`), renderSlice(locale, "dashboard", dash));
  console.log(`${locale}: ${pub.length} public, ${dash.length} dashboard`);
}

splitLocale("en");
splitLocale("fr");