#!/usr/bin/env node
/**
 * Fail when marketing components reference i18n keys missing from generated bundles.
 * Run: node scripts/check-i18n-keys.mjs
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const I18N = join(ROOT, "src", "lib", "i18n");
const SCAN_ROOTS = [
  join(ROOT, "src", "app", "(marketing)"),
  join(ROOT, "src", "components", "marketing"),
];

const KEY_RE = /^[a-z][a-z0-9_]*(\.[a-z0-9_]+)+$/i;

/** Keys built from template literals in marketing pages. */
const DYNAMIC_KEY_EXPANSIONS = [
  ...["gpus", "mcp", "serverless", "instances", "hosting", "xcelai", "volumes"].flatMap((product) => [
    `features.prod_${product}_badge`,
    `features.prod_${product}_title`,
    `features.prod_${product}_desc`,
    `features.prod_${product}_cta`,
    `features.prod_${product}_b1`,
    `features.prod_${product}_b2`,
  ]),
  ...["pick", "provision", "pulse"].flatMap((step) => [
    `gpus.velocity_${step}`,
    `gpus.velocity_${step}_desc`,
  ]),
  "mcp.landing.solution_1",
  "mcp.landing.solution_2",
  "mcp.landing.solution_3",
];

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    const stat = statSync(path);
    if (stat.isDirectory()) walk(path, out);
    else if (entry.endsWith(".tsx") || entry.endsWith(".ts")) out.push(path);
  }
  return out;
}

function loadDictionary() {
  const enPublic = readFileSync(join(I18N, "en-public.ts"), "utf8");
  const enDashboard = readFileSync(join(I18N, "en-dashboard.ts"), "utf8");
  const keys = new Set();
  const re = /"([^"]+)":/g;
  for (const source of [enPublic, enDashboard]) {
    let m;
    while ((m = re.exec(source)) !== null) keys.add(m[1]);
  }
  return keys;
}

function extractKeys(source) {
  const keys = new Set();
  for (const m of source.matchAll(/t\(\s*["']([a-z][a-z0-9_.]*)["']/gi)) {
    const key = m[1];
    if (KEY_RE.test(key)) keys.add(key);
  }
  return keys;
}

const dictionary = loadDictionary();
const used = new Set(DYNAMIC_KEY_EXPANSIONS);
for (const file of SCAN_ROOTS.flatMap((dir) => walk(dir))) {
  const source = readFileSync(file, "utf8");
  for (const key of extractKeys(source)) used.add(key);
}

const missing = [...used].filter((key) => !dictionary.has(key)).sort();
if (missing.length) {
  console.error(`Missing ${missing.length} i18n key(s) in en-public/en-dashboard:`);
  for (const key of missing) console.error(`  - ${key}`);
  process.exit(1);
}

console.log(`OK - ${used.size} marketing keys present in dictionary.`);