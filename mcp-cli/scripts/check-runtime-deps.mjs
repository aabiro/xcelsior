#!/usr/bin/env node
// Fails the build when the copied server code imports a package this wrapper
// does not declare, or declares it at a different range than ../mcp does.
//
// dist/ is ../mcp's compiled output, but npm installs only the dependencies
// listed here. The wrapper listed two of the eight packages the stdio entry
// imports, so a published copy would have crashed on its first import.

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const wrapper = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8")).dependencies ?? {};
const server = JSON.parse(readFileSync(resolve(root, "../mcp/package.json"), "utf8")).dependencies ?? {};

const seen = new Set();
const reached = new Map();
function walk(file) {
    if (seen.has(file)) return;
    seen.add(file);
    const src = readFileSync(file, "utf8");
    const pattern = /(?:import|export)\s[^"']*?from\s*["']([^"']+)["']|import\s*["']([^"']+)["']|import\(\s*["']([^"']+)["']\s*\)/g;
    const specifiers = [];
    for (let m; (m = pattern.exec(src)); ) specifiers.push(m[1] || m[2] || m[3]);
    for (const spec of specifiers) {
        if (spec.startsWith(".")) walk(resolve(dirname(file), spec));
        else if (!spec.startsWith("node:")) {
            const name = spec.startsWith("@") ? spec.split("/").slice(0, 2).join("/") : spec.split("/")[0];
            if (!reached.has(name)) reached.set(name, file.slice(root.length + 1));
        }
    }
}
walk(resolve(root, "dist/stdio.js"));

const problems = [];
for (const [name, via] of reached) {
    if (!(name in wrapper)) problems.push(`${name} is imported (${via}) but not in dependencies; ../mcp has ${server[name] ?? "nothing"}`);
    else if (server[name] && wrapper[name] !== server[name]) problems.push(`${name} is ${wrapper[name]} here but ${server[name]} in ../mcp`);
}
if (problems.length) {
    console.error("mcp-cli dependencies do not match what dist/stdio.js imports:\n  " + problems.join("\n  "));
    process.exit(1);
}
console.log(`mcp-cli dependencies cover all ${reached.size} packages dist/stdio.js imports (${seen.size} files).`);
