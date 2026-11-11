import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const cliPkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
const serverPkg = JSON.parse(readFileSync(resolve(root, "../mcp/package.json"), "utf8"));
const readme = readFileSync(resolve(root, "README.md"), "utf8");

test("cli package metadata and entrypoint are valid", () => {
  assert.equal(cliPkg.name, "@xcelsior-gpu/mcp");
  assert.equal(cliPkg.bin["xcelsior-mcp"], "bin/xcelsior-mcp.js");
  assert.ok(existsSync(resolve(root, "bin/xcelsior-mcp.js")));
});

test("runtime dependencies stay aligned with @xcelsior-gpu/mcp-server", () => {
  for (const [dep, range] of Object.entries(cliPkg.dependencies ?? {})) {
    assert.equal(
      range,
      serverPkg.dependencies?.[dep],
      `dependency ${dep} drifted from ../mcp/package.json`,
    );
  }
});

test("README documents privacy policy and required auth token", () => {
  assert.match(readme, /https:\/\/xcelsior\.ca\/privacy/);
  assert.match(readme, /XCELSIOR_ACCESS_TOKEN/);
});
