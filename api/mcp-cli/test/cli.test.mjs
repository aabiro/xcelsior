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

// ── `connect`: browser sign-in through mcp-remote, with Xcelsior's sign-off ──

const connectModule = await import(resolve(root, "bin/connect.js"));

test("the sign-off follows the ready line exactly once, however the log is chunked", () => {
  const { createLogRelay, SIGN_OFF } = connectModule;
  let out = "";
  const relay = createLogRelay((t) => (out += t));
  const log = "[42] Connected to remote server\r\n[42] Press Ctrl+C to exit\n[42] later line\n[42] Press Ctrl+C to exit\n";
  for (const piece of [log.slice(0, 7), log.slice(7, 40), log.slice(40, 52), log.slice(52)]) relay.push(piece);
  relay.end();
  assert.equal(
    out,
    "[42] Connected to remote server\n[42] Press Ctrl+C to exit\n" +
      `${SIGN_OFF}\n[42] later line\n[42] Press Ctrl+C to exit\n`,
  );
});

test("connect passes the MCP channel through and signs off after mcp-remote is up", async () => {
  const { spawn } = await import("node:child_process");
  const { mkdtempSync, writeFileSync } = await import("node:fs");
  const { tmpdir } = await import("node:os");
  const dir = mkdtempSync(resolve(tmpdir(), "xcl-connect-"));
  const fake = resolve(dir, "fake-mcp-remote.mjs");
  // Stands in for mcp-remote: logs like it, echoes stdin to stdout, then exits.
  writeFileSync(
    fake,
    `process.stderr.write("[7] Connected to remote server using StreamableHTTPClientTransport\\n[7] Proxy established successfully\\n");
process.stderr.write("[7] Press Ctrl+C to exit\\n");
process.stdin.on("data", (d) => process.stdout.write("echo:" + d));
process.stdin.on("end", () => { process.stderr.write("[7] url=" + process.argv[2] + "\\n"); process.exit(3); });
`,
  );
  const child = spawn(process.execPath, [resolve(root, "bin/xcelsior-mcp.js"), "connect"], {
    env: { ...process.env, XCELSIOR_MCP_REMOTE_COMMAND: `${process.execPath} ${fake}` },
  });
  let out = "";
  let err = "";
  child.stdout.on("data", (d) => (out += d));
  child.stderr.on("data", (d) => (err += d));
  child.stdin.end('{"jsonrpc":"2.0","id":1,"method":"ping"}\n');
  const code = await new Promise((done) => child.on("exit", done));
  assert.equal(out, 'echo:{"jsonrpc":"2.0","id":1,"method":"ping"}\n', "stdout must be untouched");
  const lines = err.trim().split("\n");
  const ready = lines.indexOf("[7] Press Ctrl+C to exit");
  assert.ok(ready > 0, err);
  assert.equal(lines[ready + 1], connectModule.SIGN_OFF);
  assert.equal(lines.filter((l) => l === connectModule.SIGN_OFF).length, 1);
  assert.ok(lines.includes(`[7] url=${connectModule.DEFAULT_URL}`), "defaults to the hosted server");
  assert.equal(code, 3, "the child's exit code is ours");
});
