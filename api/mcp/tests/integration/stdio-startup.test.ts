import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { describe, expect, it } from "vitest";

// Regression gate for the "MCP start pops up an OAuth browser flow" defect.
//
// The fix moves the local client onto the stdio transport (no HTTP → no 401 →
// no OAuth discovery) AND makes stdio startup tolerant of a missing/expired
// token: it must still start and serve `initialize` + `tools/list`, surfacing
// auth failures per-call rather than exiting. A hard exit would show the server
// as "failed to start", which some clients recover from by launching the exact
// interactive auth flow we are avoiding.
//
// This drives the *built* dist/stdio.js — the artifact the VS Code config runs —
// so a build that regresses the behaviour fails here.

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const STDIO_ENTRY = path.resolve(__dirname, "../../dist/stdio.js");

function runStdio(input: string, env: Record<string, string>): Promise<{ code: number | null; stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [STDIO_ENTRY], {
      env: { ...process.env, ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (d) => (stdout += d.toString()));
    child.stderr.on("data", (d) => (stderr += d.toString()));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
    child.stdin.write(input);
    child.stdin.end();
    // Safety: never let a hung child stall the suite.
    setTimeout(() => child.kill("SIGKILL"), 12_000).unref();
  });
}

const INIT = JSON.stringify({
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "vitest", version: "1" } },
});
const LIST = JSON.stringify({ jsonrpc: "2.0", id: 2, method: "tools/list" });

describe("stdio startup robustness (no OAuth flow)", () => {
  it("starts and serves initialize + tools/list with NO token", async () => {
    const { code, stdout, stderr } = await runStdio(`${INIT}\n${LIST}\n`, {
      XCELSIOR_ACCESS_TOKEN: "",
      XCELSIOR_OAUTH_TOKEN: "",
      // Unreachable API URL: proves startup never depends on a live backend or a
      // valid token (best-effort validateBearer must not be fatal).
      XCELSIOR_API_URL: "http://127.0.0.1:1",
    });
    // Did not hard-exit before serving.
    expect(code === 0 || code === null).toBe(true);
    expect(stdout).toContain('"serverInfo"');
    expect(stdout).toContain("xcelsior-mcp");
    // Full tool registry is available even without auth (auth is enforced per
    // call by the API, not by hiding the catalog).
    expect(stdout).toContain('"list_available_gpus"');
    expect(stdout).toContain('"create_instance"');
    // Operator gets an actionable hint, not a crash.
    expect(stderr).toContain("Quick Connect");
  }, 15_000);

  it("starts with an invalid token instead of exiting (would-be 401 deferred to calls)", async () => {
    const { code, stdout } = await runStdio(`${INIT}\n`, {
      XCELSIOR_ACCESS_TOKEN: "definitely-not-a-valid-token",
      XCELSIOR_API_URL: "http://127.0.0.1:1",
    });
    expect(code === 0 || code === null).toBe(true);
    expect(stdout).toContain('"serverInfo"');
  }, 15_000);
});
