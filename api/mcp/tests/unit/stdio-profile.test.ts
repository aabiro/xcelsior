import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

// The stdio entry once passed `undefined` where the profile goes, so
// XCELSIOR_MCP_TOOL_PROFILE=operator still served the customer surface and
// the HPC tools never appeared on a stdio client, while the HTTP entry had
// always passed the profile. Both entries must hand the configured profile
// to createMcpServer; this reads the source because stdio.ts starts the
// server on import.
describe("entry points hand their tool profile to the server", () => {
  for (const file of ["src/stdio.ts", "src/index.ts"]) {
    it(`${file} passes config.toolProfile to createMcpServer`, () => {
      const source = readFileSync(new URL(`../../${file}`, import.meta.url), "utf8");
      const calls = source.match(/createMcpServer\([^;]*\)/gs) ?? [];
      expect(calls.length).toBeGreaterThan(0);
      for (const call of calls) {
        expect(call).toContain("config.toolProfile");
      }
    });
  }
});
