import { describe, expect, it } from "vitest";
import { redactedArgumentHash } from "../../src/audit/context.js";

describe("MCP audit redaction", () => {
  it("hashes sensitive inputs without retaining their value", () => {
    const secret = "never-appear-in-audit";
    const first = redactedArgumentHash({
      name: "training",
      init_script: secret,
      environment: { API_TOKEN: secret },
      registry_password: secret,
    });
    const second = redactedArgumentHash({
      name: "training",
      init_script: "different",
      environment: { API_TOKEN: "different" },
      registry_password: "different",
    });
    expect(first).toMatch(/^[0-9a-f]{64}$/);
    expect(first).toBe(second);
    expect(first).not.toContain(secret);
  });

  it("changes when a non-sensitive canonical argument changes", () => {
    expect(redactedArgumentHash({ name: "a", init_script: "x" }))
      .not.toBe(redactedArgumentHash({ name: "b", init_script: "x" }));
  });
});
