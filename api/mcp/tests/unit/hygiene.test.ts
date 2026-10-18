import { describe, expect, it } from "vitest";
import { REDACTED, scrubResponse, scrubText } from "../../src/lib/hygiene.js";

describe("response hygiene", () => {
  it("removes auth material at any depth", () => {
    const { value, removed } = scrubResponse({
      instance: {
        job_id: "job-1",
        registry_password: "hunter2",
        env: { API_KEY: "abc", NAME: "trainer" },
      },
      client_secret: "s3cret",
      access_token: "xoa_deadbeef",
    });
    const serialised = JSON.stringify(value);
    expect(serialised).not.toContain("hunter2");
    expect(serialised).not.toContain("s3cret");
    expect(serialised).not.toContain("abc");
    expect(serialised).toContain("job-1");
    expect(serialised).toContain("trainer");
    expect(removed).toContain("client_secret");
    expect(removed).toContain("access_token");
  });

  it("masks credential-shaped values even under an innocent key", () => {
    // The failure this catches: an upstream field named `value` or `detail`
    // that happens to carry a live token.
    const { value, removed } = scrubResponse({
      note: "use Bearer xoa_AbCdEf0123456789AbCdEf to authenticate",
      detail: "sk_live_0123456789abcdefABCDEF failed",
    });
    expect(JSON.stringify(value)).not.toContain("xoa_AbCdEf");
    expect(JSON.stringify(value)).not.toContain("sk_live_0123");
    expect(JSON.stringify(value)).toContain(REDACTED);
    expect(removed.length).toBe(2);
  });

  it("removes debug payloads and internal identifiers", () => {
    const { value } = scrubResponse({
      ok: false,
      traceback: "File \"db.py\", line 1",
      sql: "SELECT * FROM users",
      internal_id: 42,
      _private: "internal",
      detail: "instance not found",
    });
    expect(value).toEqual({ ok: false, detail: "instance not found" });
  });

  it("preserves the MCP protocol's own _meta namespace", () => {
    // `_meta` starts with an underscore but is published on purpose — it
    // carries our tool version and contract metadata.
    const { value } = scrubResponse({ _meta: { "xcelsior/toolVersion": "2.0.0" }, ok: true });
    expect(value).toEqual({ _meta: { "xcelsior/toolVersion": "2.0.0" }, ok: true });
  });

  it("leaves ordinary tool output untouched", () => {
    const payload = {
      instances: [{ job_id: "j1", status: "running", gpu_model: "RTX 4090", host_id: "h1" }],
      next_cursor: "MTAw",
      estimate: { currency: "CAD", estimate_micros: 1_234_000 },
      idempotency_key: "5b1f0e6a-0000-4000-8000-000000000000",
      plan_id: "11111111-1111-4111-8111-111111111111",
      approval_url: "https://xcelsior.ca/approve/1",
    };
    const { value, removed } = scrubResponse(payload);
    expect(removed).toEqual([]);
    expect(value).toEqual(payload);
  });

  it("does not mangle a plain summary string", () => {
    const { text, masked } = scrubText("Launch plan 1111 executed.");
    expect(masked).toBe(false);
    expect(text).toBe("Launch plan 1111 executed.");
  });

  it("walks arrays", () => {
    const { value } = scrubResponse({ rows: [{ id: 1, password: "x" }, { id: 2 }] });
    expect(value).toEqual({ rows: [{ id: 1 }, { id: 2 }] });
  });
});
