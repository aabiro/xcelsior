/**
 * A failed call says what the API said.
 *
 * The API answers most errors with its own envelope,
 * `{ ok: false, error: { code, message, details? } }` (every HTTPException and
 * every request-validation error), and some with an RFC 9457 problem. The
 * client recognised only the problem, so for the envelope it lost the code and
 * kept the whole body as JSON for the detail; `simulate_instance_placement`
 * called without a `name` then answered only "simulate_instance_placement
 * failed.", which is the text most clients show.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import { XcelsiorApiClient } from "../../src/client/api.js";
import { ApiError, problemFromBody } from "../../src/client/errors.js";
import { registerDiagnosticTools } from "../../src/tools/diagnostics.js";
import type { AuthUser } from "../../src/auth/bearer.js";

afterEach(() => vi.unstubAllGlobals());

const VALIDATION = {
  ok: false,
  error: {
    code: "validation_error",
    message: "Request validation failed",
    details: [{ loc: ["body", "name"], type: "missing", msg: "Field required" }],
  },
};

function answer(status: number, body: unknown) {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify(body), { status })));
}

describe("the API client", () => {
  it("reads the API's envelope: its code, and which field failed", async () => {
    answer(422, VALIDATION);
    const client = new XcelsiorApiClient({ baseUrl: "https://api.example", bearer: "redacted" });
    const error = await client.post("/api/v1/placements/simulate", {}).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).problem).toMatchObject({
      status: 422,
      code: "validation_error",
      detail: "Request validation failed: name: Field required",
      errors: VALIDATION.error.details,
    });
  });

  it("reads an HTTPException's message", () => {
    const body = { ok: false, error: { code: "http_error", message: "Instance not found" } };
    expect(problemFromBody(body, 404)).toMatchObject({ code: "http_error", detail: "Instance not found" });
  });

  it("names the fields a launch-spec problem lists", () => {
    const body = {
      type: "about:blank", status: 422, code: "invalid_spec",
      detail: "the launch spec failed validation",
      errors: [{ code: "out_of_range", field: "gpu_count", detail: "must be at least 1" }],
    };
    expect(problemFromBody(body, 422)?.detail).toBe(
      "the launch spec failed validation: gpu_count: must be at least 1",
    );
  });

  it("keeps a plain RFC 9457 problem as it was", () => {
    const body = { type: "https://docs.xcelsior.ca/problems/x", code: "version_conflict", detail: "stale version" };
    expect(problemFromBody(body, 409)).toEqual(body);
  });

  it("turns FastAPI's own list detail into words, not [object Object]", () => {
    const body = { detail: [{ loc: ["body", "spec", "gpu_count"], msg: "Input should be a valid integer" }] };
    expect(problemFromBody(body, 422)?.detail).toBe("spec.gpu_count: Input should be a valid integer");
  });

  it("leaves a body it cannot read to the caller", () => {
    expect(problemFromBody("upstream timed out", 502)).toBeUndefined();
    expect(problemFromBody({ ok: false }, 500)).toBeUndefined();
  });
});

describe("simulate_instance_placement", () => {
  it("says which field the API refused, in the text a client shows", async () => {
    answer(422, VALIDATION);
    type Handler = (args: Record<string, unknown>) => Promise<{
      content: { text: string }[];
      structuredContent: Record<string, unknown>;
      isError?: boolean;
    }>;
    const handlers = new Map<string, Handler>();
    const server = { registerTool: (name: string, _config: unknown, handler: Handler) => handlers.set(name, handler) };
    const client = new XcelsiorApiClient({ baseUrl: "https://api.example", bearer: "redacted" });
    registerDiagnosticTools(
      server as unknown as Parameters<typeof registerDiagnosticTools>[0],
      client,
      { scopes: ["instances:read", "gpu:read"] } as AuthUser,
    );

    const result = await handlers.get("simulate_instance_placement")!({ spec: { gpu_model: "RTX 3060" } });

    expect(result.isError).toBe(true);
    expect(result.content[0].text).toBe(
      "simulate_instance_placement failed: Request validation failed: name: Field required",
    );
    expect(result.structuredContent).toMatchObject({ ok: false, code: "validation_error", status: 422 });
  });
});
