import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

/**
 * The journey script is the executable form of gate P1 clause 1 and gate P2
 * clause 1, and P2 states the constraint that makes it meaningful:
 *
 *   "completes using **only tool calls** ... A journey that needs a raw HTTP
 *    call or a dashboard click fails the gate."
 *
 * A script that quietly reaches for the REST API when a tool is missing proves
 * the opposite of what it claims, and that is the natural thing to do the first
 * time a step does not work. So the constraint is enforced here rather than
 * left to discipline.
 */
const SOURCE = readFileSync(
  resolve(__dirname, "../../src/gate-journey.ts"),
  "utf8",
);

/** Stripped of comments — the file *discusses* not using fetch, at length. */
const CODE = SOURCE.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");

describe("the gate journey uses only tool calls", () => {
  it("performs no HTTP of its own", () => {
    for (const forbidden of ["fetch(", "axios", "XMLHttpRequest", "node:http", "undici"]) {
      expect(
        CODE.includes(forbidden),
        `gate-journey.ts references ${forbidden}. Gate P2 fails a journey that ` +
          `needs a raw HTTP call; reaching for one to work around a missing tool ` +
          `proves the opposite of what the gate asks.`,
      ).toBe(false);
    }
  });

  it("names no REST paths", () => {
    const paths = CODE.match(/["'`]\/api\/[^"'`]*["'`]/g) ?? [];
    expect(
      paths,
      `gate-journey.ts names REST paths (${paths.join(", ")}), which only a raw ` +
        `HTTP call would need.`,
    ).toEqual([]);
  });

  it("requires every tool it depends on, up front", () => {
    // A missing tool must fail by name before anything launches, not surface
    // later as a confusing call error against a half-built journey.
    expect(CODE).toContain("listTools");
    for (const tool of [
      "create_instance",
      "watch_instance",
      "open_instance_access",
      "terminate_instance",
      "top_up_wallet",
      "list_payment_methods",
    ]) {
      expect(CODE, `${tool} is not in the journey's required set`).toContain(tool);
    }
  });

  it("terminates what it launched even when a clause fails", () => {
    // A gate that leaks a running GPU costs money on exactly the runs nobody is
    // watching — the failing ones.
    const finallyIndex = CODE.indexOf("} finally {");
    expect(finallyIndex, "the journey has no finally block").toBeGreaterThan(-1);

    // The actual call, not the string anywhere after the block: the first
    // version of this check matched `terminate_instance` inside a log message
    // and passed even when the call itself had been renamed away.
    const tail = CODE.slice(finallyIndex);
    expect(
      /callTool\(\s*\{\s*name:\s*["'`]terminate_instance["'`]/.test(tail),
      "terminate_instance is not *called* from the finally block, so a failed " +
        "journey leaves the instance running and billing — on exactly the runs " +
        "nobody is watching",
    ).toBe(true);
  });

  it("reads the token from stdin, never argv", () => {
    expect(CODE).toContain("readFileSync(0");
    expect(
      /process\.argv\[\d\][^\n]*token/i.test(CODE),
      "the token is taken from argv, where it lands in shell history, process " +
        "listings and CI logs",
    ).toBe(false);
  });

  it("does not treat an SCA challenge as a pass", () => {
    // Gate P1 asks for a completion with NO browser. A 3DS decline is a
    // legitimate outcome of the call and a failure of the clause.
    expect(CODE).toContain("authentication_required");
  });
});
