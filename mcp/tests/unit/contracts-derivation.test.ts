/**
 * `TOOL_CONTRACTS` is derived from `TOOL_SCOPES`, and that must stay true.
 *
 * P0 names five artifacts that have to agree: contracts, scopes, annotations,
 * descriptions and the published snapshot. Two of them already cannot drift —
 * `contracts.ts` computes `TOOL_CONTRACTS` at *import time* from `TOOL_SCOPES`,
 * and no script writes that file. That is a structural guarantee rather than a
 * convention, which is why P0.3's remaining work is scoped to the other three.
 *
 * But the guarantee is undocumented in any test. A refactor that replaced the
 * `Object.fromEntries(Object.entries(TOOL_SCOPES).map(...))` with a literal —
 * for readability, or to give one tool a bespoke requirement — would remove it
 * and nothing would notice until the two lists disagreed in production. The
 * OpenAPI generator read its own output for months on exactly that basis: a
 * derivation that was true, then quietly wasn't.
 *
 * So this asserts the derivation itself, not merely that the two currently
 * happen to match:
 *
 *   - every scoped tool has a contract, and vice versa
 *   - each contract's `scopeRequirement` is the *same object* as its
 *     `TOOL_SCOPES` entry, which a hand-written literal cannot satisfy
 *   - the advertised `requiredScopes` is that requirement flattened, never an
 *     independently maintained list
 *
 * The third matters because `requiredScopes` is metadata for advertising while
 * `scopeRequirement` is what authorizes. Flattening `allOf` into a flat list is
 * exactly how the any-one-of bug read, and `contracts.ts` says so at the point
 * the flattening happens.
 */
import { describe, expect, it } from "vitest";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import { TOOL_SCOPES, scopeUnion } from "../../src/auth/scopes.js";

describe("TOOL_CONTRACTS is derived, not maintained", () => {
  it("covers exactly the tools in TOOL_SCOPES", () => {
    expect(Object.keys(TOOL_CONTRACTS).sort()).toEqual(Object.keys(TOOL_SCOPES).sort());
  });

  it("reuses the scope requirement by reference, which a literal cannot fake", () => {
    // Reference equality is the point. Deep equality would pass for a
    // hand-written copy that happens to match today and drifts tomorrow.
    for (const [name, requirement] of Object.entries(TOOL_SCOPES)) {
      expect(
        TOOL_CONTRACTS[name].scopeRequirement,
        `${name}: scopeRequirement is not the TOOL_SCOPES entry itself — the ` +
          "derivation has been replaced by a copy, and the two can now drift",
      ).toBe(requirement);
    }
  });

  it("advertises the flattened union, never a separate list", () => {
    for (const [name, requirement] of Object.entries(TOOL_SCOPES)) {
      expect(TOOL_CONTRACTS[name].requiredScopes, name).toEqual(scopeUnion(requirement));
    }
  });

  it("never advertises a tool as requiring nothing", () => {
    const empty = Object.entries(TOOL_CONTRACTS)
      .filter(([, contract]) => contract.requiredScopes.length === 0)
      .map(([name]) => name);
    expect(empty, "tools advertising no required scope").toEqual([]);
  });

  it("finds a meaningful number of tools, so the checks are not vacuous", () => {
    // A broken import would leave both objects empty and every assertion above
    // would pass on nothing — the shape of a guard that reports clean because
    // it looked in the wrong place.
    expect(Object.keys(TOOL_SCOPES).length).toBeGreaterThanOrEqual(30);
  });
});
