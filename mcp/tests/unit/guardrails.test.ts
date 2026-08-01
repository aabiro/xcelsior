import { describe, expect, it } from "vitest";
import { evaluateShouldIRunThis } from "../../src/lib/guardrails.js";

describe("evaluateShouldIRunThis", () => {
  const estimate = {
    gpu_model: "RTX 4090",
    duration_hours: 2,
    rate_cad_per_hour: 1.5,
    gross_cost_cad: 3.0,
    currency: "CAD",
  };

  it("approves when wallet covers cost and rate is within budget", () => {
    const result = evaluateShouldIRunThis(
      { gpu_model: "RTX 4090", duration_hours: 2, max_hourly_cad: 2 },
      estimate,
      10,
    );
    expect(result.approved).toBe(true);
    expect(result.reasons).toHaveLength(0);
    expect(result.wallet_balance_cad).toBe(10);
  });

  it("rejects when wallet is insufficient", () => {
    const result = evaluateShouldIRunThis(
      { gpu_model: "RTX 4090", duration_hours: 2 },
      estimate,
      1,
    );
    expect(result.approved).toBe(false);
    expect(result.reasons.some((r) => r.includes("wallet"))).toBe(true);
  });

  it("rejects when hourly rate exceeds max", () => {
    const result = evaluateShouldIRunThis(
      { gpu_model: "RTX 4090", duration_hours: 2, max_hourly_cad: 1 },
      estimate,
      50,
    );
    expect(result.approved).toBe(false);
    expect(result.reasons.some((r) => r.includes("max_hourly_cad"))).toBe(true);
  });

  it("includes a jurisdiction note naming the requested residency", () => {
    const result = evaluateShouldIRunThis(
      { gpu_model: "RTX 4090", duration_hours: 1, require_residency: "CA" },
      estimate,
      50,
    );
    expect(result.jurisdiction_note).toContain("CA");
    expect(result.jurisdiction_note).toContain("Do not assume");
  });

  it("omits the jurisdiction note when residency is unconstrained", () => {
    const result = evaluateShouldIRunThis(
      { gpu_model: "RTX 4090", duration_hours: 1 },
      estimate,
      50,
    );
    expect(result.jurisdiction_note).toBeUndefined();
  });
});