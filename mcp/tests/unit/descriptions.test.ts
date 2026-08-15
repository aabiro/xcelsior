import { describe, expect, it } from "vitest";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import { TOOL_DESCRIPTIONS } from "../../src/tools/descriptions.js";

const NAMES = Object.keys(TOOL_CONTRACTS).sort();

/** Words that state what a call costs or changes. */
const IMPACT = /\b(bill|bills|billing|cost|costs|spend|spends|charge|charged|free|per million tokens|hourly)\b/i;
/** Words that warn a call cannot simply be undone. */
const IRREVERSIBILITY = /\b(irreversible|cannot be (recovered|resumed)|is lost|destroyed|preempt)/i;

describe("tool descriptions", () => {
  it("covers every tool and nothing else", () => {
    // A tool with no entry throws at registration; an orphan entry is a
    // description for a tool that no longer exists, which is worse than none.
    expect(Object.keys(TOOL_DESCRIPTIONS).sort()).toEqual(NAMES);
  });

  it.each(NAMES)("%s says what it does, when to use it, and what it costs", (name) => {
    const description = TOOL_DESCRIPTIONS[name];
    expect(description.length, `${name} description is too thin to choose on`).toBeGreaterThan(120);
    expect(description, `${name} never says when to use it`).toMatch(
      /\bUse (this |it |only )?(when|to |before|after|instead)/i,
    );
    expect(description, `${name} never states its cost or impact`).toMatch(IMPACT);
    expect(description.trim().endsWith("."), `${name} description is truncated`).toBe(true);
  });

  it("tells the model that read-only tools are read-only and free", () => {
    for (const name of NAMES) {
      if (!TOOL_CONTRACTS[name].annotations.readOnlyHint) continue;
      expect(TOOL_DESCRIPTIONS[name], `${name} is read-only but does not say so`)
        .toMatch(/Read-only/i);
    }
  });

  it("says out loud that a mutating tool changes something", () => {
    // The failure this guards against is a write tool that reads like a query.
    // A model that cannot tell them apart will call the write one to answer a
    // question — and a reviewer will find out by doing exactly that.
    for (const name of NAMES) {
      if (TOOL_CONTRACTS[name].annotations.readOnlyHint) continue;
      expect(TOOL_DESCRIPTIONS[name], `${name} mutates but reads like a query`).toMatch(
        /\b(spend|spends|launch|launching|create|creates|cancel|cancelling|terminate|destroy|destroyed|retry|retrying|reissue|reissues|preempt|remove|removal|stop|stops|corrects|resume|resumes|enqueue)\b/i,
      );
    }
  });

  it("warns explicitly on every destructive tool", () => {
    for (const name of NAMES) {
      if (!TOOL_CONTRACTS[name].annotations.destructiveHint) continue;
      expect(TOOL_DESCRIPTIONS[name], `${name} is destructive but reads as routine`)
        .toMatch(IRREVERSIBILITY);
    }
  });

  it("labels every operator tool as platform-wide", () => {
    // A tenant reading "drain a host" could reasonably think it means their
    // host. It does not — it is the platform's.
    for (const name of NAMES) {
      if (TOOL_CONTRACTS[name].tenantClass !== "operator") continue;
      expect(TOOL_DESCRIPTIONS[name], `${name} does not identify itself as an operator tool`)
        .toMatch(/Operator tool|platform/i);
    }
  });

  it("reads as prose, not as concatenated fragments", () => {
    // Descriptions are built by joining string literals, and a fragment
    // appended without a leading space silently produces "instance starts.Not
    // idempotent:". Eight descriptions carried that at once, from one helper
    // that dropped the space — invisible in the source, visible to every model
    // that reads the surface.
    const glued: string[] = [];
    const doubled: string[] = [];
    for (const [name, text] of Object.entries(TOOL_DESCRIPTIONS)) {
      const g = text.match(/.{0,24}[a-z][.!?][A-Z].{0,24}/);
      if (g) glued.push(`${name}: ...${g[0]}...`);
      const d = text.match(/.{0,24}  .{0,24}/);
      if (d) doubled.push(`${name}: ...${d[0]}...`);
    }
    expect(glued, "sentence boundary with no space between concatenated parts").toEqual([]);
    expect(doubled, "double space from a fragment joined twice").toEqual([]);
  });

  it("says that a non-idempotent tool must not be blindly retried", () => {
    for (const name of NAMES) {
      const contract = TOOL_CONTRACTS[name];
      if (contract.annotations.idempotentHint || contract.idempotency !== "none") continue;
      expect(TOOL_DESCRIPTIONS[name], `${name} is not idempotent but never says so`)
        .toMatch(/idempotent|do not retry|twice/i);
    }
  });

  it("points approval-gated tools at the approval, not at confirm", () => {
    // `confirm:true` expresses intent and never substitutes for approval. A
    // description that implies otherwise teaches the model to skip the gate.
    for (const name of ["create_instance", "create_serverless_endpoint", "evict_host_workloads"]) {
      expect(TOOL_DESCRIPTIONS[name], name).toMatch(/approval|plan_id/i);
    }
    expect(TOOL_DESCRIPTIONS.create_instance).toMatch(/never substitutes for approval/i);
  });
});
