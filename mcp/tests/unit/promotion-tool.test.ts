/**
 * The promotion tool must never let a model say "saved" when it means "started".
 *
 * §3.6 of `docs/artifact-promotion-plan.md` chose an async handle over a
 * blocking call, because a large checkpoint exceeds any sane tool timeout and
 * "a timeout reads as a failure of a promotion that is still running". That
 * choice creates the risk this file guards: the tool returns quickly and
 * successfully while the copy has barely begun.
 *
 * The plan names the failure directly — *"an agent that reports 'saved' when it
 * means 'started' is the failure this whole phase exists to prevent"* — and the
 * only mechanism standing between the model and that mistake is what the
 * description and the response say. So they are tested, not assumed.
 */
import { describe, expect, it } from "vitest";

import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import { TOOL_DESCRIPTIONS } from "../../src/tools/descriptions.js";
import { TOOL_SCOPES } from "../../src/auth/scopes.js";

describe("promote_artifact_to_volume", () => {
  it("says the copy has started rather than finished", () => {
    const text = TOOL_DESCRIPTIONS.promote_artifact_to_volume;
    expect(text).toMatch(/\bSTARTS?\b/);
    expect(text).toMatch(/not when it finishes|still running/i);
  });

  it("points at the tool that answers 'is it done yet'", () => {
    // Telling a model "this is not finished" without telling it how to find out
    // leaves it guessing, and the cheapest guess is to assume success.
    expect(TOOL_DESCRIPTIONS.promote_artifact_to_volume).toMatch(/get_promotion_status/);
    expect(TOOL_SCOPES.get_promotion_status).toBeDefined();
  });

  it("says asking twice does not copy twice", () => {
    // Gate P3: "a repeated call produces one volume, not two". A model that
    // believes a retry duplicates the work will avoid retrying after a timeout,
    // which is exactly when a retry is correct.
    expect(TOOL_DESCRIPTIONS.promote_artifact_to_volume).toMatch(
      /does not copy twice|not copy twice|one promotion/i,
    );
  });

  it("needs both the read and the write scope", () => {
    // Promotion reads artifacts and writes a volume. Without the read half, a
    // caller could copy artifacts it may not read onto a volume it may write.
    const scopes = TOOL_SCOPES.promote_artifact_to_volume as { allOf?: string[] };
    expect(scopes.allOf).toContain("volumes:write");
    expect(scopes.allOf).toContain("artifacts:read");
  });

  it("is not marked read-only, and its status tool is", () => {
    expect(TOOL_CONTRACTS.promote_artifact_to_volume.annotations.readOnlyHint).toBe(false);
    expect(TOOL_CONTRACTS.get_promotion_status.annotations.readOnlyHint).toBe(true);
  });

  it("is idempotent by contract, because the server keys it", () => {
    // The idempotency is real — `(tenant, job, key)` is unique and the key
    // defaults to the manifest digest — so the annotation is a fact about the
    // server rather than a hope about the model.
    expect(TOOL_CONTRACTS.promote_artifact_to_volume.annotations.idempotentHint).toBe(true);
  });

  it("only ever mentions the files being safe as something to defer", () => {
    // The first version of this asserted the description never contains
    // "are safe". It does — in "check get_promotion_status *before* telling
    // anyone their files are safe", which is the instruction, not a claim. A
    // regex cannot tell a claim from an instruction about the claim, so the
    // assertion is that every such phrase is governed by a deferring word.
    //
    // The description was right and the test was wrong; weakening the wording
    // to satisfy a bad assertion would have been the worse repair.
    const text = TOOL_DESCRIPTIONS.promote_artifact_to_volume;
    const claims = [...text.matchAll(/\b(?:is|are)\s+(?:now\s+)?(?:saved|safe|backed up|preserved)\b/gi)];
    for (const match of claims) {
      const preceding = text.slice(Math.max(0, match.index! - 60), match.index!);
      expect(preceding, `unqualified completion claim: "${match[0]}"`).toMatch(
        /\b(before|until|not|once|when)\b/i,
      );
    }
  });
});
