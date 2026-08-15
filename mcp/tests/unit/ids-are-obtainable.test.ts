import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describeToolSurface } from "../../src/tools/surface.js";
import { TOOL_DESCRIPTIONS } from "../../src/tools/descriptions.js";

/**
 * A tool that requires an id nobody can obtain is a tool that cannot be called.
 *
 * `create_image_sweep` shipped requiring an `image_id` when no tool on the
 * surface could produce or find one — its own input schema referred callers to
 * `list_user_images`, which did not exist. It passed every test we had: the
 * registry was complete, the annotations agreed, the snapshot matched. Nothing
 * asked whether an agent could actually reach it.
 *
 * ## The property
 *
 * For every **required** input that looks like an opaque identifier, some tool
 * description must name that field. Descriptions are what a model reads in
 * `tools/list`, so this is the only place the surface can explain where an id
 * comes from. A runtime message in the tool's *response* does not count and is
 * precisely how `promotion_id` went unexplained: `promote_artifact_to_volume`
 * told the caller about `get_promotion_status` in text the model only sees
 * *after* it has already guessed.
 *
 * ## What this cannot check
 *
 * That the named tool genuinely returns the id — only that the surface claims a
 * source. Outputs are not in the published snapshot (`hasOutputSchema` is a
 * boolean), so a description could name the wrong producer and this would pass.
 * It catches the gap that actually occurred: nothing at all.
 */

const SURFACE = JSON.parse(
  readFileSync(join(import.meta.dirname, "..", "..", "tool-surface.json"), "utf8"),
) as { tools: Array<{ name: string; input: Record<string, { optional?: boolean }> }> };

/** `foo_id` — an opaque handle a model cannot invent, unlike a name or a region. */
const ID_LIKE = /^(.+)_id$/;

function requiredIds(): Map<string, string[]> {
  const out = new Map<string, string[]>();
  for (const tool of SURFACE.tools) {
    for (const [field, spec] of Object.entries(tool.input ?? {})) {
      if (spec?.optional === false && ID_LIKE.test(field)) {
        out.set(field, [...(out.get(field) ?? []), tool.name]);
      }
    }
  }
  return out;
}

describe("every required id can be obtained from the surface", () => {
  it("finds required id inputs at all", () => {
    // Calibration: an empty map satisfies the assertion below for free, and
    // the surface shape has changed under a parser here before.
    const ids = requiredIds();
    expect(ids.size).toBeGreaterThan(4);
    expect([...ids.keys()]).toContain("job_id");
  });

  it("names a source for each one in some description", () => {
    const orphans: string[] = [];
    for (const [field, consumers] of requiredIds()) {
      const named = Object.entries(TOOL_DESCRIPTIONS)
        .filter(([name, text]) => text.includes(field) && !consumers.includes(name))
        .map(([name]) => name);
      // A consumer may legitimately explain its own input — `attach_volume`
      // saying job_id comes from list_instances is exactly right.
      const selfExplained = consumers.filter((name) =>
        (TOOL_DESCRIPTIONS[name] ?? "").includes(field),
      );
      if (!named.length && !selfExplained.length) {
        orphans.push(`${field} (required by ${consumers.join(", ")})`);
      }
    }
    expect(
      orphans,
      "these ids are required and no description says where to get one, so a " +
        "model must guess a value it cannot invent",
    ).toEqual([]);
  });

  it("uses one name for the instance identifier", () => {
    // `attach_volume` was the only tool of seventeen calling it `instance_id`.
    // A model holding ids from list_instances had no reason to think they fit.
    // The alias still works through its deprecation window, so what is asserted
    // is that `job_id` is offered everywhere — not that `instance_id` is gone.
    const offenders = SURFACE.tools
      .filter((t) => {
        const fields = Object.keys(t.input ?? {});
        return fields.includes("instance_id") && !fields.includes("job_id");
      })
      .map((t) => t.name);
    expect(
      offenders,
      "these tools take instance_id and do not accept job_id, which is what " +
        "the other sixteen instance tools call the same identifier",
    ).toEqual([]);
  });
});
