/**
 * Regenerate `mcp/tool-surface.json`.
 *
 *   npm run surface:update
 *
 * Run this deliberately, in the same commit as the change it records — that is
 * what makes the diff reviewable. `tests/unit/surface.test.ts` fails if the
 * committed snapshot and the code disagree, so a forgotten run is caught rather
 * than shipped.
 */
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describeToolSurface } from "../src/tools/surface.js";

const here = dirname(fileURLToPath(import.meta.url));
const target = join(here, "..", "tool-surface.json");

const snapshot = {
  $comment:
    "Published MCP tool surface. Regenerate with `npm run surface:update` in the " +
    "same commit as the change. A breaking change requires a toolVersion bump — " +
    "see docs/mcp-tool-versioning.md.",
  generatedFor: "customer",
  tools: describeToolSurface("customer"),
};

writeFileSync(target, `${JSON.stringify(snapshot, null, 2)}\n`);
console.log(`wrote ${target} (${snapshot.tools.length} tools)`);
