import { describe, expect, it } from "vitest";
import { scopesForProfile } from "../../src/tools/profiles.js";

/**
 * Changing `scopes_supported` signs out every running mcp-remote.
 *
 * The list is derived from the tool registry, so adding a tool with a new scope
 * changes it silently. mcp-remote reads it once, when a process starts, and
 * stamps every token it saves with the scope string it asked for. A process
 * that started before the change compares a token saved by a newer process
 * against its own stale copy, calls the scopes "changed", deletes the shared
 * token file and starts a browser sign-in, which the newer processes then wait
 * on. A user with several clients open (Desktop, Code sessions, IDEs) is
 * locked out until every old process is restarted. That happened on
 * 2026-11-25, after `providers:onboard` was added (9972dd1a).
 *
 * So a change here must be deliberate: update this pin, add the release note
 * that connectors must be restarted after the deploy, and say so to the user
 * who deploys it. The order matters too: mcp-remote compares the joined string.
 */
const PINNED_CUSTOMER_SCOPES = [
  "artifacts:read",
  "billing:read",
  "billing:topup",
  "email",
  "events:read",
  "gpu:read",
  "inference:read",
  "inference:write",
  "instances:connect",
  "instances:operate",
  "instances:read",
  "instances:write",
  "marketplace:read",
  "profile",
  "providers:onboard",
  "providers:read",
  "reputation:read",
  "ssh:read",
  "ssh:write",
  "volumes:read",
  "volumes:write",
];

describe("advertised scopes", () => {
  it("are the pinned list, in order: changing them signs out every running connector", () => {
    expect(scopesForProfile("customer", { companyKnowledge: false }).join(" ")).toBe(
      PINNED_CUSTOMER_SCOPES.join(" "),
    );
  });
});
