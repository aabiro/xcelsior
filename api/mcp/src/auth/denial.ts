import type { AuthUser } from "./bearer.js";
import {
  TOOL_SCOPES,
  describeScopeRequirement,
  satisfiesScope,
  scopeUnion,
  type ScopeRequirement,
} from "./scopes.js";

/**
 * What a tool answers when the credential may not use it.
 *
 * One place, because it was nine copies of the same three fields, and those
 * fields told a model *that* it was refused and nothing about what to do next.
 * A refusal the agent cannot act on reaches the user as "it didn't work".
 *
 * `missing` is what the credential lacks out of what the tool needs, and
 * `how_to_fix` says who can change that. It differs by credential: a Quick
 * Connect key can ask its owner (`request_permission`), and its permissions
 * change in place, so the same session carries on. A signed-in connector was
 * granted its permissions on a consent page and has to go back through it.
 */
export function scopeDenial(
  tool: string,
  user: AuthUser | undefined,
  required: ScopeRequirement | undefined = TOOL_SCOPES[tool],
): Record<string, unknown> | null {
  if (satisfiesScope(user?.scopes, required)) return null;
  const granted = new Set(user?.scopes ?? []);
  const missing = scopeUnion(required).filter((scope) => !granted.has(scope));
  const isKey = user?.auth_type === "agent_api_key";
  return {
    error: "insufficient_scope",
    required: scopeUnion(required),
    missing,
    message: `This tool requires ${describeScopeRequirement(required)}`,
    how_to_fix: isKey
      ? "Tell the user which permission this connection lacks. Call request_permission with the " +
        "missing permissions to give them an approval link, or they can turn it on themselves on " +
        "the Connect MCP page of the Xcelsior dashboard. Either way the same key keeps working: " +
        "retry this tool once they say it is done. Do not try another tool to get around it."
      : "Tell the user which permission this connection lacks. They granted this connector's " +
        "permissions when they signed in; to add one they reconnect it and approve the permission " +
        "on the consent page.",
  };
}
