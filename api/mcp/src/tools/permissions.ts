import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import type { AuthUser } from "../auth/bearer.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { TOOL_SCOPES, satisfiesScope, scopeUnion } from "../auth/scopes.js";
import { scopeDenial } from "../auth/denial.js";
import { TOOL_CONTRACTS } from "./contracts.js";

/**
 * What this connection may do, and how its owner can change that.
 *
 * A Quick Connect key is issued the moment the dashboard page opens, with no
 * consent screen, and until these tools existed nothing told the user what the
 * agent holding it could do. `get_permissions` is what the server instructions
 * ask an agent to call first and show. `request_permission` is how the agent
 * asks for more without being able to grant it: the API records a request and
 * returns a link, and only a person signed in to the dashboard can approve it.
 */

interface PermissionItem {
  scope: string;
  description: string;
  how?: string;
}

interface PermissionsResponse {
  credential?: string;
  permission_preset?: string;
  granted?: PermissionItem[];
  not_granted?: PermissionItem[];
  can_request?: boolean;
  manage_url?: string;
  pending_requests?: unknown[];
}

/**
 * Customer tools this credential cannot use, grouped by the permission that
 * would unlock them. Built from the scope registry so it cannot disagree with
 * what the tools themselves enforce.
 */
export function toolsLockedBy(scopes: string[]): Record<string, string[]> {
  const held = new Set(scopes);
  const locked: Record<string, string[]> = {};
  for (const [name, contract] of Object.entries(TOOL_CONTRACTS)) {
    if (contract.tenantClass !== "tenant") continue;
    const required = TOOL_SCOPES[name];
    if (satisfiesScope(scopes, required)) continue;
    // A tool needing two permissions is listed under each one it lacks.
    for (const scope of scopeUnion(required).filter((s) => !held.has(s))) {
      (locked[scope] ??= []).push(name);
    }
  }
  for (const names of Object.values(locked)) names.sort();
  return locked;
}

function summarize(data: PermissionsResponse): string {
  const can = (data.granted ?? []).map((item) => item.description);
  const cannot = (data.not_granted ?? []).map((item) => item.description);
  const lines = [`This connection can: ${can.length ? can.join("; ") : "nothing yet"}.`];
  if (cannot.length) lines.push(`It cannot: ${cannot.join("; ")}.`);
  if (data.manage_url) lines.push(`The user can change this at ${data.manage_url}.`);
  return lines.join(" ");
}

export function registerPermissionTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "get_permissions",
    {
      inputSchema: z.object({}),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async () => {
      const denial = scopeDenial("get_permissions", user);
      if (denial) return jsonText(denial);
      try {
        const data = await client.get<PermissionsResponse>("/api/mcp/permissions");
        const granted = (data.granted ?? []).map((item) => item.scope);
        return jsonText({
          ...data,
          summary: summarize(data),
          unavailable_tools: toolsLockedBy(granted),
          tell_the_user:
            "Show the user what this connection can and cannot do, in plain words, and where to " +
            "change it. Do this once, when you first connect.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "request_permission",
    {
      inputSchema: z.object({
        permissions: z
          .array(z.string().regex(/^[a-z_]+:[a-z_]+$/))
          .min(1)
          .max(8)
          .describe(
            "The permissions to ask for, as scope names from get_permissions' not_granted list " +
              "or a refused tool's `missing` field, e.g. [\"instances:write\"]",
          ),
        reason: z
          .string()
          .max(240)
          .optional()
          .describe("One sentence, shown to the user on the approval page, on why you need it"),
      }),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async ({ permissions, reason }) => {
      const denial = scopeDenial("request_permission", user);
      if (denial) return jsonText(denial);
      try {
        const data = await client.post<Record<string, unknown>>("/api/mcp/permission-requests", {
          scopes: permissions,
          reason: reason ?? "",
        });
        if (data.status === "already_granted") {
          return jsonText({ ...data, next_step: "This connection already holds these. Retry the tool." });
        }
        return jsonText({
          ...data,
          next_step:
            "Nothing is granted yet. Show the user approve_url exactly as given and say what it " +
            "would allow. Only they can approve, signed in to their dashboard: do not open the " +
            "link or approve for them. When they say they approved, retry the tool that was " +
            "refused; this same connection will then hold the permission. If they decline, stop " +
            "and do not ask again unless they bring it up.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}
