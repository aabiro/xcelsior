import { ApiError, formatApiError } from "../client/errors.js";
import { jsonText } from "./format.js";

type ToolResult = ReturnType<typeof jsonText>;

/**
 * Read what a destructive preview would act on, before describing the act.
 *
 * A preview that answers "set confirm:true to delete this" about something
 * that is not there, or is another account's, tells the agent the next call
 * will do something when it will not, and invites a confirm that can only
 * fail. Reading the target first makes the preview name the real thing, and
 * makes a wrong id a refusal at the preview, where the agent can still fix it.
 *
 * 403 and 404 both read as "not on this account": a preview must not say
 * whether someone else's resource exists.
 */
export async function readPreviewTarget(
  read: () => Promise<unknown>,
  what: string,
  listTool: string,
): Promise<{ found: true; value: Record<string, unknown> } | { found: false; result: ToolResult }> {
  try {
    const value = (await read()) as Record<string, unknown> | null;
    return { found: true, value: value ?? {} };
  } catch (error) {
    if (error instanceof ApiError && (error.status === 404 || error.status === 403)) {
      return { found: false, result: notOnThisAccount(what, listTool) };
    }
    return { found: false, result: jsonText({ ok: false, error: formatApiError(error) }) };
  }
}

export function notOnThisAccount(what: string, listTool: string): ToolResult {
  return jsonText({
    ok: false,
    error: "not_found",
    message: `No ${what} is on this account — ${listTool} shows what is. Nothing has changed.`,
  });
}
