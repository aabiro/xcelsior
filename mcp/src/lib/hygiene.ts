/**
 * Response hygiene: what a tool is allowed to hand back to a model.
 *
 * Anything a tool returns goes into a third party's context window, is usually
 * logged by that provider, and is frequently shown to the user verbatim. Our
 * audit layer already redacts what goes *in* (arguments); this is the other
 * direction, and it is the one a directory reviewer checks against the privacy
 * policy line by line.
 *
 * Three classes are removed, chosen to be high-confidence rather than broad —
 * a filter that strips legitimate fields would corrupt tool output, which is a
 * worse failure than the one it prevents:
 *
 *  1. **Auth material.** Tokens, secrets, passwords, cookies. Matched by key
 *     name *and* by value shape, because an upstream field named `value` can
 *     still hold an `xoa_` token.
 *  2. **Debug payloads.** Tracebacks, SQL, internal error dumps. These leak
 *     implementation detail and occasionally credentials inside a connection
 *     string.
 *  3. **Undisclosed user fields.** Password hashes, MFA secrets, verification
 *     tokens — fields the privacy policy never says we share.
 *
 * Removal is recorded, not silent: a tool that starts leaking should show up
 * as a metric and a log line, because the filter is the safety net and the
 * tool is the bug.
 */

/** Key names that must never appear in tool output, at any depth. */
const FORBIDDEN_KEY =
  /(^|_)(token|secret|password|passphrase|credential|credentials|apikey|private_key|authorization|cookie|salt)($|_)|^(api_key|client_secret|session_token|access_token|refresh_token|id_token|set-cookie|password_hash|password_salt|mfa_secret|totp_secret|email_verification_token|ssh_private_key|traceback|stacktrace|stack_trace|sql|debug|__debug__|internal_id|internal_error|row_id|db_id|oid)$/i;

/**
 * Value shapes that are credentials regardless of the key they arrive under.
 *
 * Kept narrow and anchored: Xcelsior's own prefixed credentials, Stripe keys,
 * and a three-segment JWT. A loose "long base64-ish string" rule would eat
 * legitimate ids and log lines.
 */
const CREDENTIAL_VALUE =
  /\b(xoa_[A-Za-z0-9_-]{16,}|xcel_ai_[A-Za-z0-9_-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{16,}|rk_(?:live|test)_[A-Za-z0-9]{16,}|whsec_[A-Za-z0-9]{16,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b/g;

export const REDACTED = "[REDACTED]";

export interface HygieneReport {
  value: unknown;
  /** Dotted paths that were removed or masked, for the metric and the log. */
  removed: string[];
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Mask credential-shaped substrings inside a free-text field. */
export function scrubText(text: string): { text: string; masked: boolean } {
  CREDENTIAL_VALUE.lastIndex = 0;
  if (!CREDENTIAL_VALUE.test(text)) return { text, masked: false };
  CREDENTIAL_VALUE.lastIndex = 0;
  return { text: text.replace(CREDENTIAL_VALUE, REDACTED), masked: true };
}

/**
 * Recursively remove forbidden keys and mask credential-shaped values.
 *
 * `_meta` is preserved: it is the MCP protocol's own namespace and carries our
 * tool version and contract metadata, which is published on purpose.
 */
export function scrubResponse(value: unknown, path = ""): HygieneReport {
  const removed: string[] = [];

  const walk = (node: unknown, at: string): unknown => {
    if (typeof node === "string") {
      const { text, masked } = scrubText(node);
      if (masked) removed.push(at || "(root)");
      return text;
    }
    if (Array.isArray(node)) return node.map((item, index) => walk(item, `${at}[${index}]`));
    if (!isPlainObject(node)) return node;
    const result: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(node)) {
      const here = at ? `${at}.${key}` : key;
      if (key === "_meta") {
        result[key] = item;
        continue;
      }
      if (FORBIDDEN_KEY.test(key) || (key.startsWith("_") && key !== "_meta")) {
        removed.push(here);
        continue;
      }
      result[key] = walk(item, here);
    }
    return result;
  };

  return { value: walk(value, path), removed };
}
