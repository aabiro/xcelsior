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

/**
 * A private key, wherever it turns up and whatever the field is called.
 *
 * `FORBIDDEN_KEY` already drops a field *named* `private_key` or
 * `ssh_private_key`. That is not enough, and Gate P2 asks for the difference to
 * be asserted rather than assumed: the realistic leak is a key body inside a
 * field with an innocent name. `get_instance_logs` returns whatever the
 * instance printed, and a bootstrap script that echoes a key, a `cat` of the
 * wrong file, or a config dump all arrive as ordinary text under `logs`,
 * `output`, or `detail`. Probed before this existed: a PEM block under
 * `bootstrap_output` was reported as `removed: []` and reached the model intact.
 *
 * Matched from BEGIN to END, or to the end of the string when the block is
 * truncated — a log tail is routinely cut mid-key, and half a private key is
 * still key material. The header alternation covers OpenSSH, PKCS#1 (`RSA`),
 * PKCS#8 (bare), `EC`, `DSA`, `ENCRYPTED`, and PGP's `PRIVATE KEY BLOCK`.
 *
 * Deliberately *not* matched: `PUBLIC KEY` and `CERTIFICATE` blocks. Publishing
 * a public key is the point of `register_ssh_key`, and redacting it would break
 * the tool whose output a user needs to verify.
 */
const PRIVATE_KEY_BLOCK =
  /-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?(?:-----END (?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----|$)/g;

export const REDACTED = "[REDACTED]";

export interface HygieneReport {
  value: unknown;
  /** Dotted paths that were removed or masked, for the metric and the log. */
  removed: string[];
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Mask credential-shaped substrings inside a free-text field.
 *
 * Whole key blocks first, then credential-shaped values in what remains: a PEM
 * body is base64 and can contain something that looks like another credential,
 * and redacting the block as one unit says what actually happened.
 *
 * Written as replace-and-compare rather than `.test()` then `.replace()`.
 * Both patterns carry the `g` flag, and a `g` regex's `.test()` advances
 * `lastIndex` — so the old form depended on two manual resets bracketing every
 * call, and a third pattern added without them would silently start skipping
 * the first match of every other string. There is no state to reset here.
 */
export function scrubText(text: string): { text: string; masked: boolean } {
  const withoutKeys = text.replace(PRIVATE_KEY_BLOCK, REDACTED);
  const scrubbed = withoutKeys.replace(CREDENTIAL_VALUE, REDACTED);
  return { text: scrubbed, masked: scrubbed !== text };
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
