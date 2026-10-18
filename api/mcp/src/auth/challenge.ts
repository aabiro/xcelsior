/**
 * RFC 6750 / RFC 9728 `WWW-Authenticate` challenge for the MCP edge.
 *
 * Every unauthenticated request — including the very first `initialize` — has to
 * come back as a 401 carrying this header. It is the only breadcrumb an MCP
 * client has: the spec tells clients to read `resource_metadata` off the
 * challenge, fetch the protected-resource document, and start OAuth from the
 * authorization server it names. Without the header a connector reports
 * "couldn't connect" while our logs show a perfectly ordinary 401, which is the
 * worst possible failure shape — nothing looks broken on either side.
 */

/** Escape a value for an RFC 7235 quoted-string. */
function quoted(value: string): string {
  return `"${value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
}

export interface ChallengeOptions {
  realm: string;
  resourceMetadataUrl: string;
  /** Omitted on the no-credentials branch: RFC 6750 §3 sends bare params there. */
  error?: "invalid_token" | "invalid_request" | "insufficient_scope";
  errorDescription?: string;
  /** Space-delimited, only meaningful alongside `insufficient_scope`. */
  scope?: string;
}

export function buildWwwAuthenticate(options: ChallengeOptions): string {
  const params: string[] = [
    `realm=${quoted(options.realm)}`,
    `resource_metadata=${quoted(options.resourceMetadataUrl)}`,
  ];
  if (options.error) {
    params.push(`error=${quoted(options.error)}`);
    if (options.errorDescription) {
      // CR/LF would let a description split the header; strip rather than trust.
      params.push(`error_description=${quoted(options.errorDescription.replace(/[\r\n]+/g, " "))}`);
    }
  }
  if (options.scope) params.push(`scope=${quoted(options.scope)}`);
  return `Bearer ${params.join(", ")}`;
}
