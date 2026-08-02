import { loadRateLimitConfig, type RateLimitConfig } from "./rate-limit.js";

export interface AppConfig {
  apiUrl: string;
  host: string;
  port: number;
  mcpPath: string;
  rateLimitPerMinute: number;
  rateLimit: RateLimitConfig;
  publicUrl: string;
  /** Canonical RFC 8707 resource identifier — the exact URL a user pastes. */
  resourceAudience: string;
  /** Pre-migration origin audience, honoured until `legacyAudienceSunset`. */
  legacyResourceAudience: string;
  legacyAudienceSunset: number | null;
  /** Where the 401 challenge points clients to discover how to authenticate. */
  resourceMetadataUrl: string;
  authRealm: string;
  oauthIssuer: string;
  oauthJwksUrl: string;
  /** Which tool profile this deployment exposes (adoption plan §4a). */
  toolProfile: "customer" | "operator";
  /**
   * ChatGPT company-knowledge `search`/`fetch` tools (adoption plan X1.11).
   *
   * Off by default. The track is explicitly optional and must not delay the
   * base plugin submission, so the reviewed surface stays exactly what it was
   * until someone decides to pursue company knowledge and turns it on.
   */
  companyKnowledge: boolean;
  /** Public marketing site — the human-openable half of a citation. */
  siteUrl: string;
  /** Published documentation site. */
  docsUrl: string;
  /**
   * Token the OpenAI plugin submission portal issues for domain verification.
   * Empty until the portal issues one; the route 404s rather than inventing a
   * value (adoption plan BLOCKER 4).
   */
  openaiAppsChallenge: string;
}

/** Origin of an absolute URL, or "" when it cannot be parsed. */
function originOf(url: string): string {
  try {
    return new URL(url).origin;
  } catch {
    return "";
  }
}

function parseSunset(raw: string | undefined): number | null {
  const value = (raw ?? "").trim();
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? null : parsed;
}

export function loadConfig(): AppConfig {
  const apiUrl = (process.env.XCELSIOR_MCP_API_URL || process.env.XCELSIOR_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const rateLimit = loadRateLimitConfig(process.env);
  const resourceAudience = (
    process.env.XCELSIOR_MCP_RESOURCE_AUDIENCE || "https://mcp.xcelsior.ca/mcp"
  ).replace(/\/$/, "");
  const resourceOrigin = originOf(resourceAudience);
  // The old identifier was the bare origin. Derive it rather than requiring
  // every deployment to restate it, but only when the canonical value actually
  // has a path — otherwise "legacy" and "canonical" would be the same string
  // and the sunset would be meaningless.
  const derivedLegacy = resourceOrigin && resourceOrigin !== resourceAudience ? resourceOrigin : "";
  const profile = (process.env.XCELSIOR_MCP_TOOL_PROFILE || "customer").trim().toLowerCase();
  return {
    apiUrl,
    host: process.env.MCP_HOST || "0.0.0.0",
    port: Number(process.env.MCP_PORT || "8770"),
    mcpPath: process.env.MCP_PATH || "/mcp",
    rateLimitPerMinute: rateLimit.perMinute,
    rateLimit,
    publicUrl: (process.env.XCELSIOR_MCP_PUBLIC_URL || "http://127.0.0.1:8770/mcp").replace(/\/$/, ""),
    resourceAudience,
    legacyResourceAudience: (
      process.env.XCELSIOR_MCP_LEGACY_RESOURCE_AUDIENCE ?? derivedLegacy
    ).replace(/\/$/, ""),
    legacyAudienceSunset: parseSunset(
      process.env.XCELSIOR_MCP_LEGACY_AUDIENCE_SUNSET ?? "2026-11-30T00:00:00Z",
    ),
    resourceMetadataUrl:
      process.env.XCELSIOR_MCP_RESOURCE_METADATA_URL ||
      `${resourceOrigin || resourceAudience}/.well-known/oauth-protected-resource`,
    authRealm: process.env.XCELSIOR_MCP_AUTH_REALM || "xcelsior",
    oauthIssuer: (process.env.XCELSIOR_OAUTH_ISSUER || `${apiUrl}/`).replace(/\/$/, ""),
    oauthJwksUrl: process.env.XCELSIOR_OAUTH_JWKS_URL || `${apiUrl}/.well-known/jwks.json`,
    toolProfile: profile === "operator" ? "operator" : "customer",
    companyKnowledge: ["1", "true", "yes"].includes(
      (process.env.XCELSIOR_MCP_COMPANY_KNOWLEDGE || "").trim().toLowerCase(),
    ),
    siteUrl: (process.env.XCELSIOR_PUBLIC_URL || "https://xcelsior.ca").replace(/\/$/, ""),
    docsUrl: (process.env.XCELSIOR_DOCS_URL || "https://docs.xcelsior.ca").replace(/\/$/, ""),
    // Trimmed: a trailing newline from a shell heredoc or a secrets manager
    // would make the response body differ from the token by one byte, which is
    // exactly the kind of failure that reads as "verification just doesn't work".
    openaiAppsChallenge: (process.env.XCELSIOR_MCP_OPENAI_APPS_CHALLENGE || "").trim(),
  };
}

/**
 * Audience values a presented token may carry. The legacy origin drops out of
 * the set on its own at the sunset instant, so the compatibility window closes
 * without anyone remembering to close it.
 */
export function acceptedAudiences(config: AppConfig, now: number = Date.now()): string[] {
  const accepted = [config.resourceAudience];
  if (
    config.legacyResourceAudience &&
    config.legacyResourceAudience !== config.resourceAudience &&
    (config.legacyAudienceSunset === null || now < config.legacyAudienceSunset)
  ) {
    accepted.push(config.legacyResourceAudience);
  }
  return accepted;
}
