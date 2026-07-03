import { XcelsiorApiClient, XcelsiorApiEnvironment } from "@xcelsior-gpu/sdk";

/**
 * Pre-configured Xcelsior SDK client for server-side / scripting use.
 *
 * Auth: the generated Fern SDK takes credentials as request headers rather
 * than a first-class `apiKey` field, so we attach `Authorization: Bearer …`
 * from the environment. Set `XCELSIOR_API_TOKEN` (the value also lives in the
 * repo `.env`), never hard-code a key in source.
 *
 * Usage (Node / route handler / script, NOT the browser, since it carries a
 * secret token):
 *
 *   import { xcelsior } from "@/lib/xcelsior-sdk";
 *   const { hosts } = await xcelsior.hosts.list();
 *
 * For end-user installs prefer the CLI wizard instead of a global npm install:
 *   npx @xcelsior-gpu/wizard@latest
 */
export function createXcelsiorClient(token = process.env.XCELSIOR_API_TOKEN): XcelsiorApiClient {
  if (!token) {
    throw new Error(
      "XCELSIOR_API_TOKEN is not set, add it to your environment (.env) before using the SDK client.",
    );
  }
  return new XcelsiorApiClient({
    environment: process.env.XCELSIOR_API_BASE_URL || XcelsiorApiEnvironment.Production,
    headers: { Authorization: `Bearer ${token}` },
  });
}

/** Lazily-instantiated singleton, throws on first use if no token is configured. */
let _client: XcelsiorApiClient | null = null;
export function xcelsiorClient(): XcelsiorApiClient {
  return (_client ??= createXcelsiorClient());
}
