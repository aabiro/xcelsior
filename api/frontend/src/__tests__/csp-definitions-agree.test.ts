import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * The Content-Security-Policy is written out twice: once in `next.config.ts`
 * as a static header, and once in `src/proxy.ts`, which sets it at runtime and
 * therefore wins. Two copies of a long allowlist drift, and the drift is
 * invisible — the browser simply refuses a request and the feature looks broken
 * for some other reason.
 *
 * It had already happened twice. `proxy.ts` was missing
 * `https://us.i.posthog.com`, the host `NEXT_PUBLIC_POSTHOG_HOST` points at, so
 * every analytics and error-tracking call was blocked in production while
 * `next.config.ts` said they were allowed. And neither copy carried
 * `https://connect-js.stripe.com`, which the embedded Stripe Connect payouts
 * panel on /dashboard/earnings loads.
 *
 * This compares the two source-of-truth strings directive by directive. It does
 * not check that any particular host is present — that is what the
 * feature-specific CSP tests do — only that the two copies say the same thing,
 * so a host added to one cannot silently miss the other.
 */

const ROOT = join(__dirname, "..", "..");

function directivesOf(source: string): Map<string, Set<string>> {
  const out = new Map<string, Set<string>>();
  const pattern = /"([a-z-]+) ([^"]*?)(?:; )?"/g;
  for (const match of source.matchAll(pattern)) {
    const [, name, value] = match;
    if (!name.endsWith("-src") && name !== "frame-ancestors") continue;
    const sources = value
      .replace(/; $/, "")
      .split(/\s+/)
      .filter((token) => /^(https?:|wss:|')/.test(token))
      .map((token) => token.replace(/;$/, ""));
    const bucket = out.get(name) ?? new Set<string>();
    for (const source of sources) bucket.add(source);
    out.set(name, bucket);
  }
  return out;
}

describe("the two CSP definitions agree", () => {
  it("lists the same sources in next.config.ts and src/proxy.ts", () => {
    const config = directivesOf(readFileSync(join(ROOT, "next.config.ts"), "utf8"));
    const proxy = directivesOf(readFileSync(join(ROOT, "src", "proxy.ts"), "utf8"));

    expect(config.size).toBeGreaterThan(0);
    expect(proxy.size).toBeGreaterThan(0);

    for (const name of new Set([...config.keys(), ...proxy.keys()])) {
      const inConfig = [...(config.get(name) ?? [])].sort();
      const inProxy = [...(proxy.get(name) ?? [])].sort();
      expect(
        inProxy,
        `${name} differs between next.config.ts and src/proxy.ts — proxy.ts is ` +
          `the header actually served, so anything missing there is blocked in ` +
          `the browser regardless of what next.config.ts allows`,
      ).toEqual(inConfig);
    }
  });
});
