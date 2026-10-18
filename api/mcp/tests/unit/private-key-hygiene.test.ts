/**
 * Gate P2: "No private key material appears in any tool result. Asserted, not
 * assumed."
 *
 * The assumption that needed testing was that `FORBIDDEN_KEY` covered this. It
 * drops a field *named* `private_key`, which handles a well-behaved upstream
 * and nothing else. The realistic leak has an innocent field name:
 * `get_instance_logs` returns whatever the instance printed, so a bootstrap
 * script echoing a key, a `cat` of the wrong file, or a config dump all arrive
 * as ordinary text. Probed before the fix — a PEM block under
 * `bootstrap_output` came back `removed: []`, intact.
 *
 * These assert at `scrubText`/`scrubResponse` because that is genuinely the
 * chokepoint: `applyResponseHygiene` in `src/audit/context.ts` runs on every
 * tool result, scrubbing `structuredContent` through `scrubResponse` and each
 * `content.text` through `scrubText`. Asserting here is asserting for all of
 * them, which is what "any tool result" requires.
 */

import { describe, expect, it } from "vitest";
import { REDACTED, scrubResponse, scrubText } from "../../src/lib/hygiene.js";

const BODY = "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt";

const BLOCK = (label: string) =>
  `-----BEGIN ${label}-----\n${BODY}\n${BODY}\n-----END ${label}-----`;

describe("private key material never reaches the model", () => {
  it.each([
    "OPENSSH PRIVATE KEY",
    "RSA PRIVATE KEY",
    "PRIVATE KEY",
    "EC PRIVATE KEY",
    "DSA PRIVATE KEY",
    "ENCRYPTED PRIVATE KEY",
    "PGP PRIVATE KEY BLOCK",
  ])("redacts a %s block", (label) => {
    const { text, masked } = scrubText(`before\n${BLOCK(label)}\nafter`);
    expect(masked).toBe(true);
    expect(text).not.toContain(BODY);
    expect(text).not.toContain("BEGIN");
    // The surrounding log is preserved: this masks a secret, it does not
    // discard the output the user asked for.
    expect(text).toContain("before");
    expect(text).toContain("after");
    expect(text).toContain(REDACTED);
  });

  it("redacts a truncated block, because a log tail cuts mid-key", () => {
    // Half a private key is still key material, and `logs` is a tail by nature.
    const truncated = `starting up\n-----BEGIN OPENSSH PRIVATE KEY-----\n${BODY}`;
    const { text, masked } = scrubText(truncated);
    expect(masked).toBe(true);
    expect(text).not.toContain(BODY);
    expect(text).toContain("starting up");
  });

  it("leaves public keys and certificates alone", () => {
    // register_ssh_key's whole output is a public key. Redacting it would break
    // the tool a user needs in order to verify what was registered.
    const pub = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB5wUu1kFVJ0 laptop";
    expect(scrubText(pub).masked).toBe(false);
    const cert = BLOCK("CERTIFICATE");
    expect(scrubText(cert).masked).toBe(false);
    const pubBlock = BLOCK("PUBLIC KEY");
    expect(scrubText(pubBlock).masked).toBe(false);
  });

  it("catches it under an innocuous field name, at any depth", () => {
    // The exact shape that got through before: not named like a secret.
    const report = scrubResponse({
      instance: {
        id: "i-1",
        logs: ["boot ok", `writing deploy key\n${BLOCK("OPENSSH PRIVATE KEY")}`],
        bootstrap_output: BLOCK("RSA PRIVATE KEY"),
      },
    });
    const wire = JSON.stringify(report.value);
    expect(wire).not.toContain(BODY);
    expect(wire).not.toContain("BEGIN");
    expect(report.removed).toContain("instance.bootstrap_output");
    expect(report.removed).toContain("instance.logs[1]");
    // The legitimate payload survives.
    expect(wire).toContain("boot ok");
    expect(wire).toContain("i-1");
  });

  it("still masks the credential shapes it already caught", () => {
    // Regression guard on the rewrite of scrubText, not on the new pattern.
    const { text, masked } = scrubText("connecting with xoa_LeakedTokenValue0123456789");
    expect(masked).toBe(true);
    expect(text).toContain(REDACTED);
    expect(text).not.toContain("xoa_LeakedTokenValue");
  });

  it("does not depend on a regex reset between calls", () => {
    // Both patterns carry `g`. The previous implementation called `.test()` on
    // a `g` regex, which advances lastIndex, and worked only because two manual
    // resets bracketed it. Calling twice with the same input is the cheapest
    // way to catch that class of bug returning.
    const input = `log\n${BLOCK("OPENSSH PRIVATE KEY")}`;
    const first = scrubText(input);
    const second = scrubText(input);
    expect(second).toEqual(first);
    expect(second.masked).toBe(true);
  });

  it("stays linear on a large log", () => {
    // A scrubber is on the path of every tool result, and a lazy `[\s\S]*?`
    // next to an end-of-input alternation is exactly where quadratic blowup
    // hides. 40 KB with an unterminated block is the adversarial case: nothing
    // to anchor the lazy match against.
    const haystack = "x".repeat(40_000);
    const started = performance.now();
    scrubText(`-----BEGIN OPENSSH PRIVATE KEY-----\n${haystack}`);
    scrubText(haystack);
    const elapsed = performance.now() - started;
    expect(elapsed).toBeLessThan(250);
  });
});
