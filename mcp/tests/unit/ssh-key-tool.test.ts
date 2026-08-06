/**
 * The one tool whose wrong argument is a secret.
 *
 * `register_ssh_key` asks for a public key, and the file next to it on every
 * developer's disk is the private one. Two properties are worth a test:
 *
 *  1. A private key is *classified* as private — not merely rejected as
 *     malformed, which would tell the user to paste their key file again.
 *  2. A private key never reaches the network. The refusal is upstream of
 *     `client.post`, so the handler is exercised with a client that records
 *     every call and the assertion is that there were none. Asserting only on
 *     the returned message would keep passing if someone reordered the
 *     validation below the request.
 */

import { describe, expect, it } from "vitest";
import { z } from "zod";
import { inspectSshKeyInput } from "../../src/lib/ssh-key.js";
import { registerComputeTools } from "../../src/tools/compute.js";
import type { AuthUser } from "../../src/auth/bearer.js";

const PUBLIC_ED25519 =
  "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIB5wUu1kFVJ0lXn4pxJmkQ0z2h9BqQ2m1cVQZ1mQ0aBc laptop";

const PRIVATE_OPENSSH = [
  "-----BEGIN OPENSSH PRIVATE KEY-----",
  "b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAABAAAAMwAAAAtzc2gt",
  "-----END OPENSSH PRIVATE KEY-----",
].join("\n");

describe("inspectSshKeyInput", () => {
  it("accepts a pasted .pub file and returns just the key line", () => {
    const result = inspectSshKeyInput(`\n${PUBLIC_ED25519}\n\n`);
    expect(result.verdict).toBe("public");
    expect(result.key).toBe(PUBLIC_ED25519);
  });

  it.each([
    ["OpenSSH", PRIVATE_OPENSSH],
    ["PEM RSA", "-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END RSA PRIVATE KEY-----"],
    ["PKCS#8", "-----BEGIN PRIVATE KEY-----\nMIIEvQ==\n-----END PRIVATE KEY-----"],
    ["encrypted", "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFH==\n-----END ENCRYPTED PRIVATE KEY-----"],
    ["PuTTY", "PuTTY-User-Key-File-3: ssh-ed25519\nEncryption: none\nPrivate-Lines: 1\nAAAA"],
  ])("names a %s private key as private, not as malformed", (_label, material) => {
    const result = inspectSshKeyInput(material);
    expect(result.verdict).toBe("private");
    expect(result.key).toBe("");
    expect(result.message).toMatch(/PRIVATE key/);
    expect(result.message).toMatch(/not sent anywhere/i);
  });

  it.each([
    ["empty", ""],
    ["prose", "here is my key i think"],
    ["a fingerprint", "SHA256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU"],
    ["a bare base64 blob", "AAAAC3NzaC1lZDI1NTE5AAAAIB5wUu1kFVJ0lXn4pxJmkQ0z"],
  ])("rejects %s as unrecognized", (_label, material) => {
    expect(inspectSshKeyInput(material).verdict).toBe("unrecognized");
  });
});

/** Minimal stand-ins: enough surface for the handler, nothing more. */
type Handler = (args: Record<string, unknown>) => Promise<{ content: { text: string }[] }>;

function harness(scopes: string[]) {
  const posts: { path: string; body: unknown }[] = [];
  const gets: string[] = [];
  const handlers = new Map<string, Handler>();
  const schemas = new Map<string, z.ZodObject<z.ZodRawShape>>();

  const server = {
    registerTool(name: string, config: { inputSchema: z.ZodObject<z.ZodRawShape> }, handler: Handler) {
      handlers.set(name, handler);
      schemas.set(name, config.inputSchema);
    },
  };
  const client = {
    async post(path: string, body?: unknown) { posts.push({ path, body }); return { ok: true, id: "sshk-test" }; },
    async get(path: string) { gets.push(path); return {}; },
  };

  registerComputeTools(
    server as unknown as Parameters<typeof registerComputeTools>[0],
    client as unknown as Parameters<typeof registerComputeTools>[1],
    { scopes } as AuthUser,
  );

  const call = async (name: string, args: Record<string, unknown>) => {
    const handler = handlers.get(name);
    if (!handler) throw new Error(`${name} was never registered`);
    // Through the declared schema, so defaults apply exactly as at runtime.
    const parsed = schemas.get(name)!.parse(args);
    const result = await handler(parsed as Record<string, unknown>);
    return JSON.parse(result.content[0].text) as Record<string, unknown>;
  };

  return { call, posts, gets, handlers };
}

describe("register_ssh_key", () => {
  it("is registered at all", () => {
    // Prove the reach: every assertion below is vacuous if the tool is missing.
    expect(harness(["ssh:write"]).handlers.has("register_ssh_key")).toBe(true);
  });

  it("sends a public key to the account's key endpoint", async () => {
    const { call, posts } = harness(["ssh:write"]);
    const result = await call("register_ssh_key", { public_key: PUBLIC_ED25519, name: "laptop" });
    expect(posts).toEqual([
      { path: "/api/ssh/keys", body: { public_key: PUBLIC_ED25519, name: "laptop" } },
    ]);
    expect(result.ok).toBe(true);
  });

  it("does not send a private key anywhere", async () => {
    const { call, posts, gets } = harness(["ssh:write"]);
    const result = await call("register_ssh_key", { public_key: PRIVATE_OPENSSH });
    expect(posts, "a private key reached the API").toEqual([]);
    expect(gets, "a private key reached the API").toEqual([]);
    expect(result.error).toBe("private_key_supplied");
    expect(String(result.message)).toMatch(/compromised/i);
  });

  it("refuses without ssh:write, before looking at the argument", async () => {
    // The scope check must not be reachable-around by sending a valid key.
    const { call, posts } = harness(["instances:read", "instances:connect"]);
    const result = await call("register_ssh_key", { public_key: PUBLIC_ED25519 });
    expect(posts).toEqual([]);
    expect(JSON.stringify(result)).toMatch(/ssh:write/);
  });
});
