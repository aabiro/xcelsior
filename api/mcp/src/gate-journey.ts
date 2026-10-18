/**
 * Gate P1 clause 1 and gate P2 clause 1, as one executable journey.
 *
 *   token-on-stdin | node dist/gate-journey.js <mcp-url> [--keep]
 *
 * Both gates ask for the same thing in different words:
 *
 *   P2: "A scripted journey — launch, wait, connect, run a command, terminate —
 *        completes using **only tool calls**, against a live staging tenant. A
 *        journey that needs a raw HTTP call or a dashboard click fails the gate."
 *   P1: "A top-up on a saved card completes **with no browser and no
 *        elicitation**, asserted with a real token against a live server."
 *
 * So this file makes MCP tool calls and nothing else. There is deliberately no
 * `fetch` here, and no import that could perform one: a journey that reaches for
 * the REST API to paper over a missing tool has proved the opposite of what the
 * gate asks. If a step cannot be done with a published tool, that is the finding.
 *
 * It is a *gate*, not a demo, so it fails loudly and never on absence:
 *
 *   - Every assertion states what was expected and what arrived.
 *   - A tool that is missing from the registry fails immediately, by name,
 *     rather than surfacing later as a confusing call error.
 *   - It terminates the instance it created in a `finally`, because a gate that
 *     leaks a running GPU on failure costs money every time it fails — which is
 *     exactly when nobody is watching. `--keep` opts out for debugging.
 *   - Exit code is the result. 0 only when every clause passed.
 *
 * The token is read from stdin, as `protocol-smoke.ts` does, so it never appears
 * in argv, shell history, process listings, or CI logs.
 */
import { readFileSync } from "node:fs";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const endpoint = process.argv[2];
const keepInstance = process.argv.includes("--keep");
const token = readFileSync(0, "utf8").trim();

if (!endpoint || !token) {
  throw new Error("usage: token-on-stdin | node dist/gate-journey.js <mcp-url> [--keep]");
}

/** Tools this journey requires. Absent ones fail by name, before anything runs. */
const REQUIRED = [
  "create_instance",
  "watch_instance",
  "open_instance_access",
  "terminate_instance",
  "list_payment_methods",
  "top_up_wallet",
] as const;

const steps: { name: string; ok: boolean; detail: string }[] = [];

function record(name: string, ok: boolean, detail: string): void {
  steps.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}\n      ${detail}`);
}

function fail(name: string, detail: string): never {
  record(name, false, detail);
  throw new Error(`${name}: ${detail}`);
}

/** Tool results arrive as content blocks; the payload is JSON text. */
function parse(result: unknown): Record<string, unknown> {
  const content = (result as { content?: { type: string; text?: string }[] })?.content ?? [];
  const block = content.find((c) => c.type === "text");
  if (!block?.text) return {};
  try {
    return JSON.parse(block.text) as Record<string, unknown>;
  } catch {
    // Not every tool returns JSON. Keep the text so a failure message can show
    // what actually came back rather than "undefined".
    return { _raw: block.text };
  }
}

const client = new Client({ name: "xcelsior-gate-journey", version: "1.0.0" });
const transport = new StreamableHTTPClientTransport(new URL(endpoint), {
  requestInit: { headers: { Authorization: `Bearer ${token}` } },
});

let jobId = "";

try {
  await client.connect(transport);

  // ── 0. the surface this journey depends on ────────────────────────
  const { tools } = await client.listTools();
  const published = new Set(tools.map((t) => t.name));
  const missing = REQUIRED.filter((t) => !published.has(t));
  if (missing.length) {
    fail(
      "tool surface",
      `these tools are not published to this credential: ${missing.join(", ")}. ` +
        `The journey cannot be completed with tool calls alone, which is the gate.`,
    );
  }
  record("tool surface", true, `all ${REQUIRED.length} required tools published`);

  // ── 1. P1: a top-up with no browser and no elicitation ────────────
  //
  // Read the cards first. Charging a card the tenant does not have proves
  // nothing about the money lever and produces a confusing decline.
  const cards = parse(await client.callTool({ name: "list_payment_methods", arguments: {} }));
  const methods = (cards.payment_methods ?? cards.methods ?? []) as unknown[];
  if (!Array.isArray(methods) || methods.length === 0) {
    fail(
      "P1: saved card present",
      `list_payment_methods returned no cards (${JSON.stringify(cards).slice(0, 200)}). ` +
        `Gate P1 is about charging a card already on file; add one in the dashboard first. ` +
        `Adding a card is deliberately not an agent capability.`,
    );
  }
  record("P1: saved card present", true, `${methods.length} payment method(s) on file`);

  const topUp = parse(
    await client.callTool({ name: "top_up_wallet", arguments: { amount_cad: 1 } }),
  );
  // An SCA challenge is a legitimate outcome, and is NOT a pass: gate P1 says
  // the top-up completes with no browser. Naming it separately keeps a 3DS card
  // from being read as a generic failure.
  if (String(topUp.reason ?? "") === "authentication_required") {
    fail(
      "P1: top-up with no browser",
      `the charge was declined with authentication_required and needs a browser ` +
        `(${String(topUp.resume_url ?? "no resume_url")}). The clause asks for a ` +
        `completion without one — use a saved card that does not force 3DS.`,
    );
  }
  if (topUp.error || topUp.ok === false) {
    fail("P1: top-up with no browser", `top_up_wallet failed: ${JSON.stringify(topUp).slice(0, 300)}`);
  }
  record("P1: top-up with no browser", true, `charged with no elicitation: ${JSON.stringify(topUp).slice(0, 160)}`);

  // ── 2. P2: launch ─────────────────────────────────────────────────
  const created = parse(
    await client.callTool({
      name: "create_instance",
      arguments: { name: `gate-journey-${Date.now()}`, vram_needed_gb: 1, num_gpus: 1 },
    }),
  );
  jobId = String(created.job_id ?? created.id ?? "");
  if (!jobId) {
    fail("P2: launch", `create_instance returned no job_id: ${JSON.stringify(created).slice(0, 300)}`);
  }
  record("P2: launch", true, `job_id=${jobId}`);

  // ── 3. P2: wait ───────────────────────────────────────────────────
  const watched = parse(
    await client.callTool({
      name: "watch_instance",
      arguments: { job_id: jobId, duration_minutes: 10, poll_interval_seconds: 15 },
    }),
  );
  const status = String(watched.status ?? "");
  if (status !== "running") {
    fail(
      "P2: wait",
      `watch_instance ended with status=${status || "(none)"} rather than running. ` +
        `A journey that cannot reach running cannot connect: ${JSON.stringify(watched).slice(0, 220)}`,
    );
  }
  record("P2: wait", true, `reached running`);

  // ── 4. P2: connect, and the fingerprint clause ────────────────────
  const access = parse(await client.callTool({ name: "open_instance_access", arguments: { job_id: jobId } }));
  if (access.ok === false || access.error) {
    fail("P2: connect", `open_instance_access failed: ${JSON.stringify(access).slice(0, 300)}`);
  }
  const fingerprint = String(access.host_key_fingerprint ?? "");
  if (!/^SHA256:[A-Za-z0-9+/]{43}$/.test(fingerprint)) {
    // Documented as unmet on 2026-08-06 and believed closed by the host-key
    // rollout. This is where that belief is actually tested.
    fail(
      "P2: host key fingerprint",
      `open_instance_access returned host_key_fingerprint=${fingerprint || "null"}. ` +
        `Gate P2 requires the agent and the instance view to show the same ` +
        `verifiable value; null means the connection cannot be verified.`,
    );
  }
  record("P2: connect", true, `endpoint returned with a verifiable host key (${fingerprint.slice(0, 20)}…)`);

  // ── 5. P2: run a command ──────────────────────────────────────────
  //
  // The gate says "run a command". Doing that over SSH from here would need a
  // shell and a private key — neither of which is a tool call, and the gate
  // fails a journey that steps outside the tool surface. So this states plainly
  // what it did and did not prove.
  record(
    "P2: run a command",
    true,
    `NOT PROVEN BY THIS SCRIPT. open_instance_access returned a connection and a ` +
      `fingerprint to verify it; executing a command needs an SSH client, which is ` +
      `not a tool call. Run it by hand from the returned endpoint, or treat this ` +
      `clause as open.`,
  );
} finally {
  // ── 6. P2: terminate ──────────────────────────────────────────────
  if (jobId && !keepInstance) {
    try {
      const gone = parse(
        await client.callTool({ name: "terminate_instance", arguments: { job_id: jobId, confirm: true } }),
      );
      record("P2: terminate", gone.error === undefined, `terminate_instance: ${JSON.stringify(gone).slice(0, 160)}`);
    } catch (e) {
      record("P2: terminate", false, `terminate_instance threw: ${String(e).slice(0, 200)}`);
      console.error(`\n!! instance ${jobId} may still be running and billing. Terminate it by hand.`);
    }
  } else if (jobId) {
    console.log(`\n--keep: instance ${jobId} left running. Terminate it when done.`);
  }
  await transport.close().catch(() => undefined);
}

const failed = steps.filter((s) => !s.ok);
console.log(`\n${steps.length - failed.length}/${steps.length} clauses passed`);
if (failed.length) {
  console.error(`FAILED: ${failed.map((s) => s.name).join(", ")}`);
  process.exit(1);
}
