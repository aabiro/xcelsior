import type { XcelsiorApiClient } from "../client/api.js";

const TERMINAL = new Set(["completed", "failed", "cancelled", "terminated", "preempted"]);

export interface WaitForInstanceOptions {
  timeoutMs?: number;
  pollIntervalMs?: number;
  targetStatus?: string;
}

export async function waitForInstance(
  client: XcelsiorApiClient,
  jobId: string,
  opts: WaitForInstanceOptions = {},
): Promise<{ ok: boolean; instance: Record<string, unknown>; timedOut: boolean }> {
  const timeoutMs = opts.timeoutMs ?? 300_000;
  const pollIntervalMs = opts.pollIntervalMs ?? 5_000;
  const target = opts.targetStatus ?? "running";
  const deadline = Date.now() + timeoutMs;

  let last: Record<string, unknown> = {};
  while (Date.now() < deadline) {
    const data = (await client.get(`/instance/${encodeURIComponent(jobId)}`)) as Record<
      string,
      unknown
    >;
    const instance = (data.instance as Record<string, unknown>) || data;
    last = instance;
    const status = String(instance.status || "");
    if (status === target) {
      return { ok: true, instance: last, timedOut: false };
    }
    if (TERMINAL.has(status)) {
      return { ok: false, instance: last, timedOut: false };
    }
    await new Promise((r) => setTimeout(r, pollIntervalMs));
  }

  return { ok: false, instance: last, timedOut: true };
}