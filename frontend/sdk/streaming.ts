/**
 * Hand-written streaming companion for generated @xcelsior-gpu/sdk (C2.3).
 * Use for OpenAI-compatible chat completions SSE and serverless job streams.
 */

export type SseHandler = (event: { data: string; event?: string }) => void;

export interface StreamRequestOptions {
  baseUrl: string;
  path: string;
  token: string;
  body?: unknown;
  onEvent: SseHandler;
  signal?: AbortSignal;
}

function parseSseChunk(buffer: string): { events: Array<{ event?: string; data: string }>; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  const events: Array<{ event?: string; data: string }> = [];
  for (const block of parts) {
    let ev: string | undefined;
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (!line || line.startsWith(":")) continue;
      const i = line.indexOf(":");
      const field = i === -1 ? line : line.slice(0, i);
      const value = i === -1 ? "" : line.slice(i + 1).replace(/^ /, "");
      if (field === "event") ev = value;
      if (field === "data") data.push(value);
    }
    if (data.length) events.push({ event: ev, data: data.join("\n") });
  }
  return { events, rest };
}

export async function streamSseRequest(opts: StreamRequestOptions): Promise<void> {
  const res = await fetch(`${opts.baseUrl.replace(/\/$/, "")}${opts.path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${opts.token}`,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: opts.body ? JSON.stringify(opts.body) : undefined,
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: HTTP ${res.status}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parsed = parseSseChunk(buffer);
    buffer = parsed.rest;
    for (const ev of parsed.events) opts.onEvent(ev);
  }
  if (buffer.trim()) {
    const parsed = parseSseChunk(buffer + "\n\n");
    for (const ev of parsed.events) opts.onEvent(ev);
  }
}

export function streamChatCompletions(
  baseUrl: string,
  token: string,
  body: Record<string, unknown>,
  onEvent: SseHandler,
  signal?: AbortSignal,
): Promise<void> {
  return streamSseRequest({
    baseUrl,
    path: "/v1/chat/completions",
    token,
    body: { ...body, stream: true },
    onEvent,
    signal,
  });
}

export function streamServerlessJob(
  baseUrl: string,
  token: string,
  jobId: string,
  onEvent: SseHandler,
  signal?: AbortSignal,
): Promise<void> {
  return streamSseRequest({
    baseUrl,
    path: `/api/v2/serverless/jobs/${encodeURIComponent(jobId)}/stream`,
    token,
    onEvent,
    signal,
  });
}