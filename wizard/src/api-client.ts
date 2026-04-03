// API client for the Xcelsior AI assistant.
// Connects to the existing /api/ai/chat SSE endpoint from the terminal.

import http from "node:http";
import https from "node:https";

export interface SSEEvent {
    type: "meta" | "token" | "tool_call" | "tool_result" | "confirmation_required" | "done" | "error";
    content?: string;
    conversation_id?: string;
    name?: string;
    input?: Record<string, unknown>;
    output?: Record<string, unknown>;
    confirmation_id?: string;
    tool_name?: string;
    tool_args?: Record<string, unknown>;
    message?: string;
}

export interface ApiClientConfig {
    baseUrl: string;       // e.g. "https://xcelsior.ca" or "http://localhost:9500"
    apiKey: string;        // Bearer token for auth
    pageContext?: string;  // optional context hint
}

/**
 * Stream a chat message to the Xcel AI assistant.
 * Yields parsed SSE events as they arrive.
 */
export async function* streamChat(
    config: ApiClientConfig,
    message: string,
    conversationId?: string,
): AsyncGenerator<SSEEvent> {
    const body = JSON.stringify({
        message,
        conversation_id: conversationId ?? null,
        page_context: config.pageContext ?? "cli-wizard",
    });

    const url = new URL("/api/ai/chat", config.baseUrl);
    const isHttps = url.protocol === "https:";
    const transport = isHttps ? https : http;

    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
        const req = transport.request(
            {
                hostname: url.hostname,
                port: url.port || (isHttps ? 443 : 80),
                path: url.pathname,
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${config.apiKey}`,
                    "Accept": "text/event-stream",
                },
            },
            resolve,
        );
        req.on("error", reject);
        req.write(body);
        req.end();
    });

    if (response.statusCode !== 200) {
        const chunks: Buffer[] = [];
        for await (const chunk of response) chunks.push(chunk as Buffer);
        const text = Buffer.concat(chunks).toString();
        yield { type: "error", message: `HTTP ${response.statusCode}: ${text}` };
        return;
    }

    // Parse SSE stream
    let buffer = "";
    for await (const chunk of response) {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? ""; // keep incomplete line in buffer

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;
            try {
                const event = JSON.parse(jsonStr) as SSEEvent;
                yield event;
            } catch {
                // skip malformed SSE lines
            }
        }
    }
}

/**
 * Confirm or reject a pending write action.
 * Returns the streamed response from the AI summarizing the result.
 */
export async function* confirmAction(
    config: ApiClientConfig,
    confirmationId: string,
    approved: boolean,
): AsyncGenerator<SSEEvent> {
    const body = JSON.stringify({ confirmation_id: confirmationId, approved });
    const url = new URL("/api/ai/confirm", config.baseUrl);
    const isHttps = url.protocol === "https:";
    const transport = isHttps ? https : http;

    const response = await new Promise<http.IncomingMessage>((resolve, reject) => {
        const req = transport.request(
            {
                hostname: url.hostname,
                port: url.port || (isHttps ? 443 : 80),
                path: url.pathname,
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${config.apiKey}`,
                    "Accept": "text/event-stream",
                },
            },
            resolve,
        );
        req.on("error", reject);
        req.write(body);
        req.end();
    });

    let buffer = "";
    for await (const chunk of response) {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const jsonStr = line.slice(6).trim();
            if (!jsonStr || jsonStr === "[DONE]") continue;
            try {
                yield JSON.parse(jsonStr) as SSEEvent;
            } catch {
                // skip
            }
        }
    }
}
