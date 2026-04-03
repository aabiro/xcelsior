// Mock for api-client.ts — used in tests to avoid real HTTP calls.
// Returns configurable SSE event sequences.

import type { SSEEvent, ApiClientConfig } from "../api-client.js";

let _chatEvents: SSEEvent[] = [
    { type: "meta", conversation_id: "test-conv-1" },
    { type: "token", content: "I can help with that! " },
    { type: "token", content: "Here's what you need to know." },
    { type: "done" },
];

let _confirmEvents: SSEEvent[] = [
    { type: "token", content: "Action completed successfully." },
    { type: "done" },
];

/** Override the mock chat response sequence */
export function setMockChatEvents(events: SSEEvent[]) {
    _chatEvents = events;
}

/** Override the mock confirm response sequence */
export function setMockConfirmEvents(events: SSEEvent[]) {
    _confirmEvents = events;
}

/** Reset to defaults */
export function resetApiMock() {
    _chatEvents = [
        { type: "meta", conversation_id: "test-conv-1" },
        { type: "token", content: "I can help with that! " },
        { type: "token", content: "Here's what you need to know." },
        { type: "done" },
    ];
    _confirmEvents = [
        { type: "token", content: "Action completed successfully." },
        { type: "done" },
    ];
}

export async function* streamChat(
    _config: ApiClientConfig,
    _message: string,
    _conversationId?: string,
): AsyncGenerator<SSEEvent> {
    for (const event of _chatEvents) {
        yield event;
    }
}

export async function* confirmAction(
    _config: ApiClientConfig,
    _confirmationId: string,
    _approved: boolean,
): AsyncGenerator<SSEEvent> {
    for (const event of _confirmEvents) {
        yield event;
    }
}
