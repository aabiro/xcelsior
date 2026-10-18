/** RFC-compliant SSE frame parser with CRLF normalization. */

export interface SseFrame {
    event?: string;
    data: string;
    id?: string;
}

export function normalizeSseChunk(buffer: string): string {
    return buffer.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

export function parseSseBuffer(buffer: string): { frames: SseFrame[]; remainder: string } {
    const normalized = normalizeSseChunk(buffer);
    const parts = normalized.split("\n\n");
    const remainder = parts.pop() ?? "";
    const frames: SseFrame[] = [];

    for (const block of parts) {
        if (!block.trim()) continue;
        let event: string | undefined;
        let id: string | undefined;
        const dataLines: string[] = [];

        for (const line of block.split("\n")) {
            if (!line || line.startsWith(":")) continue; // comment / heartbeat
            const colon = line.indexOf(":");
            const field = colon === -1 ? line : line.slice(0, colon);
            const value = colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");

            switch (field) {
                case "event":
                    event = value;
                    break;
                case "data":
                    dataLines.push(value);
                    break;
                case "id":
                    id = value;
                    break;
            }
        }

        if (dataLines.length) {
            frames.push({ event, id, data: dataLines.join("\n") });
        }
    }

    return { frames, remainder };
}

export function* parseSseStream(buffer: string): Generator<{ frame: SseFrame; remainder: string }> {
    let working = buffer;
    while (true) {
        const normalized = normalizeSseChunk(working);
        const splitAt = normalized.indexOf("\n\n");
        if (splitAt === -1) break;
        const block = normalized.slice(0, splitAt);
        working = normalized.slice(splitAt + 2);
        const { frames } = parseSseBuffer(block + "\n\n");
        for (const frame of frames) {
            yield { frame, remainder: working };
        }
    }
}