import React from "react";
import { EventEmitter } from "node:events";
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, Text } from "ink";
import { useRepaintOnNarrow } from "../use-terminal-width.js";

function fakeTty(columns: number) {
    const stdout = Object.assign(new EventEmitter(), {
        isTTY: true, columns, rows: 30, written: [] as string[],
        write(chunk: string) { stdout.written.push(String(chunk)); return true; },
    });
    return stdout;
}

function Screen() {
    useRepaintOnNarrow();
    return <Text>Hexara</Text>;
}

const CLEAR = "\u001b[2J\u001b[H";
afterEach(() => vi.useRealTimers());

describe("useRepaintOnNarrow", () => {
    it("clears and redraws once the terminal has settled narrower", async () => {
        vi.useFakeTimers();
        const stdout = fakeTty(96);
        const app = render(<Screen />, { stdout: stdout as unknown as NodeJS.WriteStream, patchConsole: false });
        await vi.advanceTimersByTimeAsync(50);
        stdout.columns = 80; stdout.emit("resize");
        stdout.columns = 64; stdout.emit("resize");
        await vi.advanceTimersByTimeAsync(100);
        expect(stdout.written.join("")).not.toContain(CLEAR);
        await vi.advanceTimersByTimeAsync(100);
        const after = stdout.written.join("");
        expect(after.split(CLEAR)).toHaveLength(2);
        // Ink follows the clear with the current frame.
        expect(after.slice(after.indexOf(CLEAR))).toContain("Hexara");
        app.unmount();
    });

    it("leaves the screen alone when the terminal widens", async () => {
        vi.useFakeTimers();
        const stdout = fakeTty(64);
        const app = render(<Screen />, { stdout: stdout as unknown as NodeJS.WriteStream, patchConsole: false });
        stdout.columns = 110; stdout.emit("resize");
        await vi.advanceTimersByTimeAsync(300);
        expect(stdout.written.join("")).not.toContain(CLEAR);
        app.unmount();
    });
});
