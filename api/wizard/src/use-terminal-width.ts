// use-terminal-width.ts — shared hook tracking terminal width with resize.

import { useEffect, useState } from "react";
import { useStdout } from "ink";

/** Track terminal column count, updating on resize. */
export function useTerminalWidth(): number {
    const { stdout } = useStdout();
    const [width, setWidth] = useState(stdout?.columns ?? 80);
    useEffect(() => {
        if (!stdout) return;
        const onResize = () => setWidth(stdout.columns ?? 80);
        stdout.on("resize", onResize);
        return () => {
            stdout.off?.("resize", onResize);
        };
    }, [stdout]);
    return width;
}

/** Track terminal columns and rows, updating on resize. */
export function useTerminalSize(): { columns: number; rows: number } {
    const { stdout } = useStdout();
    const read = () => ({ columns: stdout?.columns ?? 80, rows: stdout?.rows ?? 24 });
    const [size, setSize] = useState(read);
    useEffect(() => {
        if (!stdout) return;
        const onResize = () => setSize(read());
        stdout.on("resize", onResize);
        return () => {
            stdout.off?.("resize", onResize);
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [stdout]);
    return size;
}

/**
 * Repaint from a clean screen when the terminal narrows.
 *
 * Ink 5 erases its previous frame line by line before drawing the next one.
 * A narrower terminal re-wraps those lines onto more rows than Ink erases, so
 * the leftover rows stay above every later frame (a doubled title bar, a
 * second half of Hexara). Ink also renders once immediately on resize, before
 * React has the new width, and throttles renders to 32 ms. So wait for the
 * resize to settle and the frame laid out for the new width to be drawn, then
 * ask Ink to write a screen clear; Ink follows it with that frame.
 */
export function useRepaintOnNarrow(settleMs = 150): void {
    const { stdout, write } = useStdout();
    useEffect(() => {
        if (!stdout?.isTTY) return;
        let width = stdout.columns ?? 80;
        let narrowed = false;
        let timer: ReturnType<typeof setTimeout> | undefined;
        const onResize = () => {
            const next = stdout.columns ?? 80;
            if (next < width) narrowed = true;
            width = next;
            clearTimeout(timer);
            timer = setTimeout(() => {
                if (narrowed) write("\u001b[2J\u001b[H");
                narrowed = false;
            }, settleMs);
        };
        stdout.on("resize", onResize);
        return () => {
            clearTimeout(timer);
            stdout.off?.("resize", onResize);
        };
    }, [stdout, write, settleMs]);
}
