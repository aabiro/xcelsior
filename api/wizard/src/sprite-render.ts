// sprite-render.ts — Draw Hexara as terminal text.
//
// Each terminal cell holds two pixels stacked vertically: "▀" with the upper
// pixel as foreground and the lower as background. That is ordinary coloured
// text, so it works in every colour terminal (VS Code, iTerm, GNOME, Windows
// Terminal) and Ink lays it out like any other line — unlike the Sixel overlay
// it replaces, which needed terminal image support and was drawn outside Ink's
// layout with saved-cursor writes.
//
// Colour depth follows chalk: truecolor where available, else the nearest
// 256- or 16-colour approximation.

import chalk, { type ChalkInstance } from "chalk";
import { PALETTE, type Frame } from "../sprites/wizard/wizard-frames.js";

export const TRANSPARENT = ".";
/** Stage-only palette keys; the generator only emits [0-9a-zA-Z]. */
export const FLOOR_KEY = "_";
export const SHADOW_KEY = "=";

/** Mirror a frame left↔right so Hexara can face the way he is walking. */
export function flipFrame(frame: Frame): Frame {
    return frame.map((row) => [...row].reverse().join(""));
}

function cell(top: string | undefined, bottom: string | undefined, c: ChalkInstance): string {
    if (!top && !bottom) return " ";
    if (top && !bottom) return c.hex(top)("▀");
    if (!top && bottom) return c.hex(bottom)("▄");
    if (top === bottom) return c.hex(top!)("█");
    return c.hex(top!).bgHex(bottom!)("▀");
}

/**
 * Render pixel rows to `ceil(rows / 2)` lines, each exactly one cell per
 * pixel column. Transparent pixels stay unpainted, so the terminal background
 * shows through in light and dark themes alike.
 */
export function renderPixels(
    rows: readonly string[],
    palette: Readonly<Record<string, string>> = PALETTE,
    c: ChalkInstance = chalk,
): string[] {
    const lines: string[] = [];
    for (let y = 0; y < rows.length; y += 2) {
        const top = rows[y];
        const bottom = rows[y + 1] ?? "";
        let line = "";
        for (let x = 0; x < top.length; x++) {
            const t = top[x] === TRANSPARENT ? undefined : palette[top[x]];
            const b = bottom[x] === undefined || bottom[x] === TRANSPARENT ? undefined : palette[bottom[x]];
            line += cell(t, b, c);
        }
        lines.push(line);
    }
    return lines;
}

export interface StageScene {
    frame: Frame;
    facingLeft: boolean;
    /** Stage width in columns (= pixels). */
    width: number;
    /** Stage height in pixel rows, floor included; even. */
    height: number;
    /** Column of the frame's left edge. */
    left: number;
    /** Pixels the frame is lifted off its resting place (jumps, floating). */
    lift: number;
    /** Frame row that rests on the floor when `lift` is 0. */
    restRow: number;
    /** Shadow span on the floor, in stage columns (inclusive); omitted for none. */
    shadow?: { from: number; to: number };
}

/**
 * Compose Hexara onto his stage in pixel space — frame at its position, a
 * one-pixel floor along the bottom with his shadow on it — and render the lot.
 * Composing before rendering is what lets the feet meet the floor exactly:
 * half-block cells would otherwise only allow two-pixel steps.
 */
export function renderStage(scene: StageScene, floorColor: string, shadowColor: string, c: ChalkInstance = chalk): string[] {
    const { width, height } = scene;
    const canvas: string[][] = Array.from({ length: height }, () => Array<string>(width).fill(TRANSPARENT));
    const frame = scene.facingLeft ? flipFrame(scene.frame) : scene.frame;
    const floorRow = height - 1;
    const top = floorRow - scene.restRow - scene.lift;
    for (let r = 0; r < frame.length; r++) {
        const y = top + r;
        if (y < 0 || y >= floorRow) continue;
        const row = frame[r];
        for (let x = 0; x < row.length; x++) {
            const sx = scene.left + x;
            if (sx < 0 || sx >= width || row[x] === TRANSPARENT) continue;
            canvas[y][sx] = row[x];
        }
    }
    const floor = canvas[floorRow];
    floor.fill(FLOOR_KEY);
    if (scene.shadow) {
        for (let x = Math.max(0, scene.shadow.from); x <= Math.min(width - 1, scene.shadow.to); x++) floor[x] = SHADOW_KEY;
    }
    const palette = { ...PALETTE, [FLOOR_KEY]: floorColor, [SHADOW_KEY]: shadowColor };
    return renderPixels(canvas.map((row) => row.join("")), palette, c);
}
