// capability.ts — Can this terminal show Hexara?
//
// Hexara is drawn as coloured half-block text (sprite-render.ts), so what he
// needs is an interactive terminal with colour — not Sixel. This replaced a
// DA1 round-trip that asked the terminal about Sixel support: it had to put
// stdin into raw mode before Ink started, and it answered "no" in terminals
// such as VS Code's, where he would now render fine.

import { supportsColor } from "chalk";

/** chalk colour levels: 0 none, 1 basic 16, 2 ansi-256, 3 truecolor. */
export type ColorLevel = 0 | 1 | 2 | 3;

/**
 * Pure decision, exported for tests. Paint only on a TTY with at least
 * 256 colours (16 colours flattens the pixel art into mud), never for
 * TERM=dumb, and never when the user opted out with XCELSIOR_NO_SPRITE=1.
 */
export function heuristicCapable(
    env: NodeJS.ProcessEnv = process.env,
    stdout: { isTTY?: boolean } = process.stdout,
    colorLevel: ColorLevel = (supportsColor ? supportsColor.level : 0) as ColorLevel,
): boolean {
    if (env.XCELSIOR_NO_SPRITE === "1") return false;
    if ((env.TERM ?? "").toLowerCase() === "dumb") return false;
    if (!stdout || stdout.isTTY !== true) return false;
    return colorLevel >= 2;
}

/** Whether to draw Hexara in this process. */
export function spriteCapable(): boolean {
    return heuristicCapable();
}
