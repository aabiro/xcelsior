#!/usr/bin/env tsx
/**
 * generate-wizard-frames.ts — Build-time PNG → Sixel sprite converter
 *
 * Reads PNG frames from sprites/wizard/, crops to a global bounding box,
 * encodes as Sixel graphics strings, writes src/wizard-frames.ts.
 * Sixel allows rendering full-resolution pixel art directly in the terminal.
 *
 * Usage: npx tsx scripts/generate-wizard-frames.ts
 */

import { readFileSync, writeFileSync, readdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

// ── Types ────────────────────────────────────────────────────
interface RGBA { r: number; g: number; b: number; a: number }
type Grid = RGBA[][];

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const SPRITES_DIR = join(ROOT, "..", "sprites", "wizard");
const OUTPUT = join(ROOT, "src", "wizard-frames.ts");

// ── PNG handling ─────────────────────────────────────────────
function readPng(path: string): Grid {
    const PNG = require("pngjs").PNG;
    const data = readFileSync(path);
    const img = PNG.sync.read(data);
    const grid: Grid = [];
    for (let y = 0; y < img.height; y++) {
        const row: RGBA[] = [];
        for (let x = 0; x < img.width; x++) {
            const i = (img.width * y + x) * 4;
            row.push({ r: img.data[i], g: img.data[i + 1], b: img.data[i + 2], a: img.data[i + 3] });
        }
        grid.push(row);
    }
    return grid;
}

/** Get bounding box of non-transparent pixels */
function getBounds(grid: Grid): { top: number; bottom: number; left: number; right: number } {
    const h = grid.length;
    const w = grid[0]?.length ?? 0;
    let top = h, bottom = 0, left = w, right = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            if (grid[y][x].a >= 32) {
                if (y < top) top = y;
                if (y > bottom) bottom = y;
                if (x < left) left = x;
                if (x > right) right = x;
            }
        }
    }
    return { top, bottom, left, right };
}

/** Crop grid to given bounds */
function cropTo(grid: Grid, top: number, bottom: number, left: number, right: number): Grid {
    return grid.slice(top, bottom + 1).map(row => row.slice(left, right + 1));
}

// ── Sixel encoding ───────────────────────────────────────────
/** Encode a pixel grid as a Sixel escape sequence.
 *  Transparent pixels are not drawn (terminal background shows through).
 *  Uses RLE compression to minimize string size. */
function rgbaToSixel(grid: Grid): string {
    const h = grid.length;
    const w = grid[0]?.length ?? 0;
    if (!h || !w) return "";

    // Build indexed bitmap + palette
    const colorMap = new Map<string, number>();
    const palette: RGBA[] = [];
    const indexed: number[][] = [];

    for (let y = 0; y < h; y++) {
        const row: number[] = [];
        for (let x = 0; x < w; x++) {
            const px = grid[y][x];
            if (px.a < 32) { row.push(-1); continue; }
            const key = `${px.r},${px.g},${px.b}`;
            let ci = colorMap.get(key);
            if (ci === undefined) {
                ci = palette.length;
                colorMap.set(key, ci);
                palette.push(px);
            }
            row.push(ci);
        }
        indexed.push(row);
    }

    if (palette.length === 0) return `\x1bPq"1;1;${w};${h}\x1b\\`; // all transparent — valid empty Sixel

    // DCS q with raster attributes (1:1 aspect, width x height)
    let s = `\x1bPq"1;1;${w};${h}`;

    // Color registers (Sixel RGB uses 0-100 scale)
    for (let i = 0; i < palette.length; i++) {
        const c = palette[i];
        s += `#${i};2;${Math.round(c.r * 100 / 255)};${Math.round(c.g * 100 / 255)};${Math.round(c.b * 100 / 255)}`;
    }

    // Encode pixel bands (each band = 6 pixel rows)
    const bands = Math.ceil(h / 6);
    for (let band = 0; band < bands; band++) {
        const y0 = band * 6;
        for (let ci = 0; ci < palette.length; ci++) {
            const raw: number[] = [];
            let lastNonZero = -1;
            for (let x = 0; x < w; x++) {
                let bits = 0;
                for (let dy = 0; dy < 6; dy++) {
                    const y = y0 + dy;
                    if (y < h && indexed[y][x] === ci) bits |= (1 << dy);
                }
                raw.push(bits);
                if (bits > 0) lastNonZero = x;
            }

            if (lastNonZero < 0) continue; // no pixels for this color in this band

            // RLE encode, trimming trailing zeros
            let data = "";
            let runVal = raw[0];
            let runLen = 1;
            for (let x = 1; x <= lastNonZero; x++) {
                if (raw[x] === runVal) {
                    runLen++;
                } else {
                    data += runLen > 3
                        ? `!${runLen}${String.fromCharCode(63 + runVal)}`
                        : String.fromCharCode(63 + runVal).repeat(runLen);
                    runVal = raw[x];
                    runLen = 1;
                }
            }
            data += runLen > 3
                ? `!${runLen}${String.fromCharCode(63 + runVal)}`
                : String.fromCharCode(63 + runVal).repeat(runLen);

            s += `#${ci}${data}$`;
        }
        if (band < bands - 1) s += "-";
    }

    s += `\x1b\\`;
    return s;
}

// ── File grouping ────────────────────────────────────────────
type Group = "intro" | "idle" | "pace" | "think" | "wave" | "cast" | "outro"
    | "eureka" | "celebrate" | "error" | "sleep" | "levitate" | "dance" | "bow";

function classify(name: string): Group | null {
    const n = name.toLowerCase().replace(/^wizard-/, "");
    if (n.startsWith("intro")) return "intro";
    if (n.startsWith("idle")) return "idle";
    if (n.startsWith("pace")) return "pace";
    if (n.startsWith("think")) return "think";
    if (n.startsWith("wave")) return "wave";
    if (n.startsWith("cast")) return "cast";
    if (n.startsWith("outro")) return "outro";
    if (n.startsWith("eureka")) return "eureka";
    if (n.startsWith("celebrate")) return "celebrate";
    if (n.startsWith("error")) return "error";
    if (n.startsWith("sleep")) return "sleep";
    if (n.startsWith("levitate")) return "levitate";
    if (n.startsWith("dance")) return "dance";
    if (n.startsWith("bow")) return "bow";
    return null;
}

// ── Serialization ────────────────────────────────────────────
function serializeFrames(name: string, frames: string[]): string {
    const inner = frames.map(f => `  ${JSON.stringify(f)}`).join(",\n");
    return `export const ${name}: Frame[] = [\n${inner},\n];`;
}

// ── Main ─────────────────────────────────────────────────────
const ALL_GROUPS: Group[] = [
    "intro", "idle", "pace", "think", "wave", "cast", "outro",
    "eureka", "celebrate", "error", "sleep", "levitate", "dance", "bow",
];
const groups: Record<Group, string[]> = {
    intro: [], idle: [], pace: [], think: [], wave: [], cast: [], outro: [],
    eureka: [], celebrate: [], error: [], sleep: [], levitate: [], dance: [], bow: [],
};

const rawByGroup: Record<Group, { file: string; grid: Grid }[]> = {
    intro: [], idle: [], pace: [], think: [], wave: [], cast: [], outro: [],
    eureka: [], celebrate: [], error: [], sleep: [], levitate: [], dance: [], bow: [],
};

if (existsSync(SPRITES_DIR)) {
    const files = readdirSync(SPRITES_DIR)
        .filter((f) => f.endsWith(".png"))
        .sort((a, b) => {
            const na = a.match(/(\d+)\.png$/)?.[1] ?? "0";
            const nb = b.match(/(\d+)\.png$/)?.[1] ?? "0";
            const prefix = a.replace(/\d+\.png$/, "").localeCompare(b.replace(/\d+\.png$/, ""));
            return prefix || parseInt(na) - parseInt(nb);
        });

    if (files.length > 0) {
        console.log(`🖼  Found ${files.length} PNG(s) in sprites/wizard/\n`);
        for (const file of files) {
            const group = classify(file);
            if (!group) {
                console.log(`  ⚠  Skipping ${file} — unrecognized prefix`);
                continue;
            }
            rawByGroup[group].push({ file, grid: readPng(join(SPRITES_DIR, file)) });
        }
    }
}

// Global bounding box across ALL frames for consistent sprite dimensions
let gTop = Infinity, gBottom = 0, gLeft = Infinity, gRight = 0;
for (const g of ALL_GROUPS) {
    for (const { grid } of rawByGroup[g]) {
        const b = getBounds(grid);
        if (b.top <= b.bottom) {
            gTop = Math.min(gTop, b.top);
            gBottom = Math.max(gBottom, b.bottom);
            gLeft = Math.min(gLeft, b.left);
            gRight = Math.max(gRight, b.right);
        }
    }
}

// Tighten the bottom to the core animation groups (not intro/outro particles).
// This keeps the wizard's feet at the bottom edge, aligned with the text baseline.
const CORE_GROUPS: Group[] = ["idle", "pace", "think", "wave", "cast",
    "eureka", "celebrate", "error", "sleep", "levitate", "dance", "bow"];
let coreBottom = 0;
for (const g of CORE_GROUPS) {
    for (const { grid } of rawByGroup[g]) {
        const b = getBounds(grid);
        if (b.top <= b.bottom) coreBottom = Math.max(coreBottom, b.bottom);
    }
}
if (coreBottom > 0 && coreBottom < gBottom) {
    console.log(`  📐 Tightened bottom from row ${gBottom} → ${coreBottom} (clipping intro/outro particles)`);
    gBottom = coreBottom;
}

const cropW = gTop <= gBottom ? gRight - gLeft + 1 : 0;
const cropH = gTop <= gBottom ? gBottom - gTop + 1 : 0;
console.log(`  📐 Global bbox: ${cropW}×${cropH} px (rows ${gTop}–${gBottom}, cols ${gLeft}–${gRight})\n`);

// Crop all frames to global bbox then encode as Sixel
for (const g of ALL_GROUPS) {
    for (const { file, grid } of rawByGroup[g]) {
        const cropped = (gTop <= gBottom) ? cropTo(grid, gTop, gBottom, gLeft, gRight) : grid;
        const sixel = rgbaToSixel(cropped);
        groups[g].push(sixel);
        console.log(`  ✓  ${file} → ${g} (${grid[0].length}×${grid.length} → ${cropW}×${cropH} sixel, ${sixel.length} bytes)`);
    }
}

const total = Object.values(groups).reduce((n, g) => n + g.length, 0);
if (total === 0) {
    console.log("No PNGs found in sprites/wizard/ — keeping placeholder wizard-frames.ts.");
    console.log("\nTo generate real frames:");
    console.log("  1. Drop PNGs in sprites/wizard/ (wizard-intro-1.png, wizard-idle-1.png, etc.)");
    console.log("  2. Run: npm run generate-frames");
    process.exit(0);
}

if (groups.idle.length === 0) {
    const fallback = ALL_GROUPS.find((g) => groups[g].length > 0);
    if (fallback) {
        groups.idle = [groups[fallback][0]];
        console.log(`  ℹ  No idle PNGs — using ${fallback} frame as fallback`);
    }
}

const NAMES: Record<Group, string> = {
    intro: "INTRO_FRAMES",
    idle: "IDLE_FRAMES",
    pace: "PACE_FRAMES",
    think: "THINK_FRAMES",
    wave: "WAVE_FRAMES",
    cast: "CAST_FRAMES",
    outro: "OUTRO_FRAMES",
    eureka: "EUREKA_FRAMES",
    celebrate: "CELEBRATE_FRAMES",
    error: "ERROR_FRAMES",
    sleep: "SLEEP_FRAMES",
    levitate: "LEVITATE_FRAMES",
    dance: "DANCE_FRAMES",
    bow: "BOW_FRAMES",
};

// Estimate terminal cell dimensions (typical ~8x16 px per cell)
const spriteCols = Math.ceil(cropW / 8);
const spriteRows = Math.ceil(cropH / 16);

const sections = ALL_GROUPS
    .filter((g) => groups[g].length > 0)
    .map((g) => serializeFrames(NAMES[g], groups[g]))
    .join("\n\n");

const output = `// Auto-generated by scripts/generate-wizard-frames.ts — DO NOT EDIT
// Re-generate: npm run generate-frames
// Source: sprites/wizard/*.png → Sixel graphics (full pixel resolution)
//
// 14 frame groups: 7 core + 7 branch reactions
//   Core: INTRO → IDLE → [PACE → THINK → WAVE → CAST → IDLE] loop → OUTRO
//   Branch: EUREKA, CELEBRATE, ERROR, SLEEP, LEVITATE, DANCE, BOW

/** Sixel escape sequence string */
export type Frame = string;

/** Estimated terminal cell width of sprite */
export const SPRITE_COLS = ${spriteCols};
/** Estimated terminal cell height of sprite */
export const SPRITE_ROWS = ${spriteRows};
/** Sprite pixel dimensions */
export const SPRITE_PX = { w: ${cropW}, h: ${cropH} };

${sections}
`;

writeFileSync(OUTPUT, output, "utf-8");
console.log(`\n✓ Wrote ${OUTPUT} (${total} frames, ${spriteCols}×${spriteRows} estimated cells)`);
