#!/usr/bin/env tsx
/**
 * generate-wizard-frames.ts — Build-time PNG → pixel-frame converter
 *
 * Reads PNG frames from sprites/wizard/, crops them to one global bounding box,
 * and writes sprites/wizard/wizard-frames.ts: a shared colour palette plus
 * every frame as rows of palette keys. src/sprite-render.ts turns a frame into
 * truecolor half-block text (two pixel rows per terminal cell), so Hexara draws
 * in any colour terminal — no Sixel support, no cursor tricks.
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
const OUTPUT = join(ROOT, "sprites", "wizard", "wizard-frames.ts");

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

// ── Pixel encoding ───────────────────────────────────────────
/** Below this alpha a pixel is background; at or above it is drawn solid. */
const ALPHA_THRESHOLD = 32;
/** One character per palette entry; "." is reserved for transparent. */
const KEYS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const palette = new Map<string, string>(); // "#rrggbb" → key

function hex(px: RGBA): string {
    return "#" + [px.r, px.g, px.b].map((v) => v.toString(16).padStart(2, "0")).join("");
}

function encodeFrame(grid: Grid): string[] {
    return grid.map((row) => row.map((px) => {
        if (px.a < ALPHA_THRESHOLD) return ".";
        const color = hex(px);
        let key = palette.get(color);
        if (key === undefined) {
            if (palette.size >= KEYS.length) throw new Error(`More than ${KEYS.length} sprite colours`);
            key = KEYS[palette.size];
            palette.set(color, key);
        }
        return key;
    }).join(""));
}

// ── File grouping ────────────────────────────────────────────
type Group = "intro" | "idle" | "pace" | "think" | "wave" | "cast" | "outro"
    | "eureka" | "celebrate" | "error" | "sleep" | "levitate" | "dance" | "bow"
    | "peek" | "type" | "nod";

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
    if (n.startsWith("peek")) return "peek";
    if (n.startsWith("type")) return "type";
    if (n.startsWith("nod")) return "nod";
    return null;
}

// ── Serialization ────────────────────────────────────────────
// Identical frames (the neutral bookends every move shares) are stored once.
const uniqueFrames: string[][] = [];
const frameIndex = new Map<string, number>();
function frameRef(frame: string[]): string {
    const id = frame.join("\n");
    let i = frameIndex.get(id);
    if (i === undefined) {
        i = uniqueFrames.length;
        uniqueFrames.push(frame);
        frameIndex.set(id, i);
    }
    return `F[${i}]`;
}
function serializeFrames(name: string, frames: string[][]): string {
    return `export const ${name}: Frame[] = [${frames.map(frameRef).join(", ")}];`;
}

// ── Main ─────────────────────────────────────────────────────
const ALL_GROUPS: Group[] = [
    "intro", "idle", "pace", "think", "wave", "cast", "outro",
    "eureka", "celebrate", "error", "sleep", "levitate", "dance", "bow",
    "peek", "type", "nod",
];
const groups: Record<Group, string[][]> = {
    intro: [], idle: [], pace: [], think: [], wave: [], cast: [], outro: [],
    eureka: [], celebrate: [], error: [], sleep: [], levitate: [], dance: [], bow: [],
    peek: [], type: [], nod: [],
};

const rawByGroup: Record<Group, { file: string; grid: Grid }[]> = {
    intro: [], idle: [], pace: [], think: [], wave: [], cast: [], outro: [],
    eureka: [], celebrate: [], error: [], sleep: [], levitate: [], dance: [], bow: [],
    peek: [], type: [], nod: [],
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

// Interstitial choreography frames: inject neutral start/stop frames
// so all animations begin and end perfectly in sync with idle.
const neutralGrid = rawByGroup["idle"][0]?.grid;
if (neutralGrid) {
    const noNeutral = new Set(["intro", "outro", "idle"]);
    for (const g of ALL_GROUPS) {
        if (noNeutral.has(g)) continue;
        if (rawByGroup[g].length > 0) {
            rawByGroup[g].unshift({ file: `wizard-${g}-00.png`, grid: neutralGrid });
            rawByGroup[g].push({ file: `wizard-${g}-99.png`, grid: neutralGrid });
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
    "eureka", "celebrate", "error", "sleep", "levitate", "dance", "bow", "peek", "type", "nod"];
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

// Crop all frames to the global bbox, then encode against the shared palette
for (const g of ALL_GROUPS) {
    for (const { file, grid } of rawByGroup[g]) {
        const cropped = (gTop <= gBottom) ? cropTo(grid, gTop, gBottom, gLeft, gRight) : grid;
        groups[g].push(encodeFrame(cropped));
        console.log(`  ✓  ${file} → ${g} (${grid[0].length}×${grid.length} → ${cropW}×${cropH})`);
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
    peek: "PEEK_FRAMES",
    type: "TYPE_FRAMES",
    nod: "NOD_FRAMES",
};

const spriteCols = cropW;
const spriteRows = Math.ceil(cropH / 2);

const sections = ALL_GROUPS
    .filter((g) => groups[g].length > 0)
    .map((g) => serializeFrames(NAMES[g], groups[g]))
    .join("\n");

const paletteBody = [...palette.entries()].map(([color, key]) => `  "${key}": "${color}",`).join("\n");
const framesBody = uniqueFrames
    .map((rows) => `  [\n${rows.map((r) => `    "${r}",`).join("\n")}\n  ],`)
    .join("\n");

const output = `// Auto-generated by scripts/generate-wizard-frames.ts — DO NOT EDIT
// Re-generate: npm run generate-frames
// Source: sprites/wizard/*.png → palette-keyed pixel rows (full resolution)
//
// Core: INTRO → IDLE → [PACE → THINK → WAVE → CAST → IDLE] loop → OUTRO
// Branch reactions: EUREKA, CELEBRATE, ERROR, SLEEP, LEVITATE, DANCE, BOW,
// PEEK, TYPE, NOD. src/sprite-render.ts draws them as half-block text.

/** One frame: SPRITE_PX.h rows of SPRITE_PX.w palette keys; "." is transparent. */
export type Frame = readonly string[];

/** Palette key → hex colour. */
export const PALETTE: Readonly<Record<string, string>> = {
${paletteBody}
};

/** Sprite pixel dimensions */
export const SPRITE_PX = { w: ${cropW}, h: ${cropH} } as const;
/** Terminal cells: one column per pixel, two pixel rows per cell. */
export const SPRITE_COLS = ${spriteCols};
export const SPRITE_ROWS = ${spriteRows};

const F: Frame[] = [
${framesBody}
];

${sections}
`;

writeFileSync(OUTPUT, output, "utf-8");
console.log(`\n✓ Wrote ${OUTPUT} (${total} frames, ${uniqueFrames.length} unique, ${palette.size} colours, ${spriteCols}×${spriteRows} cells)`);
