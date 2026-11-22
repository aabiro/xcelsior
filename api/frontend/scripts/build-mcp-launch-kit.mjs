#!/usr/bin/env node
/**
 * Re-cut the MCP launch-kit images that show the tool count.
 *
 * The images were designed on a canvas; four of them draw numbers that come
 * from api/mcp/tool-surface.json (the count, and the tools/list screenshot's
 * split and grid). This script erases those regions in each PNG (1x and @2x)
 * and redraws them from the surface, with the fonts and positions measured
 * from the originals. Then it updates baked-in.json, which
 * tests/test_release_facts.py holds to the surface.
 *
 *   node scripts/build-mcp-launch-kit.mjs --date "22 NOV 2026" [--live tools.json]
 *
 * --date is the day the tools/list screenshot's list was checked against the
 * live service. --live is a saved tools/list result; when given, its tool
 * names must equal the surface's or nothing is written.
 *
 * Run from api/frontend. CHROME_PATH overrides the Chromium binary.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FE = path.resolve(HERE, "..");
const ROOT = path.resolve(FE, "../..");
const KIT = process.env.MCP_KIT_DIR || path.join(FE, "public/brand-system/assets/social/mcp-launch");
const FONTS = {
  GeistKit: path.join(FE, "node_modules/next/dist/next-devtools/server/font/geist-latin.woff2"),
  GeistMonoKit: path.join(FE, "public/brand-system/GeistMono-Regular.ttf"),
};

function arg(name) {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : undefined;
}

const date = arg("--date");
if (!date || !/^\d{2} [A-Z]{3} \d{4}$/.test(date)) {
  console.error('Pass --date "DD MON YYYY", the day the list was checked against the live service.');
  process.exit(2);
}

const surface = JSON.parse(fs.readFileSync(path.join(ROOT, "api/mcp/tool-surface.json"), "utf8")).tools;
const tools = [...surface].sort((a, b) => a.name.localeCompare(b.name));
const live = arg("--live");
if (live) {
  const body = JSON.parse(fs.readFileSync(live, "utf8"));
  const names = (body.result?.tools ?? body.tools ?? []).map((t) => t.name).sort();
  const expected = tools.map((t) => t.name);
  if (JSON.stringify(names) !== JSON.stringify(expected)) {
    console.error("The live tools/list differs from tool-surface.json; deploy first.");
    process.exit(1);
  }
}
const kind = (t) =>
  t.annotations?.readOnlyHint ? "readOnly" : t.annotations?.destructiveHint ? "confirm" : "other";
const count = tools.length;
const split = { readOnly: 0, confirm: 0, other: 0 };
for (const t of tools) split[kind(t)] += 1;
for (const v of Object.values(split)) {
  if (v < 10 || v > 99) throw new Error("the header split is laid out for two-digit counts");
}

// Every coordinate is in @2x pixels, measured from the originals; 1x halves them.
const SANS = "GeistKit";
const MONO = "GeistMonoKit";
const SLATE = "rgb(132,146,166)";
const ROWS = 23;
const DOT = { readOnly: "rgb(0,212,255)", confirm: "rgb(245,158,11)", other: "rgb(70,86,111)" };

const IMAGES = [
  {
    file: "x/header-1500x500",
    erase: [[1550, 215, 2190, 285]],
    text: [{ s: `${count} tools · any MCP client`, x: 1559, y: 266, font: [400, 54.5, SANS], color: SLATE }],
  },
  {
    file: "github/social-preview-1280x640",
    erase: [[240, 1066, 760, 1124]],
    text: [{ s: `Hosted MCP · ${count} tools`, x: 248, y: 1112, font: [500, 48, SANS], color: "rgb(213,221,232)" }],
  },
  {
    file: "producthunt/gallery-4-capabilities-1270x760",
    erase: [[630, 122, 880, 170]],
    text: [{ s: `${count} TOOLS`, x: 645, y: 159, font: [400, 37, MONO], ls: 6, color: SLATE }],
  },
  {
    file: "screenshots/mcp-tools-list-1270x760",
    erase: [
      [975, 84, 1300, 128], // capture date
      [80, 155, 1310, 265], // headline
      [1588, 332, 1642, 372], // read-only count
      [1978, 332, 2030, 372], // confirm-first count
      [2366, 332, 2420, 372], // other-actions count
      [120, 425, 2420, 1405], // the grid
    ],
    text: [
      { s: date, x: 983, y: 119, font: [400, 37, MONO], ls: 6, color: SLATE },
      { s: `${count} tools`, x: 88, y: 240, font: [600, 96, SANS], ls: -4, color: "rgb(232,237,245)", then: { s: " your agent can call.", color: SLATE } },
      { s: String(split.readOnly), x: 1595, y: 362, font: [400, 32, MONO], color: "rgb(183,194,211)" },
      { s: String(split.confirm), x: 1984, y: 362, font: [400, 32, MONO], color: "rgb(183,194,211)" },
      { s: String(split.other), x: 2374, y: 362, font: [400, 32, MONO], color: "rgb(183,194,211)" },
    ],
    grid: tools.map((t) => ({ name: t.name, dot: DOT[kind(t)] })),
  },
];
if (tools.length > ROWS * 4) throw new Error(`the grid holds ${ROWS * 4} tools`);

const browser = await chromium.launch(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {});
const page = await browser.newPage();
await page.setContent("<html><body></body></html>");
const fonts = Object.fromEntries(
  Object.entries(FONTS).map(([family, file]) => [family, fs.readFileSync(file).toString("base64")]),
);

for (const image of IMAGES) {
  for (const scale of [2, 1]) {
    const file = path.join(KIT, `${image.file}${scale === 2 ? "@2x" : ""}.png`);
    const png = fs.readFileSync(file).toString("base64");
    const out = await page.evaluate(
      async ({ png, image, scale, fonts, ROWS }) => {
        for (const [family, b64] of Object.entries(fonts)) {
          const face = new FontFace(family, `url(data:font/woff2;base64,${b64})`, { weight: "100 900" });
          await face.load();
          document.fonts.add(face);
        }
        const k = scale / 2;
        const img = new Image();
        img.src = `data:image/png;base64,${png}`;
        await img.decode();
        const canvas = document.createElement("canvas");
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);

        // Erase: each column blends from the row above the box to the row below,
        // so a background gradient carries through.
        for (const box of image.erase) {
          const [x0, y0, x1, y1] = box.map((v) => Math.round(v * k));
          const w = x1 - x0;
          const h = y1 - y0;
          const top = ctx.getImageData(x0, y0 - 1, w, 1).data;
          const bottom = ctx.getImageData(x0, y1, w, 1).data;
          const fill = ctx.createImageData(w, h);
          for (let y = 0; y < h; y++) {
            const t = (y + 1) / (h + 1);
            for (let x = 0; x < w; x++) {
              for (let c = 0; c < 4; c++) {
                fill.data[(y * w + x) * 4 + c] = Math.round(top[x * 4 + c] * (1 - t) + bottom[x * 4 + c] * t);
              }
            }
          }
          ctx.putImageData(fill, x0, y0);
        }

        const draw = ({ s, x, y, font, ls = 0, color, then }) => {
          const [weight, size, family] = font;
          ctx.font = `${weight} ${size * k}px ${family}`;
          ctx.letterSpacing = `${ls * k}px`;
          ctx.textBaseline = "alphabetic";
          ctx.fillStyle = color;
          ctx.fillText(s, x * k, y * k);
          if (then) {
            const width = ctx.measureText(s).width;
            ctx.fillStyle = then.color;
            ctx.fillText(then.s, x * k + width, y * k);
          }
        };
        for (const item of image.text) draw(item);

        if (image.grid) {
          const columns = [166, 756, 1362, 1951];
          image.grid.forEach((tool, i) => {
            const x = columns[Math.floor(i / ROWS)];
            const y = 458 + 42 * (i % ROWS);
            ctx.fillStyle = tool.dot;
            ctx.beginPath();
            ctx.arc((x - 24.5) * k, (y - 10.5) * k, 6 * k, 0, Math.PI * 2);
            ctx.fill();
            draw({ s: tool.name, x, y, font: [400, 28, "GeistMonoKit"], color: "rgb(213,221,232)" });
          });
        }
        return canvas.toDataURL("image/png").split(",")[1];
      },
      { png, image, scale, fonts, ROWS },
    );
    fs.writeFileSync(file, Buffer.from(out, "base64"));
    console.log("wrote", path.relative(FE, file));
  }
}
await browser.close();

const bakedPath = path.join(KIT, "baked-in.json");
const baked = JSON.parse(fs.readFileSync(bakedPath, "utf8"));
baked.mcpToolCount.value = count;
baked.toolsListSplit.readOnly = split.readOnly;
baked.toolsListSplit.previewThenConfirm = split.confirm;
baked.toolsListSplit.otherActions = split.other;
baked.toolsListSplit.checkedAgainstLive = date;
fs.writeFileSync(bakedPath, `${JSON.stringify(baked, null, 2)}\n`);
console.log(`baked-in.json: ${count} tools, split ${split.readOnly}/${split.confirm}/${split.other}`);
