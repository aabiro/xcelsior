#!/usr/bin/env node
/**
 * Draw the images for a product-update email (core/product_update_issues/).
 *
 *   node scripts/build-product-update-assets.mjs 2026-10
 *
 * Writes public/email/product-updates/<slug>/{hero,mcp,billing}.png and
 * host-check.gif. Each scene is a small HTML page in the brand's own colours
 * and typeface, screenshotted with Playwright at 2x; the GIF's frames are
 * joined with Pillow (python3). Mail clients cannot run CSS animation or load
 * web fonts, which is why these are images.
 *
 * The scenes below are the October 2026 issue. A new issue adds its scenes
 * to SCENES under its slug.
 */
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = join(FRONTEND, "public");
// Inlined: a page set with setContent() may not load file:// fonts.
const font = (file) =>
  `data:font/ttf;base64,${readFileSync(join(PUBLIC, "brand-system", file)).toString("base64")}`;
const LOCKUP = readFileSync(join(PUBLIC, "site-assets", "assets", "lockup-light.svg"), "utf8");
// The mark from site-assets/assets/icon-gradient.svg.
const X_PATH =
  "M56.5,54 L92.5,54 L122.9,102 L168.5,54 L204.5,54 L138,120 L183.6,186 L147.6,186 L117.2,138 L71.6,186 L35.5,186 L102,120 Z";

const mark = (size, fill = "url(#xg)") => `
<svg viewBox="0 0 240 240" width="${size}" height="${size}" aria-hidden="true">
  <defs><linearGradient id="xg" x1="120" y1="48" x2="120" y2="192" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="#00d4ff"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
  <path d="${X_PATH}" fill="${fill}"/>
</svg>`;

const BASE_CSS = `
@font-face { font-family: Geist; font-weight: 500; src: url("${font("Geist-Medium.ttf")}"); }
@font-face { font-family: Geist; font-weight: 600; src: url("${font("Geist-SemiBold.ttf")}"); }
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: #0a0e1a; }
body { font-family: Geist, system-ui, sans-serif; font-weight: 500; color: #e8edf5; }
.frame {
  width: 536px; height: 307px; padding: 22px 36px; overflow: hidden;
  background:
    radial-gradient(120% 90% at 50% 118%, rgba(0,212,255,.55) 0%, rgba(124,58,237,.55) 34%, rgba(124,58,237,0) 70%),
    #0b1020;
}
.panel {
  width: 100%; height: 100%; border-radius: 14px; background: #0d1320;
  border: 1px solid #2a3548; display: flex; align-items: center; justify-content: center;
  position: relative; overflow: hidden;
}
.tile { width: 104px; height: 104px; border-radius: 24px; display: flex; align-items: center; justify-content: center; }
.dot { width: 6px; height: 6px; border-radius: 1px; background: #7e8fa0; margin: 0 12px; }
.pill {
  padding: 7px 18px; border-radius: 999px; font-size: 19px; font-weight: 600; color: #fff;
  background: linear-gradient(135deg, #00d4ff, #7c3aed);
}
.mono { font-family: "JetBrains Mono", ui-monospace, Menlo, monospace; }
`;

const page = (css, body) =>
  `<!doctype html><html><head><meta charset="utf-8"><style>${BASE_CSS}${css}</style></head><body>${body}</body></html>`;

const connector = (tile) => `
  <div class="tile" style="background:#05070d;border:1px solid #2a3548">${mark(tile * 0.72)}</div>
  <div class="dot"></div><div class="pill">MCP</div><div class="dot"></div>
  <div class="tile" style="background:linear-gradient(135deg,#00d4ff,#7c3aed)">
    <svg viewBox="0 0 48 48" width="${tile * 0.5}" height="${tile * 0.5}" fill="none" stroke="#fff" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M10 14l10 10-10 10"/><path d="M24 36h14"/>
    </svg>
  </div>`;

const hero = page(
  `
  .hero { width: 600px; height: 220px; position: relative; overflow: hidden;
    background:
      radial-gradient(70% 120% at 100% 100%, rgba(124,58,237,.50) 0%, rgba(124,58,237,0) 62%),
      radial-gradient(60% 110% at 62% 118%, rgba(0,212,255,.34) 0%, rgba(0,212,255,0) 60%),
      #0a0e1a; }
  .grid { position: absolute; inset: 0;
    background-image: radial-gradient(rgba(232,237,245,.10) 1px, transparent 1px);
    background-size: 14px 14px; mask-image: linear-gradient(90deg, transparent 20%, #000 100%); }
  .lockup { position: absolute; left: 32px; top: 30px; }
  .lockup svg { height: 26px; width: auto; display: block; }
  .tag { position: absolute; left: 32px; bottom: 34px; }
  .tag p { font-size: 15px; font-weight: 600; color: #00d4ff; letter-spacing: .02em; }
  .tag h1 { font-size: 30px; line-height: 1.12; font-weight: 600; margin-top: 6px; }
  .row { position: absolute; right: 30px; top: 68px; display: flex; align-items: center; }
  .row .tile { width: 78px; height: 78px; border-radius: 18px; }
  .row .pill { font-size: 14px; padding: 5px 13px; }
  .row .dot { margin: 0 8px; width: 5px; height: 5px; }
  `,
  `<div class="hero"><div class="grid"></div>
     <div class="lockup">${LOCKUP}</div>
     <div class="tag"><p>October 2026</p><h1>What&rsquo;s new<br>in Xcelsior</h1></div>
     <div class="row">${connector(78)}</div>
   </div>`,
);

const mcp = page("", `<div class="frame"><div class="panel">${connector(104)}</div></div>`);

const TRANSACTIONS = [
  ["Wallet top-up", "Oct 28", "+$50.00", "#10b981"],
  ["RTX 4090 · 2h 14m", "Oct 27", "−$1.84", "#e8edf5"],
  ["A100 80GB · 5h 02m", "Oct 25", "−$9.31", "#e8edf5"],
  ["Volume storage · 40 GB", "Oct 24", "−$0.62", "#e8edf5"],
];

const billing = page(
  `
  .panel { flex-direction: column; align-items: stretch; justify-content: flex-start; padding: 16px 22px; }
  .head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
  .head b { font-size: 16px; font-weight: 600; }
  .head span { font-size: 12px; color: #7e8fa0; }
  .tx { display: flex; align-items: center; padding: 9px 0; border-top: 1px solid #1a2235; font-size: 13.5px; }
  .tx .what { flex: 1; }
  .tx .when { width: 62px; color: #7e8fa0; font-size: 12px; }
  .tx .amount { width: 70px; text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; }
  .foot { display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px; }
  .btn { font-size: 12.5px; font-weight: 600; padding: 6px 14px; border-radius: 8px; border: 1px solid #2a3548; color: #e8edf5; }
  .btn.primary { border: 0; color: #fff; background: linear-gradient(135deg, #2f7df6, #7c3aed); }
  `,
  `<div class="frame"><div class="panel">
     <div class="head"><b>Transactions</b><span>1&ndash;20 of 134</span></div>
     ${TRANSACTIONS.map(
       ([what, when, amount, colour]) =>
         `<div class="tx"><span class="what">${what}</span><span class="when">${when}</span><span class="amount" style="color:${colour}">${amount}</span></div>`,
     ).join("")}
     <div class="foot"><span class="btn" style="opacity:.45">Newer</span><span class="btn primary">Older</span></div>
   </div></div>`,
);

const CHECKS = ["Linux host", "NVIDIA driver", "Docker runtime", "Ready to host"];

/** The checklist with the first `done` rows ticked. */
const hostCheck = (done) =>
  page(
    `
    .list { width: 300px; background: #111827; border: 1px solid #2a3548; border-radius: 12px; padding: 8px 20px;
      box-shadow: 0 18px 40px rgba(0,0,0,.45); }
    .item { display: flex; align-items: center; gap: 12px; padding: 11px 0; font-size: 17px; color: #7e8fa0; }
    .item.done { color: #e8edf5; }
    .mark { width: 22px; height: 22px; border-radius: 50%; border: 1.5px solid #4b5a70; flex: none;
      display: flex; align-items: center; justify-content: center; font-size: 11px; color: #7e8fa0; }
    .item.done .mark { border: 0; background: linear-gradient(135deg, #00d4ff, #7c3aed); }
    `,
    `<div class="frame"><div class="panel"><div class="list">
       ${CHECKS.map((label, n) =>
         n < done
           ? `<div class="item done"><span class="mark"><svg viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg></span>${label}</div>`
           : `<div class="item"><span class="mark">${n + 1}</span>${label}</div>`,
       ).join("")}
     </div></div></div>`,
  );

const SCENES = {
  "2026-11": {
    stills: [
      { file: "hero.png", html: hero.replace("October 2026", "November 2026"), width: 600, height: 220 },
      { file: "billing.png", html: billing.replace("1&ndash;20 of 134", "1&ndash;10 of 134"), width: 536, height: 307 },
    ],
    animations: [],
  },
  "2026-10": {
    stills: [
      { file: "hero.png", html: hero, width: 600, height: 220 },
      { file: "mcp.png", html: mcp, width: 536, height: 307 },
      { file: "billing.png", html: billing, width: 536, height: 307 },
    ],
    animations: [
      {
        file: "host-check.gif",
        width: 536,
        height: 307,
        // [html, milliseconds on screen]. The finished list comes first and
        // holds longest: a client that does not animate shows only frame one.
        frames: [
          [hostCheck(4), 2600],
          [hostCheck(0), 900],
          [hostCheck(1), 800],
          [hostCheck(2), 800],
          [hostCheck(3), 800],
        ],
      },
    ],
  },
};

const slug = process.argv[2];
const scenes = SCENES[slug];
if (!scenes) {
  console.error(`usage: build-product-update-assets.mjs <${Object.keys(SCENES).join("|")}>`);
  process.exit(2);
}

const out = join(PUBLIC, "email", "product-updates", slug);
mkdirSync(out, { recursive: true });
const scratch = mkdtempSync(join(tmpdir(), "xcelsior-email-assets-"));
// Same override as playwright.config.ts, for a machine whose downloaded
// browser does not match the installed Playwright.
const executablePath = process.env.XCELSIOR_PLAYWRIGHT_EXECUTABLE_PATH?.trim();
const browser = await chromium.launch(executablePath ? { executablePath } : undefined);
try {
  const shoot = async (html, width, height, path, scale) => {
    const context = await browser.newContext({ viewport: { width, height }, deviceScaleFactor: scale });
    const tab = await context.newPage();
    await tab.setContent(html, { waitUntil: "load" });
    await tab.evaluate(() => document.fonts.ready);
    await tab.screenshot({ path, clip: { x: 0, y: 0, width, height } });
    await context.close();
  };

  for (const still of scenes.stills) {
    await shoot(still.html, still.width, still.height, join(out, still.file), 2);
    console.log(`wrote ${join("public/email/product-updates", slug, still.file)}`);
  }

  for (const animation of scenes.animations) {
    const frames = [];
    for (const [n, [html]] of animation.frames.entries()) {
      const path = join(scratch, `${animation.file}-${n}.png`);
      // 1.5x keeps the GIF sharp on a phone and well under mail size limits.
      await shoot(html, animation.width, animation.height, path, 1.5);
      frames.push(path);
    }
    const durations = animation.frames.map(([, ms]) => ms);
    execFileSync("python3", [
      "-c",
      `import json, sys
from PIL import Image
paths, durations, target = json.loads(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
frames = [Image.open(p).convert("RGB") for p in paths]
# One palette for every frame, so the background does not shimmer between them.
palette = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
frames = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
frames[0].save(target, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)`,
      JSON.stringify(frames),
      JSON.stringify(durations),
      join(out, animation.file),
    ]);
    console.log(`wrote ${join("public/email/product-updates", slug, animation.file)}`);
  }
} finally {
  await browser.close();
  rmSync(scratch, { recursive: true, force: true });
}
