#!/usr/bin/env node
/**
 * Draw the X images for the automated Xcelsior Compute account, which posts
 * release notes.
 *
 *   node scripts/build-bot-social-assets.mjs
 *   node scripts/build-bot-social-assets.mjs --release release.json --out <dir>
 *
 * With no arguments, writes the header, the pinned intro post and the sample
 * release post (SAMPLE_RELEASE) to public/brand-system/assets/social/bot/x/,
 * each at 1x and @2x. With --release, writes only the release card for that
 * release ({ "area", "title", "body", "url" }) to <dir>. Either way it prints
 * the text that goes with each post and fails if a card or post overflows.
 *
 * The profile picture beside them (profile-1000.png) is not drawn here.
 */
import { mkdirSync, readFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { chromium } from "@playwright/test";

const FRONTEND = join(dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = join(FRONTEND, "public");
const OUT = join(PUBLIC, "brand-system", "assets", "social", "bot", "x");
// Inlined: a page set with setContent() may not load file:// fonts.
const font = (file) =>
  `data:font/ttf;base64,${readFileSync(join(PUBLIC, "brand-system", file)).toString("base64")}`;
// The mark from brand-system/icon-gradient.svg.
const X_PATH =
  "M56.5,54 L92.5,54 L122.9,102 L168.5,54 L204.5,54 L138,120 L183.6,186 L147.6,186 L117.2,138 L71.6,186 L35.5,186 L102,120 Z";

// The release the sample card shows. Live for customers since November 2026
// (core/product_update_issues/2026-11.json, first feature).
const SAMPLE_RELEASE = {
  area: "Billing",
  title: "Itemized usage statements",
  body: "Pick a month under Usage statements to see compute and storage line items, quantities, rates and the total. Download the same period as CSV.",
  url: "https://xcelsior.ca/dashboard/billing",
};

const INTRO_TEXT =
  "This is Xcelsior's automated account. It posts release notes: new features and fixes, once customers can use them, each with a link to where to find it.\n\n" +
  "For help, write to support@xcelsior.ca.";

const mark = (size) => `
<svg viewBox="0 0 240 240" width="${size}" height="${size}" aria-hidden="true">
  <defs><linearGradient id="xg" x1="120" y1="48" x2="120" y2="192" gradientUnits="userSpaceOnUse">
    <stop offset="0" stop-color="#00d4ff"/><stop offset="1" stop-color="#7c3aed"/></linearGradient></defs>
  <path d="${X_PATH}" fill="url(#xg)"/>
</svg>`;

// The profile picture's badge: white disc, dark ring, flat robot head.
const badge = (size) => `
<svg viewBox="-14 -14 268 268" width="${size}" height="${size}" aria-hidden="true">
  <circle cx="120" cy="120" r="134" fill="#060a13"/>
  <circle cx="120" cy="120" r="120" fill="#ffffff"/>
  <g transform="translate(120 120) scale(1.12) translate(-120 -120)" fill="#060a13">
    <rect x="116" y="65" width="8" height="26"/><circle cx="120" cy="61" r="11"/>
    <rect x="52" y="115" width="14" height="34" rx="5"/><rect x="174" y="115" width="14" height="34" rx="5"/>
    <rect x="64" y="91" width="112" height="82" rx="20"/>
    <rect x="84" y="117" width="20" height="20" rx="4" fill="#fff"/><rect x="136" y="117" width="20" height="20" rx="4" fill="#fff"/>
  </g>
</svg>`;

// The mark as a faint outline, for the right-hand side of the posts.
const outline = (width) => `
<svg viewBox="35.5 54 169 132" width="${width}" height="${(width * 132) / 169}" aria-hidden="true">
  <defs><linearGradient id="og" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0" stop-color="#00d4ff" stop-opacity=".55"/><stop offset="1" stop-color="#7c3aed" stop-opacity=".55"/></linearGradient></defs>
  <path d="${X_PATH}" fill="#0d1320" fill-opacity=".55" stroke="url(#og)" stroke-width="1.6" vector-effect="non-scaling-stroke"/>
</svg>`;

const BASE_CSS = `
@font-face { font-family: Geist; font-weight: 500; src: url("${font("Geist-Medium.ttf")}"); }
@font-face { font-family: Geist; font-weight: 600; src: url("${font("Geist-SemiBold.ttf")}"); }
@font-face { font-family: "Geist Mono"; font-weight: 400; src: url("${font("GeistMono-Regular.ttf")}"); }
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { background: #060a13; }
body { font-family: Geist, system-ui, sans-serif; font-weight: 500; color: #e8edf5; }
.canvas {
  position: relative; overflow: hidden;
  background:
    radial-gradient(55% 75% at 100% 0%, rgba(124,58,237,.20), rgba(124,58,237,0) 70%),
    radial-gradient(45% 70% at 0% 100%, rgba(0,212,255,.08), rgba(0,212,255,0) 70%),
    #060a13;
}
.grid {
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(148,163,184,.045) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,.045) 1px, transparent 1px);
  background-size: 50px 50px;
}
.lockup { display: flex; align-items: center; gap: .32em; }
.word { font-weight: 600; letter-spacing: -.02em; color: #f8fafc; }
.pill {
  font-family: "Geist Mono", ui-monospace, monospace; letter-spacing: .16em; color: #c3cddb;
  border: 1.5px solid #2a3548; border-radius: .4em; padding: .3em .55em .28em .7em; margin-left: .25em;
}
.sep { width: 1.5px; height: 1.1em; background: #2a3548; margin: 0 .3em 0 .45em; }
.area { color: #8492a6; }
.h1 { font-weight: 600; letter-spacing: -.035em; line-height: 1.06; color: #f8fafc; }
.grad { background: linear-gradient(90deg, #00d4ff, #7c3aed); -webkit-background-clip: text; background-clip: text; color: transparent; }
.muted { color: #8492a6; }
.chip {
  display: inline-flex; align-items: center; gap: .55em;
  font-family: "Geist Mono", ui-monospace, monospace; color: #e8edf5;
  border: 1.5px solid #2a3548; border-radius: .45em; padding: .42em .7em; background: rgba(13,19,32,.7);
}
.chip::before { content: ""; width: .38em; height: .38em; border-radius: 50%; background: #00d4ff; box-shadow: 0 0 .5em #00d4ff; }
.mono { font-family: "Geist Mono", ui-monospace, monospace; }
`;

const page = (css, body) =>
  `<!doctype html><html><head><meta charset="utf-8"><style>${BASE_CSS}${css}</style></head><body>${body}</body></html>`;

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

// A feed of release cards with the newest on top. Bars, not words: nothing
// here claims a release that did not happen.
const feed = () => {
  const rows = [
    [1, 172, 128],
    [0.62, 150, 176],
    [0.38, 184, 120],
    [0.2, 160, 150],
  ];
  return rows
    .map(([opacity, b, c], i) => {
      const top = i === 0;
      return `
      <div class="row" style="opacity:${opacity}">
        <div class="node${top ? " hot" : ""}"></div>
        <div class="card${top ? " hot" : ""}">
          <div class="bar" style="width:54px;height:8px;background:${top ? "#00d4ff" : "#2a3548"}"></div>
          <div class="bar" style="width:${b}px;height:11px;background:${top ? "#c3cddb" : "#3b4a63"}"></div>
          <div class="bar" style="width:${c}px;height:8px;background:#1f2a3d"></div>
        </div>
      </div>`;
    })
    .join("");
};

const header = page(
  `
  .copy { position: absolute; left: 440px; top: 92px; }
  .copy .lockup { font-size: 40px; }
  .copy .pill { font-size: 17px; }
  .copy .h1 { font-size: 60px; margin-top: 30px; }
  .copy .chip { font-size: 25px; margin-top: 34px; }
  .feed { position: absolute; left: 1188px; top: 64px; width: 290px; }
  .feed::before {
    content: ""; position: absolute; left: 8px; top: 30px; width: 2px; height: 330px;
    background: linear-gradient(#00d4ff, rgba(0,212,255,0));
  }
  .row { position: relative; height: 92px; }
  .node {
    position: absolute; left: 2px; top: 25px; width: 14px; height: 14px; border-radius: 50%;
    background: #060a13; border: 2px solid #2a3548;
  }
  .node.hot { background: linear-gradient(135deg, #00d4ff, #7c3aed); border: 0; box-shadow: 0 0 0 5px rgba(0,212,255,.15), 0 0 18px rgba(0,212,255,.6); }
  .card {
    position: absolute; left: 36px; top: 0; width: 240px; height: 66px; border-radius: 10px;
    background: rgba(13,19,32,.75); border: 1px solid #1a2235;
    display: flex; flex-direction: column; justify-content: center; gap: 7px; padding: 0 16px;
  }
  .card.hot { border-color: rgba(0,212,255,.45); box-shadow: 0 0 28px rgba(0,212,255,.12); }
  .bar { border-radius: 4px; }
`,
  `<div class="canvas" style="width:1500px;height:500px">
    <div class="grid"></div>
    <div class="feed">${feed()}</div>
    <div class="copy">
      <div class="lockup">${mark(46)}<span class="word">Xcelsior</span><span class="pill">AUTOMATED</span></div>
      <h1 class="h1">What's <span class="grad">new</span> in Xcelsior,<br><span class="muted">posted automatically.</span></h1>
      <div class="chip">xcelsior.ca</div>
    </div>
  </div>`,
);

const POST_CSS = `
  .copy { position: absolute; left: 104px; top: 92px; right: 104px; }
  .copy .lockup { font-size: 46px; }
  .copy .pill { font-size: 20px; }
  .mark-outline { position: absolute; left: 1040px; top: 36px; }
  .mark-badge { position: absolute; left: 1368px; top: 342px; }
`;

const intro = page(
  `${POST_CSS}
  .copy .h1 { font-size: 92px; margin-top: 72px; }
  .cols { position: absolute; left: 104px; right: 104px; top: 600px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 48px; }
  .col { border-top: 2px solid #2a3548; padding-top: 28px; }
  .col h2 { font-size: 40px; font-weight: 600; letter-spacing: -.02em; color: #f8fafc; display: flex; align-items: baseline; gap: 14px; }
  .col h2 .mono { font-size: 28px; font-weight: 400; color: #00d4ff; }
  .col p { font-size: 31px; line-height: 1.38; color: #b6c2d2; margin-top: 16px; }
  .col p .mono { font-size: .92em; color: #e8edf5; }
`,
  `<div class="canvas" style="width:1600px;height:900px">
    <div class="grid"></div>
    <div class="mark-outline">${outline(500)}</div>
    <div class="mark-badge">${badge(150)}</div>
    <div class="copy">
      <div class="lockup">${mark(56)}<span class="word">Xcelsior</span><span class="pill">AUTOMATED</span></div>
      <h1 class="h1" data-max-lines="2">What's <span class="grad">new</span> in Xcelsior,<br><span class="muted">posted automatically.</span></h1>
    </div>
    <div class="cols">
      <div class="col"><h2><span class="mono">01</span>What it posts</h2>
        <p>New features and fixes, once customers can use them.</p></div>
      <div class="col"><h2><span class="mono">02</span>Every post</h2>
        <p>Says what changed and links to where to find it.</p></div>
      <div class="col"><h2><span class="mono">03</span>Questions</h2>
        <p>This account is automated. Write to <span class="mono">support@xcelsior.ca</span></p></div>
    </div>
  </div>`,
);

const releaseCard = ({ area, title, body, url }) =>
  page(
    `${POST_CSS}
  /* Up and to the right, clear of a three-line body. */
  .mark-outline { opacity: .7; left: 1190px; top: -40px; }
  .copy .title { font-size: 96px; margin-top: 92px; max-width: 1180px; }
  .copy .body { font-size: 40px; line-height: 1.4; color: #b6c2d2; margin-top: 36px; max-width: 1120px; }
  .link { position: absolute; left: 104px; bottom: 92px; font-size: 32px; }
`,
    `<div class="canvas" style="width:1600px;height:900px">
    <div class="grid"></div>
    <div class="mark-outline">${outline(500)}</div>
    <div class="copy">
      <div class="lockup">${mark(56)}<span class="word">Xcelsior</span><span class="pill">RELEASE</span><span class="sep"></span><span class="area">${esc(area)}</span></div>
      <h1 class="h1 title" data-max-lines="2">${esc(title)}</h1>
      <p class="body" data-max-lines="3">${esc(body)}</p>
    </div>
    <div class="chip link" data-below=".body">${esc(url.replace(/^https?:\/\//, ""))}</div>
  </div>`,
  );

// X counts every link as 23 characters, whatever its length.
const xLength = (text) => text.replace(/https?:\/\/\S+/g, "x".repeat(23)).length;

const releaseText = ({ title, body, url }) => `${title}\n\n${body}\n\n${url}`;

const checkText = (name, text) => {
  const n = xLength(text);
  if (n > 280) throw new Error(`${name}: post text is ${n} characters on X, over the 280 limit`);
  console.log(`\n--- ${name} (${n}/280) ---\n${text}\n`);
};

const { values } = parseArgs({ options: { release: { type: "string" }, out: { type: "string" } } });
let jobs;
if (values.release) {
  const release = JSON.parse(readFileSync(values.release, "utf8"));
  for (const key of ["area", "title", "body", "url"]) {
    if (typeof release[key] !== "string" || !release[key].trim()) throw new Error(`release.${key} is missing`);
  }
  checkText("release post", releaseText(release));
  jobs = [{ html: releaseCard(release), width: 1600, height: 900, file: "post-release-1600x900", out: values.out ?? "." }];
} else {
  checkText("intro post", INTRO_TEXT);
  checkText("release post", releaseText(SAMPLE_RELEASE));
  jobs = [
    { html: header, width: 1500, height: 500, file: "header-1500x500", out: OUT },
    { html: intro, width: 1600, height: 900, file: "post-intro-1600x900", out: OUT },
    { html: releaseCard(SAMPLE_RELEASE), width: 1600, height: 900, file: "post-release-1600x900", out: OUT },
  ];
}

// Same override as playwright.config.ts, for a machine whose downloaded
// browser does not match the installed Playwright.
const executablePath = process.env.XCELSIOR_PLAYWRIGHT_EXECUTABLE_PATH?.trim();
const browser = await chromium.launch(executablePath ? { executablePath } : undefined);
try {
  for (const job of jobs) {
    mkdirSync(job.out, { recursive: true });
    for (const scale of [1, 2]) {
      const context = await browser.newContext({ viewport: { width: job.width, height: job.height }, deviceScaleFactor: scale });
      const tab = await context.newPage();
      await tab.setContent(job.html, { waitUntil: "load" });
      await tab.evaluate(() => document.fonts.ready);
      const problems = await tab.evaluate(() => {
        const found = [];
        for (const el of document.querySelectorAll("[data-max-lines]")) {
          const lines = Math.round(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight));
          if (lines > Number(el.dataset.maxLines)) found.push(`${el.className} runs to ${lines} lines (max ${el.dataset.maxLines})`);
        }
        for (const el of document.querySelectorAll("[data-below]")) {
          const above = document.querySelector(el.dataset.below);
          if (above && above.getBoundingClientRect().bottom > el.getBoundingClientRect().top - 32) {
            found.push(`${el.dataset.below} runs into the link`);
          }
        }
        return found;
      });
      if (problems.length) throw new Error(`${job.file}: ${problems.join("; ")}`);
      const path = join(job.out, `${job.file}${scale === 2 ? "@2x" : ""}.png`);
      await tab.screenshot({ path, clip: { x: 0, y: 0, width: job.width, height: job.height } });
      await context.close();
      console.log(`wrote ${relative(process.cwd(), path)}`);
    }
  }
} finally {
  await browser.close();
}
