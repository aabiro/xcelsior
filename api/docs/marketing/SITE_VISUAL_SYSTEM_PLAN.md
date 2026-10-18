# Xcelsior Site Visual System Plan

Source bundle: `/Users/aaryn/xcelsior-site-redesign` on `aaryn@100.64.0.3`.

The source bundle is the visual and interaction contract for the marketing surface. The Next.js app should match its spacing, color tokens, light/dark behavior, animations, transitions, and asset usage while preserving the current production page content, links, SEO intent, live data, and product claims.

## Imported Assets

Runtime assets live under functional public paths:

- `frontend/public/site-assets/assets/**` for the marketing site logo, wordmark, icons, and PNG distribution assets.
- `frontend/public/site-assets/reference/site-system.dc.html` and `site-system.html` for local visual comparison only.
- `frontend/public/site-assets/reference/logo-system.dc.html` and `logo-system.html` for logo-system comparison only.
- `frontend/public/site-assets/reference/support.js` for those reference files.
- `frontend/public/brand-system/**` for specialty brand artifacts that are not part of the primary site asset bundle.

Mirror updates from the Mac source into a temporary directory first, then copy only the functional runtime assets into the repo:

```bash
rsync -az --delete \
  aaryn@100.64.0.3:/Users/aaryn/xcelsior-site-redesign/ \
  /tmp/xcelsior-site-source/
```

Do not expose duplicate process HTML files in `public`. Keep production asset URLs neutral and stable.

## Asset Contract

Use the site asset bundle directly. Do not replace these with approximate Lucide icons or older logo files.

Logo and wordmark:

- `assets/icon-gradient.svg`
- `assets/icon-navy.svg`
- `assets/icon-white.svg`
- `assets/wordmark-light.svg`
- `assets/wordmark-dark.svg`
- `assets/lockup-light.svg`
- `assets/lockup-dark.svg`

App icon SVGs:

- `assets/app-rounded-dark.svg`
- `assets/app-rounded-light.svg`
- `assets/app-circle-dark.svg`
- `assets/app-circle-light.svg`
- `assets/app-square-dark.svg`
- `assets/app-square-light.svg`
- `assets/app-gradient-rounded.svg`
- `assets/app-gradient-circle.svg`

Custom dark/light UI icons:

- `assets/icons/activity.dark.svg`, `activity.light.svg`
- `badge`, `bolt`, `bot`, `check-circle`, `cloud`, `coins`, `dollar`
- `gauge`, `gear`, `globe`, `gpu`, `grid`, `leaf`, `lock`
- `receipt`, `route`, `server`, `shield-check`, `shield`
- `sparkle`, `star`, `terminal`, `users`

PNG distribution assets:

- `assets/png/og-image-1200x630.png`
- `assets/png/favicon-16.png`, `favicon-32.png`, `favicon-48.png`
- `assets/png/apple-touch-icon-180.png`
- `assets/png/icon-192.png`, `icon-512.png`, `icon-maskable-512.png`
- `assets/png/icon-gradient-512.png`, `icon-navy-512.png`, `icon-white-512.png`
- `assets/png/app-gradient-rounded-512.png`
- `assets/png/twitter-header-1500x500.png`
- `assets/png/x-post-1600x900.png`
- `assets/png/linkedin-banner-1584x396.png`
- `assets/png/facebook-cover-1640x624.png`
- `assets/png/instagram-post-1080.png`
- `assets/png/instagram-portrait-1080x1350.png`
- `assets/png/profile-dark-1000.png`
- `assets/png/profile-gradient-1000.png`

Wire the PNGs into metadata, manifest, service worker notification defaults, WalletConnect metadata, and stable legacy root files such as `/favicon.ico` and `/og-image.png`.

## Theme Tokens

Copy the source token block exactly into `frontend/src/components/marketing/marketing-theme.css`. The implementation uses `data-theme="dark"` and `data-theme="light"` on `.marketing-site`, not Tailwind approximations.

Dark tokens:

```css
--bg:#05070e; --panel:rgba(5,7,14,0.6); --panel-2:rgba(8,11,20,0.5);
--text:#f0f4fa; --text-2:#dbe2ec; --text-3:#9aa7bd; --text-4:#7a879d; --text-5:#5a6578;
--line:rgba(255,255,255,0.06); --edge:rgba(255,255,255,0.12); --grid:rgba(255,255,255,0.028);
--surface-1:rgba(255,255,255,0.02); --surface-2:rgba(255,255,255,0.05); --surface-3:rgba(255,255,255,0.1);
--cyan:#38dbff; --violet:#9b6dff; --violet-2:#b79dff; --green:#34d399; --gold:#ffbf5e; --coral:#ff8f8f;
--mesh-a:rgba(56,219,255,0.10); --mesh-b:rgba(155,109,255,0.15); --mesh-c:rgba(155,109,255,0.05);
--cta-bg:#ffffff; --cta-fg:#0b0f19;
--shadow-hero:0 40px 90px rgba(0,0,0,0.55),inset 0 1px 0 rgba(255,255,255,0.14);
--shadow-cta:0 12px 34px rgba(0,0,0,0.3);
--shadow-panel:0 24px 60px -30px rgba(0,0,0,0.6);
```

Light tokens:

```css
--bg:#e7ebf5; --panel:rgba(255,255,255,0.82); --panel-2:rgba(255,255,255,0.7);
--text:#0f1830; --text-2:#2a3550; --text-3:#4d5975; --text-4:#6c7891; --text-5:#939eb4;
--line:rgba(16,26,48,0.10); --edge:rgba(16,26,48,0.15); --grid:rgba(16,26,48,0.045);
--surface-1:rgba(255,255,255,0.66); --surface-2:rgba(255,255,255,0.9); --surface-3:rgba(255,255,255,0.6);
--cyan:#0e7d95; --violet:#6d28d9; --violet-2:#6d28d9; --green:#047d5a; --gold:#b45309; --coral:#be123c;
--mesh-a:rgba(34,197,235,0.20); --mesh-b:rgba(139,80,245,0.20); --mesh-c:rgba(236,72,153,0.11);
--cta-bg:linear-gradient(100deg,#0e7d95,#6d28d9); --cta-fg:#ffffff;
--shadow-hero:0 34px 74px -26px rgba(23,37,84,0.30),0 12px 26px -14px rgba(23,37,84,0.18),inset 0 1px 0 rgba(255,255,255,0.95);
--shadow-cta:0 16px 32px -10px rgba(109,40,217,0.42);
--shadow-panel:0 20px 46px -24px rgba(23,37,84,0.24),0 6px 16px -12px rgba(23,37,84,0.12);
```

The theme toggle must match the source:

- 62x32 rounded pill
- 24x24 gradient knob
- dark knob transform `translateX(0px)`
- light knob transform `translateX(30px)`
- persisted in `localStorage` under `xcelsior-theme`

## Animations

Carry these keyframes unchanged:

- `auroraA`
- `auroraB`
- `siteShimmer`
- `blink`
- `floaty`
- `barflow`
- `heroUp`

The hero card keeps `floaty 7s ease-in-out infinite`. GPU telemetry bars keep `barflow 1.4s linear infinite`. Status dots keep the blink pulse.

## Route Mapping

Split the source views into route-aware React while preserving the rendered hierarchy and style values.

| Source view | Next route | Implementation target |
|---|---|---|
| home | `/` | `frontend/src/app/(marketing)/page.tsx` |
| features | `/features` | current features content, restyled with site rails/cards/type |
| pricing | `/pricing` | current pricing content/live data, restyled with site rails/cards/tables |
| dashboard style | `/dashboard` | adapt real dashboard shell/cards without replacing authenticated functionality |

The marketing navbar routes to real URLs:

- Home -> `/`
- Features -> `/features`
- Pricing -> `/pricing`
- GPUs -> `/gpu-availability`
- MCP -> `/mcp`
- Download -> `/download`
- Docs -> `https://docs.xcelsior.ca`
- Sign In -> `/login`

## Component Plan

Use the existing marketing component structure:

```text
frontend/src/components/marketing/
  MarketingShell.tsx
  marketing-theme.css
  navbar.tsx
  footer.tsx
```

Route content stays in `frontend/src/app/(marketing)/**` so the repo layout remains conventional.

Rules:

- Preserve source style values as CSS class declarations or React style objects. Do not reinterpret the design with Tailwind colors.
- Use current production page content as the content contract. The source controls visual form; existing pages control copy, links, live data, SEO claims, and section intent.
- Keep the 1240px max width, 28px outer padding, 66px nav height, and bordered vertical content rails.
- Preserve the hero grid: `grid-template-columns:minmax(0,1fr) 300px`, `gap:32px`, `padding:56px 48px 52px`.
- Preserve the KPI strip and section borders.
- Keep the comparison table, fleet table, AI Compute Access Fund callout, and CTA in the source order.
- Use custom site assets for brand and icons.

## Implementation Order

1. Mirror the Mac source into a temporary folder and copy functional runtime assets into `frontend/public/site-assets`.
2. Add `SITE_ASSETS`, `BRAND_ASSETS`, and `siteIcon`, then wire metadata/PWA/social PNGs to the site asset bundle.
3. Add `Geist_Mono` in `frontend/src/app/layout.tsx` and include both font variables on `<body>`.
4. Add `marketing-theme.css` with the exact token and keyframe blocks from the source.
5. Replace marketing shell/navbar/footer styling with the site shell, using real Next links.
6. Restyle `/`, `/features`, and `/pricing` with the source visual system while keeping current production content and data.
7. Apply the visual language to real dashboard shell/cards without removing auth, routing, live data, or app controls.
8. Run visual diff screenshots against the local reference files in both dark and light themes.
9. Run `npm test`, targeted lint, and `npm run build`.

## Verification

Required checks:

```bash
cd frontend
npm test -- src/__tests__/pwa-manifest.test.ts
npx eslint src/components/marketing src/app/layout.tsx src/app/manifest.ts src/app/sw.ts
npm run build
```

Required visual captures:

- Reference source dark: `frontend/public/site-assets/reference/site-system.dc.html`
- Next `/` dark and light
- Next `/features` dark and light
- Next `/pricing` dark and light

Compare at:

- 1440x4000 desktop
- 1024x1600 tablet
- 430x1200 mobile

Acceptance:

- No missing assets.
- No placeholder icons.
- No legacy visual styling visible on `/`, `/features`, or `/pricing`; current production content remains present.
- No Tailwind color approximations in site-system surfaces.
- Theme toggle changes every token-driven surface.
- Animations run continuously where the source runs them.
- Build and lint pass.
