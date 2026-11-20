# Social images

Every social and launch image, grouped by set and then by platform. Each PNG is
exactly the size in its name; `@2x` files are the same image at double
resolution for platforms that accept larger uploads.

```
social/
├── brand/            Evergreen "Agent-Native Compute" card set
└── mcp-launch/       Xcelsior MCP launch kit (Product Hunt, Nov 2026)
    ├── x/
    ├── linkedin/
    ├── instagram/
    ├── github/
    ├── link-preview/
    ├── producthunt/
    └── screenshots/  Real output proving it is an MCP server
```

## brand/

Generated from `public/og-agent-native.svg` by `api/scripts/generate_social_cards.py`
(SVG beside each PNG). `tests/test_svg_generators_that_can_be_gated_are.py` fails if
the SVGs drift from the generator; re-shoot the PNGs after regenerating.

| File | Size | Use |
|---|---|---|
| `twitter-header-1500x500` | 1500×500 | X profile header |
| `x-post-1600x900` | 1600×900 | X feed post |
| `linkedin-banner-1584x396` | 1584×396 | LinkedIn profile banner |
| `facebook-cover-1640x624` | 1640×624 | Facebook cover |
| `instagram-post-1080` | 1080×1080 | Instagram square post |
| `instagram-portrait-1080x1350` | 1080×1350 | Instagram portrait post |
| `og-image-1200x630` | 1200×630 | Site-wide link preview |
| `profile-dark-1000`, `profile-gradient-1000` | 1000×1000 | Profile pictures |
| `profile-bot-1000` | 1000×1000 | Profile picture for the automated Xcelsior Compute account on X. `icon-gradient.svg` unchanged at 60% width on solid `#060a13`, with a white robot-head badge lower-right inside the circular crop |

## mcp-launch/

Designed on the "Xcelsior MCP Social Kit" canvas. Headline is tagline option 1 from
the Product Hunt package ("Your agent rents the GPU. You approve the bill."). The
MCP tool count is drawn into the X header, GitHub preview, gallery 4 and
`screenshots/mcp-tools-list`. [`mcp-launch/baked-in.json`](mcp-launch/baked-in.json)
records the value those images show, and `tests/test_release_facts.py` fails when the
published surface moves away from it: re-cut the images, then update that file.

These images are seen far smaller than their pixel size: X shows the 1500px header
at about 600px wide on desktop and 390px on a phone (40% / 26%), a link preview or
X post in a feed is 20–40% of its width, and a Product Hunt gallery image about 60%
on desktop. So no text is under 24px on the covers, link preview, GitHub preview and
Instagram post, or under 18px on the gallery (the boxed "MCP" tag is the one
exception, at 20–24px in capitals). The screenshots keep their verbatim output at
14–17px, as large as the longest line allows. Keep to that when re-cutting them.

| File | Size | Use |
|---|---|---|
| `x/header-1500x500` | 1500×500 | X profile header. Text clears the avatar (bottom-left). |
| `x/post-1600x900` | 1600×900 | Launch post on X |
| `linkedin/banner-1584x396` | 1584×396 | LinkedIn profile banner. Text sits right of the photo. |
| `link-preview/og-1200x630` | 1200×630 | `og:image` / `twitter:image` for xcelsior.ca/mcp, and LinkedIn/Facebook/Slack posts |
| `github/social-preview-1280x640` | 1280×640 | GitHub repo → Settings → Social preview |
| `instagram/post-1080x1350` | 1080×1350 | Instagram 4:5 post; safe for the 3:4 profile-grid crop |
| `producthunt/gallery-1-hero-1270x760` | 1270×760 | Product Hunt gallery, first image |
| `producthunt/gallery-2-how-it-works-1270x760` | 1270×760 | Connect, sign in, ask |
| `producthunt/gallery-3-guardrails-1270x760` | 1270×760 | Plan before spend, preview before delete, scoped access |
| `producthunt/gallery-4-capabilities-1270x760` | 1270×760 | What your agent can do |
| `producthunt/thumbnail-240x240` | 240×240 | Product Hunt thumbnail |

### screenshots/

Captured 18 Nov 2026 from the live service. Each command shown was run exactly as
displayed and the output is verbatim; nothing is mocked. No prices or GPU counts
appear, so they stay accurate while supply changes.

| File | Shows |
|---|---|
| `mcp-handshake-oauth-1270x760` | `/mcp/health` (server and protocol version as captured; see `baked-in.json`), the 401 + `WWW-Authenticate` OAuth challenge on `initialize`, and the protected-resource metadata |
| `mcp-registry-listing-1270x760` | The official MCP registry entry `io.github.aabiro/xcelsior`: streamable-http remote, status active |
| `mcp-tools-list-1270x760` | Every tool from the live `tools/list`, split into read-only, preview-then-confirm and other actions (counts in `baked-in.json`) |

## Runtime copy

The site serves the MCP link preview from
`/site-assets/assets/png/og-mcp-1200x630.png`, not from here: `robots.txt` disallows
`/brand-system/`, and X's crawler honours robots.txt, so a card image under this
folder would not render. `tests/test_mcp_link_preview_is_the_kit_card.py` keeps the
two files identical.
