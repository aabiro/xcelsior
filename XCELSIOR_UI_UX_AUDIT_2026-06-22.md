# Xcelsior UI / UX Audit Report

Prepared for: Aaryn  
Date: June 22, 2026  
Scope: Full-experience UI/UX audit of Xcelsior.ca, scored toward a 10/10 feel, with concrete fixes. Notes cover layout, hierarchy, conversion, responsiveness, theming, navigation, and trust.

## Site

**XCELSIOR.CA** - Sovereign GPU compute, B2B

## Overall Score

**9.0 / 10**

Verdict: Genuinely strong, near-product-grade marketing site. Clear positioning, excellent information design, working light/dark themes, real responsive behavior, and a thoughtful pricing experience. A few inconsistencies hold it just short of a flawless 10.

## What It's For

Convert technical buyers, especially ML and infrastructure teams, by communicating three things quickly:

- It is the cheapest GPU compute in Canada.
- It is fully compliant and sovereign: PIPEDA, Law 25, PHIPA, and no US CLOUD Act exposure.
- It uses clean hydro power and bills in CAD.

The site nails this messaging.

## What Works Well

- Outstanding hero: a sharp, benefit-led headline, "The cheapest GPU compute in Canada - fully compliant," with a gradient emphasis on "fully compliant," a one-line value prop, primary and secondary CTAs, and three trust chips for price, PIPEDA, and hydro.
- Clear differentiation strategy: the "sovereign middle ground between expensive hyperscalers and ungoverned P2P marketplaces" framing is excellent positioning.
- The "How We Compare" table, comparing Xcelsior against AWS Canada, Vast.ai, and RunPod, is the standout asset. It is concrete, scannable, and credibility-building with real price points and check/cross marks.
- Excellent pricing page: transparent CAD pricing, spot/on-demand/reserved tiers across the full GPU fleet, including H200, H100, A100, L40S, RTX 4090, and others; three plan tiers with SLAs; and an interactive Savings Calculator. This is best-in-class for the category.
- GPU Availability page adds real product texture: live availability framing, MCP/agent integration, "Natural language to real GPUs," a three-step "pick -> provision -> pulse" flow, and a live-updating price dock.
- Theming: both dark and light modes are fully implemented and look polished. Light mode has good contrast and a tasteful gradient wash.
- Responsive behavior is strong: clean hamburger menu on mobile containing full nav, language toggle, theme toggle, Sign In, and a prominent Get Started CTA.
- Nice touches: tasteful scroll-reveal animations, a custom 404 page with Go Home / Dashboard actions, bilingual support, BETA labeling for honest expectation-setting, and a support chat widget.

## Issues & Fixes

### 1. High - Pricing inconsistency across pages

The home/hero and pricing page say "from $0.30 CAD/hr," but the GPU Availability page shows "Spot from $0.30/hr" in one place and "$0.12 CAD/hr - SPOT FROM" in the stat block.

Two different floor prices on the same site undermine the core "cheapest, transparent, no surprises" promise.

**Fix:** Define one canonical starting price and use it everywhere.

### 2. Medium - Savings calculator clarity

On the pricing calculator, "On-Demand Monthly $968.00" and "Reserved 1yr Monthly $968.00 -> $532.40" can read confusingly. The same `$968` appears twice, once as the reserved strikethrough.

**Fix:** Clearly label the struck price as "on-demand equivalent" so the 45% savings math is unambiguous.

### 3. Medium - BETA badge contrast in light mode

The "BETA" tag next to the logo becomes very faint / low-contrast in light mode.

**Fix:** Give it a solid background chip that meets contrast in light theme. Do not change the dark-theme color.

### 4. Low - Mobile trust badges clip

On the mobile hero, the bottom badge row, "From $0.30 CAD/hr - PIPEDA Compliant - Hydro-Powered," gets cut off at the right edge instead of wrapping.

**Fix:** Allow the row to wrap, or convert it to a horizontally scrollable or stacked group.

### 5. Low - Fleet availability shows "On request" for every tier

For a "Live Availability" page, all-identical "On request" statuses weaken the live feel.

**Fix:** Show real or representative availability states. Keep the live availability framing; do not rename the page away from "Live Availability."

### 6. Low - Nav path hygiene

The "GPUs" nav item routes to `/gpu-availability`; `/gpus` itself 404s. This is not user-facing when the nav link works, but it is worth fixing for SEO and shareability.

**Fix:** Add `/gpus` as a redirect to `/gpu-availability`.

### 7. Low - GPU fleet naming consistency

The comparison table and some copy reference RTX 3090 as a headline value GPU, while the pricing page leads with RTX 4090 as "Best Value."

**Fix:** Align the flagship-value story across pages.

## Path To 10/10

Unify the starting-price number everywhere. This is the single most important fix. Then clarify the calculator's strikethrough labeling, fix the light-mode BETA contrast, and make the mobile trust-badge row wrap. Address the all-"On request" availability state so the live page feels live.

After that, this is a 10/10-feel site.

## Summary

Xcelsior.ca scores **9.0 / 10**. It is an excellent B2B compute site with standout comparison and pricing UX. The priority fix is unifying the inconsistent starting price across pages.
