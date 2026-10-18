// theme.ts — Centralized design tokens for the Xcelsior wizard.
//
// One source of truth for the brand gradient, semantic state colors, and the
// glyph set used across every pane (gate, tasks, learn, status). Keeping these
// here means a single edit re-skins the whole TUI and guarantees consistency
// between components (Part I — brand & motion polish).

import type { WizardState } from "../sprites/wizard/wizard-sprite.js";
import { STATE_COLORS } from "../sprites/wizard/wizard-sprite.js";

export { STATE_COLORS };
export type { WizardState };

/** Brand gradient — cyan → purple → red. Used for spinners, charts, accents. */
export const BRAND_GRADIENT = [
    "#00d4ff", // cyan
    "#2ea9f8", // cyan-blue
    "#5c6ff2", // blue-purple
    "#7c3aed", // purple
    "#9c306c", // purple-red
    "#dc2626", // red
] as const;

/** Pick a brand-gradient color by fractional position 0..1 (clamped). */
export function gradientAt(t: number): string {
    if (!Number.isFinite(t)) return BRAND_GRADIENT[0];
    const clamped = Math.max(0, Math.min(1, t));
    const idx = Math.round(clamped * (BRAND_GRADIENT.length - 1));
    return BRAND_GRADIENT[idx];
}

/** Semantic colors — referenced by name instead of scattering hex literals. */
export const COLORS = {
    brand: "#00d4ff",
    brandDim: "#0ea5e9",
    accent: "#a78bfa", // Hexara purple
    accentDim: "#7c3aed",
    success: "#22c55e",
    warning: "#fbbf24",
    warningAlt: "#facc15",
    error: "#ef4444",
    gold: "#ffcc00",
    muted: "#94a3b8",
    border: "#374151",
    pink: "#f472b6",
} as const;

/** State → glyph mapping for service / task status rows. */
export const GLYPHS = {
    // task lifecycle
    todo: "☐",
    active: "▶",
    done: "■",
    failed: "✗",
    // inline status
    ok: "✓",
    cross: "✗",
    bullet: "◆",
    arrow: "▸",
    dots: "⋮",
    spinnerHint: "⟳",
    up: "↑",
    down: "↓",
    swap: "↔",
} as const;

/** Service-state → color, used by the preflight gate and status rows. */
export const SERVICE_STATE_COLORS: Record<string, string> = {
    operational: COLORS.success,
    degraded: COLORS.warning,
    down: COLORS.error,
    unknown: COLORS.muted,
};

/** Verdict → color, used by the gate header. */
export const VERDICT_COLORS: Record<string, string> = {
    operational: COLORS.success,
    degraded: COLORS.warning,
    blocked: COLORS.error,
};

/**
 * Detect whether the terminal can render 24-bit truecolor. Ink/chalk degrade
 * gracefully on their own, but charts that interpolate the gradient look muddy
 * on 16-color terminals, so we expose a capability flag callers can branch on.
 */
export function supportsTruecolor(
    env: NodeJS.ProcessEnv = process.env,
): boolean {
    const colorterm = (env.COLORTERM ?? "").toLowerCase();
    if (colorterm.includes("truecolor") || colorterm.includes("24bit")) return true;
    const term = (env.TERM ?? "").toLowerCase();
    if (term.includes("256color") || term.includes("direct")) return true;
    // iTerm, kitty, WezTerm, vscode set their own markers
    if (env.TERM_PROGRAM || env.KITTY_WINDOW_ID || env.WT_SESSION) return true;
    return false;
}
