// chrome.tsx — Persistent frame: branded title bar + context-sensitive keybind
// footer (Part I). PostHog has a title bar with version + feedback and a footer
// of context keys on every screen; this is our equivalent, with Hexara's
// gradient. Both span the full terminal width and survive resize.

import React from "react";
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Box, Text } from "ink";
import { COLORS, GLYPHS, gradientAt, supportsTruecolor } from "./theme.js";
import { useTerminalWidth } from "./use-terminal-width.js";
import type { StepType } from "./wizard-flow.js";
import type { GatePhase } from "./useWizardFlow.js";

/** Read the real version from the nearest package.json (no hardcoded drift). */
function readVersion(): string {
    try {
        let dir = dirname(fileURLToPath(import.meta.url));
        for (let i = 0; i < 6; i++) {
            const p = join(dir, "package.json");
            if (existsSync(p)) {
                const v = JSON.parse(readFileSync(p, "utf-8")).version;
                if (typeof v === "string" && v) return v;
            }
            const parent = dirname(dir);
            if (parent === dir) break;
            dir = parent;
        }
    } catch {
        // fall through
    }
    return "0.0.0";
}

export const WIZARD_VERSION = readVersion();

/** A full-width gradient rule (degrades to a single accent on flat terminals). */
function GradientRule({ width }: { width: number }) {
    const n = Math.max(1, Math.min(width, 200));
    const truecolor = supportsTruecolor();
    if (!truecolor) {
        return <Text color={COLORS.brand}>{"─".repeat(n)}</Text>;
    }
    // Build contiguous spans so we don't emit one <Text> per char unnecessarily.
    const cells = Array.from({ length: n }, (_, i) => gradientAt(n > 1 ? i / (n - 1) : 0));
    return (
        <Text>
            {cells.map((c, i) => <Text key={i} color={c}>─</Text>)}
        </Text>
    );
}

export function TitleBar() {
    const width = Math.max(20, useTerminalWidth());
    const left = "✦ Xcelsior Setup";
    const right = "Ask Hexara: type ?";
    const version = `v${WIZARD_VERSION}`;
    // Spread left (title+version) and right hint across the width.
    const used = left.length + version.length + 1 + right.length;
    const gap = Math.max(2, width - used);
    return (
        <Box flexDirection="column" width={width}>
            <Box>
                <Text bold color={COLORS.brand}>{left}</Text>
                <Text color={COLORS.muted}> {version}</Text>
                <Text>{" ".repeat(gap)}</Text>
                <Text color={COLORS.accent}>{right}</Text>
            </Box>
            <GradientRule width={width} />
        </Box>
    );
}

export interface FooterContext {
    gatePhase: GatePhase;
    stepType?: StepType;
    isComplete: boolean;
    aiAvailable: boolean;
    isWorkStep: boolean;
    canRetry: boolean;
    awaitContinue: boolean;
    showAiPrompt: boolean;
    aiResponseOpen: boolean;
}

export interface Hint {
    key: string;
    label: string;
}

/**
 * Pure: the context-appropriate keybind hints for the current screen.
 * Exported for tests so footer content is verified without rendering.
 */
export function footerHints(ctx: FooterContext): Hint[] {
    const hints: Hint[] = [];

    // An open AI prompt / response owns input.
    if (ctx.aiResponseOpen) {
        return [{ key: "enter", label: "continue" }, { key: "?", label: "ask again" }];
    }
    if (ctx.showAiPrompt) {
        return [{ key: "type", label: "your question" }, { key: "esc", label: "cancel" }];
    }

    if (ctx.gatePhase !== "passed") {
        if (ctx.gatePhase === "blocked") {
            hints.push({ key: "r", label: "re-check" }, { key: "c", label: "continue anyway" });
        } else if (ctx.gatePhase === "ready") {
            hints.push({ key: "enter", label: "continue" }, { key: "r", label: "re-check" });
        } else {
            hints.push({ key: "…", label: "checking services" });
        }
        hints.push({ key: "q", label: "quit" });
        return hints;
    }

    if (ctx.isComplete) {
        return [{ key: "↑↓", label: "navigate" }, { key: "enter", label: "select" }, { key: "q", label: "exit" }];
    }

    switch (ctx.stepType) {
        case "select":
            hints.push({ key: "↑↓", label: "navigate" }, { key: "enter", label: "select" });
            break;
        case "text":
            hints.push({ key: "enter", label: "submit" });
            break;
        case "confirm":
            hints.push({ key: "y", label: "confirm" }, { key: "n", label: "cancel" });
            break;
        case "device-auth":
            hints.push({ key: "enter", label: "continue" }, { key: "m", label: "manual paste" });
            break;
        case "auto-check":
        case "auto-fetch":
        case "payment-gate":
            if (ctx.isWorkStep) hints.push({ key: "Tab/←→", label: "switch tab" });
            if (ctx.canRetry) hints.push({ key: "enter", label: "retry" }, { key: "s", label: "skip" });
            else if (ctx.awaitContinue) hints.push({ key: "enter", label: "continue" });
            break;
        default:
            break;
    }

    if (ctx.aiAvailable) hints.push({ key: "?", label: "ask Hexara" });
    hints.push({ key: "q", label: "quit" });
    return hints;
}

export function KeybindFooter({ ctx }: { ctx: FooterContext }) {
    const width = Math.max(20, useTerminalWidth());
    const hints = footerHints(ctx);
    return (
        <Box width={width} flexDirection="column">
            <GradientRule width={width} />
            <Box>
                {hints.map((h, i) => (
                    <Text key={`${h.key}-${i}`}>
                        {i > 0 ? <Text color={COLORS.muted}>  {GLYPHS.dots}  </Text> : null}
                        <Text bold color={COLORS.gold}>{h.key}</Text>
                        <Text color={COLORS.muted}> {h.label}</Text>
                    </Text>
                ))}
            </Box>
        </Box>
    );
}
