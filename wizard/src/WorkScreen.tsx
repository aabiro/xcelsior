// WorkScreen.tsx — Two-pane "Learn + Tasks" layout (Part B).
//
// Composes the Learn slideshow (left) and the live task checklist (right) with
// an Ink row layout, gated by index.tsx to the long-running steps where there's
// latency to fill. Degrades gracefully: below ~100 columns it reflows to a
// single column (Tasks first — the actionable pane — then Learn), and it
// survives terminal resize via a width listener.

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { LearnSlides } from "./LearnSlides.js";
import { TaskListPane } from "./TaskListPane.js";
import { COLORS, GLYPHS } from "./theme.js";
import { useTerminalWidth } from "./use-terminal-width.js";
import type { WizardTask } from "./task-model.js";
import type { LearnSlide } from "./learn-content.js";

export { useTerminalWidth };

/** Below this width we reflow the two panes into one column. */
export const REFLOW_COLUMNS = 100;
const LEARN_WIDTH = 44;

/** Pure: whether the two-pane layout should collapse to one column. */
export function shouldReflow(width: number): boolean {
    return width < REFLOW_COLUMNS;
}

interface WorkScreenProps {
    tasks: WizardTask[];
    slides?: LearnSlide[];
    /** Override the slideshow interval (used by tests). */
    slideIntervalMs?: number;
    /** Rolling message log (the Status tab). */
    log?: string[];
    /** Per-item check output (the Tail logs tab). */
    tail?: string[];
    /** Step-aware time expectation for the Learn pane. */
    intro?: string | null;
    /** Interactive nudge for the Learn pane. */
    nudge?: string | null;
    /** Whether Hexara is reachable (affects the Ask Hexara tab). */
    aiAvailable?: boolean;
}

type StatusTab = "status" | "tail" | "ask";
const STATUS_TABS: { id: StatusTab; label: string }[] = [
    { id: "status", label: "Status" },
    { id: "tail", label: "Tail logs" },
    { id: "ask", label: "Ask Hexara" },
];

/** Pure: next tab id when cycling forward/back. Exported for tests. */
export function cycleTab(current: StatusTab, dir: 1 | -1): StatusTab {
    const idx = STATUS_TABS.findIndex((t) => t.id === current);
    const next = (idx + dir + STATUS_TABS.length) % STATUS_TABS.length;
    return STATUS_TABS[next].id;
}

interface StatusPaneProps {
    /** Message-level log (the Status tab). */
    log: string[];
    /** Per-item check output (the Tail logs tab). */
    tail: string[];
    /** Whether Hexara is reachable — changes the Ask Hexara tab copy. */
    aiAvailable?: boolean;
    /** Disable key handling (e.g. when an overlay owns input). */
    inputActive?: boolean;
}

/**
 * Tabbed bottom strip (Part E): Status · Tail logs, switched with Tab / ←→.
 * "Ask Hexara (?)" is shown as a hint — the ? key is handled globally.
 * Only mounted on work steps, where arrows/Tab don't collide with menus.
 */
function StatusPane({ log, tail, aiAvailable = true, inputActive = true }: StatusPaneProps) {
    const [tab, setTab] = useState<StatusTab>("status");

    useInput((_input, key) => {
        if (!inputActive) return;
        if (key.tab || key.rightArrow) setTab((t) => cycleTab(t, 1));
        else if (key.leftArrow) setTab((t) => cycleTab(t, -1));
    });

    const askLines = aiAvailable
        ? [
            "Press A or ? to ask Hexara about this step.",
            "She can read your live check results and walk you through any fix.",
        ]
        : [
            "Hexara is offline for this run.",
            "See the Status / Tail logs tabs and any on-screen fixes for guidance.",
        ];
    const lines = tab === "status" ? log.slice(-4) : tab === "tail" ? tail.slice(-6) : askLines;
    const showPlaceholder = tab !== "ask" && lines.length === 0;

    return (
        <Box flexDirection="column" marginTop={1} borderStyle="single" borderColor={COLORS.border} paddingX={1}>
            <Box>
                {STATUS_TABS.map((t, i) => (
                    <Text key={t.id}>
                        {i > 0 ? <Text color={COLORS.muted}>  </Text> : null}
                        <Text
                            bold={t.id === tab}
                            color={t.id === tab ? COLORS.gold : COLORS.muted}
                            underline={t.id === tab}
                        >
                            {t.label}
                        </Text>
                    </Text>
                ))}
            </Box>
            <Box flexDirection="column" marginTop={1}>
                {showPlaceholder ? (
                    <Text color={COLORS.muted} dimColor>
                        {tab === "tail" ? "(no check output yet)" : "(waiting…)"}
                    </Text>
                ) : (
                    lines.map((line, i) => {
                        const isCurrent = tab === "status" && i === lines.length - 1;
                        const color = tab === "ask" ? COLORS.accent : isCurrent ? COLORS.brand : COLORS.muted;
                        const glyph = tab === "ask" ? GLYPHS.bullet : isCurrent ? GLYPHS.arrow : GLYPHS.dots;
                        return (
                            <Text key={`${i}-${line}`} color={color} dimColor={tab !== "ask" && !isCurrent}>
                                {glyph} {line}
                            </Text>
                        );
                    })
                )}
            </Box>
            <Box marginTop={1}>
                <Text color={COLORS.muted} dimColor>
                    {GLYPHS.swap} <Text bold>Tab</Text>/<Text bold>←→</Text> switch tab
                </Text>
            </Box>
        </Box>
    );
}

export function WorkScreen({ tasks, slides, slideIntervalMs, log = [], tail = [], intro, nudge, aiAvailable = true }: WorkScreenProps) {
    const width = useTerminalWidth();
    const narrow = shouldReflow(width);

    if (narrow) {
        // Single column — Tasks first (actionable), then Learn.
        return (
            <Box flexDirection="column" width={Math.min(width - 2, 72)}>
                <TaskListPane tasks={tasks} />
                <Box marginTop={1}>
                    <LearnSlides slides={slides} intervalMs={slideIntervalMs} intro={intro} nudge={nudge} />
                </Box>
                <StatusPane log={log} tail={tail} aiAvailable={aiAvailable} />
            </Box>
        );
    }

    return (
        <Box flexDirection="column" width={Math.min(width - 2, 110)}>
            <Box flexDirection="row">
                <Box width={LEARN_WIDTH} flexShrink={0} marginRight={2}>
                    <LearnSlides slides={slides} intervalMs={slideIntervalMs} intro={intro} nudge={nudge} />
                </Box>
                <Box flexGrow={1}>
                    <TaskListPane tasks={tasks} />
                </Box>
            </Box>
            <StatusPane log={log} tail={tail} />
        </Box>
    );
}
