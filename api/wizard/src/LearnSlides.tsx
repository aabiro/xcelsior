// LearnSlides.tsx — The left "Learn/Tips" pane (Part C).
//
// Cycles Xcelsior concept cards on a timer while long-running steps work,
// hiding latency behind education and marketing the product. Charts are
// rendered in the brand gradient (per-row) for the "wow" factor; on a flat
// terminal the gradient degrades to a single accent color.

import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { COLORS, gradientAt, supportsTruecolor } from "./theme.js";
import { LEARN_SLIDES, nextSlideIndex, type LearnSlide } from "./learn-content.js";

interface LearnSlidesProps {
    /** Slides to cycle (defaults to all). */
    slides?: LearnSlide[];
    /** Ms between advances (default 6000). */
    intervalMs?: number;
    /** Heading shown above the cycling card. */
    title?: string;
    /** Step-aware time expectation, shown above the card (Part C). */
    intro?: string | null;
    /** Interactive nudge surfaced during the wait (e.g. "Press A to ask Hexara"). */
    nudge?: string | null;
}

/** Render chart lines with a top→bottom brand gradient (or flat accent). */
function Chart({ lines }: { lines: string[] }) {
    const truecolor = supportsTruecolor();
    return (
        <Box flexDirection="column">
            {lines.map((line, i) => {
                const color = truecolor
                    ? gradientAt(lines.length > 1 ? i / (lines.length - 1) : 0)
                    : COLORS.brand;
                return <Text key={i} color={color}>{line}</Text>;
            })}
        </Box>
    );
}

export function LearnSlides({ slides = LEARN_SLIDES, intervalMs = 6000, title, intro, nudge }: LearnSlidesProps) {
    const [idx, setIdx] = useState(0);

    useEffect(() => {
        if (slides.length <= 1) return;
        const id = setInterval(() => {
            setIdx((i) => nextSlideIndex(i, slides.length));
        }, intervalMs);
        return () => clearInterval(id);
    }, [slides.length, intervalMs]);

    const slide = slides[Math.min(idx, slides.length - 1)];
    if (!slide) return null;

    const heading = title ?? (slide.mode === "tips" ? "Tips" : "Learn");

    return (
        <Box flexDirection="column">
            <Text bold color={COLORS.warning}>{heading}</Text>
            {intro && (
                <Box marginTop={1}>
                    <Text color={COLORS.brand}>⏱ {intro}</Text>
                </Box>
            )}
            <Box flexDirection="column" marginTop={1}>
                <Text bold color={COLORS.accent}>{slide.heading}</Text>
                <Box flexDirection="column" marginTop={1}>
                    {slide.lines.map((line, i) => (
                        <Text key={i} color={COLORS.muted}>{line}</Text>
                    ))}
                </Box>
                {slide.chart && slide.chart.length > 0 && (
                    <Box flexDirection="column" marginTop={1}>
                        {slide.chartCaption && (
                            <Text color={COLORS.brand}>{slide.chartCaption}</Text>
                        )}
                        <Chart lines={slide.chart} />
                    </Box>
                )}
            </Box>
            {nudge && (
                <Box marginTop={1}>
                    <Text color={COLORS.accent}>◆ {nudge}</Text>
                </Box>
            )}
        </Box>
    );
}
