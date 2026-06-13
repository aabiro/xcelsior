// StatusGate.tsx — Preflight service-health gate (Part A).
//
// Renders before the onboarding flow. Three phases:
//   checking — spinner while fetchServiceStatus runs
//   ready    — degraded verdict: show badges, Enter proceeds
//   blocked  — a required service is down: status link + "continue anyway"
// Operational verdicts never render this (the flow falls straight through).

import React from "react";
import { Box, Text, useInput } from "ink";
import { BrandSpinner } from "./steps.js";
import { COLORS, GLYPHS, SERVICE_STATE_COLORS, VERDICT_COLORS } from "./theme.js";
import type { StatusReport, ServiceState } from "./preflight.js";

export type GatePhase = "checking" | "ready" | "blocked";

interface StatusGateProps {
    phase: GatePhase;
    report: StatusReport | null;
    statusUrl: string;
    onProceed: () => void;
    onRecheck: () => void;
    onContinueAnyway: () => void;
}

const STATE_GLYPH: Record<ServiceState, string> = {
    operational: GLYPHS.ok,
    degraded: GLYPHS.dots,
    down: GLYPHS.cross,
    unknown: "·",
};

function ServiceRow({ name, state, detail, required }: StatusReport["services"][number]) {
    const color = SERVICE_STATE_COLORS[state] ?? COLORS.muted;
    return (
        <Text>
            <Text color={color}>{STATE_GLYPH[state]}</Text>
            <Text bold> {name}</Text>
            {required ? <Text color={COLORS.muted}> (required)</Text> : null}
            <Text color={COLORS.muted}> — {detail || state}</Text>
        </Text>
    );
}

export function StatusGate({ phase, report, statusUrl, onProceed, onRecheck, onContinueAnyway }: StatusGateProps) {
    useInput((input, key) => {
        if (phase === "checking") return;
        if ((input === "r" || input === "R")) { onRecheck(); return; }
        if (phase === "ready" && (key.return || input === "\r")) { onProceed(); return; }
        if (phase === "blocked" && (input === "c" || input === "C")) { onContinueAnyway(); return; }
    });

    if (phase === "checking") {
        return (
            <Box width={70} flexDirection="column">
                <BrandSpinner label="Checking Xcelsior service health…" />
            </Box>
        );
    }

    const verdict = report?.verdict ?? "blocked";
    const verdictColor = VERDICT_COLORS[verdict] ?? COLORS.error;
    const headline = verdict === "blocked"
        ? "Some required services are unavailable"
        : "Heads up — running in degraded mode";

    return (
        <Box width={70} flexDirection="column" borderStyle="round" borderColor={verdictColor} paddingX={2} paddingY={1}>
            <Text bold color={verdictColor}>
                {verdict === "blocked" ? GLYPHS.cross : GLYPHS.dots} {headline}
            </Text>
            {report?.fallback && (
                <Text color={COLORS.muted}>(status endpoint unavailable — basic probe)</Text>
            )}
            <Box flexDirection="column" marginTop={1}>
                {(report?.services ?? []).map((s) => (
                    <ServiceRow key={s.name} {...s} />
                ))}
            </Box>
            <Box flexDirection="column" marginTop={1}>
                {verdict === "blocked" ? (
                    <>
                        <Text>
                            Status & updates: <Text color={COLORS.brand} underline>{statusUrl}</Text>
                        </Text>
                        <Text color={COLORS.muted}>
                            Press <Text bold color={COLORS.warning}>r</Text> to re-check ·{" "}
                            <Text bold color={COLORS.warning}>c</Text> to continue anyway (best-effort) ·{" "}
                            <Text bold>q</Text> to quit
                        </Text>
                    </>
                ) : (
                    <Text color={COLORS.muted}>
                        Press <Text bold color={COLORS.success}>Enter</Text> to continue ·{" "}
                        <Text bold color={COLORS.warning}>r</Text> to re-check
                    </Text>
                )}
            </Box>
        </Box>
    );
}
