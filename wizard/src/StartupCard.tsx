// StartupCard.tsx — Opening card shown during the preflight check.
//
// Mirrors PostHog's startup: a value prop, a privacy reassurance, and a
// "detecting your environment…" spinner with the detected stack. Replaces the
// bare gate spinner while services are being checked.

import React from "react";
import { Box, Text } from "ink";
import { BrandSpinner } from "./steps.js";
import { COLORS, GLYPHS } from "./theme.js";
import { describeEnvironment, type EnvInfo } from "./environment.js";

interface StartupCardProps {
    env: EnvInfo;
}

export function StartupCard({ env }: StartupCardProps) {
    return (
        <Box width={70} flexDirection="column" borderStyle="round" borderColor={COLORS.accent} paddingX={2} paddingY={1}>
            <Text bold color={COLORS.brand}>Welcome to Xcelsior</Text>
            <Box marginTop={1} flexDirection="column">
                <Text>Rent GPUs, share yours, or integrate the SDK — Hexara guides every track.</Text>
                <Text color={COLORS.muted}>Auth, benchmarks, verification, launch, and API setup from here.</Text>
            </Box>
            <Box marginTop={1}>
                <Text color={COLORS.success}>{GLYPHS.ok} </Text>
                <Text color={COLORS.muted}>Private: your </Text>
                <Text bold>.env</Text>
                <Text color={COLORS.muted}> and tokens never leave this machine.</Text>
            </Box>
            <Box marginTop={1}>
                <BrandSpinner label="Detecting your environment & checking services…" />
            </Box>
            <Box marginTop={1}>
                <Text color={COLORS.muted}>{GLYPHS.bullet} Detected: </Text>
                <Text color={COLORS.brand}>{describeEnvironment(env)}</Text>
            </Box>
        </Box>
    );
}
