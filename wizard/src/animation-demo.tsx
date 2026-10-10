#!/usr/bin/env node
/** A local animation preview: no accounts, network requests or setup actions. */
import React, { useEffect, useState } from "react";
import { Box, Text, render, useApp, useInput, useStdout } from "ink";
import { HexaraStage, roamFor } from "./HexaraStage.js";
import { HexaraDirector, type BranchId, type WizardMood } from "./useWizardAnimation.js";
import { spriteCapable } from "./capability.js";

const scenes: { at: number; title: string; mood: WizardMood; branch?: BranchId }[] = [
    { at: 0, title: "Arrival", mood: "idle" },
    { at: 5_000, title: "Hello", mood: "presenting", branch: "wave" },
    { at: 9_000, title: "A little victory dance", mood: "success", branch: "dance" },
    { at: 15_000, title: "Working on a spell", mood: "working", branch: "cast" },
    { at: 20_000, title: "Something went wrong", mood: "error", branch: "error" },
    { at: 24_000, title: "Ready to try again", mood: "waiting", branch: "nod" },
    { at: 28_000, title: "That worked", mood: "success", branch: "dance" },
    { at: 34_000, title: "Until next time", mood: "idle", branch: "bow" },
];

export function AnimationDemo() {
    const { exit } = useApp();
    const { stdout } = useStdout();
    const [columns, setColumns] = useState(stdout.columns || 80);
    const [title, setTitle] = useState(scenes[0].title);
    const [director] = useState(() => new HexaraDirector({ animate: spriteCapable() }));
    const [manual, setManual] = useState(false);
    const width = Math.max(1, Math.min(columns - 4, 76));
    const capable = spriteCapable();

    useEffect(() => {
        const resize = () => setColumns(stdout.columns || 80);
        stdout.on("resize", resize);
        return () => { stdout.off("resize", resize); };
    }, [stdout]);
    useEffect(() => { director.setRoam(roamFor(width)); }, [director, width]);
    useEffect(() => {
        director.onDone = exit;
        director.start();
        return () => director.stop();
    }, [director, exit]);
    useEffect(() => {
        if (manual || !capable) return;
        const timers = scenes.map((scene) => setTimeout(() => {
            setTitle(scene.title);
            director.setMood(scene.mood);
            if (scene.branch) director.trigger(scene.branch);
        }, scene.at));
        return () => timers.forEach(clearTimeout);
    }, [director, manual, capable]);
    useInput((input, key) => {
        if (input === "q" || key.escape) {
            director.setExiting(true, { immediate: !capable });
            return;
        }
        const scene = ({ d: scenes[2], c: scenes[3], e: scenes[4], r: scenes[5] } as const)[input as "d"];
        if (!scene) return;
        setManual(true);
        setTitle(scene.title);
        director.setMood(scene.mood);
        if (scene.branch) director.trigger(scene.branch);
    });
    return <Box flexDirection="column" paddingX={1} width={Math.max(1, columns - 1)}>
        <Text bold color="cyan">Hexara · animation preview</Text>
        <Text dimColor>Local preview · no onboarding actions</Text>
        {capable && width >= 36
            ? <HexaraStage director={director} width={width} caption={title} />
            : <Text>{capable ? "Widen this terminal to see Hexara." : "Open in a terminal with 256-colour support to see Hexara."}</Text>}
        <Text dimColor>d dance · c cast · e error · r recover · q exit</Text>
    </Box>;
}

// This entry is deliberately separate from the real onboarding application.
render(<AnimationDemo />);
