// useWizardAnimation — Choreographed sprite animation sequencer.
//
// Core sequence: INTRO → SETTLE → mood-driven loop → OUTRO
// Branch reactions: EUREKA, CELEBRATE, ERROR, SLEEP, LEVITATE, DANCE, WAVE, CAST, BOW
//
// Branches fire at act boundaries, or immediately for urgent reactions (success,
// error, dance, wave, cast) so Hexara feels responsive during long PACE/CAST acts.

import { useState, useEffect, useRef, useCallback } from "react";
import { writeSync, appendFileSync } from "fs";
import {
    INTRO_FRAMES,
    IDLE_FRAMES,
    PACE_FRAMES,
    THINK_FRAMES,
    WAVE_FRAMES,
    CAST_FRAMES,
    OUTRO_FRAMES,
    EUREKA_FRAMES,
    CELEBRATE_FRAMES,
    ERROR_FRAMES,
    SLEEP_FRAMES,
    LEVITATE_FRAMES,
    DANCE_FRAMES,
    BOW_FRAMES,
    SPRITE_ROWS,
    SPRITE_COLS,
    type Frame,
} from "../sprites/wizard/wizard-frames.js";

import { spriteCapable } from "./capability.js";
import { PEEK_FRAMES, TYPE_FRAMES, NOD_FRAMES } from "./hexara-moves.js";

const BASE_FRAME_MS = 160;

/** Branch animation identifiers — core loop acts + one-shot reactions. */
export type BranchId =
    | "eureka" | "celebrate" | "error" | "sleep"
    | "levitate" | "dance" | "bow"
    | "wave" | "cast"
    | "peek" | "type" | "nod";

/** Continuous idle-loop character — changes which acts Hexara cycles through. */
export type WizardMood = "idle" | "working" | "waiting" | "presenting" | "success" | "error";

// Pre-built act sequences — stretch short groups so every mood reads distinctly.
const SETTLE = [...IDLE_FRAMES, ...IDLE_FRAMES];
const RECOVERY = [...IDLE_FRAMES];

const THINK_ACT = [...THINK_FRAMES, ...THINK_FRAMES, ...THINK_FRAMES];
const WAVE_ACT = [...WAVE_FRAMES, ...WAVE_FRAMES, ...WAVE_FRAMES];
const CAST_ACT = [...CAST_FRAMES];
const DANCE_ACT = [...DANCE_FRAMES];
const LEVITATE_ACT = [...LEVITATE_FRAMES];

const PRELUDE: readonly Frame[][] = [INTRO_FRAMES, SETTLE];

/** Default explore loop — pace, ponder, greet, cast, breathe. */
const LOOP_DEFAULT: readonly Frame[][] = [PACE_FRAMES, THINK_ACT, WAVE_ACT, CAST_ACT, RECOVERY];
/** Long checks / API calls — more thinking and spell-casting, full levitation pass. */
const LOOP_WORKING: readonly Frame[][] = [PACE_FRAMES, THINK_ACT, CAST_ACT, LEVITATE_ACT, RECOVERY];
/** Browser/device waits — gentle breathing and drowsy sway. */
const LOOP_WAITING: readonly Frame[][] = [IDLE_FRAMES, SLEEP_FRAMES, IDLE_FRAMES, RECOVERY];
/** Selections & confirms — wave, cast, show off the dance moves. */
const LOOP_PRESENTING: readonly Frame[][] = [WAVE_ACT, CAST_ACT, DANCE_ACT, RECOVERY];
/** Milestones — celebrate, dance, eureka, wave. */
const LOOP_SUCCESS: readonly Frame[][] = [CELEBRATE_FRAMES, DANCE_ACT, EUREKA_FRAMES, WAVE_ACT, RECOVERY];
/** Failures — stumble, pace it off, think through the fix. */
const LOOP_ERROR: readonly Frame[][] = [ERROR_FRAMES, PACE_FRAMES, THINK_ACT, RECOVERY];

const LOOP_BY_MOOD: Record<WizardMood, readonly Frame[][]> = {
    idle: LOOP_DEFAULT,
    working: LOOP_WORKING,
    waiting: LOOP_WAITING,
    presenting: LOOP_PRESENTING,
    success: LOOP_SUCCESS,
    error: LOOP_ERROR,
};

const EXIT_SEQ: readonly Frame[][] = [RECOVERY, OUTRO_FRAMES];

/** Urgent branches interrupt the current loop act instead of waiting for PACE/CAST to finish. */
const URGENT_BRANCHES: ReadonlySet<BranchId> = new Set([
    "eureka", "celebrate", "error", "dance", "wave", "cast", "bow",
    "peek", "type", "nod",
]);

const FRAME_MS_BY_MOOD: Record<WizardMood, number> = {
    idle: BASE_FRAME_MS,
    working: 120,
    waiting: 200,
    presenting: 140,
    success: 130,
    error: 150,
};

export function frameMsForMood(mood: WizardMood): number {
    return FRAME_MS_BY_MOOD[mood] ?? BASE_FRAME_MS;
}

function loopForMood(mood: WizardMood): readonly Frame[][] {
    return LOOP_BY_MOOD[mood] ?? LOOP_DEFAULT;
}

const BRANCH_FRAMES: Record<BranchId, readonly Frame[]> = {
    eureka: EUREKA_FRAMES,
    celebrate: CELEBRATE_FRAMES,
    error: ERROR_FRAMES,
    sleep: SLEEP_FRAMES,
    levitate: LEVITATE_FRAMES,
    dance: DANCE_FRAMES,
    bow: BOW_FRAMES,
    wave: WAVE_FRAMES,
    cast: CAST_FRAMES,
    peek: PEEK_FRAMES,
    type: TYPE_FRAMES,
    nod: NOD_FRAMES,
};

const BRANCH_NEXT: Record<BranchId, "loop" | "settle" | "exit"> = {
    eureka: "loop",
    celebrate: "settle",
    error: "loop",
    sleep: "settle",
    levitate: "loop",
    dance: "settle",
    bow: "exit",
    wave: "settle",
    cast: "loop",
    peek: "settle",
    type: "loop",
    nod: "settle",
};

export type Phase = "prelude" | "loop" | "exit" | "branch" | "settle-to-loop" | "done";

export interface AnimState {
    phase: Phase;
    actIdx: number;
    frameIdx: number;
    branchId?: BranchId;
}

/** @internal exported for testing */
export function getSeq(state: AnimState, mood: WizardMood = "idle"): readonly Frame[][] {
    switch (state.phase) {
        case "prelude": return PRELUDE;
        case "loop": return loopForMood(mood);
        case "exit": return EXIT_SEQ;
        case "settle-to-loop": return [SETTLE];
        case "branch": {
            const frames = state.branchId ? BRANCH_FRAMES[state.branchId] : [];
            return frames.length ? [frames as Frame[]] : [];
        }
        default: return [];
    }
}

/** @internal exported for testing */
export function advance(prev: AnimState, wantExit: boolean, pendingBranch: BranchId | null, mood: WizardMood = "idle"): AnimState {
    if (prev.phase === "done") return prev;

    const seq = getSeq(prev, mood);
    const act = seq[prev.actIdx];
    if (!act || act.length === 0) return { phase: "done", actIdx: 0, frameIdx: 0 };

    const nextFrame = prev.frameIdx + 1;

    // Exit at act boundary takes priority over branch reactions.
    if (prev.phase === "loop" && wantExit && nextFrame >= act.length) {
        return { phase: "exit", actIdx: 0, frameIdx: 0 };
    }

    // Urgent reactions cut in immediately (don't wait 16-frame PACE to finish).
    if (prev.phase === "loop" && pendingBranch && URGENT_BRANCHES.has(pendingBranch) && !wantExit) {
        return { phase: "branch", actIdx: 0, frameIdx: 0, branchId: pendingBranch };
    }

    if (nextFrame < act.length) {
        return { ...prev, frameIdx: nextFrame };
    }

    if (prev.phase === "loop" && pendingBranch) {
        return { phase: "branch", actIdx: 0, frameIdx: 0, branchId: pendingBranch };
    }

    const nextActIdx = prev.actIdx + 1;
    if (nextActIdx < seq.length) {
        return { ...prev, actIdx: nextActIdx, frameIdx: 0 };
    }

    switch (prev.phase) {
        case "prelude":
            return { phase: "loop", actIdx: 0, frameIdx: 0 };
        case "loop":
            return { phase: "loop", actIdx: 0, frameIdx: 0 };
        case "settle-to-loop":
            return { phase: "loop", actIdx: 0, frameIdx: 0 };
        case "branch": {
            const next = prev.branchId ? BRANCH_NEXT[prev.branchId] : "loop";
            if (next === "exit") return { phase: "exit", actIdx: 0, frameIdx: 0 };
            if (next === "settle") return { phase: "settle-to-loop", actIdx: 0, frameIdx: 0 };
            return { phase: "loop", actIdx: 0, frameIdx: 0 };
        }
        case "exit":
            return { phase: "done", actIdx: 0, frameIdx: 0 };
        default:
            return prev;
    }
}

export const WIZARD_ROW = 2;
const PAD_ROWS = 1;

const LOG_FILE = "/tmp/wizard-debug.log";
function dbg(msg: string): void {
    try { appendFileSync(LOG_FILE, `${Date.now()} ${msg}\n`); } catch (_) { }
}

export function setupWizardRegion(): void {
    if (!spriteCapable()) {
        dbg("setup: sprite disabled (incapable terminal) — skipping scroll region");
        return;
    }
    const inkStart = WIZARD_ROW + SPRITE_ROWS + PAD_ROWS;
    dbg(`setup: inkStart=${inkStart} WIZARD_ROW=${WIZARD_ROW} SPRITE_ROWS=${SPRITE_ROWS}`);

    writeSync(1, "\x1b[2J\x1b[H");
    writeSync(1, "\x1b[?80h");
    writeSync(1, `\x1b[${inkStart};999r`);
    writeSync(1, `\x1b[${inkStart};1H`);
}

export function resetWizardRegion(): void {
    if (!spriteCapable()) return;
    writeSync(1, "\x1b[?80l");
    writeSync(1, "\x1b[r");
}

export interface WizardAnimationResult {
    done: boolean;
    triggerBranch: (branch: BranchId) => void;
}

export function useWizardAnimation(exiting: boolean, mood: WizardMood = "idle"): WizardAnimationResult {
    const exitRef = useRef(false);
    const branchRef = useRef<BranchId | null>(null);
    const moodRef = useRef<WizardMood>(mood);
    const paintRef = useRef(spriteCapable());
    const stateRef = useRef<AnimState>({
        phase: "prelude",
        actIdx: 0,
        frameIdx: 0,
    });
    const [done, setDone] = useState(false);

    if (exiting && !exitRef.current) exitRef.current = true;

    if (moodRef.current !== mood) {
        moodRef.current = mood;
        if (stateRef.current.phase === "loop") {
            stateRef.current = { phase: "loop", actIdx: 0, frameIdx: 0 };
        }
    }

    const triggerBranch = useCallback((branch: BranchId) => {
        branchRef.current = branch;
    }, []);

    useEffect(() => {
        if (done) return;

        const tick = () => {
            const prev = stateRef.current;
            if (prev.phase === "done") return;

            const next = advance(prev, exitRef.current, branchRef.current, moodRef.current);
            if (next.phase === "branch" && branchRef.current) {
                branchRef.current = null;
            }
            stateRef.current = next;

            if (next.phase === "done") {
                setDone(true);
                return;
            }

            const seq = getSeq(next, moodRef.current);
            const frame = seq[next.actIdx]?.[next.frameIdx];
            if (frame && paintRef.current) {
                const cup = `\x1b[${WIZARD_ROW};${spriteCol()}H`;
                writeSync(1, "\x1b7" + cup + frame + "\x1b8");
            }
        };

        tick();
        const ms = frameMsForMood(moodRef.current);
        const id = setInterval(tick, ms);

        return () => {
            clearInterval(id);
        };
    }, [done, mood]);

    return { done, triggerBranch };
}

function spriteCol(): number {
    const cols = process.stdout.columns || 80;
    return Math.max(1, Math.floor((cols - SPRITE_COLS) / 2) + 1);
}