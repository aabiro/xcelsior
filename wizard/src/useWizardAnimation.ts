// useWizardAnimation — Choreographed sprite animation sequencer.
//
// Core sequence: INTRO → SETTLE → [ PACE → THINK → WAVE → CAST → RECOVERY ] loop → OUTRO
// Branch reactions: EUREKA, CELEBRATE, ERROR, SLEEP, LEVITATE, DANCE, BOW
//
// Branches are triggered externally via `branch`. When a branch is set,
// it plays at the next recovery point, then transitions per the spec:
//   EUREKA → resume loop (or → CELEBRATE)
//   CELEBRATE → SETTLE or → OUTRO
//   ERROR → THINK or PACE
//   SLEEP → SETTLE (on any activity)
//   LEVITATE → EUREKA (success) or ERROR (failure)
//   DANCE → SETTLE or → BOW
//   BOW → OUTRO or → SETTLE
//
// When `exiting` becomes true, the current act finishes, then
// RECOVERY (idle) + OUTRO play, and `done` becomes true.

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
    type Frame,
} from "./wizard-frames.js";

const FRAME_MS = 160;

/** Branch animation identifiers that can be triggered externally */
export type BranchId = "eureka" | "celebrate" | "error" | "sleep" | "levitate" | "dance" | "bow";

// Pre-built act sequences
const SETTLE = [...IDLE_FRAMES, ...IDLE_FRAMES]; // 6 frames (2 breath cycles)
const RECOVERY = [...IDLE_FRAMES];                // 3 frames (1 breath cycle)

// Repeat short acts so they're visually distinct (THINK: 5→15 = 1.8s, WAVE: 4→12 = 1.4s)
const THINK_ACT = [...THINK_FRAMES, ...THINK_FRAMES, ...THINK_FRAMES];
const WAVE_ACT = [...WAVE_FRAMES, ...WAVE_FRAMES, ...WAVE_FRAMES];

const PRELUDE: readonly Frame[][] = [INTRO_FRAMES, SETTLE];
const LOOP: readonly Frame[][] = [PACE_FRAMES, THINK_ACT, WAVE_ACT, CAST_FRAMES, RECOVERY];
const EXIT_SEQ: readonly Frame[][] = [RECOVERY, OUTRO_FRAMES];

const BRANCH_FRAMES: Record<BranchId, readonly Frame[]> = {
    eureka: EUREKA_FRAMES,
    celebrate: CELEBRATE_FRAMES,
    error: ERROR_FRAMES,
    sleep: SLEEP_FRAMES,
    levitate: LEVITATE_FRAMES,
    dance: DANCE_FRAMES,
    bow: BOW_FRAMES,
};

// What to do after a branch finishes
const BRANCH_NEXT: Record<BranchId, "loop" | "settle" | "exit"> = {
    eureka: "loop",       // resume loop
    celebrate: "settle",  // settle idle then loop
    error: "loop",        // pace it off
    sleep: "settle",      // wake up → idle
    levitate: "loop",     // resume loop (caller can chain eureka/error)
    dance: "settle",      // settle after fun
    bow: "exit",          // graceful exit
};

type Phase = "prelude" | "loop" | "exit" | "branch" | "settle-to-loop" | "done";

interface AnimState {
    phase: Phase;
    actIdx: number;
    frameIdx: number;
    branchId?: BranchId;
}

function getSeq(state: AnimState): readonly Frame[][] {
    switch (state.phase) {
        case "prelude": return PRELUDE;
        case "loop": return LOOP;
        case "exit": return EXIT_SEQ;
        case "settle-to-loop": return [SETTLE];
        case "branch": {
            const frames = state.branchId ? BRANCH_FRAMES[state.branchId] : [];
            return frames.length ? [frames as Frame[]] : [];
        }
        default: return [];
    }
}

function advance(prev: AnimState, wantExit: boolean, pendingBranch: BranchId | null): AnimState {
    if (prev.phase === "done") return prev;

    const seq = getSeq(prev);
    const act = seq[prev.actIdx];
    if (!act || act.length === 0) return { phase: "done", actIdx: 0, frameIdx: 0 };

    const nextFrame = prev.frameIdx + 1;

    // Still within current act
    if (nextFrame < act.length) {
        return { ...prev, frameIdx: nextFrame };
    }

    // Current act finished — check for exit or branch at act boundaries
    if (prev.phase === "loop" && wantExit) {
        return { phase: "exit", actIdx: 0, frameIdx: 0 };
    }

    // Check for pending branch at recovery points (end of any act in the loop)
    if (prev.phase === "loop" && pendingBranch) {
        return { phase: "branch", actIdx: 0, frameIdx: 0, branchId: pendingBranch };
    }

    // More acts in this phase
    const nextActIdx = prev.actIdx + 1;
    if (nextActIdx < seq.length) {
        return { ...prev, actIdx: nextActIdx, frameIdx: 0 };
    }

    // Phase finished — transition
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

/** Row where wizard sprite is drawn (CUP row, 1-based) */
export const WIZARD_ROW = 1;
/** Extra blank rows between sprite bottom and Ink content */
const PAD_ROWS = 0;

const LOG_FILE = "/tmp/wizard-debug.log";
function dbg(msg: string): void {
    try { appendFileSync(LOG_FILE, `${Date.now()} ${msg}\n`); } catch (_) { }
}

/**
 * Set up terminal scroll region before Ink starts.
 * Reserves the top rows for the wizard sprite + padding, confines Ink below.
 * Call this BEFORE render(<App />).
 */
export function setupWizardRegion(): void {
    const inkStart = WIZARD_ROW + SPRITE_ROWS + PAD_ROWS;
    dbg(`setup: inkStart=${inkStart} WIZARD_ROW=${WIZARD_ROW} SPRITE_ROWS=${SPRITE_ROWS}`);

    // Clear screen and move cursor home
    writeSync(1, "\x1b[2J\x1b[H");

    // Enable Sixel Display Mode (DECSDM) — prevents ghost/duplicate from sixel scrolling
    writeSync(1, "\x1b[?80h");

    // Set scroll region: wizard sprite above, Ink content below
    writeSync(1, `\x1b[${inkStart};999r`);

    // Move cursor into scroll region
    writeSync(1, `\x1b[${inkStart};1H`);
}

/** Reset scroll region to full terminal (call on exit). */
export function resetWizardRegion(): void {
    writeSync(1, "\x1b[?80l");  // Disable DECSDM
    writeSync(1, "\x1b[r");
}

export interface WizardAnimationResult {
    done: boolean;
    /** Trigger a branch animation (plays at next act boundary) */
    triggerBranch: (branch: BranchId) => void;
}

/**
 * Runs the wizard animation via direct fd writes.
 * Uses writeSync(1, ...) to paint Sixel at WIZARD_ROW (above Ink's scroll region).
 * Ink can't erase the wizard because its rendering is confined to the scroll region below.
 */
export function useWizardAnimation(exiting: boolean): WizardAnimationResult {
    const exitRef = useRef(false);
    const branchRef = useRef<BranchId | null>(null);
    const stateRef = useRef<AnimState>({
        phase: "prelude",
        actIdx: 0,
        frameIdx: 0,
    });
    const [done, setDone] = useState(false);

    // Latch exit signal (never un-set)
    if (exiting && !exitRef.current) exitRef.current = true;

    const triggerBranch = useCallback((branch: BranchId) => {
        branchRef.current = branch;
    }, []);

    useEffect(() => {
        if (done) return;

        const tick = () => {
            const prev = stateRef.current;
            if (prev.phase === "done") return;

            const next = advance(prev, exitRef.current, branchRef.current);
            if (next.phase === "branch" && branchRef.current) {
                branchRef.current = null;
            }
            stateRef.current = next;

            if (next.phase === "done") {
                setDone(true);
                return;
            }

            const seq = getSeq(next);
            const frame = seq[next.actIdx]?.[next.frameIdx];
            if (frame) {
                // Save cursor, draw frame at WIZARD_ROW, restore cursor
                const cup = `\x1b[${WIZARD_ROW};1H`;
                writeSync(1, "\x1b7" + cup + frame + "\x1b8");
            }
        };

        tick();
        const id = setInterval(tick, FRAME_MS);

        return () => {
            clearInterval(id);
        };
    }, [done]);

    return { done, triggerBranch };
}
