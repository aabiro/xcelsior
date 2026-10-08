// useWizardAnimation — Hexara's choreography: which frame plays, and where
// on the stage he stands while it does.
//
// Core sequence: INTRO → SETTLE → mood-driven loop → OUTRO
// Branch reactions: EUREKA, CELEBRATE, ERROR, SLEEP, LEVITATE, DANCE, WAVE, CAST, BOW
//
// Branches fire at act boundaries, or immediately for urgent reactions (success,
// error, dance, wave, cast) so Hexara feels responsive during long PACE/CAST acts.
//
// On top of the frames, each act has a stage motion: he strolls across the
// stage while pacing (facing the way he walks), hops side to side when he
// dances, jumps when he celebrates, floats when he levitates. `HexaraDirector`
// runs the clock and publishes a snapshot that HexaraStage draws.

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
    type Frame,
} from "../sprites/wizard/wizard-frames.js";

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

// ── Stage motion ─────────────────────────────────────────────────────

/** What is playing right now — the key the stage motion is chosen by. */
export type ActId =
    | "intro" | "settle" | "recovery" | "outro"
    | "pace" | "think" | "idle" | "sleep"
    | BranchId;

const ACT_IDS: Record<string, ActId> = {};
function nameActs(id: ActId, ...acts: readonly Frame[][]): void {
    // Acts are identified by array identity: SETTLE, RECOVERY and the *_ACT
    // repeats are distinct arrays even where their frames coincide.
    for (const act of acts) ACT_IDS[actKey(act)] = id;
}
const actKeys = new WeakMap<readonly Frame[], string>();
let nextActKey = 0;
function actKey(act: readonly Frame[]): string {
    let key = actKeys.get(act);
    if (!key) {
        key = `a${nextActKey++}`;
        actKeys.set(act, key);
    }
    return key;
}
nameActs("intro", INTRO_FRAMES);
nameActs("settle", SETTLE);
nameActs("recovery", RECOVERY);
nameActs("outro", OUTRO_FRAMES);
nameActs("pace", PACE_FRAMES);
nameActs("think", THINK_ACT, THINK_FRAMES);
nameActs("wave", WAVE_ACT, WAVE_FRAMES);
nameActs("cast", CAST_ACT, CAST_FRAMES);
nameActs("dance", DANCE_ACT, DANCE_FRAMES);
nameActs("levitate", LEVITATE_ACT, LEVITATE_FRAMES);
nameActs("idle", IDLE_FRAMES);
nameActs("sleep", SLEEP_FRAMES);
nameActs("celebrate", CELEBRATE_FRAMES);
nameActs("eureka", EUREKA_FRAMES);
nameActs("error", ERROR_FRAMES);
nameActs("bow", BOW_FRAMES);
nameActs("peek", PEEK_FRAMES);
nameActs("type", TYPE_FRAMES);
nameActs("nod", NOD_FRAMES);

/** The act being played in `state`, or null once the sequence is over. */
export function currentAct(state: AnimState, mood: WizardMood = "idle"): ActId | null {
    if (state.phase === "branch" && state.branchId) return state.branchId;
    const act = getSeq(state, mood)[state.actIdx];
    return act ? (ACT_IDS[actKey(act)] ?? "idle") : null;
}

/** Where an act wants Hexara this frame, relative to his home spot. */
export interface MotionTarget {
    /** Columns from home; negative is left. */
    dx: number;
    /** Rows above the floor. */
    dy: number;
}

/**
 * Stage choreography. `amp` is how far he may roam either side of home.
 * Every path starts and ends at home, so acts chain without a jump; the
 * director eases him toward each target, so a branch that cuts in mid-stroll
 * walks him back rather than teleporting him.
 */
export function motionFor(act: ActId | null, frameIdx: number, frameCount: number, amp: number): MotionTarget {
    const n = Math.max(1, frameCount);
    // 0 on the first frame, 1 on the last: every path is home at both ends,
    // where the frames are the shared neutral pose.
    const t = n > 1 ? frameIdx / (n - 1) : 0;
    const col = (v: number) => Math.round(v) + 0; // + 0 turns -0 into 0
    switch (act) {
        case "pace":
            // A stroll: out to the left, back past home, out to the right, home.
            return { dx: col(-amp * Math.sin(2 * Math.PI * t)), dy: 0 };
        case "dance": {
            // Two side-to-side hops, with a bounce on every other beat.
            const reach = Math.max(1, Math.round(amp * 0.7));
            return { dx: col(reach * Math.sin(4 * Math.PI * t)), dy: frameIdx % 2 === 1 && frameIdx < n - 1 ? 1 : 0 };
        }
        case "celebrate":
            // One big jump.
            return { dx: 0, dy: Math.round(2 * Math.sin(Math.PI * (frameIdx / Math.max(1, n - 1)))) };
        case "eureka":
            return { dx: 0, dy: frameIdx > 0 && frameIdx < n - 1 && frameIdx % 3 !== 0 ? 1 : 0 };
        case "levitate":
            // Rise, hover with a slow bob, settle.
            if (frameIdx === 0 || frameIdx === n - 1) return { dx: 0, dy: 0 };
            return { dx: 0, dy: frameIdx === 1 || frameIdx === n - 2 ? 1 : 2 - (frameIdx % 4 === 0 ? 1 : 0) };
        case "error":
            // A stumble: a quick shake on the spot.
            return { dx: frameIdx > 0 && frameIdx < n - 1 ? (frameIdx % 2 === 0 ? 1 : -1) : 0, dy: 0 };
        default:
            return { dx: 0, dy: 0 };
    }
}

/** Move `from` toward `to` by at most `max` per tick. */
export function easeToward(from: number, to: number, max: number): number {
    if (from === to) return from;
    return from < to ? Math.min(to, from + max) : Math.max(to, from - max);
}

// ── Director ─────────────────────────────────────────────────────────

/** Rows of headroom above the sprite so he can jump and float. */
export const STAGE_HEADROOM = 2;
/** Frame interval while exiting — the outro should not hold up a quit. */
const EXIT_FRAME_MS = 70;

export interface HexaraSnapshot {
    frame: Frame;
    /** Columns from the stage's left edge to the sprite's left edge. */
    x: number;
    /** Rows above the floor. */
    y: number;
    facingLeft: boolean;
    act: ActId | null;
    done: boolean;
}

/**
 * Runs Hexara's clock outside React so the stage can move between layouts
 * (beside or above the steps) without restarting his routine, and so a tick
 * re-renders only the stage. Subscribe with `useSyncExternalStore`.
 */
export class HexaraDirector {
    private state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: 0 };
    private mood: WizardMood = "idle";
    private pending: BranchId | null = null;
    private exiting = false;
    private animate: boolean;
    private amp = 0;
    private x = 0;
    private y = 0;
    private facingLeft = false;
    private timer: ReturnType<typeof setTimeout> | null = null;
    private listeners = new Set<() => void>();
    private snapshot: HexaraSnapshot;
    private finished = false;

    constructor(options: { animate: boolean; onDone?: () => void }) {
        this.animate = options.animate;
        this.onDone = options.onDone ?? null;
        this.snapshot = this.compose();
    }

    /** Called once the exit sequence has played (or at once when not animating). */
    onDone: (() => void) | null;

    subscribe = (listener: () => void): (() => void) => {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    };

    getSnapshot = (): HexaraSnapshot => this.snapshot;

    start(): void {
        if (this.animate && !this.timer && !this.finished) this.schedule();
    }

    stop(): void {
        if (this.timer) clearTimeout(this.timer);
        this.timer = null;
    }

    setMood(mood: WizardMood): void {
        if (mood === this.mood) return;
        this.mood = mood;
        // A new mood starts its own loop from the top rather than finishing
        // an act chosen for the old one.
        if (this.state.phase === "loop") this.state = { phase: "loop", actIdx: 0, frameIdx: 0 };
    }

    trigger(branch: BranchId): void {
        // Already on stage: a re-trigger (the step's message changed) must not
        // queue the same move again, or a busy step replays it forever.
        if (this.state.phase === "branch" && this.state.branchId === branch) return;
        this.pending = branch;
    }

    /** How far he may wander either side of home, in columns. */
    setRoam(amp: number): void {
        this.amp = Math.max(0, Math.floor(amp));
        this.x = Math.max(-this.amp, Math.min(this.amp, this.x));
        this.publish();
    }

    /** Play the farewell, then call onDone; `immediate` skips it (he is off screen). */
    setExiting(exiting: boolean, options: { immediate?: boolean } = {}): void {
        if (!exiting || this.finished) return;
        if (!this.animate || options.immediate) {
            this.exiting = true;
            this.finish();
            return;
        }
        if (this.exiting) return;
        this.exiting = true;
        // Restart the clock at the faster exit pace.
        this.stop();
        this.schedule();
    }

    private schedule(): void {
        const ms = this.exiting ? EXIT_FRAME_MS : frameMsForMood(this.mood);
        this.timer = setTimeout(() => {
            this.timer = null;
            this.tick();
            if (!this.finished) this.schedule();
        }, ms);
    }

    /** @internal advance one frame; exported behaviour is driven by start(). */
    tick(): void {
        if (this.finished) return;
        const next = advance(this.state, this.exiting, this.pending, this.mood);
        if (next.phase === "branch" && this.state.phase !== "branch") this.pending = null;
        this.state = next;
        if (next.phase === "done") {
            this.finish();
            return;
        }
        const act = currentAct(next, this.mood);
        const frames = getSeq(next, this.mood)[next.actIdx] ?? [];
        if (act === "intro" || act === "outro") {
            this.x = 0;
            this.y = 0;
            this.facingLeft = false;
        } else {
            const target = motionFor(act, next.frameIdx, frames.length, this.amp);
            const nx = easeToward(this.x, Math.max(-this.amp, Math.min(this.amp, target.dx)), 2);
            if (nx !== this.x) this.facingLeft = nx < this.x;
            this.x = nx;
            this.y = easeToward(this.y, Math.max(0, Math.min(STAGE_HEADROOM, target.dy)), 1);
        }
        this.publish();
    }

    private finish(): void {
        this.finished = true;
        this.stop();
        this.state = { phase: "done", actIdx: 0, frameIdx: 0 };
        this.publish();
        this.onDone?.();
    }

    private compose(): HexaraSnapshot {
        const seq = getSeq(this.state, this.mood);
        const frame = seq[this.state.actIdx]?.[this.state.frameIdx] ?? IDLE_FRAMES[0];
        return {
            frame,
            x: this.amp + this.x,
            y: this.y,
            facingLeft: this.facingLeft,
            act: currentAct(this.state, this.mood),
            done: this.finished,
        };
    }

    private publish(): void {
        this.snapshot = this.compose();
        for (const listener of this.listeners) listener();
    }
}
