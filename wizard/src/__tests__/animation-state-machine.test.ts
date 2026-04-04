// Tests for the animation state machine — advance() and getSeq()
// Validates phase transitions, branch routing, exit handling, and loop wrapping.

import { describe, it, expect } from "vitest";
import {
    advance,
    getSeq,
    type AnimState,
    type BranchId,
    WIZARD_ROW,
} from "../useWizardAnimation.js";
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
} from "../../sprites/wizard/wizard-frames.js";

// Pre-built act sequences (mirrored from useWizardAnimation.ts)
const SETTLE_LEN = IDLE_FRAMES.length * 2;      // 6 frames
const RECOVERY_LEN = IDLE_FRAMES.length;          // 3 frames
const THINK_ACT_LEN = THINK_FRAMES.length * 3;   // 15 frames
const WAVE_ACT_LEN = WAVE_FRAMES.length * 3;     // 12 frames

const PRELUDE_ACTS = 2;   // INTRO, SETTLE
const LOOP_ACTS = 5;      // PACE, THINK_ACT, WAVE_ACT, CAST, RECOVERY
const EXIT_ACTS = 2;      // RECOVERY, OUTRO

// ── getSeq ──────────────────────────────────────────────────────────

describe("getSeq", () => {
    it("prelude returns 2 acts", () => {
        const seq = getSeq({ phase: "prelude", actIdx: 0, frameIdx: 0 });
        expect(seq).toHaveLength(PRELUDE_ACTS);
    });

    it("loop returns 5 acts", () => {
        const seq = getSeq({ phase: "loop", actIdx: 0, frameIdx: 0 });
        expect(seq).toHaveLength(LOOP_ACTS);
    });

    it("exit returns 2 acts", () => {
        const seq = getSeq({ phase: "exit", actIdx: 0, frameIdx: 0 });
        expect(seq).toHaveLength(EXIT_ACTS);
    });

    it("settle-to-loop returns 1 act", () => {
        const seq = getSeq({ phase: "settle-to-loop", actIdx: 0, frameIdx: 0 });
        expect(seq).toHaveLength(1);
    });

    it("branch with valid branchId returns 1 act", () => {
        const branches: BranchId[] = ["eureka", "celebrate", "error", "sleep", "levitate", "dance", "bow"];
        for (const id of branches) {
            const seq = getSeq({ phase: "branch", actIdx: 0, frameIdx: 0, branchId: id });
            expect(seq).toHaveLength(1);
            expect(seq[0].length).toBeGreaterThan(0);
        }
    });

    it("branch without branchId returns empty", () => {
        const seq = getSeq({ phase: "branch", actIdx: 0, frameIdx: 0 });
        expect(seq).toHaveLength(0);
    });

    it("done returns empty", () => {
        const seq = getSeq({ phase: "done", actIdx: 0, frameIdx: 0 });
        expect(seq).toHaveLength(0);
    });

    it("prelude first act is INTRO_FRAMES", () => {
        const seq = getSeq({ phase: "prelude", actIdx: 0, frameIdx: 0 });
        expect(seq[0]).toHaveLength(INTRO_FRAMES.length);
    });

    it("loop first act is PACE_FRAMES", () => {
        const seq = getSeq({ phase: "loop", actIdx: 0, frameIdx: 0 });
        expect(seq[0]).toHaveLength(PACE_FRAMES.length);
    });

    it("exit second act is OUTRO_FRAMES", () => {
        const seq = getSeq({ phase: "exit", actIdx: 0, frameIdx: 0 });
        expect(seq[1]).toHaveLength(OUTRO_FRAMES.length);
    });

    it("branch eureka returns EUREKA_FRAMES", () => {
        const seq = getSeq({ phase: "branch", actIdx: 0, frameIdx: 0, branchId: "eureka" });
        expect(seq[0]).toHaveLength(EUREKA_FRAMES.length);
    });

    it("branch bow returns BOW_FRAMES", () => {
        const seq = getSeq({ phase: "branch", actIdx: 0, frameIdx: 0, branchId: "bow" });
        expect(seq[0]).toHaveLength(BOW_FRAMES.length);
    });
});

// ── advance — basic frame advancement ───────────────────────────────

describe("advance — frame advancement", () => {
    it("advances frameIdx within same act", () => {
        const state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: 0 };
        const next = advance(state, false, null);
        expect(next.phase).toBe("prelude");
        expect(next.actIdx).toBe(0);
        expect(next.frameIdx).toBe(1);
    });

    it("advances multiple frames sequentially", () => {
        let state: AnimState = { phase: "loop", actIdx: 0, frameIdx: 0 };
        for (let i = 0; i < 5; i++) {
            state = advance(state, false, null);
        }
        expect(state.frameIdx).toBe(5);
        expect(state.phase).toBe("loop");
    });

    it("moves to next act when current act ends", () => {
        // At last frame of INTRO (act 0 in prelude)
        const state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: INTRO_FRAMES.length - 1 };
        const next = advance(state, false, null);
        expect(next.actIdx).toBe(1);
        expect(next.frameIdx).toBe(0);
        expect(next.phase).toBe("prelude");
    });

    it("done state is terminal — returns same state", () => {
        const state: AnimState = { phase: "done", actIdx: 0, frameIdx: 0 };
        const next = advance(state, false, null);
        expect(next).toEqual(state);
    });

    it("done ignores exit and branch signals", () => {
        const state: AnimState = { phase: "done", actIdx: 0, frameIdx: 0 };
        const next = advance(state, true, "eureka");
        expect(next.phase).toBe("done");
    });
});

// ── advance — phase transitions ─────────────────────────────────────

describe("advance — phase transitions", () => {
    it("prelude → loop after SETTLE (act 1) finishes", () => {
        const state: AnimState = { phase: "prelude", actIdx: 1, frameIdx: SETTLE_LEN - 1 };
        const next = advance(state, false, null);
        expect(next.phase).toBe("loop");
        expect(next.actIdx).toBe(0);
        expect(next.frameIdx).toBe(0);
    });

    it("loop wraps to start when last act finishes", () => {
        const state: AnimState = { phase: "loop", actIdx: LOOP_ACTS - 1, frameIdx: RECOVERY_LEN - 1 };
        const next = advance(state, false, null);
        expect(next.phase).toBe("loop");
        expect(next.actIdx).toBe(0);
        expect(next.frameIdx).toBe(0);
    });

    it("exit → done after OUTRO finishes", () => {
        const state: AnimState = { phase: "exit", actIdx: 1, frameIdx: OUTRO_FRAMES.length - 1 };
        const next = advance(state, false, null);
        expect(next.phase).toBe("done");
    });

    it("settle-to-loop → loop after settle finishes", () => {
        const state: AnimState = { phase: "settle-to-loop", actIdx: 0, frameIdx: SETTLE_LEN - 1 };
        const next = advance(state, false, null);
        expect(next.phase).toBe("loop");
        expect(next.actIdx).toBe(0);
    });
});

// ── advance — exit handling ─────────────────────────────────────────

describe("advance — exit handling", () => {
    it("exit at loop act boundary → exit phase", () => {
        const state: AnimState = { phase: "loop", actIdx: 0, frameIdx: PACE_FRAMES.length - 1 };
        const next = advance(state, true, null);
        expect(next.phase).toBe("exit");
        expect(next.actIdx).toBe(0);
        expect(next.frameIdx).toBe(0);
    });

    it("exit mid-frame is NOT triggered — only at act boundary", () => {
        const state: AnimState = { phase: "loop", actIdx: 0, frameIdx: 5 };
        const next = advance(state, true, null);
        expect(next.phase).toBe("loop"); // still in loop
        expect(next.frameIdx).toBe(6);
    });

    it("exit takes priority over branch at act boundary", () => {
        const state: AnimState = { phase: "loop", actIdx: 2, frameIdx: WAVE_ACT_LEN - 1 };
        const next = advance(state, true, "eureka");
        expect(next.phase).toBe("exit"); // exit wins
    });

    it("exit from any loop act transitions to exit phase", () => {
        // Test from each act position
        const actLastFrames = [
            PACE_FRAMES.length - 1,
            THINK_ACT_LEN - 1,
            WAVE_ACT_LEN - 1,
            CAST_FRAMES.length - 1,
            RECOVERY_LEN - 1,
        ];
        for (let actIdx = 0; actIdx < LOOP_ACTS; actIdx++) {
            const state: AnimState = { phase: "loop", actIdx, frameIdx: actLastFrames[actIdx] };
            const next = advance(state, true, null);
            expect(next.phase).toBe("exit");
        }
    });

    it("exit signal in prelude is ignored", () => {
        const state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: INTRO_FRAMES.length - 1 };
        const next = advance(state, true, null);
        // Should advance to next act, not transition to exit
        expect(next.phase).toBe("prelude");
        expect(next.actIdx).toBe(1);
    });
});

// ── advance — branch handling ───────────────────────────────────────

describe("advance — branch handling", () => {
    it("branch triggered at loop act boundary", () => {
        const state: AnimState = { phase: "loop", actIdx: 1, frameIdx: THINK_ACT_LEN - 1 };
        const next = advance(state, false, "eureka");
        expect(next.phase).toBe("branch");
        expect(next.branchId).toBe("eureka");
        expect(next.actIdx).toBe(0);
        expect(next.frameIdx).toBe(0);
    });

    it("branch NOT triggered mid-frame", () => {
        const state: AnimState = { phase: "loop", actIdx: 0, frameIdx: 3 };
        const next = advance(state, false, "dance");
        expect(next.phase).toBe("loop");
        expect(next.frameIdx).toBe(4);
    });

    it("eureka branch → back to loop", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: EUREKA_FRAMES.length - 1, branchId: "eureka" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("loop");
    });

    it("celebrate branch → settle-to-loop", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: CELEBRATE_FRAMES.length - 1, branchId: "celebrate" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("settle-to-loop");
    });

    it("error branch → back to loop", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: ERROR_FRAMES.length - 1, branchId: "error" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("loop");
    });

    it("sleep branch → settle-to-loop", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: SLEEP_FRAMES.length - 1, branchId: "sleep" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("settle-to-loop");
    });

    it("levitate branch → back to loop", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: LEVITATE_FRAMES.length - 1, branchId: "levitate" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("loop");
    });

    it("dance branch → settle-to-loop", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: DANCE_FRAMES.length - 1, branchId: "dance" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("settle-to-loop");
    });

    it("bow branch → exit", () => {
        const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: BOW_FRAMES.length - 1, branchId: "bow" };
        const next = advance(state, false, null);
        expect(next.phase).toBe("exit");
    });

    it("all branch types produce valid next phase", () => {
        const branches: BranchId[] = ["eureka", "celebrate", "error", "sleep", "levitate", "dance", "bow"];
        const frameLens: Record<BranchId, number> = {
            eureka: EUREKA_FRAMES.length,
            celebrate: CELEBRATE_FRAMES.length,
            error: ERROR_FRAMES.length,
            sleep: SLEEP_FRAMES.length,
            levitate: LEVITATE_FRAMES.length,
            dance: DANCE_FRAMES.length,
            bow: BOW_FRAMES.length,
        };
        for (const id of branches) {
            const state: AnimState = { phase: "branch", actIdx: 0, frameIdx: frameLens[id] - 1, branchId: id };
            const next = advance(state, false, null);
            expect(["loop", "settle-to-loop", "exit"]).toContain(next.phase);
        }
    });

    it("branch in prelude is ignored (prelude doesn't check)", () => {
        const state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: INTRO_FRAMES.length - 1 };
        const next = advance(state, false, "eureka");
        // Should advance to next prelude act, not branch
        expect(next.phase).toBe("prelude");
        expect(next.actIdx).toBe(1);
    });
});

// ── advance — full sequences ────────────────────────────────────────

describe("advance — full sequence simulation", () => {
    it("prelude plays fully from start to loop", () => {
        let state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: 0 };
        let ticks = 0;
        while (state.phase === "prelude" && ticks < 200) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("loop");
        // INTRO (16) + SETTLE (6) = 22 ticks to finish prelude
        expect(ticks).toBe(INTRO_FRAMES.length + SETTLE_LEN);
    });

    it("one complete loop cycle returns to loop start", () => {
        let state: AnimState = { phase: "loop", actIdx: 0, frameIdx: 0 };
        let ticks = 0;
        const loopLen = PACE_FRAMES.length + THINK_ACT_LEN + WAVE_ACT_LEN + CAST_FRAMES.length + RECOVERY_LEN;
        for (let i = 0; i < loopLen; i++) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("loop");
        expect(state.actIdx).toBe(0);
        expect(state.frameIdx).toBe(0);
    });

    it("exit sequence from start to done", () => {
        let state: AnimState = { phase: "exit", actIdx: 0, frameIdx: 0 };
        let ticks = 0;
        while (state.phase !== "done" && ticks < 100) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("done");
        expect(ticks).toBe(RECOVERY_LEN + OUTRO_FRAMES.length);
    });

    it("full lifecycle: prelude → loop (1 cycle) → exit → done", () => {
        let state: AnimState = { phase: "prelude", actIdx: 0, frameIdx: 0 };
        let ticks = 0;

        // Play through prelude
        while (state.phase === "prelude" && ticks < 200) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("loop");

        // Play one full loop then request exit
        const loopLen = PACE_FRAMES.length + THINK_ACT_LEN + WAVE_ACT_LEN + CAST_FRAMES.length + RECOVERY_LEN;
        for (let i = 0; i < loopLen - 1; i++) {
            state = advance(state, false, null);
            ticks++;
        }
        // Now at last frame of recovery — request exit
        state = advance(state, true, null);
        ticks++;
        expect(state.phase).toBe("exit");

        // Play through exit
        while (state.phase !== "done" && ticks < 500) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("done");
    });

    it("branch eureka plays and returns to loop", () => {
        // Start of eureka branch
        let state: AnimState = { phase: "branch", actIdx: 0, frameIdx: 0, branchId: "eureka" };
        let ticks = 0;
        while (state.phase === "branch" && ticks < 50) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("loop");
        expect(ticks).toBe(EUREKA_FRAMES.length);
    });

    it("branch bow plays and exits", () => {
        let state: AnimState = { phase: "branch", actIdx: 0, frameIdx: 0, branchId: "bow" };
        let ticks = 0;
        while (state.phase === "branch" && ticks < 50) {
            state = advance(state, false, null);
            ticks++;
        }
        expect(state.phase).toBe("exit");
        expect(ticks).toBe(BOW_FRAMES.length);
    });
});

// ── WIZARD_ROW constant ─────────────────────────────────────────────

describe("WIZARD_ROW", () => {
    it("is a positive integer", () => {
        expect(WIZARD_ROW).toBeGreaterThan(0);
        expect(Number.isInteger(WIZARD_ROW)).toBe(true);
    });
});
