// Tests for wizard sprite — choreographed Sixel frame system

import { describe, it, expect } from "vitest";
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
    SPRITE_COLS,
    SPRITE_ROWS,
    SPRITE_PX,
    type Frame,
} from "../../sprites/wizard/wizard-frames.js";
import {
    STATE_COLORS,
    type WizardState,
} from "../../sprites/wizard/wizard-sprite.js";

const ALL_STATES: WizardState[] = ["idle", "thinking", "success", "error", "waiting", "excited", "finishing"];
const CORE_GROUPS: readonly Frame[][] = [
    INTRO_FRAMES, IDLE_FRAMES, PACE_FRAMES, THINK_FRAMES,
    WAVE_FRAMES, CAST_FRAMES, OUTRO_FRAMES,
];
const BRANCH_GROUPS: readonly Frame[][] = [
    EUREKA_FRAMES, CELEBRATE_FRAMES, ERROR_FRAMES, SLEEP_FRAMES,
    LEVITATE_FRAMES, DANCE_FRAMES, BOW_FRAMES,
];
const ALL_GROUPS: readonly Frame[][] = [...CORE_GROUPS, ...BRANCH_GROUPS];
const ALL_FLAT: Frame[] = ALL_GROUPS.flat();

describe("wizard-frames", () => {
    it("INTRO has 16 frames", () => expect(INTRO_FRAMES).toHaveLength(16));
    it("IDLE has 3 frames", () => expect(IDLE_FRAMES).toHaveLength(3));
    it("PACE has 18 frames", () => expect(PACE_FRAMES).toHaveLength(18));
    it("THINK has 7 frames", () => expect(THINK_FRAMES).toHaveLength(7));
    it("WAVE has 6 frames", () => expect(WAVE_FRAMES).toHaveLength(6));
    it("CAST has 10 frames", () => expect(CAST_FRAMES).toHaveLength(10));
    it("OUTRO has 16 frames", () => expect(OUTRO_FRAMES).toHaveLength(16));

    // Branch groups
    it("EUREKA has 7 frames", () => expect(EUREKA_FRAMES).toHaveLength(7));
    it("CELEBRATE has 8 frames", () => expect(CELEBRATE_FRAMES).toHaveLength(8));
    it("ERROR has 8 frames", () => expect(ERROR_FRAMES).toHaveLength(8));
    it("SLEEP has 8 frames", () => expect(SLEEP_FRAMES).toHaveLength(8));
    it("LEVITATE has 10 frames", () => expect(LEVITATE_FRAMES).toHaveLength(10));
    it("DANCE has 10 frames", () => expect(DANCE_FRAMES).toHaveLength(10));
    it("BOW has 8 frames", () => expect(BOW_FRAMES).toHaveLength(8));

    it("every frame is a non-empty string", () => {
        for (const frame of ALL_FLAT) {
            expect(typeof frame).toBe("string");
            expect(frame.length).toBeGreaterThan(0);
        }
    });

    it("frames are Sixel sequences (start with DCS, end with ST)", () => {
        for (const frame of ALL_FLAT) {
            // DCS = ESC P, ST = ESC backslash
            expect(frame.startsWith("\x1bP")).toBe(true);
            expect(frame.endsWith("\x1b\\")).toBe(true);
        }
    });

    it("non-transparent frames contain Sixel color registers", () => {
        for (const frame of ALL_FLAT) {
            // Skip empty Sixel (all-transparent frames like outro-8)
            if (frame.length < 30) continue;
            // Color registers look like #N;2;R;G;B
            expect(frame).toMatch(/#\d+;2;\d+;\d+;\d+/);
        }
    });

    it("idle frames exist and are valid Sixel", () => {
        for (const frame of IDLE_FRAMES) {
            expect(frame.startsWith("\x1bP")).toBe(true);
            expect(frame.length).toBeGreaterThan(50);
        }
    });

    it("sprite layout constants are positive", () => {
        expect(SPRITE_COLS).toBeGreaterThan(0);
        expect(SPRITE_ROWS).toBeGreaterThan(0);
        expect(SPRITE_PX.w).toBeGreaterThan(0);
        expect(SPRITE_PX.h).toBeGreaterThan(0);
    });

    it("total frames equal 135", () => {
        expect(ALL_FLAT.length).toBe(135);
    });
});

describe("wizard-sprite", () => {
    it("defines hex colors for all 7 states", () => {
        for (const state of ALL_STATES) {
            expect(STATE_COLORS[state]).toMatch(/^#[0-9a-f]{6}$/i);
        }
    });
});
