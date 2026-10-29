// Hexara on stage: half-block rendering, stage composition, choreography, and
// where the stage goes in the layout.

import { describe, it, expect, vi, afterEach } from "vitest";
import { Chalk } from "chalk";
import { flipFrame, renderPixels, renderStage, FLOOR_KEY } from "../sprite-render.js";
import {
    HexaraDirector,
    currentAct,
    easeToward,
    getSeq,
    motionFor,
    STAGE_HEADROOM,
    type ActId,
} from "../useWizardAnimation.js";
import { chooseStageLayout } from "../index.js";
import { STAGE_ROWS, roamFor } from "../HexaraStage.js";
import { DANCE_FRAMES, IDLE_FRAMES, PACE_FRAMES, SPRITE_COLS } from "../../sprites/wizard/wizard-frames.js";

const truecolor = new Chalk({ level: 3 });
const strip = (s: string) => s.replace(/\x1b\[[0-9;]*m/g, "");

describe("renderPixels", () => {
    const palette = { a: "#ff0000", b: "#0000ff" };

    it("packs two pixel rows into one line of cells", () => {
        const lines = renderPixels(["ab", "ba", "a.", ".."], palette, truecolor);
        expect(lines).toHaveLength(2);
        expect(lines.map((l) => strip(l).length)).toEqual([2, 2]);
    });

    it("upper pixel is the foreground of ▀, lower the background", () => {
        const [line] = renderPixels(["a", "b"], palette, truecolor);
        expect(line).toContain("38;2;255;0;0");
        expect(line).toContain("48;2;0;0;255");
        expect(strip(line)).toBe("▀");
    });

    it("leaves transparent pixels unpainted so the terminal background shows", () => {
        expect(renderPixels([".a", ".."], palette, truecolor)[0].startsWith(" ")).toBe(true);
        expect(strip(renderPixels(["a", "."], palette, truecolor)[0])).toBe("▀");
        expect(strip(renderPixels([".", "a"], palette, truecolor)[0])).toBe("▄");
        expect(strip(renderPixels(["a", "a"], palette, truecolor)[0])).toBe("█");
    });

    it("draws a real frame at one column per pixel", () => {
        const lines = renderPixels(IDLE_FRAMES[0]);
        expect(lines).toHaveLength(Math.ceil(IDLE_FRAMES[0].length / 2));
        for (const line of lines) expect(strip(line)).toHaveLength(SPRITE_COLS);
    });
});

describe("flipFrame", () => {
    it("mirrors each row and is its own inverse", () => {
        expect(flipFrame(["ab.", "..c"])).toEqual([".ba", "c.."]);
        expect(flipFrame(flipFrame(IDLE_FRAMES[0]))).toEqual(IDLE_FRAMES[0]);
    });
});

describe("renderStage", () => {
    const scene = (over: Partial<Parameters<typeof renderStage>[0]> = {}) => ({
        frame: IDLE_FRAMES[0],
        facingLeft: false,
        width: 56,
        height: 30,
        left: 8,
        lift: 0,
        restRow: 24,
        ...over,
    });

    it("is height/2 lines of exactly the stage width", () => {
        const lines = renderStage(scene(), "#111111", "#222222", truecolor);
        expect(lines).toHaveLength(15);
        for (const line of lines) expect(strip(line)).toHaveLength(56);
    });

    it("stands him on the floor: his shoes sit directly above it", () => {
        const lines = renderStage(scene(), "#111111", "#222222", truecolor);
        // Last cell row = [frame row 24 | floor]; the cell above holds his shoes (row 23).
        const shoesRow = strip(lines[lines.length - 2]);
        expect(shoesRow.trim().length).toBeGreaterThan(0);
        expect(lines[lines.length - 1]).toContain("48;2;17;17;17"); // floor as background
    });

    it("a lift raises him clear of the floor", () => {
        const grounded = renderStage(scene(), "#111111", "#222222", truecolor);
        const lifted = renderStage(scene({ lift: 4 }), "#111111", "#222222", truecolor);
        expect(strip(lifted[lifted.length - 2]).trim()).toBe("");
        expect(strip(grounded[0]).trim()).toBe("");
        expect(lifted).not.toEqual(grounded);
    });

    it("paints a shadow on the floor", () => {
        const lines = renderStage(scene({ shadow: { from: 20, to: 30 } }), "#111111", "#222222", truecolor);
        expect(lines[lines.length - 1]).toContain("48;2;34;34;34");
    });

    it("reserves the floor key outside the generated palette", () => {
        expect(FLOOR_KEY).toMatch(/[^0-9a-zA-Z]/);
    });
});

describe("motionFor — choreography", () => {
    const path = (act: ActId, frames: number, amp: number) =>
        Array.from({ length: frames }, (_, i) => motionFor(act, i, frames, amp));

    it("pacing strolls out both ways and comes home", () => {
        const p = path("pace", PACE_FRAMES.length, 6);
        expect(Math.min(...p.map((m) => m.dx))).toBeLessThan(0);
        expect(Math.max(...p.map((m) => m.dx))).toBeGreaterThan(0);
        expect(p[0].dx).toBe(0);
        expect(Math.abs(p[p.length - 1].dx)).toBeLessThanOrEqual(2);
    });

    it("dancing hops side to side with a bounce", () => {
        const p = path("dance", DANCE_FRAMES.length, 6);
        expect(new Set(p.map((m) => Math.sign(m.dx)))).toEqual(new Set([-1, 0, 1]));
        expect(p.some((m) => m.dy > 0)).toBe(true);
        expect(p[p.length - 1]).toEqual({ dx: 0, dy: 0 });
    });

    it("celebrating jumps, levitating floats, within the stage headroom", () => {
        for (const act of ["celebrate", "levitate", "eureka", "dance"] as ActId[]) {
            const p = path(act, 10, 6);
            expect(Math.max(...p.map((m) => m.dy))).toBeGreaterThan(0);
            expect(Math.max(...p.map((m) => m.dy))).toBeLessThanOrEqual(STAGE_HEADROOM);
        }
    });

    it("never asks him past his roaming range", () => {
        for (const act of ["pace", "dance", "error"] as ActId[]) {
            for (const amp of [0, 3, 7]) {
                for (const m of path(act, 18, amp)) expect(Math.abs(m.dx)).toBeLessThanOrEqual(Math.max(1, amp));
            }
        }
    });

    it("everything else stands at home", () => {
        for (const act of ["think", "wave", "cast", "sleep", "bow", "nod", "intro", "outro"] as ActId[]) {
            for (const m of path(act, 8, 6)) expect(m).toEqual({ dx: 0, dy: 0 });
        }
    });

    it("easeToward moves at most `max` per tick and stops on target", () => {
        expect(easeToward(0, 5, 2)).toBe(2);
        expect(easeToward(4, 5, 2)).toBe(5);
        expect(easeToward(0, -5, 2)).toBe(-2);
        expect(easeToward(3, 3, 2)).toBe(3);
    });
});

describe("currentAct", () => {
    it("names the act being played", () => {
        expect(currentAct({ phase: "prelude", actIdx: 0, frameIdx: 0 })).toBe("intro");
        expect(currentAct({ phase: "prelude", actIdx: 1, frameIdx: 0 })).toBe("settle");
        expect(currentAct({ phase: "loop", actIdx: 0, frameIdx: 0 }, "idle")).toBe("pace");
        expect(currentAct({ phase: "loop", actIdx: 2, frameIdx: 0 }, "presenting")).toBe("dance");
        expect(currentAct({ phase: "exit", actIdx: 1, frameIdx: 0 })).toBe("outro");
        expect(currentAct({ phase: "branch", actIdx: 0, frameIdx: 0, branchId: "celebrate" })).toBe("celebrate");
        expect(currentAct({ phase: "done", actIdx: 0, frameIdx: 0 })).toBeNull();
    });
});

describe("HexaraDirector", () => {
    afterEach(() => vi.useRealTimers());

    const ticks = (d: HexaraDirector, n: number) => {
        for (let i = 0; i < n; i++) d.tick();
    };
    const preludeLength = getSeq({ phase: "prelude", actIdx: 0, frameIdx: 0 }).reduce((n, act) => n + act.length, 0);

    it("walks during the idle loop, turning to face the way he goes", () => {
        const d = new HexaraDirector({ animate: true });
        d.setRoam(6);
        ticks(d, preludeLength);
        const xs: number[] = [];
        const facings = new Set<boolean>();
        for (let i = 0; i < PACE_FRAMES.length; i++) {
            d.tick();
            xs.push(d.getSnapshot().x);
            facings.add(d.getSnapshot().facingLeft);
        }
        expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThanOrEqual(6);
        expect(facings).toEqual(new Set([true, false]));
        for (const x of xs) expect(x).toBeGreaterThanOrEqual(0);
        for (const x of xs) expect(x).toBeLessThanOrEqual(12);
    });

    it("plays a triggered dance, then settles", () => {
        const d = new HexaraDirector({ animate: true });
        d.setRoam(6);
        ticks(d, preludeLength);
        d.trigger("dance");
        d.tick();
        expect(d.getSnapshot().act).toBe("dance");
        // Re-triggering the move already playing must not queue a replay.
        d.trigger("dance");
        ticks(d, DANCE_FRAMES.length);
        expect(d.getSnapshot().act).not.toBe("dance");
    });

    it("finishes the farewell and calls onDone", () => {
        const onDone = vi.fn();
        const d = new HexaraDirector({ animate: true, onDone });
        ticks(d, preludeLength);
        d.setExiting(true);
        // Bounded: the current act, the recovery and the outro.
        ticks(d, 80);
        expect(onDone).toHaveBeenCalledTimes(1);
        expect(d.getSnapshot().done).toBe(true);
    });

    it("quits at once when he is not on screen", () => {
        const onDone = vi.fn();
        const hidden = new HexaraDirector({ animate: false, onDone });
        hidden.setExiting(true);
        expect(onDone).toHaveBeenCalledTimes(1);

        const offscreen = vi.fn();
        const d = new HexaraDirector({ animate: true, onDone: offscreen });
        d.setExiting(true, { immediate: true });
        expect(offscreen).toHaveBeenCalledTimes(1);
    });

    it("runs on its own clock once started, and stops cleanly", () => {
        vi.useFakeTimers();
        const d = new HexaraDirector({ animate: true });
        const seen: unknown[] = [];
        const unsubscribe = d.subscribe(() => seen.push(d.getSnapshot().frame));
        d.start();
        vi.advanceTimersByTime(2000);
        expect(seen.length).toBeGreaterThan(5);
        d.stop();
        const count = seen.length;
        vi.advanceTimersByTime(2000);
        expect(seen.length).toBe(count);
        unsubscribe();
    });
});

describe("chooseStageLayout", () => {
    it("puts the stage beside the steps on a wide terminal", () => {
        expect(chooseStageLayout(160, 40, true)).toBe("side");
    });

    it("above them on a tall, narrower one", () => {
        expect(chooseStageLayout(100, 60, true)).toBe("top");
    });

    it("leaves him out when he would not fit, or cannot be drawn", () => {
        expect(chooseStageLayout(80, 24, true)).toBe("compact");
        expect(chooseStageLayout(200, 80, false)).toBe("compact");
        expect(chooseStageLayout(160, STAGE_ROWS + 2, true)).toBe("compact");
    });

    it("stands above a short step on an ordinary terminal once the screen is measured", () => {
        // The opening question needs about a dozen rows without him.
        expect(chooseStageLayout(110, 36, true)).toBe("compact");
        expect(chooseStageLayout(110, 36, true, 12)).toBe("top");
        expect(chooseStageLayout(80, 34, true, 12)).toBe("top");
    });

    it("steps aside for a step that needs the room", () => {
        expect(chooseStageLayout(110, 36, true, 30)).toBe("compact");
        expect(chooseStageLayout(80, 24, true, 12)).toBe("compact");
        // Exactly filling the terminal would make Ink clear and redraw every frame.
        expect(chooseStageLayout(110, 12 + STAGE_ROWS + 3, true, 12)).toBe("compact");
        expect(chooseStageLayout(110, 12 + STAGE_ROWS + 4, true, 12)).toBe("top");
    });

    it("keeps the side stage on a wide terminal unless the steps alone overflow it", () => {
        expect(chooseStageLayout(160, 40, true, 20)).toBe("side");
        expect(chooseStageLayout(160, 40, true, 39)).toBe("compact");
    });

    it("gives him room to roam on either stage", () => {
        expect(roamFor(54)).toBeGreaterThan(0);
        expect(roamFor(64)).toBeGreaterThan(roamFor(54));
        expect(roamFor(SPRITE_COLS)).toBe(0);
    });
});
