import { describe, it, expect } from "vitest";
import { HEXARA_MOVE_FRAME_PLAN, PEEK_FRAMES, TYPE_FRAMES, NOD_FRAMES } from "../hexara-moves.js";

describe("HEXARA_MOVE_FRAME_PLAN", () => {
    it("documents three new moves with frame counts", () => {
        expect(HEXARA_MOVE_FRAME_PLAN.peek.frameCount).toBe(6);
        expect(HEXARA_MOVE_FRAME_PLAN.type.frameCount).toBe(6);
        expect(HEXARA_MOVE_FRAME_PLAN.nod.frameCount).toBe(4);
    });
});

describe("composed placeholder frames", () => {
    it("peek/type/nod placeholders are non-empty sixel strings", () => {
        for (const frames of [PEEK_FRAMES, TYPE_FRAMES, NOD_FRAMES]) {
            expect(frames.length).toBeGreaterThan(0);
            for (const f of frames) {
                expect(f.length).toBeGreaterThan(10);
            }
        }
    });
});