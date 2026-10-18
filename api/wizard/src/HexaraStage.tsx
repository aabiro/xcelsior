// HexaraStage — where Hexara performs.
//
// Composes the director's current frame onto a small stage — his position,
// a floor, his shadow — renders it as half-block text, and puts his line of
// dialogue beneath. Everything here is ordinary Ink layout: no cursor
// positioning, no writes outside the render tree.

import React, { useSyncExternalStore } from "react";
import { Box, Text } from "ink";
import { IDLE_FRAMES, SPRITE_COLS } from "../sprites/wizard/wizard-frames.js";
import { renderStage, TRANSPARENT } from "./sprite-render.js";
import { STAGE_HEADROOM, type HexaraDirector } from "./useWizardAnimation.js";

const FLOOR_COLOR = "#273449";
const SHADOW_COLOR = "#56657e";

/**
 * Frame row that rests on the floor. His shoes are on row 23 of the idle pose,
 * so row 24 is the floor; frames that bob him down a pixel tuck the shoes
 * behind it, which reads as a squash rather than as sinking.
 */
const REST_ROW = 24;

/** Stage height in pixel rows: the frame above the floor, room to jump, the floor. */
const STAGE_PIXELS = REST_ROW + 2 * STAGE_HEADROOM + 2;
/** Terminal rows the stage occupies above its caption. */
export const STAGE_ROWS = STAGE_PIXELS / 2;

/** His robe hem in the idle pose: what his shadow spans, and where his body is. */
const HEM = (() => {
    const frame = IDLE_FRAMES[0];
    let widest: { from: number; to: number } | null = null;
    for (let r = REST_ROW - 3; r < REST_ROW; r++) {
        const row = [...(frame[r] ?? "")];
        const from = row.findIndex((p) => p !== TRANSPARENT);
        if (from < 0) continue;
        const to = row.length - 1 - [...row].reverse().findIndex((p) => p !== TRANSPARENT);
        if (!widest || to - from > widest.to - widest.from) widest = { from, to };
    }
    return widest ?? { from: 10, to: 23 };
})();

/**
 * Mirroring a frame flips it about the canvas centre, which moves a body that
 * is not centred on the canvas by this many columns. The stage undoes it when
 * he faces left, so he turns on the spot instead of jumping sideways.
 */
const TURN_SHIFT = SPRITE_COLS - 1 - (HEM.from + HEM.to);

/** How far he can roam either side of home on a stage this wide. */
export function roamFor(width: number): number {
    return Math.max(0, Math.floor((width - SPRITE_COLS - Math.abs(TURN_SHIFT)) / 2));
}

export interface HexaraStageProps {
    director: HexaraDirector;
    width: number;
    caption?: string;
    captionColor?: string;
}

export function HexaraStage({ director, width, caption, captionColor }: HexaraStageProps) {
    const snap = useSyncExternalStore(director.subscribe, director.getSnapshot, director.getSnapshot);
    if (snap.done) return null;

    // Facing left draws the mirrored frame TURN_SHIFT columns over so his body
    // stays put; the base offset keeps either facing on the stage.
    const base = Math.max(0, TURN_SHIFT);
    const left = Math.max(0, Math.min(width - SPRITE_COLS, base + snap.x - (snap.facingLeft ? TURN_SHIFT : 0)));
    const hemFrom = snap.facingLeft ? SPRITE_COLS - 1 - HEM.to : HEM.from;
    const hemTo = snap.facingLeft ? SPRITE_COLS - 1 - HEM.from : HEM.to;
    // The shadow tightens while he is in the air.
    const shrink = 2 * snap.y;
    const lines = renderStage(
        {
            frame: snap.frame,
            facingLeft: snap.facingLeft,
            width,
            height: STAGE_PIXELS,
            left,
            lift: 2 * snap.y,
            restRow: REST_ROW,
            shadow: snap.act === "intro" || snap.act === "outro"
                ? undefined
                : { from: left + hemFrom + shrink, to: left + hemTo - shrink },
        },
        FLOOR_COLOR,
        SHADOW_COLOR,
    );

    return (
        <Box flexDirection="column" width={width} flexShrink={0}>
            {lines.map((line, i) => (
                <Text key={i}>{line}</Text>
            ))}
            {caption ? (
                <Box width={width} justifyContent="center">
                    <Text color={captionColor} wrap="wrap">
                        {caption}
                    </Text>
                </Box>
            ) : null}
        </Box>
    );
}
