#!/usr/bin/env node
// Minimal Sixel + Ink test — scroll region approach
// Reserve top rows for the wizard sprite, confine Ink below.
// DECSTBM (Set Top and Bottom Margins) prevents Ink's eraseLines from
// touching the wizard area. CUP can still position outside the region.
import React, { useEffect, useRef } from "react";
import { render, Box, Text } from "ink";
import { writeSync } from "fs";
import { IDLE_FRAMES, INTRO_FRAMES, SPRITE_COLS, SPRITE_ROWS } from "./sprites/wizard/wizard-frames.js";

const FRAMES = [...INTRO_FRAMES, ...IDLE_FRAMES, ...IDLE_FRAMES, ...IDLE_FRAMES];

const WIZARD_ROW = 1;             // sprite starts at row 1
const INK_START = WIZARD_ROW + SPRITE_ROWS + 1;  // Ink starts below sprite + 1 gap

// ── Set up terminal BEFORE Ink starts ──
writeSync(1, "\x1b[2J\x1b[H");                // clear screen, cursor home
writeSync(1, `\x1b[${INK_START};999r`);        // scroll region: INK_START to bottom
writeSync(1, `\x1b[${INK_START};1H`);          // cursor into the scroll region

function App() {
  const frameRef = useRef(0);

  useEffect(() => {
    // Draw initial frame immediately
    const firstFrame = FRAMES[0]!;
    writeSync(1, `\x1b7\x1b[${WIZARD_ROW};1H` + firstFrame + `\x1b8`);

    const id = setInterval(() => {
      frameRef.current = (frameRef.current + 1) % FRAMES.length;
      const frame = FRAMES[frameRef.current]!;
      // CUP to wizard row (outside scroll region), draw Sixel, restore cursor
      writeSync(1, `\x1b7\x1b[${WIZARD_ROW};1H` + frame + `\x1b8`);
    }, 120);

    return () => {
      clearInterval(id);
      // Reset scroll region on exit
      writeSync(1, "\x1b[r");
    };
  }, []);

  return (
    <Box flexDirection="column">
      <Text color="green">Wizard Sixel Test — scroll region approach</Text>
      <Text>Frames: {FRAMES.length} animating above</Text>
      <Text dimColor>Press Ctrl+C to exit</Text>
    </Box>
  );
}

render(<App />);
