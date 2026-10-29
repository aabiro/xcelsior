// Xcelsior CLI Wizard — Sprite types & message styling.
// Animation is driven by useWizardAnimation.ts (choreography sequencer).
// This module exports WizardState and STATE_COLORS for message text coloring.

export type WizardState =
    | "idle"
    | "thinking"
    | "success"
    | "error"
    | "waiting"    // slow idle — maps to sleep branch (device auth wait, payment poll)
    | "excited"    // celebratory — maps to dance branch (mode select, gpu pick)
    | "finishing"; // farewell — maps to bow branch (exit sequence)

/** State → hex color for the message text beneath the sprite */
export const STATE_COLORS: Record<WizardState, string> = {
    idle: "#00d4ff",
    thinking: "#ffcc00",
    success: "#22c55e",
    error: "#ef4444",
    waiting: "#a78bfa",   // soft purple
    excited: "#f472b6",   // pink
    finishing: "#94a3b8",  // slate
};
