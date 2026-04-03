// Xcelsior CLI Wizard — Sprite types & message styling.
// Animation is driven by useWizardAnimation.ts (choreography sequencer).
// This module exports WizardState and STATE_COLORS for message text coloring.

export type WizardState = "idle" | "thinking" | "success" | "error";

/** State → hex color for the message text beneath the sprite */
export const STATE_COLORS: Record<WizardState, string> = {
    idle: "#00d4ff",
    thinking: "#ffcc00",
    success: "#22c55e",
    error: "#ef4444",
};
