/** Shared motion config for Xcel AI side panel ↔ full-page handoff. */
export const AI_PANEL_SPRING = {
  type: "spring" as const,
  damping: 28,
  stiffness: 320,
  mass: 0.85,
};

export const AI_PANEL_CROSSFADE = {
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -10 },
  transition: AI_PANEL_SPRING,
} as const;